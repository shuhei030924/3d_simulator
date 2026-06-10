"""コマンドラインインターフェース（GUI 非依存のヘッドレス実行）。

レシピ JSON または組み込みプリセットをシミュレートし、計測レポートを
標準出力に表示する。任意で STL / 断面 PNG を書き出す。

例:
    python -m semisim --preset "MOSFET フロー"
    python -m semisim recipe.json --report out.txt --stl out.stl
    python -m semisim --list-presets
"""
from __future__ import annotations

import argparse
import json
import sys

from . import metrology, presets
from .recipe import Recipe


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="semisim",
        description="半導体プロセス 3D シミュレータ（ヘッドレス実行）",
    )
    p.add_argument("recipe", nargs="?", help="レシピ JSON ファイルのパス")
    p.add_argument("--preset", help="組み込みプリセット名を実行")
    p.add_argument(
        "--list-presets", action="store_true", help="利用可能なプリセット名を一覧表示"
    )
    p.add_argument("--report", metavar="PATH", help="計測レポートをテキスト保存")
    p.add_argument(
        "--json-report",
        metavar="PATH",
        help="計測サマリを JSON 保存（プログラム解析用）",
    )
    p.add_argument("--stl", metavar="PATH", help="固体形状を STL 出力（要 pyvista）")
    p.add_argument(
        "--png",
        metavar="PATH",
        help="中央断面(Y)を PNG 出力（要 matplotlib）",
    )
    p.add_argument(
        "--csv-column",
        metavar="PATH",
        help="中央列の縦方向材料スタックを CSV 出力（依存なし）",
    )
    p.add_argument(
        "--gif",
        metavar="PATH",
        help="工程進行の断面アニメーション GIF を出力（要 matplotlib/Pillow）",
    )
    p.add_argument(
        "--gif-axis",
        choices=["X", "Y", "Z"],
        default="Y",
        help="GIF の断面軸（既定: Y = 中央 XZ 断面）",
    )
    p.add_argument(
        "--gif-fps",
        type=float,
        default=1.25,
        help="GIF のフレーム毎秒（既定: 1.25 = 1 工程 0.8 秒）",
    )
    p.add_argument(
        "--no-resist",
        action="store_true",
        help="STL/PNG 出力時にレジストを除外",
    )
    p.add_argument(
        "--hide",
        metavar="MATERIALS",
        help="STL/PNG 出力時に非表示にする材料名（カンマ区切り）。例: 'oxide,poly'",
    )
    p.add_argument(
        "--sweep",
        metavar="SPEC",
        help=(
            "パラメータ採引: 'INDEX.FIELD:START:STOP:STEP' 形式。"
            "例: '4.depth_um:0.3:0.7:0.1' で steps[4].depth_um を掛け替えて"
            "複数実行し、CSV を標準出力に出す"
        ),
    )
    p.add_argument(
        "--sweep2",
        metavar="SPEC_A,SPEC_B",
        help=(
            "2 パラメータ同時採引（実験計画法）。'A,B' とカンマで 2 つの "
            "'INDEX.FIELD:START:STOP:STEP' を並べる。全組合せを実行し、"
            "各指標を CSV（標準出力）に出す"
        ),
    )
    p.add_argument(
        "--thermal-budget",
        action="store_true",
        help="熱工程のサーマルバジェット（実効拡散長）を表示して終了",
    )
    return p


def _load_recipe(args: argparse.Namespace) -> Recipe:
    if args.preset:
        return presets.build(args.preset)
    return Recipe.load(args.recipe)


def _hidden_ids(spec: str | None) -> list:
    """'oxide,poly' のような材料名カンマ区切りを材料 ID リストに変換する。"""
    if not spec:
        return []
    from . import materials

    ids = []
    for name in spec.split(","):
        name = name.strip()
        if not name:
            continue
        ids.append(materials.get(name).id)
    return ids


def _export_stl(wafer, path: str, include_resist: bool, hidden_ids=None) -> None:
    from . import visualize  # 遅延 import（pyvista 依存を必須にしない）

    visualize.export_stl(
        wafer, path, include_resist=include_resist, hidden_ids=hidden_ids
    )


def _export_png(wafer, path: str, include_resist: bool, hidden_ids=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import visualize

    plane, ww, hh = visualize.slice_2d(
        wafer,
        "Y",
        wafer.config.ny // 2,
        include_resist=include_resist,
        hidden_ids=hidden_ids,
    )
    cmap, norm = visualize.material_listed_cmap()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.imshow(
        plane,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[0, ww, 0, hh],
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parse_sweep(spec: str) -> tuple[int, str, list[float]]:
    """'INDEX.FIELD:START:STOP:STEP' を (index, field, 値リスト) に解釈する。"""
    try:
        head, start, stop, step = spec.split(":")
        idx_str, field = head.split(".", 1)
        index = int(idx_str)
        start_f, stop_f, step_f = float(start), float(stop), float(step)
    except ValueError as exc:
        raise ValueError(
            f"--sweep の形式が不正です: {spec!r}"
            "（'INDEX.FIELD:START:STOP:STEP' を期待）"
        ) from exc
    if step_f <= 0:
        raise ValueError("--sweep の STEP は正値である必要があります。")
    values: list[float] = []
    v = start_f
    # 浮動小数誤差を考慮して stop をわずかに超えるまで含める
    while v <= stop_f + step_f * 1e-6:
        values.append(round(v, 9))
        v += step_f
    return index, field, values


def _run_sweep(recipe, spec: str) -> int:
    """指定工程のパラメータを掛け替えて連続実行し、CSV を出力する。"""
    index, field, values = _parse_sweep(spec)
    if not 0 <= index < len(recipe.steps):
        print(f"エラー: 工程番号 {index} が範囲外です。", file=sys.stderr)
        return 1
    base = recipe.steps[index].params_dict()
    if field not in base:
        print(
            f"エラー: 工程[{index}] にパラメータ {field!r} がありません。"
            f"利用可能: {', '.join(base)}",
            file=sys.stderr,
        )
        return 1
    proc_type = recipe.steps[index].type
    proc_cls = type(recipe.steps[index])
    print(f"{field},solid_fraction,step_height_um,surface_roughness_um,cmp_uniformity_pct")
    for val in values:
        params = dict(base)
        params[field] = val
        params["type"] = proc_type
        recipe.steps[index] = proc_cls._from_params(params)
        recipe.invalidate(index)
        wafer = recipe.simulate()
        s = metrology.summary(wafer)
        print(
            f"{val},{s['solid_fraction']:.6f},{s['step_height_um']:.4f},"
            f"{s['surface_roughness_um']:.4f},{s['cmp_uniformity_pct']:.4f}"
        )
    return 0


def _set_step_param(recipe, index: int, field: str, val: float) -> None:
    """工程 index のパラメータ field を val に掛け替える（キャッシュ無効化込み）。"""
    base = recipe.steps[index].params_dict()
    params = dict(base)
    params[field] = val
    params["type"] = recipe.steps[index].type
    recipe.steps[index] = type(recipe.steps[index])._from_params(params)
    recipe.invalidate(index)


def _run_sweep2(recipe, spec: str) -> int:
    """2 パラメータの全組合せを実行し、CSV を出力する（2D 採引）。"""
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"--sweep2 はカンマ区切りで 2 つの仕様が必要です: {spec!r}"
        )
    ia, fa, va = _parse_sweep(parts[0])
    ib, fb, vb = _parse_sweep(parts[1])
    for idx, fld in ((ia, fa), (ib, fb)):
        if not 0 <= idx < len(recipe.steps):
            raise ValueError(f"工程番号 {idx} が範囲外です。")
        if fld not in recipe.steps[idx].params_dict():
            raise ValueError(f"工程[{idx}] にパラメータ {fld!r} がありません。")
    print(f"{fa},{fb},solid_fraction,step_height_um,surface_roughness_um,cmp_uniformity_pct")
    for x in va:
        _set_step_param(recipe, ia, fa, x)
        for y in vb:
            _set_step_param(recipe, ib, fb, y)
            wafer = recipe.simulate()
            s = metrology.summary(wafer)
            print(
                f"{x},{y},{s['solid_fraction']:.6f},{s['step_height_um']:.4f},"
                f"{s['surface_roughness_um']:.4f},{s['cmp_uniformity_pct']:.4f}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        for name in presets.available():
            print(name)
        return 0

    if not args.preset and not args.recipe:
        parser.error("レシピファイルまたは --preset を指定してください。")

    try:
        recipe = _load_recipe(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.sweep:
        try:
            return _run_sweep(recipe, args.sweep)
        except ValueError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1

    if args.sweep2:
        try:
            return _run_sweep2(recipe, args.sweep2)
        except ValueError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1

    if args.thermal_budget:
        tb = metrology.thermal_budget(recipe.steps)
        print("=== サーマルバジェット ===")
        for c in tb["steps"]:
            print(f"  {c['type']:9s} Dt={c['dt_um2']:.4f} um^2  {c['label']}")
        print(f"合計 Dt = {tb['total_dt_um2']:.4f} um^2")
        print(f"実効拡散長 L_eff = {tb['effective_length_um']:.4f} um")
        return 0

    # シミュレーション中の検証エラー（不正な膜厚・未知の材料など）も
    # トレースバックでなく一貫した「エラー: ...」メッセージで返す。
    try:
        wafer = recipe.simulate()
    except (ValueError, KeyError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    report = metrology.report(wafer)
    print(report)

    include_resist = not args.no_resist
    hidden = _hidden_ids(args.hide)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"レポートを保存しました: {args.report}", file=sys.stderr)
    if args.json_report:
        data = metrology.summary(wafer)
        data["electrical"] = metrology.electrical_report(wafer)
        data["defects"] = metrology.defect_report(wafer)
        rth = metrology.thermal_resistance_k_w(wafer)
        data["thermal_resistance_k_w"] = None if rth == float("inf") else rth
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON レポートを保存しました: {args.json_report}", file=sys.stderr)
    if args.stl:
        _export_stl(wafer, args.stl, include_resist, hidden)
        print(f"STL を保存しました: {args.stl}", file=sys.stderr)
    if args.png:
        _export_png(wafer, args.png, include_resist, hidden)
        print(f"PNG を保存しました: {args.png}", file=sys.stderr)
    if args.gif:
        from . import animation

        try:
            n_frames = animation.save_gif(
                recipe,
                args.gif,
                axis=args.gif_axis,
                fps=args.gif_fps,
                include_resist=include_resist,
                hidden_ids=hidden,
            )
        except ValueError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(
            f"GIF アニメーションを保存しました（{n_frames} フレーム）: {args.gif}",
            file=sys.stderr,
        )
    if args.csv_column:
        from . import export

        n = export.to_csv_column(
            wafer, args.csv_column, wafer.config.nx // 2, wafer.config.ny // 2
        )
        print(
            f"CSV 列プロファイルを保存しました（{n} 行）: {args.csv_column}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
