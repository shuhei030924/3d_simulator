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
        "--no-resist",
        action="store_true",
        help="STL/PNG 出力時にレジストを除外",
    )
    return p


def _load_recipe(args: argparse.Namespace) -> Recipe:
    if args.preset:
        return presets.build(args.preset)
    return Recipe.load(args.recipe)


def _export_stl(wafer, path: str, include_resist: bool) -> None:
    from . import visualize  # 遅延 import（pyvista 依存を必須にしない）

    visualize.export_stl(wafer, path, include_resist=include_resist)


def _export_png(wafer, path: str, include_resist: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import visualize

    plane, ww, hh = visualize.slice_2d(wafer, "Y", wafer.config.ny // 2)
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

    wafer = recipe.simulate()
    report = metrology.report(wafer)
    print(report)

    include_resist = not args.no_resist
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"レポートを保存しました: {args.report}", file=sys.stderr)
    if args.json_report:
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(metrology.summary(wafer), f, ensure_ascii=False, indent=2)
        print(f"JSON レポートを保存しました: {args.json_report}", file=sys.stderr)
    if args.stl:
        _export_stl(wafer, args.stl, include_resist)
        print(f"STL を保存しました: {args.stl}", file=sys.stderr)
    if args.png:
        _export_png(wafer, args.png, include_resist)
        print(f"PNG を保存しました: {args.png}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
