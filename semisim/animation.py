"""工程アニメーション（GUI 非依存）。

レシピを 1 工程ずつ適用した断面スナップショットを順に描画し、
GIF アニメーションとして書き出す。製造フローのレビュー・教育資料・
報告書向けに「工程がどう進むか」を動画で確認できる。

サブステップ補間: 1 工程内の途中状態は、工程の「量」パラメータ
（膜厚・深さ・サイクル数・時間）を比例配分した工程を実際に物理
エンジンで適用して生成する。画像のモーフィング等の非物理な補間は
行わない（ありえない中間状態を映さないため）。量が定義できない瞬時
工程（PHOTO/STRIP/IMPLANT 等）は 1 フレームのまま。

依存は matplotlib（フレーム描画）と Pillow（GIF 書込, matplotlib の
依存に含まれる）のみで、ヘッドレス（CI/CLI）でも動作する。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import visualize
from .grid import Wafer
from .processes import Process
from .recipe import Recipe

# 断面軸 → 横軸ラベル（slice_2d の規約に対応）
_XLABELS = {"X": "y (µm)", "Y": "x (µm)", "Z": "x (µm)"}
_YLABELS = {"X": "z (µm)", "Y": "z (µm)", "Z": "y (µm)"}


def _setup_matplotlib():
    """Agg バックエンドと日本語フォントを設定して pyplot を返す。"""
    import matplotlib

    matplotlib.use("Agg")
    # 工程名（日本語）が豆腐にならないよう日本語フォントを優先する。
    # 未インストールのファミリを指定すると matplotlib がフレーム毎に
    # findfont 警告を出すため、実在するものだけを設定する。
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [
        "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans",
        "Noto Sans CJK JP", "IPAGothic",
    ]
    family = [name for name in preferred if name in installed]
    matplotlib.rcParams["font.family"] = [*family, "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


# 工程タイプ → スケールするパラメータ名（先頭が主パラメータ＝工程の「量」）。
# 主パラメータを f 倍（0<f<1）した工程を直前状態に適用すると、物理的に意味の
# ある途中状態（薄い成膜・浅いエッチ・低い充填レベル）になる。
# 2 番目以降は工程の進行とともに「蓄積する」副次効果（横アンダーカット・
# RIE ノッチ・ディッシング・再付着等）。固定のままだと最初の途中フレームから
# 全量現れてしまうため、主パラメータと一緒に比例配分する。
# 量に比例して進まない瞬時工程（PHOTO/STRIP/LIFTOFF/IMPLANT/SPINON/DEFECT）
# は対象外（1 フレームのまま）。
_SCALABLE_PARAMS: dict[str, tuple[str, ...]] = {
    # 成膜系: 既存表面から外側へ厚みが増える。PVD は膜厚スケールだと
    # t 依存バジェットの再計算で途中状態の単調性が崩れるため、フル条件の
    # 反復成長を途中で止める growth_fraction（時間発展プレフィックス）を使う
    "CVD": ("thickness_um",), "PVD": ("growth_fraction",),
    "EPI": ("thickness_um",), "SALICIDE": ("thickness_um",),
    "OXIDE": ("thickness_um",), "ALD": ("cycles",),
    "SPACER": ("thickness_um", "overetch_um"),
    # 充填系: トレンチをボトムアップで充填レベルが上がる
    "FILL": ("level_fraction",),
    # エッチ系: 露出面から深さが進む（横バイアス/ノッチ/再付着は進行と共に蓄積）
    "DRY": ("depth_um", "lateral_um", "notch_um"),
    "WET": ("depth_um",), "KOH": ("depth_um",),
    "DRIE": ("depth_um", "redeposit_um", "microtrench_um"),
    "SPUTTER": ("depth_um",), "ALE": ("cycles",), "CLEAN": ("thickness_um",),
    # 熱・拡散系: 界面/表面から拡散フロントが進む
    "DIFFUSION": ("depth_um",), "ANNEAL": ("depth_um",), "RTP": ("depth_um",),
    # 平坦化系: 上面（CMP）/底面（裏面研削）から削れる
    "CMP": ("remove_um", "dishing_um", "erosion_um"),
    "BACKGRIND": ("thin_um",),
    "REFLOW": ("radius_um",),
}


# 拡散律速の工程（量指定モード）。結果量（酸化膜厚・シリサイド厚・拡散深さ）
# は時間の平方根に比例する（Deal-Grove の放物線律速 x∝√t、拡散長 L=√(Dt)）。
# 時間比 f のフレームには f でなく √f を掛け、「最初は速く、徐々に遅くなる」
# 実際の成長カイネティクスを再現する。時間指定モード（time_min>0）は時間
# そのものを f 倍するので、この補正は不要（モデルが √t を内包する）。
_SQRT_KINETICS = {"OXIDE", "SALICIDE", "DIFFUSION", "ANNEAL", "RTP"}


def scaled_process(proc: Process, fraction: float) -> Process | None:
    """時間比 fraction の途中状態に対応する工程の複製を返す（補間不可は None）。

    - 流量律速（成膜/エッチ/CMP 等）: 量パラメータを f 倍（結果∝t）。
    - 拡散律速（酸化/シリサイド/拡散の量指定モード）: √f 倍（結果∝√t）。
      時間指定モード（time_min>0）は時間を f 倍（モデル側が √t を計算）。
    - IMPLANT: ドーズ∝t なので規格化濃度が f 倍 ⇔ しきい値を 1/f 倍。
      注入領域が飛程 Rp の周りに現れて育つドーズ蓄積を再現する。
    - cycles（ALD/ALE）は整数に丸め、最低 1 サイクル。
    - 副パラメータ（横バイアス等の蓄積効果）は正の値のみ f 倍。
    """
    if proc.type == "IMPLANT":
        params = proc.params_dict()
        thr = float(params.get("threshold", 0.0) or 0.0)
        if thr <= 0:
            return None
        # ドーズが f 倍 → 濃度（ピーク規格化）も f 倍 → しきい値換算で thr/f。
        # 1.0 超は「まだどこも閾値未達」（ごく薄い注入の初期）として上限化。
        params["threshold"] = min(1.0, thr / max(fraction, 1e-9))
        params["type"] = proc.type
        return type(proc)._from_params(params)
    fields = _SCALABLE_PARAMS.get(proc.type)
    if not fields:
        return None
    params = proc.params_dict()
    primary = fields[0]
    eff = float(fraction)
    if proc.type in ("OXIDE", "ANNEAL") and float(params.get("time_min", 0) or 0) > 0:
        primary = "time_min"  # 時間指定モード: 時間を線形配分（√t はモデル側）
    elif proc.type in _SQRT_KINETICS:
        eff = float(fraction) ** 0.5  # 拡散律速: 結果量∝√t
    val = params.get(primary)
    if val is None or not np.isfinite(float(val)) or float(val) <= 0:
        return None
    if primary == "cycles":
        params[primary] = max(1, int(round(int(val) * eff)))
    else:
        params[primary] = float(val) * eff
    for extra in fields[1:]:
        ev = params.get(extra)
        if ev is not None and np.isfinite(float(ev)) and float(ev) > 0:
            params[extra] = float(ev) * eff
    params["type"] = proc.type
    return type(proc)._from_params(params)


def step_slices(
    recipe: Recipe,
    axis: str = "Y",
    index: int | None = None,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
    substeps: int = 1,
) -> list[tuple[str, np.ndarray, float, float]]:
    """各工程適用後の断面を順に返す（先頭は初期状態）。

    substeps>1 のとき、量をスケールできる工程は 1 工程を substeps 分割した
    途中状態（25%→50%→…→100%）を物理エンジンで生成して挿入する。
    最終フレーム（100%）は必ず元の工程そのもの＝通常シミュレーションと
    同一の状態。瞬時工程は従来どおり 1 フレーム。

    戻り値: [(見出し, 2D 材料 ID 配列, 横 µm, 縦 µm), ...]。
    Recipe のスナップショットキャッシュを使うため、全体の工程適用回数は
    O(n×substeps)（毎回先頭から再計算しない）。
    """
    if substeps < 1:
        raise ValueError(f"substeps は 1 以上（指定値: {substeps}）。")

    def _slice(wafer, title):
        if index is None:
            c = wafer.config
            idx = {"X": c.nx // 2, "Y": c.ny // 2, "Z": c.nz // 2}[str(axis).upper()]
        else:
            idx = index
        plane, w_um, h_um = visualize.slice_2d(
            wafer, axis, idx, include_resist=include_resist, hidden_ids=hidden_ids
        )
        return (title, plane, w_um, h_um)

    out: list[tuple[str, np.ndarray, float, float]] = []
    n = len(recipe.steps)
    out.append(_slice(recipe.simulate(up_to=0), "初期状態"))
    for k in range(1, n + 1):
        proc = recipe.steps[k - 1]
        head = f"工程 {k}/{n}: {proc.summary()}"
        if substeps > 1:
            for j in range(1, substeps):
                f = j / substeps
                partial = scaled_process(proc, f)
                if partial is None:
                    break  # 補間不可の工程は途中フレームなし
                base = recipe.simulate(up_to=k - 1)  # キャッシュ済み
                # config は base のもの（k-1 工程まで進化済み・呼び出しごとに
                # 独立コピー）を使う。recipe.config（初期値）だと、手前に
                # BACKGRIND がある場合に substrate_um が古く、CMP の研磨下限
                # 等が本来の途中状態と食い違う。
                w = Wafer(base.config)
                w.grid = base.grid  # simulate は独立コピーを返すのでそのまま使える
                partial.apply(w)
                out.append(_slice(w, f"{head}（{f:.0%}）"))
        out.append(_slice(recipe.simulate(up_to=k), head))
    return out


def render_frame(
    title: str,
    plane: np.ndarray,
    w_um: float,
    h_um: float,
    axis: str = "Y",
    dpi: int = 110,
    figsize: tuple[float, float] = (5.4, 4.4),
    smooth: bool = True,
) -> np.ndarray:
    """断面 1 枚を RGB (H, W, 3) uint8 配列に描画する。

    figsize/dpi が同じ限り全フレームが同寸になる（GIF 結合の前提）。
    smooth=True でボクセルの角を ±半ボクセル以内で平滑化して表示する。
    """
    plt = _setup_matplotlib()
    cmap, norm = visualize.material_listed_cmap()
    if smooth:
        plane = smooth_plane(plane)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(
        plane,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[0, w_um, 0, h_um],
        interpolation="nearest",
        aspect="equal",
    )
    a = str(axis).upper()
    ax.set_xlabel(_XLABELS.get(a, "x (µm)"))
    ax.set_ylabel(_YLABELS.get(a, "z (µm)"))
    # 長い工程サマリでもはみ出さないよう短縮する
    if len(title) > 60:
        title = title[:57] + "…"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def smooth_plane(plane: np.ndarray, factor: int = 3) -> np.ndarray:
    """材料境界を保ったまま断面の輪郭を平滑化アップサンプルする。

    各材料の 2 値マスクを双線形補間で factor 倍に拡大し、画素ごとに最も
    支配的な材料を採用する（多数決）。存在しない材料 ID を発明しないため
    物理的に安全で、表示上の平滑化は境界 ±半ボクセル以内に収まる。
    """
    ids = np.unique(plane)
    if ids.size <= 1:
        return np.repeat(np.repeat(plane, factor, axis=0), factor, axis=1)
    out = np.zeros((plane.shape[0] * factor, plane.shape[1] * factor),
                   dtype=plane.dtype)
    score = np.full(out.shape, -1.0, dtype=np.float32)
    for mid in ids:
        m = ndimage.zoom((plane == mid).astype(np.float32), factor, order=1)
        upd = m > score
        out[upd] = mid
        score[upd] = m[upd]
    return out


def save_gif(
    recipe: Recipe,
    path: str,
    axis: str = "Y",
    index: int | None = None,
    fps: float | None = None,
    hold_last: int = 3,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
    progress=None,
    substeps: int = 4,
    smooth: bool = True,
) -> int:
    """レシピの工程進行を断面 GIF アニメーションとして保存する。

    substeps>1 で 1 工程を物理的な途中状態に分割してなめらかに再生する
    （既定 4 分割）。fps 省略時は「1 工程 0.8 秒」になるよう substeps から
    自動決定する。最終フレームは hold_last 倍の時間だけ静止し、完成形を
    確認しやすくする。smooth で輪郭平滑化（±半ボクセル以内の表示処理）。
    progress に callable(done, total) を渡すと進捗を通知する（GUI 用）。
    フレーム数を返す。
    """
    from PIL import Image

    if fps is None:
        fps = 1.25 * max(1, int(substeps))  # 1 工程あたり約 0.8 秒
    if fps <= 0:
        raise ValueError(f"fps は正値である必要があります（指定値: {fps}）。")
    slices = step_slices(
        recipe, axis=axis, index=index,
        include_resist=include_resist, hidden_ids=hidden_ids,
        substeps=substeps,
    )
    total = len(slices)
    imgs: list[Image.Image] = []
    for i, (title, plane, w_um, h_um) in enumerate(slices):
        imgs.append(Image.fromarray(
            render_frame(title, plane, w_um, h_um, axis=axis, smooth=smooth)
        ))
        if progress is not None:
            progress(i + 1, total)
    # サブステップは速く流し、工程確定フレーム（タイトルに % が付かない）は
    # 1 工程ぶん静止して読み取れるようにする。
    base_ms = max(20, int(round(1000.0 / fps)))
    step_ms = base_ms * max(1, int(substeps))
    durations = [
        base_ms if title.endswith("%）") else step_ms
        for (title, _p, _w, _h) in slices
    ]
    durations[-1] = step_ms * (max(0, int(hold_last)) + 1)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return len(imgs)
