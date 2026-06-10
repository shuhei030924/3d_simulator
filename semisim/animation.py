"""工程アニメーション（GUI 非依存）。

レシピを 1 工程ずつ適用した断面スナップショットを順に描画し、
GIF アニメーションとして書き出す。製造フローのレビュー・教育資料・
報告書向けに「工程がどう進むか」を動画で確認できる。

依存は matplotlib（フレーム描画）と Pillow（GIF 書込, matplotlib の
依存に含まれる）のみで、ヘッドレス（CI/CLI）でも動作する。
"""
from __future__ import annotations

import numpy as np

from . import visualize
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


def step_slices(
    recipe: Recipe,
    axis: str = "Y",
    index: int | None = None,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
) -> list[tuple[str, np.ndarray, float, float]]:
    """各工程適用後の断面を順に返す（先頭は初期状態）。

    戻り値: [(見出し, 2D 材料 ID 配列, 横 µm, 縦 µm), ...] 長さ len(steps)+1。
    Recipe のスナップショットキャッシュを使うため、全体の適用回数は工程数と
    同じ O(n)（毎回先頭から再計算しない）。
    """
    out: list[tuple[str, np.ndarray, float, float]] = []
    n = len(recipe.steps)
    for k in range(n + 1):
        wafer = recipe.simulate(up_to=k)
        if index is None:
            c = wafer.config
            idx = {"X": c.nx // 2, "Y": c.ny // 2, "Z": c.nz // 2}[str(axis).upper()]
        else:
            idx = index
        plane, w_um, h_um = visualize.slice_2d(
            wafer, axis, idx, include_resist=include_resist, hidden_ids=hidden_ids
        )
        title = "初期状態" if k == 0 else f"工程 {k}/{n}: {recipe.steps[k - 1].summary()}"
        out.append((title, plane, w_um, h_um))
    return out


def render_frame(
    title: str,
    plane: np.ndarray,
    w_um: float,
    h_um: float,
    axis: str = "Y",
    dpi: int = 110,
    figsize: tuple[float, float] = (5.4, 4.4),
) -> np.ndarray:
    """断面 1 枚を RGB (H, W, 3) uint8 配列に描画する。

    figsize/dpi が同じ限り全フレームが同寸になる（GIF 結合の前提）。
    """
    plt = _setup_matplotlib()
    cmap, norm = visualize.material_listed_cmap()
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


def save_gif(
    recipe: Recipe,
    path: str,
    axis: str = "Y",
    index: int | None = None,
    fps: float = 1.25,
    hold_last: int = 3,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
    progress=None,
) -> int:
    """レシピの工程進行を断面 GIF アニメーションとして保存する。

    fps はフレーム毎秒（既定 1.25 = 1 工程 0.8 秒）。最終フレームは
    hold_last 倍の時間だけ静止し、完成形を確認しやすくする。
    progress に callable(done, total) を渡すと進捗を通知する（GUI 用）。
    フレーム数（工程数+1）を返す。
    """
    from PIL import Image

    if fps <= 0:
        raise ValueError(f"fps は正値である必要があります（指定値: {fps}）。")
    slices = step_slices(
        recipe, axis=axis, index=index,
        include_resist=include_resist, hidden_ids=hidden_ids,
    )
    total = len(slices)
    imgs: list[Image.Image] = []
    for i, (title, plane, w_um, h_um) in enumerate(slices):
        imgs.append(Image.fromarray(render_frame(title, plane, w_um, h_um, axis=axis)))
        if progress is not None:
            progress(i + 1, total)
    duration_ms = max(20, int(round(1000.0 / fps)))
    durations = [duration_ms] * len(imgs)
    durations[-1] = duration_ms * (max(0, int(hold_last)) + 1)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return len(imgs)
