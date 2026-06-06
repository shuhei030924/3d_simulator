"""PyVista によるボクセルボリュームの可視化ヘルパ。

設計上の重要点:
    * グリッドは充填ボリュームなので、クリップした断面は必ず中身が詰まって
      見える（空洞にならない）。
    * カテゴリ（材料 ID）ごとの色を ListedColormap で割り当てる。
    * アンチエイリアス + 高解像度メッシュで拡大時のジャギーを軽減する。
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap

try:
    import pyvista as pv
except ModuleNotFoundError:  # pyvista 無し環境でも slice_2d/PNG 出力は使えるよう任意 import
    pv = None  # 3D 系（build_image_data/solid_unstructured/export_stl）呼び出し時のみ必要

from . import materials
from .grid import Wafer


def material_colormap() -> tuple[ListedColormap, list[int]]:
    """材料 ID に対応した ListedColormap と clim を返す。"""
    colors, _ = materials.color_lookup()
    cmap = ListedColormap(colors)
    return cmap, [0, len(colors) - 1]


def build_image_data(wafer: Wafer) -> pv.ImageData:
    """ウェハから材料 ID をセルデータに持つ ImageData を生成する。"""
    nz, ny, nx = wafer.grid.shape
    p = wafer.config.pitch_um
    img = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(p, p, p))
    # VTK のセル順は x が最速。grid[z,y,x] の C-order ravel が一致する。
    img.cell_data["material"] = wafer.grid.ravel(order="C").astype(np.float32)
    return img


def solid_unstructured(
    wafer: Wafer,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
) -> pv.UnstructuredGrid:
    """空気を除いた固体セルだけの UnstructuredGrid を返す。

    include_resist=False の場合、フォトレジストも除外する。
    hidden_ids に材料 ID を渡すと、その材料も非表示（空気扱い）にする。
    """
    grid = wafer.grid
    hide = set(int(i) for i in (hidden_ids or []))
    if not include_resist:
        hide.update(m.id for m in materials.all_materials() if m.is_resist)
    if hide:
        scal = grid.copy()
        for rid in hide:
            scal[scal == rid] = materials.AIR
    else:
        scal = grid

    img = pv.ImageData(
        dimensions=(grid.shape[2] + 1, grid.shape[1] + 1, grid.shape[0] + 1),
        spacing=(wafer.config.pitch_um,) * 3,
    )
    img.cell_data["material"] = scal.ravel(order="C").astype(np.float32)
    # 材料 ID >= 0.5（=空気以外）を残す
    solid = img.threshold(0.5, scalars="material")
    return solid


def smoothed_surface(mesh, plane=None, n_iter: int = 30,
                     pass_band: float = 0.1, plane_tol: float = 0.05):
    """メッシュ表面を平滑化して返す（スムーズ表示用）。

    体積を保つ Taubin（windowed-sinc）平滑化を用いる。ラプラシアン平滑化は
    形状を縮める（小さな構造で顕著, 例: 微小グリッドで約 33% 収縮）が、Taubin は
    収縮をほぼ起こさず（±1% 程度）外形だけを滑らかにする。

    plane=(origin, normal) を渡すと、その平面（断面の切断面）上にあった頂点を
    平滑化後に平面へ射影し直す。これにより断面のクリップ面が波打たず平坦に
    保たれる（平滑化は本来クリップ面も丸めてしまうため）。
    plane_tol は「切断面上」と判定する平面からの許容距離（µm, 目安はピッチ半分）。
    平滑化で点数が変わった場合は射影をスキップする（安全側）。
    """
    surf = mesh.extract_surface()
    out = surf.smooth_taubin(n_iter=n_iter, pass_band=pass_band)
    if plane is not None and out.n_points == surf.n_points:
        origin = np.asarray(plane[0], dtype=float)
        nrm = np.asarray(plane[1], dtype=float)
        ln = float(np.linalg.norm(nrm))
        if ln > 0:
            nrm = nrm / ln
            on_plane = np.abs((surf.points - origin) @ nrm) < plane_tol
            if on_plane.any():
                pts = out.points.copy()
                pts[on_plane] -= np.outer((pts[on_plane] - origin) @ nrm, nrm)
                out.points = pts
    return out


def slice_2d(
    wafer: Wafer,
    axis: str,
    index: int,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
) -> tuple[np.ndarray, float, float]:
    """断面の 2D 材料 ID 配列と物理サイズ (横µm, 縦µm) を返す。

    axis="X": x=index で切った YZ 断面 (縦=Z, 横=Y)
    axis="Y": y=index で切った XZ 断面 (縦=Z, 横=X)
    axis="Z": z=index で切った XY 断面 (縦=Y, 横=X)
    返す配列は表示用に縦軸が下から上（Z は下が基板）になるよう整える。
    hidden_ids に材料 ID を渡すとその材料は空気として表示から除外する。
    """
    grid = wafer.grid  # [z, y, x]
    nz, ny, nx = grid.shape
    p = wafer.config.pitch_um

    axis = str(axis).upper()  # 大文字小文字を問わない
    if axis == "X":
        index = int(np.clip(index, 0, nx - 1))
        plane = grid[:, :, index]  # (z, y)
        width_um, height_um = ny * p, nz * p
    elif axis == "Y":
        index = int(np.clip(index, 0, ny - 1))
        plane = grid[:, index, :]  # (z, x)
        width_um, height_um = nx * p, nz * p
    elif axis == "Z":
        index = int(np.clip(index, 0, nz - 1))
        plane = grid[index, :, :]  # (y, x)
        width_um, height_um = nx * p, ny * p
    else:
        # 無効な軸を黙って Z 断面にせず、明示的にエラーにする（取り違え防止）。
        raise ValueError(f"axis は 'X'/'Y'/'Z' のいずれかを指定してください: {axis!r}")

    plane = plane.copy()
    hide = set(int(i) for i in (hidden_ids or []))
    if not include_resist:
        hide.update(m.id for m in materials.all_materials() if m.is_resist)
    for rid in hide:
        plane[plane == rid] = materials.AIR
    return plane, width_um, height_um


def material_listed_cmap():
    """断面表示用の ListedColormap と正規化境界を返す。"""
    from matplotlib.colors import BoundaryNorm

    colors, _ = materials.color_lookup()
    cmap = ListedColormap(colors)
    n = len(colors)
    bounds = np.arange(-0.5, n + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def export_stl(
    wafer: Wafer,
    path: str,
    include_resist: bool = True,
    hidden_ids: list[int] | None = None,
) -> None:
    """固体形状の外表面を三角形メッシュ化して STL に書き出す（形状のみ）。"""
    solid = solid_unstructured(
        wafer, include_resist=include_resist, hidden_ids=hidden_ids
    )
    surface = solid.extract_surface(algorithm="dataset_surface").triangulate()
    surface.save(path)
