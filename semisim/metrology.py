"""計測・解析ヘルパ（メトロロジ）。

ボクセルグリッドから膜厚マップ・表面高さ・段差・体積・固体率などを
物理単位（µm）で算出する。GUI 非依存で純粋にエンジンのみに依存する。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import materials
from .grid import Wafer

# 台形積分。NumPy 2.0+ は np.trapezoid、1.x は np.trapz（要件 numpy>=1.22 に対応）。
_trapz = getattr(np, "trapezoid", None) or np.trapz


def material_counts(wafer: Wafer) -> dict[str, int]:
    """存在する材料ごとのボクセル数を {材料名: 個数} で返す（空気を除く）。"""
    out: dict[str, int] = {}
    for mid in np.unique(wafer.grid):
        if int(mid) == materials.AIR:
            continue
        out[materials.BY_ID[int(mid)].name] = int((wafer.grid == mid).sum())
    return out


def material_volume_um3(wafer: Wafer, name_or_id) -> float:
    """指定材料の体積を µm³ で返す。"""
    mat = materials.get(name_or_id)
    vox = float((wafer.grid == mat.id).sum())
    return vox * (wafer.config.pitch_um ** 3)


def film_thickness_map(wafer: Wafer, name_or_id) -> np.ndarray:
    """各 (y, x) 列での指定材料の合計厚みマップ (ny, nx) を µm で返す。

    縦に連続していなくても、その列に含まれる当該材料ボクセル総数 × pitch。
    """
    mat = materials.get(name_or_id)
    count = (wafer.grid == mat.id).sum(axis=0)  # (ny, nx)
    return count.astype(np.float64) * wafer.config.pitch_um


def film_thickness_stats(wafer: Wafer, name_or_id) -> dict[str, float]:
    """指定材料が存在する列における膜厚の平均/標準偏差/最小/最大 (µm)。

    どの列にも存在しない場合はすべて 0.0 を返す。
    """
    tmap = film_thickness_map(wafer, name_or_id)
    present = tmap[tmap > 0]
    if present.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coverage": 0.0}
    coverage = float(present.size) / float(tmap.size)
    return {
        "mean": float(present.mean()),
        "std": float(present.std()),
        "min": float(present.min()),
        "max": float(present.max()),
        "coverage": coverage,
    }


def surface_height_map(wafer: Wafer) -> np.ndarray:
    """各列の充填高さ (µm) のマップ (ny, nx)。固体が無い列は NaN。"""
    z_top = wafer.top_surface_z()  # (ny, nx), 固体なし= -1
    height = (z_top + 1).astype(np.float64) * wafer.config.pitch_um
    height[z_top < 0] = np.nan
    return height


def step_height_um(wafer: Wafer) -> float:
    """固体表面の最大高さと最小高さの差 (µm)。平坦なら 0。"""
    height = surface_height_map(wafer)
    valid = height[~np.isnan(height)]
    if valid.size == 0:
        return 0.0
    return float(valid.max() - valid.min())


def planarization_dof_check(wafer: Wafer, dof_um: float = 0.2) -> dict:
    """表面トポグラフィをリソの焦点深度(DOF)と比較し焦点外れリスクを判定する。

    各列の表面高さ（surface_height_map）の中央値を最良焦点面とみなし、そこからの
    乖離が DOF/2 を超える領域を「焦点外れ」と判定する。CMP 後の平坦性がリソの
    DOF に収まるか（後続パターニングの解像可否）を検証する。返す辞書:
      - surface_range_um: 表面高低差（max−min）
      - focus_plane_um: 最良焦点面（高さ中央値）
      - out_of_focus_fraction: 焦点外れ領域の面積率（0〜1）
      - within_dof: 高低差が DOF 以内か（bool）
    固体が無ければ全て 0／True。
    """
    if dof_um <= 0:
        raise ValueError("DOF は正の値が必要です。")
    height = surface_height_map(wafer)
    valid_mask = ~np.isnan(height)
    valid = height[valid_mask]
    if valid.size == 0:
        return {"surface_range_um": 0.0, "focus_plane_um": 0.0,
                "out_of_focus_fraction": 0.0, "within_dof": True}
    focus = float(np.median(valid))
    dev = np.abs(valid - focus)
    out_frac = float((dev > dof_um / 2.0).mean())
    return {
        "surface_range_um": float(valid.max() - valid.min()),
        "focus_plane_um": focus,
        "out_of_focus_fraction": out_frac,
        "within_dof": bool((valid.max() - valid.min()) <= dof_um),
    }


def solid_fraction(wafer: Wafer) -> float:
    """全ボクセルに対する固体（空気以外）の割合。"""
    return float((wafer.grid != materials.AIR).mean())


def line_scan(wafer: Wafer, axis: str, index_a: int, index_b: int) -> np.ndarray:
    """指定した z 高さ・行に沿った材料 ID の 1 次元プロファイルを返す。

    axis="x": 高さ z=index_a, 行 y=index_b の x 方向ライン (nx,)
    axis="y": 高さ z=index_a, 列 x=index_b の y 方向ライン (ny,)
    """
    grid = wafer.grid
    nz, ny, nx = grid.shape
    z = int(np.clip(index_a, 0, nz - 1))
    if axis == "x":
        y = int(np.clip(index_b, 0, ny - 1))
        return grid[z, y, :].copy()
    if axis == "y":
        x = int(np.clip(index_b, 0, nx - 1))
        return grid[z, :, x].copy()
    raise ValueError(f"axis は 'x' または 'y' を指定してください: {axis}")


def feature_width_um(wafer: Wafer, name_or_id, z_index: int, y_index: int) -> float:
    """指定高さ・行の x 方向ラインで、対象材料が連続する最大幅 (µm) を返す。

    トレンチ/ラインの CD（限界寸法）の概算に使う。存在しなければ 0。
    """
    mat = materials.get(name_or_id)
    line = line_scan(wafer, "x", z_index, y_index)
    is_t = line == mat.id
    if not is_t.any():
        return 0.0
    # 連続 True の最大ラン長を求める
    best = run = 0
    for v in is_t:
        run = run + 1 if v else 0
        best = max(best, run)
    return float(best) * wafer.config.pitch_um


def trench_depth_um(wafer: Wafer) -> float:
    """表面高さマップの最大値と最小値の差。step_height_um と同義の深さ指標。"""
    return step_height_um(wafer)


def aspect_ratio(wafer: Wafer, name_or_id, z_index: int, y_index: int) -> float:
    """概算アスペクト比 = 段差(深さ) / 開口幅。幅 0 のときは 0 を返す。"""
    width = feature_width_um(wafer, name_or_id, z_index, y_index)
    if width <= 0:
        return 0.0
    return step_height_um(wafer) / width


def surface_roughness_um(wafer: Wafer) -> float:
    """表面高さの RMS 粗さ (µm)。平坦面ほど 0 に近い。"""
    height = surface_height_map(wafer)
    valid = height[~np.isnan(height)]
    if valid.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((valid - valid.mean()) ** 2)))


def sidewall_angle_deg(wafer: Wafer, name_or_id, y_index: int) -> float:
    """指定行(y)の断面で、対象材料が作る側壁の平均傾斜角(度)を返す。

    各高さ z での対象材料の最小 x（左壁）の z に対する変化から傾きを推定する。
    垂直壁なら 90° に近く、テーパ（KOH 等）では小さくなる。対象が無ければ 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    nz, ny, nx = grid.shape
    y = int(np.clip(y_index, 0, ny - 1))
    plane = grid[:, y, :] == mat.id  # (nz, nx)
    left_x: list[int] = []
    zs: list[int] = []
    for z in range(nz):
        xs = np.flatnonzero(plane[z])
        if xs.size:
            left_x.append(int(xs.min()))
            zs.append(z)
    if len(zs) < 2:
        return 0.0
    dz = zs[-1] - zs[0]
    dx = abs(left_x[-1] - left_x[0])
    if dx == 0:
        return 90.0
    return float(np.degrees(np.arctan2(dz, dx)))


def interface_width_um(wafer: Wafer, name_a, name_b) -> float:
    """材料 A と B が隣接する界面の接触面積相当 (µm²) を返す。

    A と B の接触ボクセル対の数 × pitch² を界面の広がりの目安とする。
    拡散層と基板の界面評価などに使う。接触が無ければ 0。
    """
    a = materials.get(name_a).id
    b = materials.get(name_b).id
    grid = wafer.grid
    mask_a = grid == a
    mask_b = grid == b
    contact = np.zeros_like(mask_a)
    for axis in range(3):
        contact |= mask_a & np.roll(mask_b, 1, axis=axis)
        contact |= mask_a & np.roll(mask_b, -1, axis=axis)
    return float(contact.sum()) * (wafer.config.pitch_um ** 2)


def trench_is_closed(wafer: Wafer, x_index: int, y_index: int) -> bool:
    """指定列 (x, y) の最上固体より下に空気が挟まれている（閉塞/ボイド）かを返す。

    Fill のオーバーハングによるボイド検出などに使う。
    """
    grid = wafer.grid
    nz, ny, nx = grid.shape
    x = int(np.clip(x_index, 0, nx - 1))
    y = int(np.clip(y_index, 0, ny - 1))
    col = grid[:, y, x]
    solid = np.flatnonzero(col != materials.AIR)
    if solid.size == 0:
        return False
    top = int(solid.max())
    return bool((col[:top] == materials.AIR).any())


def void_volume_um3(wafer: Wafer) -> float:
    """埋め込まれた空隙（ボイド）の総体積 (µm³) を返す。

    各列で最上固体より下にある空気ボクセル＝外気と繋がらない閉塞空隙とみなす。
    ダマシン/トレンチ充填の品質評価に使う。
    """
    grid = wafer.grid
    nz = grid.shape[0]
    solid = grid != materials.AIR
    # 各列の最上固体高さ（無い列は -1）
    has_solid = solid.any(axis=0)
    top = np.where(has_solid, nz - 1 - np.argmax(solid[::-1, :, :], axis=0), -1)
    z_idx = np.arange(nz)[:, None, None]
    buried_air = (grid == materials.AIR) & (z_idx < top[None, :, :])
    return float(buried_air.sum()) * (wafer.config.pitch_um ** 3)


def void_metrics(wafer: Wafer) -> dict:
    """埋め込み空隙（ボイド）の連結成分統計を返す。

    void_volume_um3 と同じく「最上固体より下にある空気」を閉塞空隙とみなし、
    6 近傍連結でラベリングして個数・最大体積・最大ボイドの縦方向高さを返す。
    シーム/ボイドの深刻度評価に使う。
    返り値: {"count", "total_um3", "largest_um3", "max_height_um"}。
    """
    grid = wafer.grid
    nz = grid.shape[0]
    solid = grid != materials.AIR
    has_solid = solid.any(axis=0)
    top = np.where(has_solid, nz - 1 - np.argmax(solid[::-1, :, :], axis=0), -1)
    z_idx = np.arange(nz)[:, None, None]
    buried_air = (grid == materials.AIR) & (z_idx < top[None, :, :])
    vox = wafer.config.pitch_um ** 3
    if not buried_air.any():
        return {"count": 0, "total_um3": 0.0, "largest_um3": 0.0, "max_height_um": 0.0}
    structure = ndimage.generate_binary_structure(3, 1)  # 6 近傍
    labels, n = ndimage.label(buried_air, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]  # ラベル 0(背景)除く
    largest = int(sizes.argmax()) + 1
    zs = np.where(labels == largest)[0]
    max_height = float(zs.max() - zs.min() + 1) * wafer.config.pitch_um
    return {
        "count": int(n),
        "total_um3": float(sizes.sum()) * vox,
        "largest_um3": float(sizes.max()) * vox,
        "max_height_um": max_height,
    }


def etch_residue_metrics(
    wafer: Wafer, name_or_id, max_island_um3: float = 0.02
) -> dict:
    """エッチ残渣／ストリンガー／ブロックエッチ残りの孤立小片を検出する。

    指定材料を 6 近傍でラベリングし、体積が max_island_um3 以下の連結成分を
    「残渣（除去しきれずに残った小片）」とみなして統計化する。本来の配線/膜
    （大きな連結成分）は閾値を超えるため除外され、段差側壁に残ったストリンガー
    や、エッチ不足で孤立して残った材料島だけを拾う。
    返り値: {"count", "total_um3", "largest_um3", "max_aspect"}。
      count: 残渣島の個数。
      total_um3: 残渣の総体積。
      largest_um3: 最大残渣の体積。
      max_aspect: 最大残渣の縦横比（高さ/最大水平広がり）。
        ストリンガー（細長く背の高い側壁残り）ほど大きい。
    残渣が無ければ全て 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    mask = grid == mat.id
    vox = wafer.config.pitch_um ** 3
    zero = {"count": 0, "total_um3": 0.0, "largest_um3": 0.0, "max_aspect": 0.0}
    if not mask.any():
        return zero
    structure = ndimage.generate_binary_structure(3, 1)  # 6 近傍
    labels, n = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]  # ラベル 0(背景)除く
    thr_vox = max_island_um3 / vox
    residue_labels = np.flatnonzero(sizes <= thr_vox) + 1
    if residue_labels.size == 0:
        return zero
    total_vox = float(sizes[residue_labels - 1].sum())
    largest_lbl = int(residue_labels[np.argmax(sizes[residue_labels - 1])])
    # 最大残渣の縦横比（高さ / 水平方向の最大広がり）
    zs, ys, xs = np.where(labels == largest_lbl)
    height = float(zs.max() - zs.min() + 1)
    span = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    aspect = height / span if span > 0 else 0.0
    return {
        "count": int(residue_labels.size),
        "total_um3": total_vox * vox,
        "largest_um3": float(sizes[largest_lbl - 1]) * vox,
        "max_aspect": float(aspect),
    }


def undercut_um(wafer: Wafer, feature_name_or_id, mask_name_or_id) -> dict:
    """マスク下の横方向エッチ進行（アンダーカット不良）を計測する。

    マスク材料の直下にある被加工材料が、マスク開口より内側に横方向へ
    後退している量を測る。等方性エッチや過剰サイドエッチで生じる代表的な
    不良モードで、マスクが被加工材料に対して庇（ひさし）状に張り出す。
    各 y 行について、マスク最下層の x 方向幅と、その直下の被加工材料上面の
    x 方向幅を比較し、片側後退量 (mask_width - feature_width)/2 × pitch を
    アンダーカット量とする。
    返り値: {"max_um", "mean_um", "n"}。
      max_um: 最悪（最大）アンダーカット量。
      mean_um: 後退が生じた行の平均アンダーカット量。
      n: アンダーカットが検出された y 行数。
    いずれかの材料が無い／重なりが無ければ全て 0。
    """
    feat = materials.get(feature_name_or_id)
    mask = materials.get(mask_name_or_id)
    grid = wafer.grid
    pitch = wafer.config.pitch_um
    zero = {"max_um": 0.0, "mean_um": 0.0, "n": 0}
    mask_any = grid == mask.id
    feat_any = grid == feat.id
    if not mask_any.any() or not feat_any.any():
        return zero
    # マスクの最下層 z（被加工材料と接する側）
    mask_zs = np.where(mask_any.any(axis=(1, 2)))[0]
    mask_bottom = int(mask_zs.min())
    undercuts = []
    ny = grid.shape[1]
    for y in range(ny):
        mrow = np.where(grid[mask_bottom, y, :] == mask.id)[0]
        if mrow.size == 0:
            continue
        mask_w = mrow.max() - mrow.min() + 1
        # マスク直下で被加工材料の上面（最大 z, mask_bottom 未満）
        col = grid[:mask_bottom, y, :] == feat.id
        if not col.any():
            continue
        top_z = int(np.where(col.any(axis=1))[0].max())
        frow = np.where(grid[top_z, y, :] == feat.id)[0]
        if frow.size == 0:
            continue
        feat_w = frow.max() - frow.min() + 1
        recess = (mask_w - feat_w) / 2.0
        if recess > 0:
            undercuts.append(recess * pitch)
    if not undercuts:
        return zero
    arr = np.asarray(undercuts, dtype=float)
    return {
        "max_um": float(arr.max()),
        "mean_um": float(arr.mean()),
        "n": int(arr.size),
    }


def cmp_uniformity_pct(wafer: Wafer) -> float:
    """表面高さの不均一性 (3σ/平均, %) を返す。CMP 平坦性評価の指標。

    値が小さいほど平坦。完全平坦面は 0。有効な表面が無ければ 0。
    """
    height = surface_height_map(wafer)
    valid = height[~np.isnan(height)]
    if valid.size == 0:
        return 0.0
    mean = float(valid.mean())
    if mean <= 0:
        return 0.0
    return float(3.0 * valid.std() / mean * 100.0)


def pinhole_metrics(wafer: Wafer, name_or_id) -> dict:
    """薄膜を貫通するピンホール（被覆欠陥）を検出する。

    指定材料が各 (x, y) 位置で膜として存在するかの被覆マップを作り、膜に
    取り囲まれているのに膜が抜けている内部の穴をピンホールとみなす。膜の
    外縁（パターンエッジ）は穴に数えず、`binary_fill_holes` で塞がる
    閉じた抜けだけを拾う。堆積カバレッジ不良やパーティクル起因の貫通欠陥
    （下地が露出しリーク/腐食の起点になる）を評価する。
    返り値: {"count", "total_area_um2", "largest_area_um2"}。
      count: ピンホールの個数。
      total_area_um2: 全ピンホールの平面投影面積。
      largest_area_um2: 最大ピンホールの面積。
    膜が無い／ピンホールが無ければ全て 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    area_per = wafer.config.pitch_um ** 2
    zero = {"count": 0, "total_area_um2": 0.0, "largest_area_um2": 0.0}
    present = (grid == mat.id).any(axis=0)  # (y, x) 被覆マップ
    if not present.any():
        return zero
    filled = ndimage.binary_fill_holes(present)
    holes = filled & ~present  # 膜に囲まれた内部の抜け
    if not holes.any():
        return zero
    labels, n = ndimage.label(holes)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "count": int(n),
        "total_area_um2": float(sizes.sum()) * area_per,
        "largest_area_um2": float(sizes.max()) * area_per,
    }


def etch_depth_uniformity(wafer: Wafer, name_or_id) -> dict:
    """指定材料の上面高さの分布統計を返す（エッチ/堆積均一性の評価）。

    返り値: {"mean_um", "std_um", "cv_pct", "min_um", "max_um"}。
    対象材料が存在しなければ全て 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    nz = grid.shape[0]
    mask = grid == mat.id
    has = mask.any(axis=0)
    if not has.any():
        return {"mean_um": 0.0, "std_um": 0.0, "cv_pct": 0.0, "min_um": 0.0, "max_um": 0.0}
    top = nz - 1 - np.argmax(mask[::-1, :, :], axis=0)
    heights = top[has].astype(float) * wafer.config.pitch_um
    mean = float(heights.mean())
    std = float(heights.std())
    cv = float(std / mean * 100.0) if mean > 0 else 0.0
    return {
        "mean_um": mean,
        "std_um": std,
        "cv_pct": cv,
        "min_um": float(heights.min()),
        "max_um": float(heights.max()),
    }


def line_edge_roughness_um(wafer: Wafer, name_or_id, z_index: int) -> float:
    """指定高さでの対象材料エッジ位置の揺らぎ (RMS, µm) を返す。

    各 y 行で対象材料が左端から最初に現れる x 位置（左エッジ）を求め、
    その y 方向の標準偏差を LER とみなす。理想直線エッジなら 0 に近い。
    対象が無い、またはエッジが取れる行が 2 未満なら 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    nz, ny, nx = grid.shape
    z = int(np.clip(z_index, 0, nz - 1))
    plane = grid[z] == mat.id  # (ny, nx)
    edges: list[int] = []
    for y in range(ny):
        xs = np.flatnonzero(plane[y])
        if xs.size:
            edges.append(int(xs.min()))
    if len(edges) < 2:
        return 0.0
    return float(np.std(edges)) * wafer.config.pitch_um


def line_width_roughness_um(wafer: Wafer, name_or_id, z_index: int) -> float:
    """指定高さでの対象材料ライン幅の揺らぎ (RMS, µm) を返す。

    各 y 行で対象材料が存在する区間の幅（右端-左端+1）を求め、その y 方向の
    標準偏差を LWR とみなす。LER がエッジ位置の揺らぎなのに対し、LWR は線幅
    そのものの揺らぎで、両エッジの相関を反映する独立した指標。
    対象が無い、または幅が取れる行が 2 未満なら 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    nz, ny, nx = grid.shape
    z = int(np.clip(z_index, 0, nz - 1))
    plane = grid[z] == mat.id  # (ny, nx)
    widths: list[int] = []
    for y in range(ny):
        xs = np.flatnonzero(plane[y])
        if xs.size:
            widths.append(int(xs.max() - xs.min() + 1))
    if len(widths) < 2:
        return 0.0
    return float(np.std(widths)) * wafer.config.pitch_um


def cd_uniformity(wafer: Wafer, name_or_id, z_index: int) -> dict:
    """指定高さでの限界寸法均一性 (CDU) を返す。

    各 y 行で対象材料が存在する区間の幅（右端-左端+1）を CD とみなし、
    その平均・標準偏差・範囲(最大-最小)・3σ・サンプル行数をまとめる。
    LWR が標準偏差のみなのに対し、CDU は平均 CD と 3σ を含むフルセットで、
    ウェハ内寸法ばらつき（フォト/エッチの均一性）を評価する代表的なファブ指標。
    対象が無ければ全て 0。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    nz, ny, nx = grid.shape
    z = int(np.clip(z_index, 0, nz - 1))
    plane = grid[z] == mat.id  # (ny, nx)
    widths: list[int] = []
    for y in range(ny):
        xs = np.flatnonzero(plane[y])
        if xs.size:
            widths.append(int(xs.max() - xs.min() + 1))
    if not widths:
        return {
            "mean_um": 0.0,
            "std_um": 0.0,
            "range_um": 0.0,
            "three_sigma_um": 0.0,
            "n": 0,
        }
    arr = np.asarray(widths, dtype=float) * wafer.config.pitch_um
    std = float(arr.std())
    return {
        "mean_um": float(arr.mean()),
        "std_um": std,
        "range_um": float(arr.max() - arr.min()),
        "three_sigma_um": 3.0 * std,
        "n": int(arr.size),
    }


def overlay_error_um(wafer: Wafer, ref_material, measure_material) -> float:
    """2 つの材料層の重心ずれ（オーバレイ誤差, µm）を xy 面内で返す。

    各材料の (x, y) 重心を求め、その水平距離を整合ずれの指標とする。
    リソ重ね合わせやセルフアライン工程の評価に使う。
    どちらかが存在しなければ 0。
    """
    a = materials.get(ref_material).id
    b = materials.get(measure_material).id
    grid = wafer.grid
    mask_a = grid == a
    mask_b = grid == b
    if not mask_a.any() or not mask_b.any():
        return 0.0
    _, ya, xa = np.nonzero(mask_a)
    _, yb, xb = np.nonzero(mask_b)
    dy = ya.mean() - yb.mean()
    dx = xa.mean() - xb.mean()
    return float(np.hypot(dx, dy)) * wafer.config.pitch_um


def feature_width_variants(
    wafer: Wafer, name_or_id, z_index: int, y_index: int
) -> dict:
    """対象材料ラインの CD を複数定義で計測して返す。

    返り値: {"max_run_um", "total_um", "gap_um"}。
      max_run_um: 連続する最大幅（feature_width_um と同義）
      total_um:   その行で対象材料が占める総 x 長（複数ラインの合計）
      gap_um:     最初と最後の対象ボクセル間に挟まれる非対象部の総長
    パターン密度や OPC 検討の概算に使う。対象が無ければ全て 0。
    """
    mat = materials.get(name_or_id)
    line = line_scan(wafer, "x", z_index, y_index)
    is_t = line == mat.id
    pitch = wafer.config.pitch_um
    if not is_t.any():
        return {"max_run_um": 0.0, "total_um": 0.0, "gap_um": 0.0}
    best = run = 0
    for v in is_t:
        run = run + 1 if v else 0
        best = max(best, run)
    idx = np.flatnonzero(is_t)
    span = int(idx.max() - idx.min()) + 1
    total = int(is_t.sum())
    gap = span - total
    return {
        "max_run_um": float(best) * pitch,
        "total_um": float(total) * pitch,
        "gap_um": float(gap) * pitch,
    }


def thermal_budget(steps) -> dict:
    """レシピの熱履歴（サーマルバジェット）を拡散長ベースで集計する。

    拡散長 L=√(Dt) より Dt ∝ L²。各熱工程（DIFFUSION/ANNEAL/RTP）の
    特性拡散長を depth_um とみなし、Dt 等価量を depth_um² で表す。
    拡散長は二乗和で合成されるため、合計 Dt から実効拡散長
    L_eff=√(ΣLᵢ²) が得られる（複数の熱処理を 1 回に換算した目安）。

    引数 steps: Process の列（recipe.steps を渡す）。
    返り値: {"total_dt_um2", "effective_length_um", "by_type", "steps"}。
      by_type: {工程タイプ: Dt 合計}
      steps:   [{"label", "type", "dt_um2"}] の寄与リスト
    """
    thermal_types = {"DIFFUSION", "ANNEAL", "RTP"}
    by_type: dict[str, float] = {}
    contributions: list[dict] = []
    total = 0.0
    for s in steps:
        stype = getattr(s, "type", "")
        if stype not in thermal_types:
            continue
        # 時間/温度モードの Anneal は拡散長を物理計算した実効深さを使う。
        if getattr(s, "time_min", 0.0) and hasattr(s, "_effective_depth_um"):
            try:
                depth = float(s._effective_depth_um("phosphorus"))
            except Exception:
                depth = float(getattr(s, "depth_um", 0.0) or 0.0)
        else:
            depth = float(getattr(s, "depth_um", 0.0) or 0.0)
        dt = depth * depth
        total += dt
        by_type[stype] = by_type.get(stype, 0.0) + dt
        contributions.append({"label": s.summary(), "type": stype, "dt_um2": dt})
    return {
        "total_dt_um2": total,
        "effective_length_um": float(np.sqrt(total)),
        "by_type": by_type,
        "steps": contributions,
    }


def via_fill_quality(wafer: Wafer, name_or_id) -> dict:
    """指定材料によるビア/トレンチ充填の品質を返す。

    充填材料を含む列について、その材料の最上面より下に挟まれる空気
    （=キーホール/シーム空隙）を数え、充填率を算出する。

    返り値: {"fill_fraction", "void_volume_um3", "void_count"}。
      fill_fraction: 材料体積 / (材料体積 + 空隙体積)。1.0 で完全充填。
    充填材料が存在しなければ fill_fraction=1.0, 空隙 0 を返す。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    nz = grid.shape[0]
    is_mat = grid == mat.id
    has = is_mat.any(axis=0)  # (ny, nx)
    if not has.any():
        return {"fill_fraction": 1.0, "void_volume_um3": 0.0, "void_count": 0}
    # 各列の材料最上高さ（無い列は -1）
    top_mat = np.where(has, nz - 1 - np.argmax(is_mat[::-1, :, :], axis=0), -1)
    z_idx = np.arange(nz)[:, None, None]
    void = (grid == materials.AIR) & (z_idx < top_mat[None, :, :]) & has[None, :, :]
    void_vox = float(void.sum())
    mat_vox = float(is_mat.sum())
    denom = mat_vox + void_vox
    fill = 1.0 if denom <= 0 else mat_vox / denom
    return {
        "fill_fraction": float(fill),
        "void_volume_um3": void_vox * (wafer.config.pitch_um ** 3),
        "void_count": int(void_vox),
    }


def sidewall_bowing_um(wafer: Wafer, y_index: int) -> float:
    """Y 断面でのトレンチ側壁のボーイング（樽型膨らみ）量を µm で返す。

    各高さ z における内部トレンチ幅（両側を固体で挟まれた空気の幅）を測り、
    最大幅と表面付近（最上で内部幅>0 の z）の開口幅との差を返す。
    値が大きいほど側壁が外側へ膨らんでいる（ボーイング/バレリング）。
    トレンチが見つからなければ 0。
    """
    grid = wafer.grid
    nz, ny, nx = grid.shape
    y = int(np.clip(y_index, 0, ny - 1))
    s = grid[:, y, :]  # (nz, nx)
    widths = np.zeros(nz, dtype=int)
    for z in range(nz):
        row = s[z]
        solid = np.flatnonzero(row != materials.AIR)
        if solid.size < 2:
            continue
        lo, hi = int(solid.min()), int(solid.max())
        if hi - lo < 2:
            continue
        interior = row[lo + 1 : hi]  # 両側を固体で挟まれた内側
        widths[z] = int((interior == materials.AIR).sum())
    nz_with = np.flatnonzero(widths > 0)
    if nz_with.size == 0:
        return 0.0
    top_z = int(nz_with.max())  # 最上（表面側）の内部幅を開口幅とみなす
    surface_w = widths[top_z]
    max_w = int(widths.max())
    return float(max(0, max_w - surface_w)) * wafer.config.pitch_um


def pattern_density_map(
    wafer: Wafer, name_or_id=None, radius_um: float = 2.0
) -> np.ndarray:
    """局所パターン密度（平面占有率）マップ (ny, nx) を 0..1 で返す。

    name_or_id を指定するとその材料のフットプリント（その材料を含む列）を、
    省略すると基板上面より上に存在する全固体（=パターン構造）の占有を用い、
    半径 radius_um の窓で平滑化する。CMP のディッシング/エロージョンや
    エッチ/デポのローディング効果（疎密差）の評価に使う。
    """
    grid = wafer.grid
    if name_or_id is None:
        sub_z = wafer.um_to_vox(wafer.config.substrate_um)
        sub_z = int(np.clip(sub_z, 0, grid.shape[0]))
        footprint = np.any(grid[sub_z:] != materials.AIR, axis=0)
    else:
        mid = materials.get(name_or_id).id
        footprint = np.any(grid == mid, axis=0)
    occ = footprint.astype(np.float64)
    r = max(1, wafer.um_to_vox(radius_um))
    dens = ndimage.uniform_filter(occ, size=2 * r + 1, mode="nearest")
    return dens


def pattern_density_stats(
    wafer: Wafer, name_or_id=None, radius_um: float = 2.0
) -> dict:
    """パターン密度マップの統計を返す。

    返り値: {"min", "max", "mean", "range"}（いずれも 0..1）。range が大きい
    ほど疎密差が大きく、CMP ディッシングやエッチローディングが顕在化しやすい。
    """
    dens = pattern_density_map(wafer, name_or_id, radius_um)
    return {
        "min": float(dens.min()),
        "max": float(dens.max()),
        "mean": float(dens.mean()),
        "range": float(dens.max() - dens.min()),
    }


def conformality_pct(wafer: Wafer, name_or_id) -> dict:
    """成膜のコンフォーマリティ（段差被覆性）を %で評価する。

    指定材料の膜について、平坦部（フィールド）の代表膜厚＝列ごと膜厚の
    中央値に対する、最薄部（窪み底・側壁など最も薄い列）の膜厚比を返す。
    コンフォーマルな CVD/ALD は 100%に近づき、指向性の PVD は窪み底が
    薄くなるため低い値になる。トレンチ埋込で局所的に厚い列があっても
    中央値・最小値ベースなので頑健。
    返り値: {"field_thickness_um", "min_thickness_um", "step_coverage_pct"}。
    膜が無ければ step_coverage_pct=0。
    """
    tmap = film_thickness_map(wafer, name_or_id)  # (ny, nx) µm
    film = tmap[tmap > 0]
    if film.size == 0:
        return {
            "field_thickness_um": 0.0,
            "min_thickness_um": 0.0,
            "step_coverage_pct": 0.0,
        }
    field_t = float(np.median(film))
    min_t = float(film.min())
    cov = 0.0 if field_t <= 0 else 100.0 * min_t / field_t
    return {
        "field_thickness_um": field_t,
        "min_thickness_um": min_t,
        "step_coverage_pct": float(np.clip(cov, 0.0, 100.0)),
    }


def film_stress_thickness(wafer: Wafer) -> dict:
    """各材料の応力×平均膜厚積（N/m）と、その合計（正味）を返す。

    各材料 m について σ_m [Pa] × <t_m> [m] を計算する。<t_m> は当該材料が
    存在する列での平均膜厚（film_thickness_map の正値平均）。応力×厚さ積は
    反り（Stoney 則）を支配する量で、引張膜（+）はウェハを凸に、圧縮膜（-）
    は凹に反らせる。

    返り値: {"per_material": {name: {"thickness_um", "stress_mpa",
    "stress_thickness_N_per_m"}}, "net_N_per_m"}。
    """
    per: dict[str, dict] = {}
    net = 0.0
    for m in materials.all_materials():
        if m.stress_mpa == 0.0 or m.id == materials.AIR:
            continue
        tmap = film_thickness_map(wafer, m.id)
        film = tmap[tmap > 0]
        if film.size == 0:
            continue
        t_mean_um = float(film.mean())
        st = float(m.stress_mpa) * 1e6 * (t_mean_um * 1e-6)  # Pa·m = N/m
        per[m.name] = {
            "thickness_um": t_mean_um,
            "stress_mpa": float(m.stress_mpa),
            "stress_thickness_N_per_m": st,
        }
        net += st
    return {"per_material": per, "net_N_per_m": float(net)}


def wafer_bow_um(
    wafer: Wafer,
    wafer_diameter_mm: float = 300.0,
    substrate_thickness_um: float = 775.0,
    substrate_biaxial_modulus_gpa: float = 180.6,
) -> float:
    """残留膜応力による等価ウェハ反り量（中心たわみ, µm）を Stoney 則で推定する。

    Stoney 則の曲率 κ = 6·Σ(σ·t_f) / (M_s·t_s²)（M_s=E/(1-ν) はシリコン基板の
    二軸弾性率, t_s は基板厚）を用い、半径 R のウェハの中心たわみ δ = κ·R²/2
    を返す。正は引張膜による凸反り、負は圧縮膜による凹反り。シミュレーション
    領域は微小なので、標準 300mm/775µm ウェハ全面に同じ膜が成膜された場合の
    等価反りとして算出する（プロセス比較用の指標）。
    """
    st = film_stress_thickness(wafer)
    sum_sigma_tf = st["net_N_per_m"]  # N/m
    t_s = substrate_thickness_um * 1e-6  # m
    m_s = substrate_biaxial_modulus_gpa * 1e9  # Pa
    if t_s <= 0 or m_s <= 0:
        return 0.0
    kappa = 6.0 * sum_sigma_tf / (m_s * t_s * t_s)  # 1/m
    radius = (wafer_diameter_mm * 1e-3) / 2.0  # m
    bow = kappa * radius * radius / 2.0  # m
    return float(bow * 1e6)  # µm


def dishing_depth_um(wafer: Wafer, name_or_id) -> float:
    """指定材料（Cu 等の軟材料）上面が周囲フィールドより凹んだ量 (µm) を返す。

    ダマシン CMP では軟らかい金属が過研磨で皿状に凹む（ディッシング）。
    凹面の最深部（中央）と周囲フィールドの代表高さ（中央値）との差を返す。
    実機のディッシングは「中央が最も深い凹み量」として規定されるため、
    軟材料側は最深部（高さの下側 5 パーセンタイル）で代表する。凹んで
    いなければ 0。
    """
    grid = wafer.grid
    nz = grid.shape[0]
    soft_id = materials.get(name_or_id).id
    z_top = wafer.top_surface_z()  # (ny, nx), 固体なし=-1
    yy, xx = np.indices(z_top.shape)
    ztc = np.clip(z_top, 0, nz - 1)
    top_id = np.where(z_top >= 0, grid[ztc, yy, xx], materials.AIR)
    height = surface_height_map(wafer)  # µm, NaN where no solid
    is_soft_top = (z_top >= 0) & (top_id == soft_id)
    field_mask = (z_top >= 0) & (top_id != soft_id)
    if not is_soft_top.any() or not field_mask.any():
        return 0.0
    field_h = float(np.median(height[field_mask]))
    # 凹面の最深部（中央）を代表値とする（5 パーセンタイルでノイズに頑健化）。
    soft_h = float(np.percentile(height[is_soft_top], 5))
    return max(0.0, field_h - soft_h)


def interface_roughness_um(wafer: Wafer, name_a, name_b) -> float:
    """材料 A（下）と材料 B（上）の境界面の高さ揺らぎ (RMS, µm) を返す。

    各列で「直上に B が乗っている最上の A ボクセル」を境界とみなし、その境界
    高さの標準偏差（RMS）を界面粗さとする。表面粗さ surface_roughness_um と
    異なり、埋もれた層間（例: 基板/エピ、バリア/Cu）の凹凸を評価できる。
    境界が 2 列未満しか取れなければ 0。
    """
    a = materials.get(name_a).id
    b = materials.get(name_b).id
    grid = wafer.grid
    nz = grid.shape[0]
    mask_a = grid == a
    # 直上が B である A ボクセル（z 方向に B が A の 1 つ上）
    b_above = np.zeros_like(mask_a)
    b_above[:-1] = grid[1:] == b
    boundary = mask_a & b_above  # (nz, ny, nx)
    has = boundary.any(axis=0)
    if int(has.sum()) < 2:
        return 0.0
    # 各列で最上の境界 z
    top_boundary = nz - 1 - np.argmax(boundary[::-1, :, :], axis=0)
    heights = top_boundary[has].astype(float) * wafer.config.pitch_um
    return float(np.sqrt(np.mean((heights - heights.mean()) ** 2)))


def junction_depth_um(wafer: Wafer, dopant) -> float:
    """ドープ層（doped_n/doped_p 等）の接合深さ (µm) の中央値を返す。

    各列で、最上固体表面から当該ドーパントの最深ボクセルまでの垂直距離を
    求め、ドーパントを含む列にわたる中央値を接合深さ Xj とみなす。
    ドーパントが無ければ 0。
    """
    did = materials.get(dopant).id
    grid = wafer.grid
    mask = grid == did
    has = mask.any(axis=0)
    if not has.any():
        return 0.0
    z_top = wafer.top_surface_z()  # 固体なし=-1
    # 各列の最深ドーパント z（最初に True になる下からの位置）
    deepest = np.argmax(mask, axis=0)
    cols = has & (z_top >= 0)
    if not cols.any():
        return 0.0
    depth_vox = (z_top[cols] - deepest[cols]).astype(float)
    depth_vox = np.clip(depth_vox, 0.0, None)
    return float(np.median(depth_vox)) * wafer.config.pitch_um


def dopant_depth_profile(wafer: Wafer, dopant) -> np.ndarray:
    """高さ z ごとの当該ドーパントを含むボクセルの割合 (0..1) を長さ nz で返す。

    縦方向の存在率プロファイル。ピーク位置（飛程 Rp 相当）や分布の広がりの
    確認に使う。ドーパントが無ければ全 0。
    """
    did = materials.get(dopant).id
    grid = wafer.grid
    nz, ny, nx = grid.shape
    frac = (grid == did).reshape(nz, ny * nx).mean(axis=1)
    return frac.astype(np.float64)


def dominant_wavelength_um(wafer: Wafer) -> float:
    """表面高さマップの 2D FFT から支配的な凹凸周期（波長, µm）を返す。

    表面高さ（NaN は平均で補間）の平均を引いて 2D FFT し、振幅が最大となる
    空間周波数に対応する波長を返す。CVD roughness や Photo edge_blur が作る
    周期的な凹凸の特徴波長の評価に使う。平坦面（有意な変動なし）は 0。
    """
    height = surface_height_map(wafer)
    valid = ~np.isnan(height)
    if valid.sum() < 4:
        return 0.0
    h = np.where(valid, height, np.nanmean(height))
    h = h - h.mean()
    ny, nx = h.shape
    if np.allclose(h, 0.0):
        return 0.0
    spec = np.abs(np.fft.fft2(h))
    fy = np.fft.fftfreq(ny)[:, None]  # サイクル/ピクセル
    fx = np.fft.fftfreq(nx)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    spec_flat = spec.copy()
    spec_flat[0, 0] = 0.0  # DC 成分を除外
    idx = int(np.argmax(spec_flat))
    f = float(fr.flat[idx])
    if f <= 0:
        return 0.0
    # 波長(ピクセル) = 1/f → µm へ
    return (1.0 / f) * wafer.config.pitch_um


def electrical_continuity(wafer: Wafer, name_or_id, axis: str = "x") -> dict:
    """導体材料が指定軸の両端を連結しているか（導通/オープン）を判定する。

    指定材料のボクセルを 6 近傍でラベリングし、指定軸（"x"/"y"）の最小端
    （index 0 の面）と最大端（最終 index の面）の両方に同時に接する連結成分が
    あれば「導通」とみなす。エッチング過多による配線断線（オープン不良）や、
    パターンが両端に届かない欠損の検出に使う。
    返り値: {"connected", "n_components", "spanning_components", "largest_um3"}。
    材料が存在しなければ全て 0/False。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    mask = grid == mat.id
    if not mask.any():
        return {
            "connected": False,
            "n_components": 0,
            "spanning_components": 0,
            "largest_um3": 0.0,
        }
    a = {"x": 2, "y": 1, "z": 0}.get(axis, 2)  # grid 軸 [z,y,x]
    structure = ndimage.generate_binary_structure(3, 1)  # 6 近傍
    labels, n = ndimage.label(mask, structure=structure)
    n_axis = grid.shape[a]
    lo = set(np.unique(np.take(labels, 0, axis=a)))
    hi = set(np.unique(np.take(labels, n_axis - 1, axis=a)))
    spanning = (lo & hi) - {0}  # 両端に接するラベル（背景0除く）
    sizes = np.bincount(labels.ravel())[1:]
    vox = wafer.config.pitch_um ** 3
    return {
        "connected": len(spanning) > 0,
        "n_components": int(n),
        "spanning_components": int(len(spanning)),
        "largest_um3": float(sizes.max()) * vox if sizes.size else 0.0,
    }


def line_resistance_ohm(wafer: Wafer, name_or_id, axis: str = "x") -> float:
    """導体配線の電気抵抗（Ω）を断面積を考慮して推定する。

    指定軸（"x"/"y"）に沿って配線を薄切りし、各スライスの断面積
    A=（材料ボクセル数×pitch²）に対し R=Σ ρ·Δl/A を積算する直列抵抗モデル。
    幅が細る箇所ほど抵抗が増える（ネッキング/エレクトロマイグレーション
    リスクの検出に有用）。途中で断面積 0 のスライスがあれば断線とみなし inf を返す。
    ρ は材料の resistivity_ohm_um。非導体（ρ=0）や材料不在は inf。
    """
    mat = materials.get(name_or_id)
    rho = mat.resistivity_ohm_um
    if rho <= 0:
        return float("inf")
    grid = wafer.grid
    mask = grid == mat.id
    if not mask.any():
        return float("inf")
    a = {"x": 2, "y": 1, "z": 0}.get(axis, 2)
    pitch = wafer.config.pitch_um
    # 軸に垂直な各スライスの断面ボクセル数
    other_axes = tuple(ax for ax in (0, 1, 2) if ax != a)
    area_vox = mask.sum(axis=other_axes)  # 長さ n_axis の配列
    # 配線が存在する範囲のみを対象（前後の空きスライスは無視）
    present = np.nonzero(area_vox > 0)[0]
    if present.size == 0:
        return float("inf")
    lo, hi = int(present.min()), int(present.max())
    total = 0.0
    for i in range(lo, hi + 1):
        av = int(area_vox[i])
        if av == 0:
            return float("inf")  # 途中で途切れ＝オープン
        # R_slice = ρ·Δl / A = ρ·pitch / (av·pitch²) = ρ / (av·pitch)
        total += rho / (av * pitch)
    return float(total)


def resistance_at_temperature(
    wafer: Wafer, name_or_id, temperature_c: float,
    axis: str = "x", ref_temp_c: float = 20.0,
) -> dict:
    """温度依存の配線抵抗 R(T)=R₀·(1+TCR·(T−T₀)) を返す。

    R₀ は基準温度 ref_temp_c での line_resistance_ohm、TCR は材料の tcr_per_k。
    金属は正 TCR のため高温で抵抗が増える（自己発熱との正帰還評価に有用）。返す辞書:
      r_ref_ohm（基準温度抵抗）/ r_t_ohm（温度 T の抵抗）/ ratio（R(T)/R₀）。
    断線/非導体では inf。TCR が物理的に負の温度では R が 0 未満にならないよう 0 でクリップ。
    """
    mat = materials.get(name_or_id)
    r0 = line_resistance_ohm(wafer, name_or_id, axis)
    if r0 == float("inf"):
        return {"r_ref_ohm": float("inf"), "r_t_ohm": float("inf"), "ratio": float("nan")}
    factor = max(0.0, 1.0 + mat.tcr_per_k * (temperature_c - ref_temp_c))
    return {
        "r_ref_ohm": float(r0),
        "r_t_ohm": float(r0 * factor),
        "ratio": float(factor),
    }


def sheet_resistance_ohm_sq(wafer: Wafer, name_or_id) -> float:
    """導体薄膜のシート抵抗（Ω/sq）を返す。標準的な薄膜評価指標。

    Rs = ρ / t（ρ=resistivity_ohm_um, t=膜の平均厚 µm）。膜が薄いほど Rs は
    大きい。4 探針測定で得られる値に対応し、line_resistance_ohm（全抵抗 Ω）と
    異なり形状に依らないシート単位の量。膜が存在しない/非導体は inf。
    """
    mat = materials.get(name_or_id)
    rho = mat.resistivity_ohm_um
    if rho <= 0:
        return float("inf")
    stats = film_thickness_stats(wafer, name_or_id)
    t = stats.get("mean", 0.0)
    if t <= 0:
        return float("inf")
    return float(rho / t)


def min_spacing_um(wafer: Wafer, name_a, name_b) -> float:
    """2 材料領域の最小間隔（µm）を返す。DRC（設計規則）チェック用。

    材料 A の各ボクセルから最も近い材料 B のボクセルまでのユークリッド距離の
    最小値を返す。距離は等方ピッチで正規化する。両材料が接触している場合は 0
    （ショート/ブリッジ）。どちらかの材料が存在しなければ inf。
    隣接配線間スペースがルールを満たすか（ショート不良リスク）の検査に使う。
    """
    a = materials.get(name_a)
    b = materials.get(name_b)
    grid = wafer.grid
    mask_a = grid == a.id
    mask_b = grid == b.id
    if not mask_a.any() or not mask_b.any():
        return float("inf")
    if (mask_a & mask_b).any():  # 同一 ID 指定など
        return 0.0
    if (mask_a & ndimage.binary_dilation(mask_b)).any():
        return 0.0  # 隣接（接触）
    # B からの距離場を求め、A 位置で最小を取る（等方ピッチ）
    dist = ndimage.distance_transform_edt(~mask_b)
    return float(dist[mask_a].min()) * wafer.config.pitch_um


def contact_area_um2(wafer: Wafer, name_a, name_b) -> float:
    """材料 A と B が接する界面の面積 (µm²) を面ペア数で正確に数える。

    interface_width_um が A ボクセル数を数えるのに対し、こちらは A-B の
    隣接「面」の総数 × pitch² を返す（1 ボクセルが複数面で接していれば複数
    カウント）。コンタクト/ビアの実効接触面積として contact_resistance_ohm に
    使う。周期境界はラップさせない。接触が無ければ 0。
    """
    a = materials.get(name_a).id
    b = materials.get(name_b).id
    grid = wafer.grid
    mask_a = grid == a
    mask_b = grid == b
    faces = 0
    for axis in range(3):
        sa = [slice(None)] * 3
        sb = [slice(None)] * 3
        sa[axis] = slice(0, -1)
        sb[axis] = slice(1, None)
        a_lo, b_hi = mask_a[tuple(sa)], mask_b[tuple(sb)]
        a_hi, b_lo = mask_a[tuple(sb)], mask_b[tuple(sa)]
        faces += int((a_lo & b_hi).sum()) + int((a_hi & b_lo).sum())
    return float(faces) * (wafer.config.pitch_um ** 2)


def contact_resistance_ohm(
    wafer: Wafer, name_a, name_b, specific_contact_resistivity_ohm_um2: float = 10.0
) -> float:
    """A-B コンタクトの接触抵抗 (Ω) を Rc = ρc / A で推定する。

    ρc は比接触抵抗（Ω·µm², 既定 10 ≈ 1e-7 Ω·cm² の金属-シリコン典型値）、
    A は contact_area_um2 で求めた実効接触面積。接触面積が大きいほど抵抗は
    小さい。接触が無ければ inf。コンタクト/ビアのサイズ不足による抵抗増大
    （オープン気味）の評価に使う。
    """
    if specific_contact_resistivity_ohm_um2 <= 0:
        raise ValueError("比接触抵抗は正の値が必要です。")
    area = contact_area_um2(wafer, name_a, name_b)
    if area <= 0:
        return float("inf")
    return float(specific_contact_resistivity_ohm_um2 / area)


# 真空誘電率 ε0 を「容量[fF] = _EPS0_FF·εr·A[µm²]/d[µm]」の係数に換算した定数。
#   ε0 = 8.854e-12 F/m → 8.854e-3 fF·µm/µm²（µm 単位系での値）。
_EPS0_FF = 8.854e-3


def _conductor_ids() -> set[int]:
    """導体（ρ>0）の材料 ID 集合を返す。"""
    return {m.id for m in materials.all_materials() if m.resistivity_ohm_um > 0}


def permittivity_field(wafer: Wafer) -> np.ndarray:
    """各ボクセルの比誘電率 εr を格納した配列を返す（容量計算の補助）。

    誘電体材料は定義済み rel_permittivity、空気は 1.0、それ以外（未設定の
    導体など）は既定 1.0。導体ボクセルは容量計算側でマスク除外される。
    """
    grid = wafer.grid
    eps = np.ones(grid.shape, dtype=float)  # 既定 = 真空/空気 1.0
    for m in materials.all_materials():
        if m.rel_permittivity > 0:
            eps[grid == m.id] = m.rel_permittivity
    return eps


def parasitic_capacitance_ff(wafer: Wafer, name_a, name_b) -> float:
    """2 導体間の寄生容量（fF）を面対向ライン走査で推定する。

    3 軸それぞれに沿ってボクセル列を走査し、導体 A の面と導体 B の面が
    「間に他導体を挟まず」直接対向する区間を見つける。各対向区間を断面
    pitch²・電極間距離 d（µm）・介在誘電体平均比誘電率 εr の平行平板素子と
    見なし、dC = ε0·εr·pitch²/d を全方向・全列で積算する。

    平行平板（面積 Aₚ・間隙 g・一様誘電体 εr）では解析値 ε0·εr·Aₚ/g に
    厳密一致し、面内で隣接する配線間の対向（カップリング容量）も自動的に
    加算される。間に第 3 の導体があるとそこで遮蔽される。配線間/対基板容量の
    評価や RC 遅延（rc_delay_ps）の入力に使う。どちらかの材料が存在しなければ
    0、同一材料指定も 0。
    """
    a = materials.get(name_a)
    b = materials.get(name_b)
    if a.id == b.id:
        return 0.0
    grid = wafer.grid
    if not (grid == a.id).any() or not (grid == b.id).any():
        return 0.0
    pitch = wafer.config.pitch_um
    eps_field = permittivity_field(wafer)
    is_cond = np.isin(grid, list(_conductor_ids()))
    pair = {a.id, b.id}
    coef0 = _EPS0_FF * (pitch ** 2)

    total = 0.0
    for axis in (0, 1, 2):
        # 走査軸を末尾へ移動して 2 次元 (線数, 線長) に展開
        g2 = np.moveaxis(grid, axis, -1).reshape(-1, grid.shape[axis])
        e2 = np.moveaxis(eps_field, axis, -1).reshape(-1, grid.shape[axis])
        c2 = np.moveaxis(is_cond, axis, -1).reshape(-1, grid.shape[axis])
        # A・B 両方を含む線のみ処理（大半の線を高速にスキップ）
        targets = np.nonzero((g2 == a.id).any(1) & (g2 == b.id).any(1))[0]
        for li in targets:
            line = g2[li]
            pos = np.nonzero(c2[li])[0]  # 導体ボクセルの位置
            for k in range(pos.size - 1):
                p, q = int(pos[k]), int(pos[k + 1])
                if {int(line[p]), int(line[q])} != pair:
                    continue  # 対向ペアでない（同材料 or 第3導体）
                sep = (q - p) * pitch  # 電極面間距離 µm
                eps_gap = float(e2[li][p + 1:q].mean()) if q - p > 1 else 1.0
                total += coef0 * eps_gap / sep
    return float(total)


def rc_delay_ps(wafer: Wafer, line_material, return_material, axis: str = "x") -> float:
    """配線の RC 遅延（ps）を集中定数モデル τ=R·C で推定する。

    R は line_resistance_ohm（指定軸の直列抵抗 Ω）、C は line_material と
    return_material（対向電極=隣接配線や基板）間の parasitic_capacitance_ff
    （fF）。τ[ps] = R[Ω]·C[F]·1e12 = R·C_fF·1e-3。断線（R=inf）や
    容量 0 では inf／0 を返す。配線遅延の一次見積りに使う。
    """
    r = line_resistance_ohm(wafer, line_material, axis)
    if r == float("inf"):
        return float("inf")
    c_ff = parasitic_capacitance_ff(wafer, line_material, return_material)
    return float(r * c_ff * 1e-3)


def _line_slice_resistances(wafer: Wafer, mat_id: int, axis: str):
    """配線を指定軸に沿って薄切りした各スライスの抵抗 [Ω] 配列を返す（内部用）。

    R_slice = ρ/(断面ボクセル数·pitch)。配線が存在する範囲のみ。断面 0（断線）が
    あれば None。配線/非導体不在も None。
    """
    mat = materials.BY_ID.get(int(mat_id))
    if mat is None or mat.resistivity_ohm_um <= 0:
        return None
    grid = wafer.grid
    mask = grid == mat.id
    if not mask.any():
        return None
    a = {"x": 2, "y": 1, "z": 0}.get(axis, 2)
    pitch = wafer.config.pitch_um
    other = tuple(ax for ax in (0, 1, 2) if ax != a)
    area_vox = mask.sum(axis=other)
    present = np.nonzero(area_vox > 0)[0]
    lo, hi = int(present.min()), int(present.max())
    seg = area_vox[lo:hi + 1]
    if (seg == 0).any():
        return None  # 途中断線
    return mat.resistivity_ohm_um / (seg.astype(float) * pitch)


def elmore_delay_ps(
    wafer: Wafer, line_material, return_material, axis: str = "x"
) -> dict:
    """分布 RC 配線の Elmore 遅延（ps）を推定する（集中定数 R·C より高精度）。

    配線を指定軸方向に薄切りし、各スライスの抵抗 rₖ（断面積から）と容量 cₖ
    （対 return_material 総容量を在線スライスへ均等配分）から、
      τ_Elmore = Σₖ (Σ_{j≤k} rⱼ)·cₖ
    を計算する。一様配線では τ≈½·R·C（集中定数の半分）になり、分布効果を捉える。
    返す辞書: elmore_delay_ps / lumped_rc_ps / resistance_ohm / capacitance_ff。
    断線・容量 0 ではそれぞれ inf／0。
    """
    line = materials.get(line_material)
    r_slices = _line_slice_resistances(wafer, line.id, axis)
    c_ff = parasitic_capacitance_ff(wafer, line_material, return_material)
    r_total = line_resistance_ohm(wafer, line_material, axis)
    lumped = float("inf") if r_total == float("inf") else float(r_total * c_ff * 1e-3)
    if r_slices is None:
        return {"elmore_delay_ps": float("inf"), "lumped_rc_ps": lumped,
                "resistance_ohm": r_total, "capacitance_ff": c_ff}
    n = len(r_slices)
    c_f = c_ff * 1e-15 / n  # 各スライスの容量 [F]（均等配分）
    cum_r = np.cumsum(r_slices)  # 各ノードまでの上流抵抗 [Ω]
    tau_s = float(np.sum(cum_r * c_f))  # Σ (ΣR)·c [s]
    return {
        "elmore_delay_ps": tau_s * 1e12,
        "lumped_rc_ps": lumped,
        "resistance_ohm": float(r_total),
        "capacitance_ff": float(c_ff),
    }


def blech_immortal(
    wafer: Wafer, conductor, current_ma: float, axis: str = "x",
    jl_threshold_a_cm: float = 4000.0,
) -> dict:
    """エレクトロマイグレーションの Blech 不死条件（j·L < (jL)_crit）を判定する。

    配線の電流密度 j（平均断面）と長さ L の積 j·L が臨界値未満なら、応力勾配が
    EM 駆動力と釣り合い空孔成長が止まる＝EM 不死（故障しない）。j·L が大きい
    （長い/高電流密度の配線）ほど故障しやすい。返す辞書:
      j_a_cm2 / length_cm / jl_product_a_cm / threshold_a_cm / immortal(bool)。
    （jL）_crit は材料・温度依存（既定 4000 A/cm ≈ Cu の代表値）。
    """
    mat = materials.get(conductor)
    grid = wafer.grid
    mask = grid == mat.id
    if not mask.any() or mat.resistivity_ohm_um <= 0:
        return {"j_a_cm2": 0.0, "length_cm": 0.0, "jl_product_a_cm": 0.0,
                "threshold_a_cm": jl_threshold_a_cm, "immortal": True}
    a = {"x": 2, "y": 1, "z": 0}.get(axis, 2)
    pitch = wafer.config.pitch_um
    other = tuple(ax for ax in (0, 1, 2) if ax != a)
    area_vox = mask.sum(axis=other)
    present = np.nonzero(area_vox > 0)[0]
    lo, hi = int(present.min()), int(present.max())
    length_um = (hi - lo + 1) * pitch
    mean_area_um2 = float(area_vox[lo:hi + 1].mean()) * (pitch ** 2)
    i_a = abs(current_ma) * 1e-3
    j = i_a / (mean_area_um2 * 1e-8) if mean_area_um2 > 0 else float("inf")
    length_cm = length_um * 1e-4
    jl = j * length_cm
    return {
        "j_a_cm2": float(j),
        "length_cm": float(length_cm),
        "jl_product_a_cm": float(jl),
        "threshold_a_cm": float(jl_threshold_a_cm),
        "immortal": bool(jl < jl_threshold_a_cm),
    }


def current_density_stats(
    wafer: Wafer, name_or_id, current_ma: float, axis: str = "x"
) -> dict:
    """配線に電流 current_ma[mA] を流したときの電流密度（A/cm²）統計を返す。

    指定軸に沿って配線を薄切りし、各スライスの断面積 A=（ボクセル数×pitch²）で
    J=I/A を求める。最小断面（ネッキング箇所）で J が最大になる。返す辞書:
      - j_max_a_cm2 / j_mean_a_cm2: 最大・平均電流密度（A/cm²）
      - area_min_um2: 最小断面積（µm²）
      - bottleneck_index: 最小断面のスライス番号（指定軸の位置）
    配線が無い/非導体/断線（断面 0）の場合は j_max=inf。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    mask = grid == mat.id
    if mat.resistivity_ohm_um <= 0:  # 非導体（誘電体）は電流路にならないため対象外
        return {"j_max_a_cm2": float("inf"), "j_mean_a_cm2": 0.0,
                "area_min_um2": 0.0, "bottleneck_index": -1}
    if not mask.any() or current_ma == 0:
        return {"j_max_a_cm2": float("inf") if not mask.any() else 0.0,
                "j_mean_a_cm2": 0.0, "area_min_um2": 0.0, "bottleneck_index": -1}
    a = {"x": 2, "y": 1, "z": 0}.get(axis, 2)
    pitch = wafer.config.pitch_um
    other = tuple(ax for ax in (0, 1, 2) if ax != a)
    area_vox = mask.sum(axis=other)
    present = np.nonzero(area_vox > 0)[0]
    lo, hi = int(present.min()), int(present.max())
    seg = area_vox[lo:hi + 1]
    if (seg == 0).any():  # 途中断線
        return {"j_max_a_cm2": float("inf"), "j_mean_a_cm2": float("inf"),
                "area_min_um2": 0.0, "bottleneck_index": int(lo + np.argmin(seg))}
    area_um2 = seg * (pitch ** 2)
    area_cm2 = area_um2 * 1e-8  # µm² → cm²
    i_a = abs(current_ma) * 1e-3  # mA → A
    j = i_a / area_cm2
    bottleneck = int(lo + np.argmin(area_um2))
    return {
        "j_max_a_cm2": float(j.max()),
        "j_mean_a_cm2": float(j.mean()),
        "area_min_um2": float(area_um2.min()),
        "bottleneck_index": bottleneck,
    }


def current_density_profile(
    wafer: Wafer, name_or_id, current_ma: float, axis: str = "x"
) -> dict:
    """配線に沿った電流密度プロファイル J(位置) を返す（EM ホットスポット可視化）。

    指定軸の各スライスの断面積から J=I/A を求めた配列を返す。最小断面で J が
    ピークになり、ネッキング箇所＝EM 故障の起点を位置付きで可視化できる。返す辞書:
      - position_um: 各スライスの軸方向位置（µm, 配線の存在範囲）
      - area_um2: 各スライスの断面積（µm²）
      - j_a_cm2: 各スライスの電流密度（A/cm²）
    配線が無い/非導体/断線では空配列。
    """
    mat = materials.get(name_or_id)
    grid = wafer.grid
    mask = grid == mat.id
    empty = {"position_um": np.array([]), "area_um2": np.array([]),
             "j_a_cm2": np.array([])}
    if mat.resistivity_ohm_um <= 0 or not mask.any():
        return empty
    a = {"x": 2, "y": 1, "z": 0}.get(axis, 2)
    pitch = wafer.config.pitch_um
    other = tuple(ax for ax in (0, 1, 2) if ax != a)
    area_vox = mask.sum(axis=other)
    present = np.nonzero(area_vox > 0)[0]
    lo, hi = int(present.min()), int(present.max())
    seg = area_vox[lo:hi + 1]
    if (seg == 0).any():
        return empty  # 断線
    area_um2 = seg * (pitch ** 2)
    i_a = abs(current_ma) * 1e-3
    j = i_a / (area_um2 * 1e-8)
    pos = (np.arange(lo, hi + 1)) * pitch
    return {"position_um": pos, "area_um2": area_um2, "j_a_cm2": j}


def tlm_extract(spacings_um, resistances_ohm, width_um: float) -> dict:
    """TLM（伝送線路法）で接触抵抗・シート抵抗・伝送長を抽出する。

    複数のコンタクト間隔 L に対する全抵抗 R_total = 2·Rc + (Rsheet/W)·L の直線
    （R vs L）を最小二乗回帰し、傾き=Rsheet/W、切片=2·Rc から抽出する。
      - sheet_resistance_ohm_sq = 傾き × W
      - contact_resistance_ohm = 切片 / 2
      - transfer_length_um = |x 切片|/2 = Rc·W/Rsheet
    返す辞書: 上記 3 値 + slope/intercept。点が 2 点未満や幅 ≤0 は ValueError。
    """
    spac = np.asarray(spacings_um, dtype=float)
    res = np.asarray(resistances_ohm, dtype=float)
    if spac.size < 2 or res.size != spac.size:
        raise ValueError("spacings と resistances は同数で 2 点以上必要です。")
    if width_um <= 0:
        raise ValueError("幅 width_um は正の値が必要です。")
    slope, intercept = np.polyfit(spac, res, 1)
    rsheet = slope * width_um
    rc = intercept / 2.0
    # 傾き(シート抵抗)が実質ゼロだと伝送長は定義できない（Lt=Rc·W/Rsheet→∞）。
    # 微小な数値ノイズで巨大値にならないよう、抵抗スケールに対する相対閾値でガード。
    slope_scale = float(np.ptp(res)) / max(float(np.ptp(spac)), 1e-30)
    if abs(slope) <= 1e-9 * max(slope_scale, 1.0):
        lt = float("inf")
    else:
        lt = abs(intercept / slope) / 2.0
    return {
        "sheet_resistance_ohm_sq": float(rsheet),
        "contact_resistance_ohm": float(rc),
        "transfer_length_um": float(lt),
        "slope_ohm_per_um": float(slope),
        "intercept_ohm": float(intercept),
    }


def electromigration_risk(
    wafer: Wafer, name_or_id, current_ma: float, axis: str = "x"
) -> dict:
    """配線のエレクトロマイグレーション（EM）リスクを判定する。

    current_density_stats の最大電流密度 J_max を、材料の許容電流密度
    em_jmax_a_cm2 と比較する。返す辞書: j_max_a_cm2 / limit_a_cm2 /
    margin（=limit/J_max, 1 未満で超過）/ fail（bool, 限界超過）。
    許容値が未設定（0）の材料や非導体では判定不可として fail=False, margin=inf。
    """
    mat = materials.get(name_or_id)
    stats = current_density_stats(wafer, name_or_id, current_ma, axis)
    j_max = stats["j_max_a_cm2"]
    limit = mat.em_jmax_a_cm2
    if limit <= 0:
        return {"j_max_a_cm2": j_max, "limit_a_cm2": 0.0,
                "margin": float("inf"), "fail": False}
    margin = float("inf") if j_max == 0 else limit / j_max
    return {
        "j_max_a_cm2": j_max,
        "limit_a_cm2": limit,
        "margin": margin,
        "fail": bool(j_max > limit),
    }


def dielectric_breakdown(
    wafer: Wafer, name_a, name_b, voltage_v: float
) -> dict:
    """2 導体間に電圧 voltage_v[V] を印加したときの絶縁破壊リスクを判定する。

    導体 A・B の最小間隔 g（min_spacing_um）を電極間距離とみなし、最大電界
    E=V/g を求めて MV/cm に換算する。間隙を占める誘電体の絶縁破壊電界
    breakdown_field_mv_cm（複数あれば最小値=最弱）と比較する。返す辞書:
      - field_mv_cm: 最大電界（MV/cm）
      - gap_um: 最小間隙（µm）
      - breakdown_field_mv_cm: 介在誘電体の破壊電界（MV/cm）
      - margin: 破壊電界/印加電界（1 未満で破壊）
      - fail: 破壊電界超過か
    接触（g=0）は field=inf, fail=True。どちらかの導体が無ければ判定不可。
    """
    g_um = min_spacing_um(wafer, name_a, name_b)
    if g_um == float("inf"):
        return {"field_mv_cm": 0.0, "gap_um": float("inf"),
                "breakdown_field_mv_cm": 0.0, "margin": float("inf"), "fail": False}
    # 介在誘電体の破壊電界（最小間隔ギャップ経路上の誘電体の最小値=最弱点）。
    # 外接直方体全体だと、オフセット配線の上方/側方に広がる空気など実際の
    # 最短ギャップと無関係な弱誘電体を拾ってしまうため、最近接 A/B ペアを結ぶ
    # 直線経路上の材料だけを対象にする。
    a = materials.get(name_a)
    b = materials.get(name_b)
    grid = wafer.grid
    mask_a = grid == a.id
    mask_b = grid == b.id
    # B からの距離場と最近傍 B のインデックスから、最小間隔を与える A/B ペアを特定
    dist, inds = ndimage.distance_transform_edt(~mask_b, return_indices=True)
    da = np.where(mask_a, dist, np.inf)
    pa = np.unravel_index(int(np.argmin(da)), da.shape)        # 最小間隔の A ボクセル
    pb = tuple(int(inds[k][pa]) for k in range(3))            # その最近傍 B ボクセル
    # pa→pb 直線上をサンプリングし、ギャップを占める材料 ID を集める
    steps = max(abs(pb[k] - pa[k]) for k in range(3)) + 1
    gap_ids = {
        int(grid[tuple(int(round(pa[k] + (pb[k] - pa[k]) * t)) for k in range(3))])
        for t in np.linspace(0.0, 1.0, steps)
    }
    fields = [
        m.breakdown_field_mv_cm
        for m in materials.all_materials()
        if m.breakdown_field_mv_cm > 0 and m.id in gap_ids
    ]
    bd = min(fields) if fields else 0.0
    if g_um <= 0:  # 接触＝即破壊
        return {"field_mv_cm": float("inf"), "gap_um": 0.0,
                "breakdown_field_mv_cm": bd, "margin": 0.0, "fail": True}
    # E[MV/cm] = V[V] / g[µm] × (1e-6 MV/V) / (1e-4 cm/µm) = V/g × 1e-2
    field_mv_cm = abs(voltage_v) / g_um * 1e-2
    margin = float("inf") if field_mv_cm == 0 else (bd / field_mv_cm if bd > 0 else float("inf"))
    return {
        "field_mv_cm": float(field_mv_cm),
        "gap_um": float(g_um),
        "breakdown_field_mv_cm": float(bd),
        "margin": float(margin),
        "fail": bool(bd > 0 and field_mv_cm > bd),
    }


def yield_estimate(
    defect_density_per_cm2: float, die_area_cm2: float, model: str = "murphy"
) -> float:
    """欠陥密度から歩留り（0〜1）を推定する。

    AD = defect_density × die_area（die あたり平均欠陥数）として、
      - "poisson":  Y = exp(-AD)
      - "murphy":   Y = ((1-exp(-AD))/AD)²  （欠陥分布のばらつきを考慮、業界標準）
      - "seeds":    Y = 1/(1+AD)            （Seeds の負の二項近似）
    を返す。AD=0 では Y=1。負の入力は ValueError。
    """
    if defect_density_per_cm2 < 0 or die_area_cm2 < 0:
        raise ValueError("欠陥密度・ダイ面積は非負である必要があります。")
    ad = defect_density_per_cm2 * die_area_cm2
    if ad <= 0:
        return 1.0
    if model == "poisson":
        return float(np.exp(-ad))
    if model == "seeds":
        return float(1.0 / (1.0 + ad))
    if model == "murphy":
        return float(((1.0 - np.exp(-ad)) / ad) ** 2)
    raise ValueError(f"未知の歩留りモデル: {model!r}（poisson/murphy/seeds）")


def killer_defect_count(wafer: Wafer) -> int:
    """defect_report からキラー欠陥（致命欠陥）の総数を数える。

    ボイド連結成分数＋各材料のピンホール数＋エッチ残渣片数の合計。
    歩留り推定の入力（ダイ内検出欠陥数）に使う。
    """
    rep = defect_report(wafer)
    n = int(rep["voids"].get("count", 0))
    for d in rep["per_material"].values():
        n += int(d["pinhole"].get("count", 0))
        n += int(d["residue"].get("count", 0))
    return n


def thermal_conductivity_field(wafer: Wafer) -> np.ndarray:
    """各ボクセルの熱伝導率 k（W/m·K）を格納した配列を返す。

    定義済み thermal_conductivity_w_mk を使用。未設定（0）の材料は空気相当
    （0.026 W/m·K）として扱う。熱抵抗計算の補助。
    """
    grid = wafer.grid
    air_k = materials.BY_NAME["air"].thermal_conductivity_w_mk
    k = np.full(grid.shape, air_k, dtype=float)
    for m in materials.all_materials():
        if m.thermal_conductivity_w_mk > 0:
            k[grid == m.id] = m.thermal_conductivity_w_mk
    return k


def thermal_resistance_map(wafer: Wafer) -> np.ndarray:
    """各 (y, x) 列の縦方向熱抵抗マップ（K/W）を返す。

    基板底（z=0）から各列の最上位固体ボクセルまでを、ボクセルを直列の
    熱抵抗 R=Δz/(k·A)（Δz=A^{1/2}=pitch）と見なして積算する。各列断面は
    pitch²。途中の空気ボイドは低 k の熱障壁として加算される（自己発熱の
    放熱経路評価）。固体の無い列は inf。
    """
    kf = thermal_conductivity_field(wafer)
    p_m = wafer.config.pitch_um * 1e-6  # µm → m
    # R_voxel = Δz/(k·A) = p_m/(k·p_m²) = 1/(k·p_m)。列方向の累積和を取る。
    csum = np.cumsum(1.0 / kf, axis=0)  # csum[z,y,x] = Σ_{0..z} 1/k
    top = wafer.top_surface_z()  # (ny, nx), 固体無しは -1
    ny, nx = top.shape
    rmap = np.full((ny, nx), np.inf, dtype=float)
    has = top >= 0
    yy, xx = np.nonzero(has)
    rmap[yy, xx] = csum[top[yy, xx], yy, xx] / p_m
    return rmap


def thermal_resistance_k_w(wafer: Wafer) -> float:
    """ウェハ全体の縦方向熱抵抗（K/W）を返す（列を並列接続）。

    thermal_resistance_map の各列を並列熱抵抗とみなし 1/R_total=Σ 1/R_col を
    取る。基板→表面の実効熱抵抗で、低 k 膜が厚いほど大きい。固体が無ければ inf。
    """
    rmap = thermal_resistance_map(wafer)
    finite = np.isfinite(rmap)
    if not finite.any():
        return float("inf")
    return float(1.0 / np.sum(1.0 / rmap[finite]))


def temperature_rise_k(wafer: Wafer, power_w: float) -> float:
    """消費電力 power_w[W] による定常温度上昇 ΔT[K] を返す。

    ΔT = P · R_th（R_th は thermal_resistance_k_w の縦方向熱抵抗）。
    放熱経路が無い（R_th=inf）場合は inf。低 k 膜が厚いほど ΔT は大きい。
    自己発熱（ジュール熱）による接合温度上昇の一次評価に使う。
    """
    if power_w < 0:
        raise ValueError("消費電力は非負である必要があります。")
    rth = thermal_resistance_k_w(wafer)
    if rth == float("inf"):
        return float("inf")
    return float(power_w * rth)


def joule_self_heating_k(
    wafer: Wafer, conductor, current_ma: float, axis: str = "x"
) -> dict:
    """配線のジュール自己発熱による温度上昇を返す。

    配線抵抗 R（line_resistance_ohm）と電流 I からジュール発熱 P=I²R を求め、
    縦方向熱抵抗 R_th を介した定常温度上昇 ΔT=P·R_th を算出する。返す辞書:
      - power_w: ジュール発熱（W）
      - resistance_ohm: 配線抵抗（Ω）
      - delta_t_k: 温度上昇（K）
    断線（R=inf）や放熱経路無しでは ΔT=inf。
    """
    r = line_resistance_ohm(wafer, conductor, axis)
    i_a = abs(current_ma) * 1e-3
    if r == float("inf"):
        return {"power_w": float("inf"), "resistance_ohm": float("inf"),
                "delta_t_k": float("inf")}
    power = i_a ** 2 * r
    return {
        "power_w": float(power),
        "resistance_ohm": float(r),
        "delta_t_k": temperature_rise_k(wafer, power),
    }


def temperature_field_2d(
    wafer: Wafer, source_mask: np.ndarray, total_power_w: float, y_index: int | None = None
) -> np.ndarray:
    """2.5D 熱拡散ソルバで XZ 断面の温度上昇分布 ΔT[K] を返す。

    指定 y 断面で定常熱伝導方程式 ∇·(k∇T)=−q を有限体積・疎行列直接解法で解く。
    基板最下行（z=0）を温度基準（ヒートシンク, ΔT=0）の Dirichlet 境界、他境界は
    断熱（Neumann）とする。発熱は source_mask（3D bool, True が発熱領域）の当該
    断面ボクセルに、単位 y 長あたり total_power_w/Ly を均等分配する。横方向の熱拡散
    （ヒートスプレッディング）を捉え、均一全面発熱では 1D 熱抵抗の ΔT=P·R_th に一致する。
    返り値は (nz, nx) の ΔT[K] 配列。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    grid = wafer.grid
    nz, ny, nx = grid.shape
    if y_index is None:
        y_index = ny // 2
    kf = thermal_conductivity_field(wafer)[:, y_index, :]  # (nz, nx) W/m·K
    src2d = source_mask[:, y_index, :]
    pitch_m = wafer.config.pitch_um * 1e-6
    ly_m = ny * pitch_m
    n_src = int(src2d.sum())
    # 単位 y 長あたり発熱 [W/m]、各発熱セルに均等分配
    p_per_depth = (total_power_w / ly_m) if ly_m > 0 else 0.0
    s_cell = (p_per_depth / n_src) if n_src > 0 else 0.0

    sink = np.zeros((nz, nx), dtype=bool)
    sink[0, :] = True  # 基板底=ヒートシンク
    idx = -np.ones((nz, nx), dtype=int)
    unk = np.argwhere(~sink)
    for k, (i, j) in enumerate(unk):
        idx[i, j] = k
    m = len(unk)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def k_face(i, j, ii, jj):
        a, c = kf[i, j], kf[ii, jj]
        return 2.0 * a * c / (a + c) if (a + c) > 0 else 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(m)
    for k, (i, j) in enumerate(unk):
        diag = 0.0
        for di, dj in neigh:
            ii, jj = i + di, j + dj
            if not (0 <= ii < nz and 0 <= jj < nx):
                continue
            kfc = k_face(i, j, ii, jj)
            diag += kfc
            if sink[ii, jj]:
                pass  # ΔT=0 のシンク → 寄与 0
            else:
                rows.append(k)
                cols.append(idx[ii, jj])
                data.append(-kfc)
        rows.append(k)
        cols.append(k)
        data.append(diag)
        if src2d[i, j]:
            b[k] += s_cell  # 発熱（W/m depth）
    mat = csr_matrix((data, (rows, cols)), shape=(m, m))
    t_unk = spsolve(mat, b)
    t = np.zeros((nz, nx))
    t[~sink] = t_unk
    return t


def peak_temperature_rise_k(
    wafer: Wafer, source_mask: np.ndarray, total_power_w: float, y_index: int | None = None
) -> float:
    """2.5D 熱拡散ソルバによる断面内の最大温度上昇 ΔT_max[K] を返す。"""
    return float(temperature_field_2d(wafer, source_mask, total_power_w, y_index).max())


def estimate_convergence_order(step_sizes, errors) -> float:
    """メッシュ刻み h に対する誤差の収束次数 p を最小二乗で推定する。

    誤差 ≈ C·hᵖ を仮定し log(error)=log(C)+p·log(h) の傾き p を返す。数値ソルバ
    （容量・熱拡散など）がメッシュ細分で解析解へ収束することの定量検証に使う。
    1 次精度の有限体積なら p≈1。要素数 2 未満や非正の値があれば ValueError。
    """
    h = np.asarray(step_sizes, dtype=float)
    e = np.asarray(errors, dtype=float)
    if h.size < 2 or e.size != h.size:
        raise ValueError("step_sizes と errors は同数で 2 点以上必要です。")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("step_sizes・errors は正の値である必要があります。")
    slope, _ = np.polyfit(np.log(h), np.log(e), 1)
    return float(slope)


# ボルツマン定数（eV/K）
_K_BOLTZMANN_EV = 8.617333e-5


def electromigration_mttf(
    j_a_cm2: float, temperature_c: float,
    *, n: float = 2.0, ea_ev: float = 0.9, a_const: float = 1.0e9,
) -> float:
    """Black の式による配線 EM の平均故障時間 MTTF を返す（相対寿命指標）。

    MTTF = A · J^(−n) · exp(Ea / (k·T))。電流密度 J[A/cm²] が高いほど、温度 T が
    高いほど寿命が短い。n は電流密度指数（Cu/Al で 1〜2）、Ea は活性化エネルギー
    （eV, Cu EM で ~0.9）、A は工程定数。返り値は a_const に依存する相対時間。
    J<=0 では inf（電流なし＝劣化なし）。
    """
    if j_a_cm2 <= 0:
        return float("inf")
    t_k = temperature_c + 273.15
    return float(a_const * j_a_cm2 ** (-n) * np.exp(ea_ev / (_K_BOLTZMANN_EV * t_k)))


def tddb_lifetime(
    field_mv_cm: float, temperature_c: float,
    *, gamma: float = 4.0, ea_ev: float = 0.6, a_const: float = 1.0e3,
) -> float:
    """E モデルによる絶縁膜の経時破壊（TDDB）寿命 TTF を返す（相対寿命指標）。

    TTF = A · exp(−γ·E) · exp(Ea / (k·T))。電界 E[MV/cm] が高いほど、温度 T が
    高いほど寿命が短い。γ は電界加速係数（cm/MV, SiO2 で ~4）、Ea は活性化
    エネルギー（eV）。返り値は a_const に依存する相対時間。E<=0 では inf。
    """
    if field_mv_cm <= 0:
        return float("inf")
    t_k = temperature_c + 273.15
    return float(a_const * np.exp(-gamma * field_mv_cm) * np.exp(ea_ev / (_K_BOLTZMANN_EV * t_k)))


def nbti_vth_shift(
    v_stress: float, temperature_c: float, time_s: float,
    *, n_exp: float = 0.16, ea_ev: float = 0.10, gamma_per_v: float = 1.5,
    a_const: float = 1.0e-3,
) -> float:
    """NBTI による pMOS しきい値電圧シフト |ΔVth|（V）を返す（経時劣化）。

    反応律速 NBTI の代表式 |ΔVth| = A·exp(γ·|V|)·exp(−Ea/(k·T))·tⁿ。ストレス電圧
    |V|（電界）、温度 T、ストレス時間 t が大きいほど ΔVth は増える。時間べき指数
    n は ~0.16（H 拡散律速）。負の入力は ValueError。
    """
    if time_s < 0:
        raise ValueError("ストレス時間は非負である必要があります。")
    if time_s == 0:
        return 0.0
    t_k = temperature_c + 273.15
    return float(a_const * np.exp(gamma_per_v * abs(v_stress))
                 * np.exp(-ea_ev / (_K_BOLTZMANN_EV * t_k)) * time_s ** n_exp)


def em_lifetime_wafer(
    wafer: Wafer, conductor, current_ma: float, temperature_c: float,
    axis: str = "x", **black_kw,
) -> dict:
    """配線の EM 寿命を Black の式で評価する（current_density_stats と結合）。

    最小断面（ネッキング箇所）の最大電流密度 J_max を用いて MTTF を求める。返す
    辞書: j_max_a_cm2 / mttf / temperature_c。断線/非導体では j_max=inf, mttf=0。
    """
    st = current_density_stats(wafer, conductor, current_ma, axis)
    j = st["j_max_a_cm2"]
    mttf = 0.0 if j == float("inf") else electromigration_mttf(j, temperature_c, **black_kw)
    return {"j_max_a_cm2": j, "mttf": mttf, "temperature_c": temperature_c}


def em_lifetime_self_heated(
    wafer: Wafer, conductor, current_ma: float, ambient_c: float = 85.0,
    axis: str = "x", **black_kw,
) -> dict:
    """自己発熱を考慮した EM 寿命（ジュール熱で上昇した接合温度で Black 評価）。

    配線のジュール発熱 P=I²R による温度上昇 ΔT（joule_self_heating_k）を周囲温度
    ambient_c に加えた接合温度で電流密度 J_max から MTTF を求める。自己発熱は
    温度を上げ EM 寿命を縮める（電流の正帰還）。返す辞書: j_max_a_cm2 /
    delta_t_k / junction_temp_c / mttf / mttf_isothermal（自己発熱無視時）。
    """
    jh = joule_self_heating_k(wafer, conductor, current_ma, axis)
    st = current_density_stats(wafer, conductor, current_ma, axis)
    j = st["j_max_a_cm2"]
    dt = jh["delta_t_k"]
    if j == float("inf") or dt == float("inf"):
        return {"j_max_a_cm2": j, "delta_t_k": dt, "junction_temp_c": float("inf"),
                "mttf": 0.0, "mttf_isothermal": 0.0}
    tj = ambient_c + dt
    return {
        "j_max_a_cm2": j,
        "delta_t_k": float(dt),
        "junction_temp_c": float(tj),
        "mttf": electromigration_mttf(j, tj, **black_kw),
        "mttf_isothermal": electromigration_mttf(j, ambient_c, **black_kw),
    }


def diode_current(
    v: float, *, i_sat_a: float = 1.0e-15, ideality: float = 1.0,
    temperature_c: float = 27.0, series_r_ohm: float = 0.0,
) -> float:
    """理想ダイオード（Shockley）電流 I（A）を返す。直列抵抗付き。

    I = Is·(exp((V−I·Rs)/(n·Vt))−1)、Vt=kT/q。順方向は指数的に増え、逆方向は
    −Is に飽和する。理想係数 n でサブスレショルド的な log(I)-V 傾斜が決まる
    （n·(kT/q)·ln10）。直列抵抗 Rs>0 は高電流で電流を制限（Newton 法で陰解）。
    """
    vt = _K_BOLTZMANN_EV * (temperature_c + 273.15)
    nvt = ideality * vt

    def shockley(vj):
        return i_sat_a * (np.exp(np.clip(vj / nvt, -200.0, 200.0)) - 1.0)

    if series_r_ohm <= 0:
        return float(shockley(v))
    i = shockley(v)  # 初期値（Rs=0）
    for _ in range(60):  # Newton 反復
        vj = v - i * series_r_ohm
        ex = np.exp(np.clip(vj / nvt, -200.0, 200.0))
        f = i_sat_a * (ex - 1.0) - i
        df = -i_sat_a * ex / nvt * series_r_ohm - 1.0
        step = f / df
        i -= step
        if abs(step) < 1e-18:
            break
    return float(i)


def diode_iv_curve(
    *, v_min: float = -1.0, v_max: float = 0.8, n_points: int = 61,
    i_sat_a: float = 1.0e-15, ideality: float = 1.0,
    temperature_c: float = 27.0, series_r_ohm: float = 0.0,
) -> dict:
    """ダイオード I-V 曲線を返す（順方向指数・逆方向飽和）。

    返す辞書: v（電圧配列）/ i（電流配列 A）/ i_sat_a。順方向 log(I)-V の傾斜から
    理想係数 n（n·60mV/dec）、逆方向飽和電流 −Is が読み取れる。
    """
    v = np.linspace(v_min, v_max, n_points)
    i = np.array([
        diode_current(float(vv), i_sat_a=i_sat_a, ideality=ideality,
                      temperature_c=temperature_c, series_r_ohm=series_r_ohm)
        for vv in v
    ])
    return {"v": v, "i": i, "i_sat_a": i_sat_a}


def antenna_ratio(
    wafer: Wafer, conductor, gate_dielectric, ratio_limit: float = 400.0
) -> dict:
    """プロセスアンテナ比（プラズマ帯電損傷リスク）を判定する。

    プラズマエッチ中、ゲート酸化膜に接続した導体（アンテナ）はその露出表面積に
    比例して電荷を集め、薄いゲート酸化膜に電圧ストレスを与える。
      アンテナ比 = 導体の露出表面積 / ゲート酸化膜面積
    が大きいほど（広い金属＋小さいゲート）ゲート絶縁破壊リスクが高い。返す辞書:
      - antenna_area_um2: 導体の空気露出表面積（電荷収集面積）
      - gate_area_um2: 導体とゲート絶縁膜の接触面積
      - ratio: アンテナ比
      - fail: ratio_limit 超過か（既定 400, ファウンドリのアンテナ則相当）
    導体かゲートが無い/未接続では ratio=0, fail=False。
    """
    cond = materials.get(conductor)
    grid = wafer.grid
    mask_c = grid == cond.id
    gate_area = contact_area_um2(wafer, gate_dielectric, conductor)
    if not mask_c.any() or gate_area <= 0:
        return {"antenna_area_um2": 0.0, "gate_area_um2": float(gate_area),
                "ratio": 0.0, "fail": False}
    # 導体の空気露出面積（面ペア数×pitch²）
    air = grid == materials.AIR
    faces = 0
    for axis in range(3):
        sa = [slice(None)] * 3
        sb = [slice(None)] * 3
        sa[axis] = slice(0, -1)
        sb[axis] = slice(1, None)
        faces += int((mask_c[tuple(sa)] & air[tuple(sb)]).sum())
        faces += int((mask_c[tuple(sb)] & air[tuple(sa)]).sum())
    antenna_area = float(faces) * (wafer.config.pitch_um ** 2)
    ratio = antenna_area / gate_area
    return {
        "antenna_area_um2": antenna_area,
        "gate_area_um2": float(gate_area),
        "ratio": float(ratio),
        "fail": bool(ratio > ratio_limit),
    }


def mos_gate_capacitance(wafer: Wafer, gate_conductor, channel="silicon") -> dict:
    """MOS ゲート積層の容量密度 Cox と等価酸化膜厚 EOT を算出する。

    ゲート電極（gate_conductor）とチャネル（channel, 既定 silicon）の間に挟まれた
    誘電体スタックを各列で検出し、直列容量の電気的厚み d_eff=Σ tᵢ/εrᵢ を求める。
      - Cox = ε0 / d_eff        （F/m² → fF/µm² に換算）
      - EOT = εr(SiO2) · d_eff   （high-k 採用で物理厚より薄い EOT になる）
    返す辞書: cox_ff_per_um2 / eot_nm / gate_area_um2 / total_cap_ff。
    ゲートとチャネルの間に導体がある列や、誘電体が無い（直接接触＝短絡）列は
    除外する。有効なゲート領域が無ければ全て 0。
    """
    gate = materials.get(gate_conductor)
    ch = materials.get(channel)
    grid = wafer.grid
    pitch_m = wafer.config.pitch_um * 1e-6
    eps_field = permittivity_field(wafer)
    cond_ids = _conductor_ids()
    eps0 = 8.854e-12
    eps_sio2 = materials.BY_NAME["oxide"].rel_permittivity

    d_effs: list[float] = []
    gate_cols = np.argwhere((grid == gate.id).any(axis=0))  # (y, x) のリスト
    for y, x in gate_cols:
        col = grid[:, y, x]
        gate_zs = np.nonzero(col == gate.id)[0]
        gate_bottom = int(gate_zs.min())
        ch_zs = np.nonzero((col == ch.id) & (np.arange(len(col)) < gate_bottom))[0]
        if ch_zs.size == 0:
            continue
        ch_top = int(ch_zs.max())
        between = range(ch_top + 1, gate_bottom)
        if len(between) == 0:
            continue  # 誘電体無し（直接接触＝短絡）
        if any(int(col[z]) in cond_ids for z in between):
            continue  # 間に別の導体 → 純粋なゲート誘電体でない
        d_eff = sum(pitch_m / eps_field[z, y, x] for z in between)
        d_effs.append(d_eff)

    if not d_effs:
        return {"cox_ff_per_um2": 0.0, "eot_nm": 0.0,
                "gate_area_um2": 0.0, "total_cap_ff": 0.0}
    d_eff_avg = float(np.mean(d_effs))
    gate_area = len(d_effs) * (wafer.config.pitch_um ** 2)
    cox_f_m2 = eps0 / d_eff_avg
    cox_ff_um2 = cox_f_m2 * 1e3  # F/m² → fF/µm²
    return {
        "cox_ff_per_um2": float(cox_ff_um2),
        "eot_nm": float(eps_sio2 * d_eff_avg * 1e9),
        "gate_area_um2": float(gate_area),
        "total_cap_ff": float(cox_ff_um2 * gate_area),
    }


# 半導体物理定数（Si, 300K）
_Q = 1.602176634e-19          # 素電荷 [C]
_NI_SI_M3 = 1.0e16            # Si 真性キャリア濃度 [m^-3]（1e10 cm^-3）
_KT_Q = 0.025852              # 熱電圧 kT/q [V] @300K
_EPS_SI = 11.7 * 8.854e-12    # Si 誘電率 [F/m]


def threshold_voltage_v(
    wafer: Wafer, gate_conductor, channel="silicon",
    *, doping_cm3: float = 1.0e17, vfb: float = 0.0,
) -> dict:
    """MOS キャパシタのしきい値電圧 Vth（空乏近似, p 型基板の NMOS）を返す。

    ゲート容量 Cox（mos_gate_capacitance）と基板ドーピング Na から、
      φF = (kT/q)·ln(Na/ni)
      Wmax = √(2·εs·2φF/(q·Na))            （最大空乏層幅）
      Vth = Vfb + 2φF + √(4·εs·q·Na·φF)/Cox
    を算出する。返す辞書: vth_v / phi_f_v / w_max_um / cox_f_m2。
    Cox=0（ゲート無し）では Vth=None。
    """
    g = mos_gate_capacitance(wafer, gate_conductor, channel)
    cox = g["cox_ff_per_um2"] * 1e-3  # fF/µm² → F/m²
    if cox <= 0:
        return {"vth_v": None, "phi_f_v": 0.0, "w_max_um": 0.0, "cox_f_m2": 0.0}
    na = doping_cm3 * 1e6  # cm^-3 → m^-3
    phi_f = _KT_Q * np.log(na / _NI_SI_M3)
    w_max = np.sqrt(2.0 * _EPS_SI * 2.0 * phi_f / (_Q * na))
    q_dep_max = np.sqrt(4.0 * _EPS_SI * _Q * na * phi_f)  # 最大空乏電荷 [C/m²]
    vth = vfb + 2.0 * phi_f + q_dep_max / cox
    return {
        "vth_v": float(vth),
        "phi_f_v": float(phi_f),
        "w_max_um": float(w_max * 1e6),
        "cox_f_m2": float(cox),
    }


def short_channel_vth_v(
    wafer: Wafer, gate_conductor, channel="silicon",
    *, channel_length_um: float, vds: float = 0.05, doping_cm3: float = 1.0e17,
    vfb: float = 0.0, dibl_v_per_v: float = 0.05, sce_amp_v: float = 0.3,
    sce_char_length_um: float = 0.05,
) -> dict:
    """短チャネル効果（SCE）と DIBL を含むしきい値電圧 Vth を返す。

    長チャネル Vth（threshold_voltage_v）から、
      Vth = Vth_long − ΔVth_SCE − DIBL·Vds,  ΔVth_SCE = sce_amp·exp(−L/λ)
    を計算する。チャネル長 L が短いほど SCE で Vth が下がり（Vth ロールオフ）、
    ドレイン電圧 Vds が高いほど DIBL で Vth が下がる（オフリーク増大の要因）。返す辞書:
      vth_v / vth_long_v / dvth_sce_v / dvth_dibl_v。
    """
    if channel_length_um <= 0:
        raise ValueError("チャネル長は正の値が必要です。")
    th = threshold_voltage_v(wafer, gate_conductor, channel,
                             doping_cm3=doping_cm3, vfb=vfb)
    vth_long = th["vth_v"]
    if vth_long is None:
        return {"vth_v": None, "vth_long_v": None,
                "dvth_sce_v": 0.0, "dvth_dibl_v": 0.0}
    dvth_sce = sce_amp_v * float(np.exp(-channel_length_um / sce_char_length_um))
    dvth_dibl = dibl_v_per_v * vds
    return {
        "vth_v": float(vth_long - dvth_sce - dvth_dibl),
        "vth_long_v": float(vth_long),
        "dvth_sce_v": float(dvth_sce),
        "dvth_dibl_v": float(dvth_dibl),
    }


def mos_cv_curve(
    wafer: Wafer, gate_conductor, channel="silicon",
    *, doping_cm3: float = 1.0e17, vfb: float = 0.0,
    v_min: float = -2.0, v_max: float = 2.0, n_points: int = 121,
) -> dict:
    """MOS キャパシタの高周波 C-V 特性曲線を返す（空乏近似, p 型基板）。

    蓄積（V<Vfb で C=Cox）→空乏（空乏層拡大で C 低下）→反転（C=Cmin で飽和）の
    古典的 HF C-V を、表面ポテンシャル ψs をパラメータに計算する。返す辞書:
      - v: 印加電圧配列 [V]
      - c_ff_per_um2: 容量密度配列 [fF/µm²]
      - c_over_cox: 規格化容量 C/Cox 配列
      - cox_ff_per_um2 / cmin_ff_per_um2 / vth_v / w_max_um
    Cox=0（ゲート無し）では空配列。
    """
    th = threshold_voltage_v(wafer, gate_conductor, channel,
                             doping_cm3=doping_cm3, vfb=vfb)
    cox = th["cox_f_m2"]
    if cox <= 0:
        return {"v": np.array([]), "c_ff_per_um2": np.array([]),
                "c_over_cox": np.array([]), "cox_ff_per_um2": 0.0,
                "cmin_ff_per_um2": 0.0, "vth_v": None, "w_max_um": 0.0}
    na = doping_cm3 * 1e6
    phi_f = th["phi_f_v"]
    w_max = th["w_max_um"] * 1e-6
    cmin = cox * (_EPS_SI / w_max) / (cox + _EPS_SI / w_max)  # 直列最小容量

    v = np.linspace(v_min, v_max, n_points)
    c = np.empty_like(v)
    vth = th["vth_v"]
    # 空乏領域の V(ψs) を作り、各 V に対し ψs を逆引きして C を求める。
    psi = np.linspace(1e-6, 2.0 * phi_f, 2000)
    w_psi = np.sqrt(2.0 * _EPS_SI * psi / (_Q * na))
    cdep = _EPS_SI / w_psi
    c_depl = cox * cdep / (cox + cdep)
    v_depl = vfb + psi + np.sqrt(2.0 * _EPS_SI * _Q * na * psi) / cox
    for k, vv in enumerate(v):
        if vv <= vfb:
            c[k] = cox                      # 蓄積
        elif vv >= vth:
            c[k] = cmin                     # 反転（高周波）
        else:
            c[k] = float(np.interp(vv, v_depl, c_depl))  # 空乏
    return {
        "v": v,
        "c_ff_per_um2": c * 1e3,
        "c_over_cox": c / cox,
        "cox_ff_per_um2": cox * 1e3,
        "cmin_ff_per_um2": cmin * 1e3,
        "vth_v": vth,
        "w_max_um": th["w_max_um"],
    }


def junction_capacitance(
    na_cm3: float, nd_cm3: float, reverse_bias_v: float = 0.0, area_um2: float = 1.0
) -> dict:
    """pn 接合（階段接合）の空乏層幅・接合容量・ビルトイン電位を返す。

    ビルトイン電位 Vbi=(kT/q)·ln(Na·Nd/ni²)、逆バイアス V_R≥0 のとき
      W = √(2·εs·(Vbi+V_R)/q · (1/Na+1/Nd))     （空乏層幅）
      Cj = εs/W                                   （単位面積容量）
    片側接合は一方を高ドープにすると再現できる（W は低ドープ側で決まる）。返す辞書:
      vbi_v / depletion_width_um / cj_ff_per_um2 / cj_total_ff / one_over_cj2。
    逆バイアスが深いほど W は広く Cj は小さい。`1/Cj²` は (Vbi+V_R) に比例（C-V
    profiling でドーピングと Vbi を抽出できる）。
    """
    if na_cm3 <= 0 or nd_cm3 <= 0 or area_um2 < 0:
        raise ValueError("ドーピングは正、面積は非負である必要があります。")
    if reverse_bias_v < 0:
        raise ValueError("reverse_bias_v は逆バイアス量(≥0)で指定してください。")
    na = na_cm3 * 1e6
    nd = nd_cm3 * 1e6
    vbi = _KT_Q * np.log(na * nd / _NI_SI_M3 ** 2)
    w = np.sqrt(2.0 * _EPS_SI * (vbi + reverse_bias_v) / _Q * (1.0 / na + 1.0 / nd))
    cj_f_m2 = _EPS_SI / w
    cj_ff_um2 = cj_f_m2 * 1e3
    return {
        "vbi_v": float(vbi),
        "depletion_width_um": float(w * 1e6),
        "cj_ff_per_um2": float(cj_ff_um2),
        "cj_total_ff": float(cj_ff_um2 * area_um2),
        "one_over_cj2": float(1.0 / cj_f_m2 ** 2),
    }


def junction_leakage_a(
    area_um2: float, temperature_c: float = 27.0,
    *, j0_a_per_um2: float = 1.0e-15, eg_ev: float = 1.12,
) -> float:
    """pn 接合の逆方向リーク電流（A）を温度依存で返す。

    発生律速の逆リークは真性キャリア濃度 ni∝T^1.5·exp(−Eg/2kT) に比例する。
    300K の基準リーク密度 j0_a_per_um2 から
      J(T) = j0 · (T/300)^1.5 · exp(−Eg/2k·(1/T − 1/300))
    を求め、面積を掛ける。温度が高いほど指数的に増え（~8〜10°C で倍増）、待機
    電力やリテンション不良の要因になる。Eg はバンドギャップ（Si=1.12eV）。
    """
    if area_um2 < 0:
        raise ValueError("面積は非負である必要があります。")
    t_k = temperature_c + 273.15
    t0 = 300.0
    j = j0_a_per_um2 * (t_k / t0) ** 1.5 * np.exp(
        -eg_ev / (2.0 * _K_BOLTZMANN_EV) * (1.0 / t_k - 1.0 / t0))
    return float(j * area_um2)


def dram_retention_time_s(
    storage_cap_ff: float, junction_area_um2: float, temperature_c: float = 85.0,
    *, sense_margin_v: float = 0.3, **leak_kw,
) -> dict:
    """DRAM セルの保持（リテンション）時間 t_ret を返す（容量＋接合リークの結合）。

    蓄積容量 C[fF] に貯めた電荷が接合リーク I_leak（junction_leakage_a）で抜け、
    センスマージン sense_margin_v だけ電圧が落ちるまでの時間 t_ret=C·ΔV/I_leak を
    求める。温度が高いほどリークが増え保持時間は指数的に短くなる（リフレッシュ
    周期の決定要因）。返す辞書: retention_s / leakage_a / storage_cap_ff。
    リークが 0 なら inf。
    """
    if storage_cap_ff < 0 or sense_margin_v <= 0:
        raise ValueError("容量は非負・センスマージンは正である必要があります。")
    i_leak = junction_leakage_a(junction_area_um2, temperature_c, **leak_kw)
    if i_leak <= 0:
        return {"retention_s": float("inf"), "leakage_a": 0.0,
                "storage_cap_ff": float(storage_cap_ff)}
    t_ret = (storage_cap_ff * 1e-15) * sense_margin_v / i_leak
    return {
        "retention_s": float(t_ret),
        "leakage_a": float(i_leak),
        "storage_cap_ff": float(storage_cap_ff),
    }


def critical_charge_fc(storage_cap_ff: float, node_voltage_v: float = 1.0) -> float:
    """ソフトエラー（SEU）の臨界電荷 Q_crit（fC）を返す。

    記憶ノードの状態反転に必要な最小電荷 Q_crit = C·V（一次近似）。粒子線が
    集めた電荷がこれを超えるとビット反転（ソフトエラー）。容量・電圧が大きいほど
    Q_crit が大きくソフトエラー耐性が高い。容量[fF]×電圧[V] = 電荷[fC]。
    """
    if storage_cap_ff < 0:
        raise ValueError("容量は非負である必要があります。")
    return float(storage_cap_ff * node_voltage_v)


def junction_cv_curve(
    na_cm3: float, nd_cm3: float,
    *, v_max_reverse: float = 5.0, n_points: int = 51, area_um2: float = 1.0,
) -> dict:
    """pn 接合の逆バイアス C-V 曲線（Cj-V と 1/Cj²-V）を返す。

    逆バイアス 0→v_max_reverse を掃引し、各点の接合容量と 1/Cj² を返す。
    1/Cj² は逆バイアスに対し直線になり、傾きから実効ドーピング、外挿切片から
    Vbi が求まる（標準の C-V ドーピングプロファイリング）。返す辞書:
      reverse_bias_v / cj_ff_per_um2 / one_over_cj2 / vbi_v。
    """
    vr = np.linspace(0.0, v_max_reverse, n_points)
    cj = np.empty_like(vr)
    inv2 = np.empty_like(vr)
    vbi = 0.0
    for k, v in enumerate(vr):
        d = junction_capacitance(na_cm3, nd_cm3, float(v), area_um2)
        cj[k] = d["cj_ff_per_um2"]
        inv2[k] = d["one_over_cj2"]
        vbi = d["vbi_v"]
    return {
        "reverse_bias_v": vr,
        "cj_ff_per_um2": cj,
        "one_over_cj2": inv2,
        "vbi_v": float(vbi),
    }


def mos_drain_current(
    wafer: Wafer, gate_conductor, channel="silicon", *, vg: float, vd: float,
    doping_cm3: float = 1.0e17, vfb: float = 0.0, mobility_cm2_vs: float = 450.0,
    w_over_l: float = 10.0, lambda_per_v: float = 0.0, subthreshold_n: float = 1.3,
) -> float:
    """長チャネル n-MOSFET のドレイン電流 Id（A）を返す（EKV 連続モデル）。

    ゲート容量 Cox（mos_gate_capacitance）としきい値 Vth（threshold_voltage_v）から、
    弱反転〜強反転・三極管〜飽和を単一の連続式で表す（区分モデルの不連続を解消）:
      Id = 2n·β·Vt²·[ ln²(1+e^(Vov/(2n·Vt))) − ln²(1+e^((Vov−Vd)/(2n·Vt))) ]·(1+λ·Vd)
    ここで β=µ·Cox·(W/L), Vov=Vg−Vth, Vt=kT/q。漸近的に
      - 弱反転: Id∝exp(Vov/(n·Vt))（SS=n·(kT/q)·ln10≈n·60mV/dec）
      - 強反転・飽和: Id→β·Vov²/(2n)（Idsat∝(Vg−Vth)²）, 三極管: Id∝(Vov·Vd−Vd²/2)
    を満たし、全領域で滑らかに接続する。Cox=0 では 0。
    """
    th = threshold_voltage_v(wafer, gate_conductor, channel,
                             doping_cm3=doping_cm3, vfb=vfb)
    cox = th["cox_f_m2"]
    vth = th["vth_v"]
    if cox <= 0 or vth is None:
        return 0.0
    mu = mobility_cm2_vs * 1e-4  # cm²/Vs → m²/Vs
    beta = mu * cox * w_over_l   # A/V²
    vt = _KT_Q
    n = subthreshold_n
    vov = vg - vth

    def _lse2(x):  # ln²(1+e^x)（オーバーフロー保護）
        return np.log1p(np.exp(np.clip(x, -60.0, 60.0))) ** 2

    spec = 2.0 * n * beta * vt * vt
    i_fwd = _lse2(vov / (2.0 * n * vt))
    i_rev = _lse2((vov - max(vd, 0.0)) / (2.0 * n * vt))
    return float(spec * (i_fwd - i_rev) * (1.0 + lambda_per_v * vd))


def mos_iv_curve(
    wafer: Wafer, gate_conductor, channel="silicon", *, vg_list=(0.5, 1.0, 1.5),
    vd_max: float = 2.0, n_points: int = 41, **kw,
) -> dict:
    """MOS 出力特性 Id-Vd 族（複数 Vg）を返す。

    返す辞書: vd（配列）/ curves（{vg: Id 配列(A)}）。三極管→飽和の遷移と、Vg を
    上げると Idsat が二乗で増える様子（飽和電流 ∝ (Vg−Vth)²）を表す。
    """
    vd = np.linspace(0.0, vd_max, n_points)
    curves = {}
    for vg in vg_list:
        curves[float(vg)] = np.array([
            mos_drain_current(wafer, gate_conductor, channel, vg=float(vg),
                              vd=float(v), **kw) for v in vd
        ])
    return {"vd": vd, "curves": curves}


def gate_switching_delay_ps(
    wafer: Wafer, gate_conductor, channel="silicon",
    *, load_cap_ff: float, vdd: float = 1.0, **mos_kw,
) -> dict:
    """ロジックゲートのスイッチング遅延 τ=C·Vdd/I_drive（CV/I モデル, ps）。

    MOS の飽和駆動電流 I_drive=mos_drain_current(Vg=Vd=Vdd) と負荷容量 load_cap_ff
    から、出力を Vdd まで充電する遅延 τ=C_load·Vdd/I_drive を求める。駆動電流が
    大きい（W/L 大・Vdd 大）ほど速く、負荷容量が大きいほど遅い。返す辞書:
      delay_ps / drive_current_a / load_cap_ff。I_drive=0 では inf。
    """
    if load_cap_ff < 0:
        raise ValueError("負荷容量は非負である必要があります。")
    i_drive = mos_drain_current(wafer, gate_conductor, channel,
                                vg=vdd, vd=vdd, **mos_kw)
    if i_drive <= 0:
        return {"delay_ps": float("inf"), "drive_current_a": float(i_drive),
                "load_cap_ff": float(load_cap_ff)}
    tau_s = (load_cap_ff * 1e-15) * vdd / i_drive
    return {
        "delay_ps": float(tau_s * 1e12),
        "drive_current_a": float(i_drive),
        "load_cap_ff": float(load_cap_ff),
    }


def hci_lifetime(vds: float, *, b_volt: float = 30.0, a_const: float = 1.0e-6) -> float:
    """ホットキャリア注入（HCI）寿命 TTF を返す（相対寿命指標, ラッキー電子モデル）。

    TTF = A·exp(B/Vds)。ドレイン電圧 Vds が高いほど高エネルギーキャリアが増え
    寿命が指数的に短くなる（NBTI/TDDB と異なりドレイン電界律速）。B は電圧加速
    係数（V）。Vds≤0 では inf。極小 Vds では指数をクリップして inf を返す
    （オーバーフロー警告を回避）。
    """
    if vds <= 0:
        return float("inf")
    expo = b_volt / vds
    if expo > 700.0:  # exp(700) ≈ 1e304（float 上限近傍）。これ以上は実質 inf。
        return float("inf")
    return float(a_const * np.exp(expo))


def critical_area_short_um2(
    wafer: Wafer, name_a, name_b, defect_diameter_um: float, z_index: int | None = None
) -> float:
    """直径 d の円形導電性欠陥が 2 導体をブリッジ（ショート）させ得る臨界面積(µm²)。

    指定 z 断面（既定=導体が存在する代表層）の上面図で、欠陥中心が置かれたとき
    両導体に同時に触れる（A までの距離 ≤ d/2 かつ B までの距離 ≤ d/2）点の面積を
    距離変換で数える。d が 2 配線間隔より小さいと 0、d が大きいほど臨界面積は増える。
    クリティカルエリア解析（CAA）によるショート歩留りの基礎量。
    """
    a = materials.get(name_a)
    b = materials.get(name_b)
    grid = wafer.grid
    pitch = wafer.config.pitch_um
    if z_index is None:
        # A・B が最も多く共存する z 層を選ぶ
        both_per_z = ((grid == a.id) | (grid == b.id)).sum(axis=(1, 2))
        if both_per_z.max() == 0:
            return 0.0
        z_index = int(np.argmax(both_per_z))
    plane = grid[z_index]
    mask_a = plane == a.id
    mask_b = plane == b.id
    if not mask_a.any() or not mask_b.any():
        return 0.0
    r = defect_diameter_um / 2.0
    tol = pitch * 1e-6  # ボクセル量子化による境界の取りこぼし防止
    dist_a = ndimage.distance_transform_edt(~mask_a) * pitch
    dist_b = ndimage.distance_transform_edt(~mask_b) * pitch
    bridges = (dist_a <= r + tol) & (dist_b <= r + tol)
    return float(bridges.sum()) * (pitch ** 2)


def caa_short_yield(
    wafer: Wafer, name_a, name_b,
    *, defect_density_per_cm2: float = 0.1, chip_area_cm2: float = 0.1,
    x0_um: float = 0.02, xmax_um: float = 1.0, n: int = 24, z_index: int | None = None,
) -> dict:
    """クリティカルエリア解析（CAA）でショート歩留りを推定する。

    シミュレートした代表レイアウトの臨界面積 A_c(x) を版面面積で割った
    「臨界面積率」を、実チップ面積 chip_area_cm2 に拡張して期待故障数を求める。
    欠陥サイズ分布 dD/dx=k/x³（x≥x0, ∫=defect_density）に対し
      λ = chip_area · ∫ frac_c(x)·(k/x³) dx
    を数値積分し、Poisson 歩留り Y=exp(−λ) を返す。配線間隔が広い/欠陥が小さい/
    欠陥密度が低いほど λ は小さく歩留りは高い。返す辞書: yield / lambda_faults /
    diameters_um / critical_area_um2 / critical_area_fraction。
    """
    if defect_density_per_cm2 < 0 or x0_um <= 0 or xmax_um <= x0_um or chip_area_cm2 < 0:
        raise ValueError("CAA パラメータが不正です（密度/面積≥0, 0<x0<xmax）。")
    cfg = wafer.config
    layout_area_um2 = cfg.nx * cfg.ny * (cfg.pitch_um ** 2)  # 上面図の版面面積
    xs = np.logspace(np.log10(x0_um), np.log10(xmax_um), n)
    ac_um2 = np.array([
        critical_area_short_um2(wafer, name_a, name_b, x, z_index) for x in xs
    ])
    frac = ac_um2 / layout_area_um2 if layout_area_um2 > 0 else ac_um2 * 0.0
    k = 2.0 * defect_density_per_cm2 * x0_um ** 2  # サイズ分布の正規化定数
    dens = k / xs ** 3  # /cm²/µm
    lam = float(chip_area_cm2 * _trapz(frac * dens, xs))
    return {
        "yield": float(np.exp(-lam)),
        "lambda_faults": lam,
        "diameters_um": xs,
        "critical_area_um2": ac_um2,
        "critical_area_fraction": frac,
    }


def stress_field_mpa(wafer: Wafer) -> np.ndarray:
    """各ボクセルの残留膜応力 σ（MPa）を格納した配列を返す。空気は 0。"""
    grid = wafer.grid
    s = np.zeros(grid.shape, dtype=float)
    for m in materials.all_materials():
        if m.stress_mpa != 0.0:
            s[grid == m.id] = m.stress_mpa
    return s


def stress_concentration_map(wafer: Wafer) -> np.ndarray:
    """局所応力集中マップ（MPa 相当）を返す。クラック/剥離リスク箇所の検出。

    各固体ボクセルについて、隣接（6 近傍）との応力ミスマッチ最大値
    Δσ=max|σ_self−σ_neighbor|（空気は σ=0 の自由表面扱い）を求め、幾何学的な
    応力集中係数 Kt=1+(異材/空気に接する面数)/2 を掛ける。界面の応力差が大きい
    箇所（例: 高応力 Si3N4 と圧縮 SiO2 の界面）や、露出面の多い凸角ほど高い値に
    なる。空気ボクセルは 0。シミュレーション境界はラップさせない。
    """
    grid = wafer.grid
    s = stress_field_mpa(wafer)
    solid = grid != materials.AIR
    mism = np.zeros(grid.shape, dtype=float)
    n_diff = np.zeros(grid.shape, dtype=int)
    for axis in (0, 1, 2):
        for shift in (1, -1):
            sn = np.roll(s, shift, axis=axis)
            gn = np.roll(grid, shift, axis=axis)
            d = np.abs(s - sn)
            diff = gn != grid
            # ラップした境界面は隣接なし扱い（偽の集中を防ぐ）
            sl = [slice(None)] * 3
            sl[axis] = 0 if shift == 1 else -1
            d[tuple(sl)] = 0.0
            diff[tuple(sl)] = False
            mism = np.maximum(mism, d)
            n_diff += diff.astype(int)
    kt = 1.0 + n_diff / 2.0
    conc = mism * kt
    conc[~solid] = 0.0
    return conc


def max_stress_concentration(wafer: Wafer) -> dict:
    """最大応力集中の値と位置・関与材料を返す（剥離/クラック最弱点）。

    返す辞書: value_mpa（集中値）/ location（z,y,x）/ material（その位置の材料名）。
    固体が無ければ value_mpa=0。
    """
    conc = stress_concentration_map(wafer)
    if not (conc > 0).any():
        return {"value_mpa": 0.0, "location": None, "material": None}
    idx = np.unravel_index(int(np.argmax(conc)), conc.shape)
    mat = materials.BY_ID.get(int(wafer.grid[idx]))
    return {
        "value_mpa": float(conc[idx]),
        "location": tuple(int(i) for i in idx),
        "material": mat.name if mat else None,
    }


def _solve_slice_capacitance(
    eps_diel: np.ndarray, is_a: np.ndarray, is_b: np.ndarray, is_cond: np.ndarray
) -> float:
    """2D 断面で ∇·(εr∇φ)=0 を解き、電極 A(φ=1)/B(φ=0) 間の単位長あたり
    容量（ε0 を除いた無次元値）を返す。面の εr は誘電体同士で調和平均、導体表面では
    誘電体側の値を使う（一様媒質で εr に厳密比例）。

    A・B 以外の導体（浮遊導体）は連結成分ごとに 1 つの等電位ノードへ束ねて未知数とし、
    その正味電荷ゼロ条件（KCL）を自動的に満たす——これにより浮遊金属を物理的に
    正しく（高 εr 近似でなく真の等電位として）扱う。境界は自然境界条件（ゼロ流束）。
    A の表面を貫く電束 Q'=Σ εr_face·(1−φ_nb) を容量（V=1）として返す。
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    h, w = eps_diel.shape
    fixed = is_a | is_b
    floating = is_cond & ~fixed
    float_lab, n_float = ndimage.label(floating)  # 浮遊導体の連結成分
    diel = ~is_cond  # 誘電体（非導体）セル

    # ノード割当: 誘電体セル各 1 + 浮遊成分各 1
    idx = -np.ones((h, w), dtype=int)
    diel_cells = np.argwhere(diel)
    for k, (i, j) in enumerate(diel_cells):
        idx[i, j] = k
    m = len(diel_cells)
    n_nodes = m + n_float
    if n_nodes == 0:
        return 0.0

    def node_of(i, j):
        if diel[i, j]:
            return idx[i, j]
        return m + int(float_lab[i, j]) - 1  # 浮遊成分ノード

    phi_fixed = np.where(is_a, 1.0, 0.0)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    _SHORT = 1.0e8  # 導体同士の接触（短絡）を表す大コンダクタンス（稀）

    def face_cond(i, j, ii, jj):
        """セル (i,j) と隣接 (ii,jj) の面コンダクタンス（誘電体側 εr 規則）。"""
        ci, cj = is_cond[i, j], is_cond[ii, jj]
        if ci and cj:
            return _SHORT  # 異材導体の接触＝短絡
        if ci:  # i が導体 → 誘電体側 (ii,jj)
            return float(eps_diel[ii, jj])
        if cj:  # j が導体 → 誘電体側 (i,j)
            return float(eps_diel[i, j])
        a, c = eps_diel[i, j], eps_diel[ii, jj]
        return 2.0 * a * c / (a + c) if (a + c) > 0 else 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n_nodes)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        data.append(v)

    # 非固定セル（誘電体 + 浮遊導体）を走査し、面寄与を該当ノード行へ加算
    for i, j in np.argwhere(~fixed):
        nc = node_of(i, j)
        for di, dj in neigh:
            ii, jj = i + di, j + dj
            if not (0 <= ii < h and 0 <= jj < w):
                continue
            ef = face_cond(i, j, ii, jj)
            if ef == 0.0:
                continue
            if fixed[ii, jj]:
                add(nc, nc, ef)
                b[nc] += ef * phi_fixed[ii, jj]
            else:
                nn = node_of(ii, jj)
                if nn == nc:
                    continue  # 同一浮遊成分の内部面（寄与なし）
                add(nc, nc, ef)
                add(nc, nn, -ef)
    mat = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    sol = spsolve(mat, b)

    phi = phi_fixed.copy().astype(float)
    phi[diel] = sol[:m]
    if n_float:
        # 浮遊導体セルへ成分ノードの解を書き戻す
        for i, j in np.argwhere(floating):
            phi[i, j] = sol[m + int(float_lab[i, j]) - 1]

    # A 表面を貫く電束（= V=1 のときの容量）
    q = 0.0
    for i, j in np.argwhere(is_a):
        for di, dj in neigh:
            ii, jj = i + di, j + dj
            if not (0 <= ii < h and 0 <= jj < w) or is_a[ii, jj]:
                continue
            q += face_cond(i, j, ii, jj) * (1.0 - phi[ii, jj])
    return float(q)


def parasitic_capacitance_field_ff(wafer: Wafer, name_a, name_b, axis: str = "y") -> float:
    """2.5D 静電界ソルバによる 2 導体間の寄生容量（fF, フリンジ込み）。

    指定軸に垂直な各断面で変係数ラプラス方程式 ∇·(εr∇φ)=0 を有限体積・疎行列
    直接解法で解き、電極 A の表面電束から単位長容量を求め、断面厚 pitch を掛けて
    積算する。`parasitic_capacitance_ff`（面対向平行平板近似）と異なり、電極端の
    フリンジ電界も捉えるため、有限幅の電極では平行平板近似より大きい容量を与える。
    A・B 以外の導体は浮遊（フローティング）金属として連結成分ごとに等電位ノードに
    束ね、正味電荷ゼロ条件で物理的に正しく扱う（接地シールドではなく浮遊導体）。
    計算量は断面サイズに依存（大規模グリッドでは重い）。どちらかの材料が
    無ければ 0。
    """
    a = materials.get(name_a)
    b = materials.get(name_b)
    if a.id == b.id:
        return 0.0
    grid = wafer.grid
    if not (grid == a.id).any() or not (grid == b.id).any():
        return 0.0
    eps_field = permittivity_field(wafer)
    cond_ids = list(_conductor_ids())
    pitch_m = wafer.config.pitch_um * 1e-6
    eps0 = 8.854e-12
    ax = {"x": 2, "y": 1, "z": 0}.get(axis, 1)

    total = 0.0
    for s in range(grid.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = s
        g2 = grid[tuple(sl)]
        is_a = g2 == a.id
        is_b = g2 == b.id
        if not (is_a.any() and is_b.any()):
            continue
        e2 = eps_field[tuple(sl)]
        is_cond = np.isin(g2, cond_ids)
        cp = _solve_slice_capacitance(e2, is_a, is_b, is_cond)  # ε0 抜き・単位長
        total += cp * eps0 * pitch_m
    return float(total * 1e15)  # F → fF


def total_net_capacitance_ff(wafer: Wafer, name_a, axis: str = "y") -> float:
    """ある導体から「他の全導体（接地）」への総容量（fF）を静電界ソルバで返す。

    対象導体 A を 1V、他の全導体材料を 0V（接地）として ∇·(εr∇φ)=0 を解き、A の
    表面電束から総負荷容量を求める。配線がドライバから見る実効負荷容量（対基板＋
    全隣接配線）に相当し、`gate_switching_delay_ps` の load_cap_ff の物理的な入力に
    なる。A が無い/他に導体が無ければ 0。
    """
    a = materials.get(name_a)
    grid = wafer.grid
    cond_ids = list(_conductor_ids())
    others = [cid for cid in cond_ids if cid != a.id]
    if not (grid == a.id).any() or not np.isin(grid, others).any():
        return 0.0
    eps_field = permittivity_field(wafer)
    pitch_m = wafer.config.pitch_um * 1e-6
    eps0 = 8.854e-12
    ax = {"x": 2, "y": 1, "z": 0}.get(axis, 1)

    total = 0.0
    for s in range(grid.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = s
        g2 = grid[tuple(sl)]
        is_a = g2 == a.id
        is_b = np.isin(g2, others)  # 他の全導体を接地電極に
        if not (is_a.any() and is_b.any()):
            continue
        is_cond = np.isin(g2, cond_ids)
        cp = _solve_slice_capacitance(eps_field[tuple(sl)], is_a, is_b, is_cond)
        total += cp * eps0 * pitch_m
    return float(total * 1e15)


def ir_drop_v(wafer: Wafer, conductor, current_ma: float, axis: str = "x") -> dict:
    """電源配線の IR ドロップ（電圧降下, V）を返す＝電源健全性検証。

    配線抵抗 R（line_resistance_ohm）と電流 I から ΔV=I·R を求める。配線が細い/
    長いほど、電流が大きいほど電圧降下が大きく、供給電圧の低下（タイミング劣化）を
    招く。返す辞書: ir_drop_v / resistance_ohm / current_ma。断線/非導体では inf。
    """
    r = line_resistance_ohm(wafer, conductor, axis)
    i_a = abs(current_ma) * 1e-3
    if r == float("inf"):
        return {"ir_drop_v": float("inf"), "resistance_ohm": float("inf"),
                "current_ma": current_ma}
    return {
        "ir_drop_v": float(i_a * r),
        "resistance_ohm": float(r),
        "current_ma": current_ma,
    }


def summary(wafer: Wafer) -> dict:
    """主要指標をまとめた辞書を返す（ログ/テスト/UI 表示用）。"""
    return {
        "solid_fraction": solid_fraction(wafer),
        "step_height_um": step_height_um(wafer),
        "surface_roughness_um": surface_roughness_um(wafer),
        "cmp_uniformity_pct": cmp_uniformity_pct(wafer),
        "void_volume_um3": void_volume_um3(wafer),
        "materials": material_counts(wafer),
    }


def report(wafer: Wafer) -> str:
    """人が読めるテキスト計測レポートを生成する。

    材料ごとの体積、固体率、段差、各材料の膜厚統計をまとめた文字列を返す。
    GUI のレポート表示やファイル出力に使う。
    """
    cfg = wafer.config
    lines: list[str] = []
    lines.append("=== 計測レポート ===")
    lines.append(
        f"グリッド: {cfg.nx}x{cfg.ny}x{cfg.nz} vox"
        f"  (pitch={cfg.pitch_um:g}µm)"
    )
    lines.append(f"固体率: {solid_fraction(wafer) * 100:.1f}%")
    lines.append(f"表面段差: {step_height_um(wafer):.3f}µm")
    lines.append(f"表面粗さ(RMS): {surface_roughness_um(wafer):.3f}µm")
    lines.append(f"CMP均一性(3σ/平均): {cmp_uniformity_pct(wafer):.2f}%")
    void = void_volume_um3(wafer)
    if void > 0:
        lines.append(f"ボイド体積: {void:.4f}µm³")
    bow = wafer_bow_um(wafer)
    if abs(bow) >= 0.001:
        sign = "凸(引張)" if bow > 0 else "凹(圧縮)"
        lines.append(f"等価ウェハ反り: {bow:+.2f}µm  {sign}")
    rth = thermal_resistance_k_w(wafer)
    if np.isfinite(rth):
        lines.append(f"縦方向熱抵抗(基板→表面): {rth:.3e}K/W")
    lines.append("")
    lines.append("材料別 体積/膜厚:")
    counts = material_counts(wafer)
    for name in sorted(counts):
        vol = material_volume_um3(wafer, name)
        stats = film_thickness_stats(wafer, name)
        lines.append(
            f"  {name:<12} 体積={vol:8.3f}µm³  "
            f"膜厚 平均={stats['mean']:.3f} 最大={stats['max']:.3f}µm  "
            f"被覆={stats['coverage'] * 100:.0f}%"
        )

    # 電気/DRC セクション: グリッドに存在する導体材料を自動検出して表示。
    conductors = [
        m
        for m in materials.all_materials()
        if m.resistivity_ohm_um > 0 and (wafer.grid == m.id).any()
    ]
    if conductors:
        lines.append("")
        lines.append("電気特性 (導体):")
        for m in conductors:
            rs = sheet_resistance_ohm_sq(wafer, m.name)
            cont = electrical_continuity(wafer, m.name, "x")
            rs_txt = "∞" if rs == float("inf") else f"{rs:.3f}"
            conn = "導通" if cont["connected"] else "オープン"
            lines.append(
                f"  {m.name:<12} シート抵抗={rs_txt}Ω/sq  "
                f"x方向={conn}  連結成分={cont['n_components']}"
            )
        # 導体ペア間の最小間隔（ショート/近接リスク）＋寄生容量
        pair_lines: list[str] = []
        for i in range(len(conductors)):
            for j in range(i + 1, len(conductors)):
                a, b = conductors[i], conductors[j]
                sp = min_spacing_um(wafer, a.name, b.name)
                cap = parasitic_capacitance_ff(wafer, a.name, b.name)
                if sp == float("inf") and cap <= 0:
                    continue
                parts = [f"  {a.name}-{b.name}"]
                if sp != float("inf"):
                    flag = " ★接触/ショート" if sp == 0.0 else ""
                    parts.append(f"最小間隔={sp:.3f}µm{flag}")
                if cap > 0:
                    parts.append(f"寄生容量={cap:.3f}fF")
                pair_lines.append("  ".join(parts))
        if pair_lines:
            lines.append("導体間 最小間隔/容量 (DRC/寄生):")
            lines.extend(pair_lines)
    return "\n".join(lines)


def electrical_report(wafer: Wafer) -> dict:
    """電気/DRC 指標を機械可読な辞書で返す（JSON 出力用）。

    グリッドに存在する導体材料を自動検出し、材料ごとにシート抵抗・x 方向導通・
    連結成分数を、導体ペアごとに最小間隔（µm, 接触は 0）をまとめる。
    導体が無ければ空の構造を返す。inf は None（JSON で表現可能）に変換する。
    """
    conductors = [
        m
        for m in materials.all_materials()
        if m.resistivity_ohm_um > 0 and (wafer.grid == m.id).any()
    ]

    def _jsonable(x: float):
        return None if x == float("inf") else x

    cond_data: dict[str, dict] = {}
    for m in conductors:
        cont = electrical_continuity(wafer, m.name, "x")
        cond_data[m.name] = {
            "sheet_resistance_ohm_sq": _jsonable(sheet_resistance_ohm_sq(wafer, m.name)),
            "connected_x": cont["connected"],
            "n_components": cont["n_components"],
        }
    spacing: dict[str, float] = {}
    capacitance: dict[str, float] = {}
    for i in range(len(conductors)):
        for j in range(i + 1, len(conductors)):
            a, b = conductors[i], conductors[j]
            sp = min_spacing_um(wafer, a.name, b.name)
            if sp != float("inf"):
                spacing[f"{a.name}-{b.name}"] = sp
            cap = parasitic_capacitance_ff(wafer, a.name, b.name)
            if cap > 0:
                capacitance[f"{a.name}-{b.name}"] = cap
    return {
        "conductors": cond_data,
        "min_spacing_um": spacing,
        "capacitance_ff": capacitance,
    }


def defect_report(wafer: Wafer) -> dict:
    """各種不良モードを一括検査した機械可読な辞書を返す。

    半導体プロセスで典型的な不良モードを横断的にチェックする：
      - voids: 埋め込みボイド（トレンチ/ビア充填不良）の連結成分統計。
      - per_material: グリッドに存在する基板/空気以外の各材料について、
          pinhole（薄膜貫通欠陥）と residue（エッチ残渣/ストリンガー）を検査。
      - dishing_um: ダマシン CMP のディッシング深さ。
      - wafer_bow_um: 膜応力起因のウェハ反り。
    不良が無ければ各項目は 0／空となり、健全であることを示す。
    """
    skip = {"silicon", "air"}
    present = [
        m
        for m in materials.all_materials()
        if m.name not in skip and m.id != materials.AIR and (wafer.grid == m.id).any()
    ]
    per_material: dict[str, dict] = {}
    for m in present:
        per_material[m.name] = {
            "pinhole": pinhole_metrics(wafer, m.name),
            "residue": etch_residue_metrics(wafer, m.name),
        }
    # ディッシングは軟金属（Cu）が存在する場合のみ評価
    dishing = 0.0
    if (wafer.grid == materials.BY_NAME["metal_cu"].id).any():
        dishing = dishing_depth_um(wafer, "metal_cu")
    return {
        "voids": void_metrics(wafer),
        "per_material": per_material,
        "dishing_um": dishing,
        "wafer_bow_um": wafer_bow_um(wafer),
    }

