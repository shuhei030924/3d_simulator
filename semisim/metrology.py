"""計測・解析ヘルパ（メトロロジ）。

ボクセルグリッドから膜厚マップ・表面高さ・段差・体積・固体率などを
物理単位（µm）で算出する。GUI 非依存で純粋にエンジンのみに依存する。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import materials
from .grid import Wafer


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

    ダマシン CMP では軟らかい金属が過研磨で凹む（ディッシング）。対象材料が
    最上面に露出する列の代表高さ（中央値）と、それ以外（フィールド）の代表
    高さの差を返す。凹んでいなければ 0。
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
    soft_h = float(np.median(height[is_soft_top]))
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
        # 導体ペア間の最小間隔（ショート/近接リスク）
        pair_lines: list[str] = []
        for i in range(len(conductors)):
            for j in range(i + 1, len(conductors)):
                a, b = conductors[i], conductors[j]
                sp = min_spacing_um(wafer, a.name, b.name)
                if sp != float("inf"):
                    flag = "  ★接触/ショート" if sp == 0.0 else ""
                    pair_lines.append(
                        f"  {a.name}-{b.name} 最小間隔={sp:.3f}µm{flag}"
                    )
        if pair_lines:
            lines.append("導体間 最小間隔 (DRC):")
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
    for i in range(len(conductors)):
        for j in range(i + 1, len(conductors)):
            a, b = conductors[i], conductors[j]
            sp = min_spacing_um(wafer, a.name, b.name)
            if sp != float("inf"):
                spacing[f"{a.name}-{b.name}"] = sp
    return {"conductors": cond_data, "min_spacing_um": spacing}

