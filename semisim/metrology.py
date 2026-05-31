"""計測・解析ヘルパ（メトロロジ）。

ボクセルグリッドから膜厚マップ・表面高さ・段差・体積・固体率などを
物理単位（µm）で算出する。GUI 非依存で純粋にエンジンのみに依存する。
"""
from __future__ import annotations

import numpy as np

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
    return "\n".join(lines)

