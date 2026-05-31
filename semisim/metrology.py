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


def summary(wafer: Wafer) -> dict:
    """主要指標をまとめた辞書を返す（ログ/テスト/UI 表示用）。"""
    return {
        "solid_fraction": solid_fraction(wafer),
        "step_height_um": step_height_um(wafer),
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

