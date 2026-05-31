"""ボクセルモデルを STL メッシュへ書き出すエクスポータ。

露出している固体ボクセル面（隣が空気／グリッド外の面）だけを三角形 2 枚の
矩形として出力する素朴な「voxel surface」メッシュ。外部 CAD・3D プリント・
ビューアでの確認に使える。追加依存なしで ASCII STL を生成する。

座標系は µm 単位（ボクセル index × pitch_um）。X=x, Y=y, Z=z。
"""
from __future__ import annotations

import numpy as np

from . import materials
from .grid import Wafer

# 6 方向の (軸シフト, 法線, 面オフセット, 面内の 4 頂点オフセット)。
# 頂点は外向き法線に対し反時計回り（CCW）になるよう並べる。
# 各ボクセルは [x, x+1] × [y, y+1] × [z, z+1]（pitch 倍前）を占める。
_FACES = [
    # -X 面
    ((0, 0, -1), (-1.0, 0.0, 0.0),
     [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
    # +X 面
    ((0, 0, 1), (1.0, 0.0, 0.0),
     [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
    # -Y 面
    ((0, -1, 0), (0.0, -1.0, 0.0),
     [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
    # +Y 面
    ((0, 1, 0), (0.0, 1.0, 0.0),
     [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
    # -Z 面（底）
    ((-1, 0, 0), (0.0, 0.0, -1.0),
     [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
    # +Z 面（上）
    ((1, 0, 0), (0.0, 0.0, 1.0),
     [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
]


def _exposed(solid: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """solid のうち、shift 方向の隣が固体でない（空気/グリッド外）面を True。"""
    neighbor = np.zeros_like(solid)
    dz, dy, dx = shift
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for axis, d in enumerate((dz, dy, dx)):
        if d > 0:
            dst[axis] = slice(0, -1)
            src[axis] = slice(1, None)
        elif d < 0:
            dst[axis] = slice(1, None)
            src[axis] = slice(0, -1)
    neighbor[tuple(dst)] = solid[tuple(src)]
    # 隣が固体なら露出していない。隣が無い（端）は neighbor=False のまま=露出。
    return solid & ~neighbor


def _facets(wafer: Wafer, name_or_id) -> list[tuple]:
    """(法線, v1, v2, v3) のリストを返す。各面を 2 三角形に分割。"""
    grid = wafer.grid
    p = wafer.config.pitch_um
    if name_or_id is None:
        solid = grid != materials.AIR
    else:
        solid = grid == materials.get(name_or_id).id
    out: list[tuple] = []
    for shift, normal, corners in _FACES:
        exposed = _exposed(solid, shift)
        idx = np.argwhere(exposed)  # (z, y, x)
        if idx.size == 0:
            continue
        for z, y, x in idx:
            # 4 頂点（µm）。corners は (dx, dy, dz)。
            verts = [
                ((x + cx) * p, (y + cy) * p, (z + cz) * p)
                for cx, cy, cz in corners
            ]
            # 矩形 → 2 三角形（0-1-2, 0-2-3）で CCW を保つ
            out.append((normal, verts[0], verts[1], verts[2]))
            out.append((normal, verts[0], verts[2], verts[3]))
    return out


def stl_string(wafer: Wafer, name_or_id=None, solid_name: str = "wafer") -> str:
    """ASCII STL 文字列を生成する。name_or_id 指定で材料を絞り込める。"""
    facets = _facets(wafer, name_or_id)
    lines = [f"solid {solid_name}"]
    for normal, v1, v2, v3 in facets:
        nx, ny, nz = normal
        lines.append(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}")
        lines.append("    outer loop")
        for vx, vy, vz in (v1, v2, v3):
            lines.append(f"      vertex {vx:.6e} {vy:.6e} {vz:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    return "\n".join(lines) + "\n"


def to_stl(wafer: Wafer, path: str, name_or_id=None, solid_name: str = "wafer") -> int:
    """STL ファイルを書き出し、出力した三角形（facet）数を返す。"""
    text = stl_string(wafer, name_or_id=name_or_id, solid_name=solid_name)
    with open(path, "w", encoding="ascii") as f:
        f.write(text)
    # facet 数 = "facet normal" 行数
    return text.count("facet normal")


def column_profile_csv(wafer: Wafer, x_index: int, y_index: int) -> str:
    """指定列 (x, y) の縦方向材料スタックを CSV 文字列で返す。

    列: z_index, z_um, material_id, material_name。底(z=0)から上へ並ぶ。
    SEM/TEM 断面や深さプロファイルとの比較に使える。
    """
    grid = wafer.grid
    nz, ny, nx = grid.shape
    x = int(np.clip(x_index, 0, nx - 1))
    y = int(np.clip(y_index, 0, ny - 1))
    p = wafer.config.pitch_um
    col = grid[:, y, x]
    lines = ["z_index,z_um,material_id,material_name"]
    for z in range(nz):
        mid = int(col[z])
        name = materials.BY_ID[mid].name if mid in materials.BY_ID else "unknown"
        lines.append(f"{z},{z * p:.4f},{mid},{name}")
    return "\n".join(lines) + "\n"


def to_csv_column(wafer: Wafer, path: str, x_index: int, y_index: int) -> int:
    """列 (x, y) の縦材料プロファイルを CSV に書き出し、行数（ヘッダ除く）を返す。"""
    text = column_profile_csv(wafer, x_index, y_index)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text.count("\n") - 1  # ヘッダ行を除いたデータ行数

