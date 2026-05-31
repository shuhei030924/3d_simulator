"""ウェハのボクセルグリッド表現。

グリッドは充填ボリュームであり、すべてのボクセルが材料 ID を持つ。
これにより断面が「空洞」にならず常に中身が詰まって表示される。

軸の規約:
    grid[z, y, x]
    z: 高さ方向（0 が底=基板側、増加方向が上=膜成長方向）
    x, y: 面内方向
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import materials


@dataclass
class WaferConfig:
    """グリッドの寸法設定。

    nx, ny, nz: 各軸のボクセル数。
    pitch_um: 1 ボクセルの一辺の長さ (µm)。
    substrate_um: 初期シリコン基板の厚み (µm)。
    """

    nx: int = 120
    ny: int = 120
    nz: int = 120
    pitch_um: float = 0.10
    substrate_um: float = 3.0

    def __post_init__(self) -> None:
        for name, val in (("nx", self.nx), ("ny", self.ny), ("nz", self.nz)):
            if not isinstance(val, (int, np.integer)) or val < 1:
                raise ValueError(
                    f"{name} は 1 以上の整数である必要があります（指定値: {val}）。"
                )
        if not np.isfinite(self.pitch_um) or self.pitch_um <= 0:
            raise ValueError(
                f"pitch_um は正の有限値である必要があります（指定値: {self.pitch_um}）。"
            )
        if not np.isfinite(self.substrate_um) or self.substrate_um < 0:
            raise ValueError(
                f"substrate_um は 0 以上の有限値である必要があります"
                f"（指定値: {self.substrate_um}）。"
            )

    def to_dict(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "pitch_um": self.pitch_um,
            "substrate_um": self.substrate_um,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WaferConfig:
        return cls(
            nx=int(d.get("nx", 120)),
            ny=int(d.get("ny", 120)),
            nz=int(d.get("nz", 120)),
            pitch_um=float(d.get("pitch_um", 0.10)),
            substrate_um=float(d.get("substrate_um", 3.0)),
        )


@dataclass
class Wafer:
    """ボクセルグリッドを保持し、工程から操作される対象。"""

    config: WaferConfig
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    # -- 基本操作 ----------------------------------------------------------
    def reset(self) -> None:
        """初期状態（底に基板、上は空気）に戻す。"""
        c = self.config
        self.grid = np.zeros((c.nz, c.ny, c.nx), dtype=np.uint8)
        sub_vox = max(1, int(round(c.substrate_um / c.pitch_um)))
        sub_vox = min(sub_vox, c.nz)
        self.grid[:sub_vox, :, :] = materials.BY_NAME["silicon"].id

    def um_to_vox(self, um: float) -> int:
        """µm をボクセル数に変換（最小 1）。"""
        return max(1, int(round(um / self.config.pitch_um)))

    # -- 形状ヘルパ --------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    def solid_mask(self) -> np.ndarray:
        """空気以外（=材料がある）のボクセルを True とするマスク。"""
        return self.grid != materials.AIR

    def resist_mask(self) -> np.ndarray:
        """フォトレジストのボクセルを True とするマスク。"""
        resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
        out = np.zeros(self.grid.shape, dtype=bool)
        for rid in resist_ids:
            out |= self.grid == rid
        return out

    def top_surface_z(self) -> np.ndarray:
        """各 (y, x) 列の最上位の固体ボクセルの z インデックスを返す。

        固体が無い列は -1。
        """
        solid = self.solid_mask()
        nz = self.grid.shape[0]
        # 上から探索して最初に固体が現れる z を求める
        any_solid = solid.any(axis=0)
        # 反転インデックスで最上面を取得
        top_from_top = np.argmax(solid[::-1, :, :], axis=0)
        z_top = (nz - 1) - top_from_top
        z_top = np.where(any_solid, z_top, -1)
        return z_top

    def column_stack(self, x: int, y: int) -> list[tuple[int, float]]:
        """指定列 (x, y) の層構成を下から上へ返す。

        連続する同一材料ボクセルをまとめ (材料ID, 厚み µm) のリストにする。
        空気 (AIR) は除外する。
        """
        col = self.grid[:, y, x]
        out: list[tuple[int, float]] = []
        run_id: int | None = None
        run_len = 0
        for v in col:
            iv = int(v)
            if iv == run_id:
                run_len += 1
                continue
            if run_id is not None and run_id != materials.AIR and run_len > 0:
                out.append((run_id, run_len * self.config.pitch_um))
            run_id = iv
            run_len = 1
        if run_id is not None and run_id != materials.AIR and run_len > 0:
            out.append((run_id, run_len * self.config.pitch_um))
        return out


