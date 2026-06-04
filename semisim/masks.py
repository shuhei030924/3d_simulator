"""フォトマスク（面内パターン）の定義。

マスクは解像度に依存しないよう、すべて 0..1 の分数座標で表現する。
複数の図形の和集合を「選択領域」とし、invert で反転できる。

選択領域 (mask_array が True の場所) の意味は工程ごとに異なる:
    - PHOTO(positive): True = 露光されレジストが除去される「開口部」
    - DRY/WET/DIFFUSION: マスク（レジスト）で保護される領域は別途決まるため、
      これらの工程は通常マスクを直接持たず、上に乗ったレジスト形状で決まる。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Shape:
    """マスクを構成する単一図形（分数座標 0..1）。"""

    kind: str  # "rect" | "circle"
    # rect: x0, y0, x1, y1 / circle: cx, cy, r
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> Shape:
        return cls(kind=d["kind"], params=dict(d.get("params", {})))

    def label(self) -> str:
        """図形の人間可読ラベル（種別アイコン＋主要パラメータ）。GUI 一覧用。"""
        p = self.params
        if self.kind == "rect":
            ang = p.get("angle", 0.0)
            ang_txt = f" ∠{ang:.0f}°" if ang else ""
            return (f"▭ 矩形 ({p.get('x0', 0):.2f},{p.get('y0', 0):.2f})-"
                    f"({p.get('x1', 1):.2f},{p.get('y1', 1):.2f}){ang_txt}")
        if self.kind == "stripe":
            return (f"▬ 帯 中心({p.get('cx', 0.5):.2f},{p.get('cy', 0.5):.2f})"
                    f" 角{p.get('angle', 0):.0f}° 幅{p.get('width', 0.2):.2f}")
        if self.kind == "grating":
            return (f"☰ 周期ライン 角{p.get('angle', 0):.0f}°"
                    f" 周期{p.get('period', 0.2):.2f} 幅{p.get('width', 0.1):.2f}")
        return (f"● 円 中心({p.get('cx', 0.5):.2f},{p.get('cy', 0.5):.2f})"
                f" r={p.get('r', 0.25):.2f}")

    def rasterize(self, nx: int, ny: int) -> np.ndarray:
        """(ny, nx) の bool 配列にラスタライズする。"""
        ys = (np.arange(ny) + 0.5) / ny
        xs = (np.arange(nx) + 0.5) / nx
        gx, gy = np.meshgrid(xs, ys)  # shape (ny, nx)
        if self.kind == "rect":
            x0 = self.params.get("x0", 0.0)
            y0 = self.params.get("y0", 0.0)
            x1 = self.params.get("x1", 1.0)
            y1 = self.params.get("y1", 1.0)
            angle = self.params.get("angle", 0.0)
            if angle:
                # 矩形中心まわりに座標を逆回転して軸並行判定に帰着
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                rad = np.deg2rad(angle)
                cos_a = np.cos(-rad)
                sin_a = np.sin(-rad)
                dx = gx - cx
                dy = gy - cy
                rx = cos_a * dx - sin_a * dy + cx
                ry = sin_a * dx + cos_a * dy + cy
                gx, gy = rx, ry
            return (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        if self.kind == "circle":
            cx = self.params.get("cx", 0.5)
            cy = self.params.get("cy", 0.5)
            r = self.params.get("r", 0.25)
            return (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2
        if self.kind == "stripe":
            # 指定角度方向に伸びる帯（ライン/トレンチ）。
            # cx, cy: 帯の中心が通る点 / angle: 帯の伸びる方向(度) / width: 帯幅
            cx = self.params.get("cx", 0.5)
            cy = self.params.get("cy", 0.5)
            angle = self.params.get("angle", 0.0)
            width = self.params.get("width", 0.2)
            rad = np.deg2rad(angle)
            # 帯の伸びる方向に直交する単位ベクトルへの射影距離で判定
            nxv = -np.sin(rad)
            nyv = np.cos(rad)
            dist = (gx - cx) * nxv + (gy - cy) * nyv
            return np.abs(dist) <= width / 2.0
        if self.kind == "grating":
            # 周期的なライン&スペース（回折格子状）。
            # angle: ラインの伸びる方向(度) / period: 周期 / width: ライン幅
            angle = self.params.get("angle", 0.0)
            period = max(1e-6, self.params.get("period", 0.2))
            width = self.params.get("width", 0.1)
            rad = np.deg2rad(angle)
            nxv = -np.sin(rad)
            nyv = np.cos(rad)
            t = gx * nxv + gy * nyv
            phase = np.mod(t, period)
            return phase < min(width, period)
        raise ValueError(f"未知の図形種別: {self.kind}")


@dataclass
class Mask:
    """複数図形の集合と反転フラグからなるフォトマスク。"""

    shapes: list[Shape] = field(default_factory=list)
    invert: bool = False

    def to_dict(self) -> dict:
        return {
            "shapes": [s.to_dict() for s in self.shapes],
            "invert": self.invert,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Mask:
        if d is None:
            return cls()
        return cls(
            shapes=[Shape.from_dict(s) for s in d.get("shapes", [])],
            invert=bool(d.get("invert", False)),
        )

    def is_empty(self) -> bool:
        return len(self.shapes) == 0

    def rasterize(self, nx: int, ny: int) -> np.ndarray:
        """選択領域を (ny, nx) bool 配列で返す。

        図形が無い場合は全面 True（=全面が対象）を返す。
        invert=True なら結果を反転する。
        """
        if not self.shapes:
            out = np.ones((ny, nx), dtype=bool)
        else:
            out = np.zeros((ny, nx), dtype=bool)
            for s in self.shapes:
                out |= s.rasterize(nx, ny)
        if self.invert:
            out = ~out
        return out

    def preview_rgb(self, size: int = 72) -> np.ndarray:
        """マスクのプレビュー画像（size×size×3, uint8, 上が y=0）を返す。

        選択領域をアクセント色、非選択を淡色で塗り分けた RGB 画像。GUI の
        マスクエディタでの即時プレビューや、テストでの検証に使う（Qt 非依存）。
        """
        sel = self.rasterize(size, size)  # (size, size) bool, 行=y
        sel = sel[::-1]  # 画像座標（上が y=0）に合わせて上下反転
        rgb = np.empty((size, size, 3), dtype=np.uint8)
        rgb[~sel] = (238, 242, 247)   # 非選択（淡いグレー）
        rgb[sel] = (45, 108, 223)     # 選択領域（アクセント青）
        return rgb
