"""半導体プロセス工程の定義と適用ロジック。

各工程は `apply(wafer)` でボクセルグリッドを変更し、`to_dict`/`from_dict`
で JSON シリアライズできる。レシピは工程の順序付きリストで、毎回初期状態
から順に再適用（リプレイ）して決定論的に断面を再現する。

すべての工程は充填ボリュームを操作するため、断面が空洞になることはない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import materials
from .grid import Wafer
from .masks import Mask

# 工程タイプ名 -> クラス の登録テーブル
_REGISTRY: dict[str, type[Process]] = {}


def register(cls: type[Process]) -> type[Process]:
    _REGISTRY[cls.type] = cls
    return cls


def _require_positive(value: float, name: str) -> float:
    """正値でなければ ValueError。工程パラメータの入力検証に使う。"""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は正の有限値である必要があります（指定値: {value}）。")
    return float(value)


def _require_non_negative(value: float, name: str) -> float:
    """0 以上でなければ ValueError。"""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} は 0 以上の有限値である必要があります（指定値: {value}）。")
    return float(value)


def _require_range(value: float, lo: float, hi: float, name: str) -> float:
    """[lo, hi] の範囲外なら ValueError。"""
    if not np.isfinite(value) or value < lo or value > hi:
        raise ValueError(
            f"{name} は {lo}〜{hi} の範囲である必要があります（指定値: {value}）。"
        )
    return float(value)


def _isotropic_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """半径 radius ボクセルでマスクを等方的（真円/真球）に膨張させる。

    binary_dilation の反復は十字構造要素により L1 距離（ひし形・45°角）に
    なってしまう。拡散やドライブインは等方的に丸く広がるのが物理的に正しいため、
    ユークリッド距離変換でしきい値処理し、丸い広がりを得る。
    """
    if radius <= 0:
        return mask.copy()
    # mask 外の各ボクセルから最近傍 mask ボクセルまでのユークリッド距離
    dist = ndimage.distance_transform_edt(~mask)
    return dist <= radius


class Process:
    """全工程の基底クラス。"""

    type: str = "base"
    label: str = "工程"

    # サブクラスで実装 ----------------------------------------------------
    def apply(self, wafer: Wafer) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def summary(self) -> str:
        """レシピ一覧に表示する 1 行サマリ。"""
        return self.label

    def params_dict(self) -> dict:
        """type を除くパラメータ辞書。サブクラスで実装。"""
        return {}

    # シリアライズ --------------------------------------------------------
    def to_dict(self) -> dict:
        d = {"type": self.type}
        d.update(self.params_dict())
        return d

    @staticmethod
    def from_dict(d: dict) -> Process:
        t = d.get("type")
        cls = _REGISTRY.get(t)
        if cls is None:
            raise ValueError(f"未知の工程タイプ: {t}")
        return cls._from_params(d)

    @classmethod
    def _from_params(cls, d: dict) -> Process:  # pragma: no cover - 抽象
        raise NotImplementedError


# --- 共通ヘルパ -------------------------------------------------------------
def _top_material(wafer: Wafer) -> tuple[np.ndarray, np.ndarray]:
    """各列の最上面 z と、その材料 ID を返す ((ny,nx), (ny,nx))。"""
    z_top = wafer.top_surface_z()
    ny, nx = z_top.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    safe_z = np.where(z_top >= 0, z_top, 0)
    top_id = wafer.grid[safe_z, yy, xx]
    top_id = np.where(z_top >= 0, top_id, materials.AIR)
    return z_top, top_id


# === PHOTO（フォトリソグラフィ）============================================
@register
@dataclass
class Photo(Process):
    """フォトレジストを塗布し、マスクで現像してパターニングする。"""

    type = "PHOTO"
    label = "フォトリソ"

    mask: Mask = field(default_factory=Mask)
    thickness_um: float = 1.0
    polarity: str = "positive"  # positive: 開口部のレジストが除去される
    # 光学解像度有限による角の丸め（OPC 前の生パターン）。マスクラスタを
    # ガウスぼかししてから 0.5 で二値化し、鋭い角を丸める。0=無効。
    edge_blur_sigma_um: float = 0.0

    def summary(self) -> str:
        pol = "ポジ" if self.polarity == "positive" else "ネガ"
        blur = "" if self.edge_blur_sigma_um <= 0 else "  +角丸め"
        return (
            f"PHOTO  厚{self.thickness_um:.2f}µm  {pol}"
            f"  図形{len(self.mask.shapes)}{blur}"
        )

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "レジスト厚")
        _require_non_negative(self.edge_blur_sigma_um, "エッジぼかし")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        resist_id = materials.BY_NAME["photoresist"].id
        t = wafer.um_to_vox(self.thickness_um)

        # 1) スピン塗布: 最も高い表面 + t の高さまで平坦に充填
        z_top = wafer.top_surface_z()  # (ny, nx)
        max_top = int(z_top.max()) if z_top.max() >= 0 else -1
        flat_top = min(nz - 1, max_top + t)

        # 各列 (top_z+1 .. flat_top) の空気をレジストで充填
        # 固体が無い列(z_top=-1)は塗布対象外（底まで充填しないようにする）
        z_idx = np.arange(nz)[:, None, None]
        has_solid = (z_top >= 0)[None, :, :]
        above_surface = z_idx > z_top[None, :, :]
        within_flat = z_idx <= flat_top
        air = grid == materials.AIR
        coat = above_surface & within_flat & air & has_solid
        grid[coat] = resist_id

        # 2) 現像: マスクの選択領域に応じて列ごとにレジストを除去
        selected = self.mask.rasterize(nx, ny)  # True = 開口（選択領域）
        if self.edge_blur_sigma_um > 0:
            # 角を丸める: ガウスぼかし後に 0.5 で再二値化
            sigma = wafer.um_to_vox(self.edge_blur_sigma_um)
            if sigma > 0:
                blurred = ndimage.gaussian_filter(
                    selected.astype(np.float32), sigma=sigma, mode="nearest"
                )
                selected = blurred >= 0.5
        remove_cols = selected if self.polarity == "positive" else ~selected
        if remove_cols.any():
            resist_vox = grid == resist_id
            remove = resist_vox & remove_cols[None, :, :]
            grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "mask": self.mask.to_dict(),
            "thickness_um": self.thickness_um,
            "polarity": self.polarity,
            "edge_blur_sigma_um": self.edge_blur_sigma_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Photo:
        return cls(
            mask=Mask.from_dict(d.get("mask")),
            thickness_um=float(d.get("thickness_um", 1.0)),
            polarity=d.get("polarity", "positive"),
            edge_blur_sigma_um=float(d.get("edge_blur_sigma_um", 0.0)),
        )


# === CVD（コンフォーマル成膜）==============================================
@register
@dataclass
class CVD(Process):
    """露出面すべてに等方的（コンフォーマル）に成膜する。"""

    type = "CVD"
    label = "CVD成膜"

    material: str = "oxide"
    thickness_um: float = 0.5
    # 負荷効果（マクロローディング, 0〜1）。パターン密度が高いほど反応種が
    # 枯渇して膜が薄くなる現象を簡易再現する。0 で従来どおり一定厚。
    loading: float = 0.0
    # 表面ラフネス（RMS, µm）。成膜表面の微小な凹凸を再現する。各列の上面を
    # 平均 0・標準偏差 roughness_um のガウス分布で上下させる。0 で平滑。
    roughness_um: float = 0.0
    seed: int = 0  # ラフネス乱数シード（再現性のため）

    def summary(self) -> str:
        m = materials.get(self.material)
        ld = "" if self.loading <= 0 else f"  負荷{self.loading:.2f}"
        rg = "" if self.roughness_um <= 0 else f"  粗さ{self.roughness_um:.2f}µm"
        return f"CVD  {m.label}  厚{self.thickness_um:.2f}µm{ld}{rg}"

    def _effective_thickness_vox(self, wafer: Wafer) -> int:
        """パターン密度に応じて負荷効果で減じた実効膜厚（ボクセル）を返す。"""
        t = wafer.um_to_vox(self.thickness_um)
        if self.loading <= 0:
            return t
        # パターン密度 = 基板上面より高い（=既にパターンがある）列の割合。
        # 平坦ブランケットでは 0、密パターンでは 1 に近づく。
        sub_top = wafer.um_to_vox(wafer.config.substrate_um)
        z_top = wafer.top_surface_z()
        density = float(np.mean(z_top > sub_top)) if z_top.size else 0.0
        t_eff = t * (1.0 - self.loading * density)
        return max(1, int(round(t_eff)))

    def _apply_roughness(self, wafer: Wafer, mat_id: int) -> None:
        """成膜表面に RMS ラフネスを付与する（膜上面のみ上下に揺らす）。"""
        grid = wafer.grid
        nz = grid.shape[0]
        sigma_r = max(0.0, float(wafer.um_to_vox(self.roughness_um)))
        if sigma_r <= 0:
            return
        rng = np.random.default_rng(int(self.seed))
        z_top, top_id = _top_material(wafer)
        is_film_top = (z_top >= 0) & (top_id == mat_id)
        delta = np.round(rng.normal(0.0, sigma_r, size=z_top.shape)).astype(int)
        ys, xs = np.nonzero(is_film_top)
        for y, x in zip(ys.tolist(), xs.tolist()):
            d = int(delta[y, x])
            zt = int(z_top[y, x])
            if d > 0:  # 凸: 上面に膜を追加
                zmax = min(nz - 1, zt + d)
                col = grid[zt + 1 : zmax + 1, y, x]
                col[col == materials.AIR] = mat_id
            elif d < 0:  # 凹: 上面の膜を削る
                zmin = max(0, zt + d + 1)
                for z in range(zt, zmin - 1, -1):
                    if grid[z, y, x] == mat_id:
                        grid[z, y, x] = materials.AIR
                    else:
                        break

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "膜厚")
        _require_range(self.loading, 0.0, 1.0, "負荷効果")
        _require_non_negative(self.roughness_um, "ラフネス")
        grid = wafer.grid
        mat_id = materials.get(self.material).id
        t = self._effective_thickness_vox(wafer)
        air = grid == materials.AIR
        # 固体表面からの距離 t 以内の空気に堆積（コンフォーマル）
        dist = ndimage.distance_transform_edt(air)
        deposit = air & (dist <= t)
        grid[deposit] = mat_id
        if self.roughness_um > 0:
            self._apply_roughness(wafer, mat_id)

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "loading": self.loading,
            "roughness_um": self.roughness_um,
            "seed": self.seed,
        }

    @classmethod
    def _from_params(cls, d: dict) -> CVD:
        return cls(
            material=d.get("material", "oxide"),
            thickness_um=float(d.get("thickness_um", 0.5)),
            loading=float(d.get("loading", 0.0)),
            roughness_um=float(d.get("roughness_um", 0.0)),
            seed=int(d.get("seed", 0)),
        )


# === ALD（原子層堆積）======================================================
@register
@dataclass
class ALD(Process):
    """原子層堆積。サイクル数×1サイクル成長量でnm精度の超コンフォーマル膜。

    CVD と同じ等方堆積だが、膜厚をサイクル数で精密制御する点が特徴。
    高アスペクト比でも均一被覆するため High-k/バリア膜に用いる。
    """

    type = "ALD"
    label = "ALD成膜"

    material: str = "hafnia"
    cycles: int = 100
    growth_per_cycle_nm: float = 1.0
    ar_coverage: float = 1.0  # 高アスペクト比底部の被覆率(1.0=完全コンフォーマル)
    ar_threshold: float = 10.0  # 被覆率が ar_coverage まで低下する AR

    @property
    def thickness_um(self) -> float:
        return self.cycles * self.growth_per_cycle_nm / 1000.0

    def summary(self) -> str:
        m = materials.get(self.material)
        cov = "" if self.ar_coverage >= 0.999 else f"  底被覆{self.ar_coverage:.0%}"
        return (
            f"ALD  {m.label}  {self.cycles}cyc×{self.growth_per_cycle_nm:.1f}nm"
            f"={self.thickness_um * 1000:.0f}nm{cov}"
        )

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.cycles, "サイクル数")
        _require_positive(self.growth_per_cycle_nm, "1サイクル成長量")
        _require_range(self.ar_coverage, 0.0, 1.0, "底被覆率")
        _require_positive(self.ar_threshold, "AR閾値")
        grid = wafer.grid
        mat_id = materials.get(self.material).id
        t = wafer.um_to_vox(self.thickness_um)
        air = grid == materials.AIR
        # 固体表面からの距離 t 以内の空気に等方堆積（超コンフォーマル）
        dist = ndimage.distance_transform_edt(air)
        if self.ar_coverage >= 0.999:
            deposit = air & (dist <= t)
        else:
            # 高 AR 窪みでは前駆体枯渇により深部ほど膜厚が減る。
            # 各列の AR(=深さ/幅)で膜厚を ar_coverage に向け線形低下させる。
            z_top = wafer.top_surface_z()
            valid = z_top >= 0
            field_level = int(z_top[valid].max()) if valid.any() else 0
            recess = valid & (z_top < field_level)
            t_field = np.full(z_top.shape, float(t))
            if recess.any():
                hw = ndimage.distance_transform_edt(recess)
                labels, n = ndimage.label(recess)
                if n > 0:
                    feat = ndimage.maximum(hw, labels, index=range(1, n + 1))
                    lut = np.concatenate(([0.0], np.asarray(feat, dtype=float)))
                    hw_feat = lut[labels]
                    width = np.maximum(2.0 * hw_feat, 1.0)
                    depth = np.where(recess, field_level - z_top, 0.0).astype(float)
                    ar = depth / width
                    cov = 1.0 - (1.0 - self.ar_coverage) * np.clip(
                        ar / self.ar_threshold, 0.0, 1.0
                    )
                    t_field = np.where(recess, np.maximum(1.0, t * cov), float(t))
            deposit = air & (dist <= t_field[None, :, :])
        grid[deposit] = mat_id

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "cycles": self.cycles,
            "growth_per_cycle_nm": self.growth_per_cycle_nm,
            "ar_coverage": self.ar_coverage,
            "ar_threshold": self.ar_threshold,
        }

    @classmethod
    def _from_params(cls, d: dict) -> ALD:
        return cls(
            material=d.get("material", "hafnia"),
            cycles=int(d.get("cycles", 100)),
            growth_per_cycle_nm=float(d.get("growth_per_cycle_nm", 1.0)),
            ar_coverage=float(d.get("ar_coverage", 1.0)),
            ar_threshold=float(d.get("ar_threshold", 10.0)),
        )


# === PVD（指向性成膜）======================================================
@register
@dataclass
class PVD(Process):
    """上方からの視線方向（指向性）に成膜する。側壁には付かない。

    step_coverage: 0..1。深い窪み（周囲が高い壁に囲まれた底）では膜が薄く
    なるシャドーイングを簡易的に再現する。1.0 で従来どおり一様堆積。
    """

    type = "PVD"
    label = "PVD成膜"

    material: str = "metal_al"
    thickness_um: float = 0.5
    step_coverage: float = 1.0
    # オーバーハング / ブレッドローフィング（0=無効）。指向性成膜では開口
    # 上端の隅にフラックスが集中して庇状に横へ張り出す。狭い開口では両側
    # の庇が合体して上部を塞ぎ、下にキーホールボイドが残る。膜厚に対する
    # 横張り出し量の比で与える。
    overhang: float = 0.0

    def summary(self) -> str:
        m = materials.get(self.material)
        sc = "" if self.step_coverage >= 0.999 else f"  被覆{self.step_coverage:.0%}"
        oh = "" if self.overhang <= 0 else f"  庇{self.overhang:.1f}"
        return f"PVD  {m.label}  厚{self.thickness_um:.2f}µm{sc}{oh}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "膜厚")
        _require_range(self.step_coverage, 0.0, 1.0, "ステップカバレッジ")
        _require_non_negative(self.overhang, "オーバーハング")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        mat_id = materials.get(self.material).id
        t = wafer.um_to_vox(self.thickness_um)
        z_top = wafer.top_surface_z()  # (ny, nx) 各列の最上固体
        valid = z_top >= 0

        # シャドーイング: 各列の堆積厚を周囲との高低差で減じる。
        # 周囲(最大プール)より低い列は、窪みの深さに比例して薄くなる。
        sc = float(np.clip(self.step_coverage, 0.0, 1.0))
        if sc < 0.999:
            zt = np.where(valid, z_top, 0).astype(np.float32)
            neigh_max = ndimage.maximum_filter(zt, size=3, mode="nearest")
            recess = np.clip(neigh_max - zt, 0, None)  # 窪みの深さ(vox)
            # 窪みが深いほど被覆率を sc に向けて線形に低下させる簡易モデル。
            # recess=0（平坦）で atten=1、recess>=膜厚で atten=sc に飽和する。
            atten = 1.0 - (1.0 - sc) * np.clip(recess / max(t, 1), 0, 1)
            local_t = np.maximum(0, np.round(t * atten)).astype(int)
        else:
            local_t = np.full((ny, nx), t, dtype=int)

        z_idx = np.arange(nz)[:, None, None]
        lo = z_top[None, :, :]
        hi = (z_top + local_t)[None, :, :]
        deposit = (z_idx > lo) & (z_idx <= hi) & (grid == materials.AIR)
        deposit &= valid[None, :, :]
        grid[deposit] = mat_id

        # オーバーハング / ブレッドローフィング: 開口上端の隅から庇状に
        # 横へ張り出す。膜の上端表面のうち「真下が空気」の隅(=開口に
        # 面した張り出し)だけを横方向へ反復成長させる。狭い開口では両側
        # の庇が合体して上部を塞ぎ、下にボイドが封じ込められる。
        oh = int(round(self.overhang * t)) if self.overhang > 0 else 0
        if oh > 0:
            lat = np.zeros((3, 3, 3), dtype=bool)
            lat[1, 1, 0] = lat[1, 1, 2] = lat[1, 0, 1] = lat[1, 2, 1] = True
            for _ in range(oh):
                film = grid == mat_id
                grow = ndimage.binary_dilation(film, structure=lat)
                grow &= grid == materials.AIR
                # 庇条件: 直下が空気のボクセルのみ(開口上に張り出す部分)
                below_air = np.zeros_like(grow)
                below_air[1:] = grid[:-1] == materials.AIR
                grow &= below_air
                if not grow.any():
                    break
                grid[grow] = mat_id

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "step_coverage": self.step_coverage,
            "overhang": self.overhang,
        }

    @classmethod
    def _from_params(cls, d: dict) -> PVD:
        return cls(
            material=d.get("material", "metal_al"),
            thickness_um=float(d.get("thickness_um", 0.5)),
            step_coverage=float(d.get("step_coverage", 1.0)),
            overhang=float(d.get("overhang", 0.0)),
        )


def _resolve_targets(targets) -> list[int]:
    """ターゲット材料名のリストを ID リストに変換。空なら全エッチ可能材料。"""
    if not targets:
        return [m.id for m in materials.all_materials() if m.etchable and not m.is_resist]
    out = []
    for name in targets:
        out.append(materials.get(name).id)
    return out


# === DRY（異方性エッチング）================================================
@register
@dataclass
class DryEtch(Process):
    """上方から垂直に削る異方性エッチング。レジスト等で保護される。

    lateral_um を指定すると、垂直エッチ後にマスク端からわずかに横方向へも
    削る（RIE のエッチバイアス/アンダーカット）。0 で完全異方性。
    """

    type = "DRY"
    label = "ドライエッチ"

    targets: list[str] = field(default_factory=list)  # 空 = 自動
    depth_um: float = 0.5
    overetch_pct: float = 0.0  # ターゲット枯渇後に下層も削る割合(%)
    lateral_um: float = 0.0  # 横方向エッチバイアス（アンダーカット）
    # 材料別エッチ選択比（材料名 -> 相対エッチ速度, 0〜1）。
    # 1.0=基準速度、低いほど削れにくい（ストップ層は 0 付近）。
    # 例: {"oxide": 0.33} は Si:SiO2 = 3:1 の選択比を表す。
    selectivity: dict[str, float] = field(default_factory=dict)
    # マスク消耗比: ターゲットを depth_um 削る間にレジストが
    # mask_erosion×depth_um だけ上面から減る（実機の 0.3〜0.5 程度）。
    mask_erosion: float = 0.0
    # 側壁テーパ角（度, 垂直からの傾き）。0=完全垂直。正でホールが上広がりの
    # 台形になる（深さ d で開口端から d×tan(taper) だけ内側に後退）。
    taper_deg: float = 0.0
    # RIE ノッチング（µm, 0=無効）。エッチがストップ層（選択比で削れない
    # 下層）に到達すると、絶縁膜の帯電で入射イオンが横へ偏向し、界面直上の
    # 側壁にノッチ（横アンダーカット）が生じる。SOI エッチ等で問題になる。
    notch_um: float = 0.0

    def summary(self) -> str:
        tgt = "/".join(self.targets) if self.targets else "露出材料"
        oe = "" if self.overetch_pct <= 0 else f"  +OE{self.overetch_pct:.0f}%"
        lat = "" if self.lateral_um <= 0 else f"  横{self.lateral_um:.2f}µm"
        sel = "  選択比あり" if self.selectivity else ""
        tp = "" if self.taper_deg <= 0 else f"  テーパ{self.taper_deg:.0f}°"
        nt = "" if self.notch_um <= 0 else f"  ノッチ{self.notch_um:.2f}µm"
        return f"DRY  {tgt}  深さ{self.depth_um:.2f}µm{oe}{lat}{sel}{tp}{nt}"

    def _rate_map(self, top_id: np.ndarray) -> np.ndarray:
        """各列の最上面材料に対する相対エッチ速度マップ（既定 1.0）。"""
        rate = np.ones(top_id.shape, dtype=float)
        for name, rv in self.selectivity.items():
            mid = materials.get(name).id
            rate[top_id == mid] = rv
        return rate

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "エッチ量")
        _require_non_negative(self.overetch_pct, "オーバーエッチ率")
        _require_non_negative(self.lateral_um, "横方向バイアス")
        _require_non_negative(self.mask_erosion, "マスク消耗比")
        _require_range(self.taper_deg, 0.0, 89.0, "テーパ角")
        _require_non_negative(self.notch_um, "RIEノッチ量")
        for name, rv in self.selectivity.items():
            _require_range(rv, 0.0, 1.0, f"選択比[{name}]")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        target_ids = set(_resolve_targets(self.targets))
        depth = wafer.um_to_vox(self.depth_um)

        # 初期の最上面材料がターゲットの列のみエッチ対象（マスク効果）
        _, top_id = _top_material(wafer)
        eligible = np.isin(top_id, list(target_ids))  # (ny, nx)

        # 側壁テーパ: 各列が削れる最大深さを開口端からの距離で制限する。
        # 深さ d での後退量 = d×tan(taper) ＝ 開口端から e ボクセル内側の列は
        # 深さ e/tan(taper) までしか削れない（上広がりの台形プロファイル）。
        if self.taper_deg > 0:
            tan_t = math.tan(math.radians(self.taper_deg))
            edge_dist = ndimage.distance_transform_edt(eligible)
            depth_cap = np.where(eligible, np.minimum(depth, edge_dist / tan_t), 0.0)
        else:
            depth_cap = np.full((ny, nx), float(depth))
        etched = np.zeros((ny, nx), dtype=float)  # 列ごとの除去ボクセル数

        # 選択比を 1 列あたりのエッチ予算（ボクセル）で表現する。
        # 削るたびに cost=1/速度 を消費し、速度<1 の材料ほど予算を多く使う
        # ＝削れる量が減る。速度1.0なら従来どおり depth ボクセル削る。
        budget = np.where(eligible, float(depth), 0.0)
        for _ in range(depth):
            z_top, top_id = _top_material(wafer)
            top_is_target = np.isin(top_id, list(target_ids))
            rate = self._rate_map(top_id)
            cost = np.where(rate > 0, 1.0 / np.where(rate > 0, rate, 1.0), np.inf)
            do = (
                eligible
                & top_is_target
                & (z_top >= 0)
                & (budget >= cost)
                & (etched < depth_cap)
            )
            if not do.any():
                break
            ys, xs = np.nonzero(do)
            grid[z_top[ys, xs], ys, xs] = materials.AIR
            budget[do] -= cost[do]
            etched[do] += 1.0

        # オーバーエッチ: ターゲット直下に露出した下層を追加で削る。
        # レジスト(is_resist)は保護膜なので削らない。
        oe = max(0, int(round(depth * self.overetch_pct / 100.0)))
        if oe > 0:
            resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
            for _ in range(oe):
                z_top, top_id = _top_material(wafer)
                exposed = eligible & (z_top >= 0) & ~np.isin(top_id, resist_ids)
                exposed &= top_id != materials.AIR
                if not exposed.any():
                    break
                ys, xs = np.nonzero(exposed)
                grid[z_top[ys, xs], ys, xs] = materials.AIR

        # 横方向エッチバイアス: 生成した空気に隣接するターゲットを lat 分削る。
        # レジストは保護されるため、マスク端下へのアンダーカットを再現する。
        lat = wafer.um_to_vox(self.lateral_um) if self.lateral_um > 0 else 0
        if lat > 0:
            resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
            air = grid == materials.AIR
            grown = ndimage.distance_transform_edt(~air) <= lat
            undercut = (
                grown
                & np.isin(grid, list(target_ids))
                & ~np.isin(grid, resist_ids)
            )
            undercut[0, :, :] = False  # 基板最下層は保護
            grid[undercut] = materials.AIR

        # RIE ノッチング: エッチがストップ層に到達した界面で、帯電による
        # イオン偏向で側壁が横方向に抉られる。トレンチ底（直下がストップ層
        # ＝ターゲットでも空気でもレジストでもない固体）の空気を界面の高さ
        # 帯で横方向に広げ、隣接ターゲット側壁を notch 分だけ削る。
        notch = wafer.um_to_vox(self.notch_um) if self.notch_um > 0 else 0
        if notch > 0:
            resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
            below_id = np.full_like(grid, materials.AIR)
            below_id[1:] = grid[:-1]
            protected = [materials.AIR, *list(target_ids), *resist_ids]
            is_stop = (below_id != materials.AIR) & ~np.isin(below_id, protected)
            floor_air = (grid == materials.AIR) & is_stop  # 界面直上の空気帯
            lat_struct = np.zeros((3, 3, 3), dtype=bool)
            lat_struct[1, 1, 1] = True
            lat_struct[1, 1, 0] = lat_struct[1, 1, 2] = True
            lat_struct[1, 0, 1] = lat_struct[1, 2, 1] = True
            region = floor_air
            for _ in range(notch):
                region = ndimage.binary_dilation(region, structure=lat_struct)
            notch_cut = (
                region
                & np.isin(grid, list(target_ids))
                & ~np.isin(grid, resist_ids)
            )
            grid[notch_cut] = materials.AIR

        # マスク消耗: レジストを上面から mask_erosion×depth 分だけ削る。
        # 実機ではエッチ中にマスクも消費され、過度だと被覆が失われ CD が崩れる。
        erode = (
            wafer.um_to_vox(self.mask_erosion * self.depth_um)
            if self.mask_erosion > 0
            else 0
        )
        if erode > 0:
            resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
            for _ in range(erode):
                z_top, top_id = _top_material(wafer)
                do = (z_top >= 0) & np.isin(top_id, resist_ids)
                if not do.any():
                    break
                ys, xs = np.nonzero(do)
                grid[z_top[ys, xs], ys, xs] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "targets": list(self.targets),
            "depth_um": self.depth_um,
            "overetch_pct": self.overetch_pct,
            "lateral_um": self.lateral_um,
            "selectivity": dict(self.selectivity),
            "mask_erosion": self.mask_erosion,
            "taper_deg": self.taper_deg,
            "notch_um": self.notch_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> DryEtch:
        return cls(
            targets=list(d.get("targets", [])),
            depth_um=float(d.get("depth_um", 0.5)),
            overetch_pct=float(d.get("overetch_pct", 0.0)),
            lateral_um=float(d.get("lateral_um", 0.0)),
            selectivity={
                str(k): float(v) for k, v in dict(d.get("selectivity", {})).items()
            },
            mask_erosion=float(d.get("mask_erosion", 0.0)),
            taper_deg=float(d.get("taper_deg", 0.0)),
            notch_um=float(d.get("notch_um", 0.0)),
        )


# === WET（等方性エッチング）================================================
@register
@dataclass
class WetEtch(Process):
    """薬液による等方性エッチング。マスク下にアンダーカットが入る。"""

    type = "WET"
    label = "ウェットエッチ"

    targets: list[str] = field(default_factory=list)
    depth_um: float = 0.5
    # 横方向アンダーカット / 縦方向エッチ の比（0〜1）。
    # 1.0=完全等方（縦と同量の横アンダーカット）。0=ほぼ垂直（撹拌や
    # 表面活性化で横方向が抑制された実プロセスを模擬）。
    lateral_ratio: float = 1.0

    def summary(self) -> str:
        tgt = "/".join(self.targets) if self.targets else "露出材料"
        lr = "" if self.lateral_ratio >= 0.999 else f"  横比{self.lateral_ratio:.2f}"
        return f"WET  {tgt}  深さ{self.depth_um:.2f}µm{lr}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "エッチ量")
        _require_range(self.lateral_ratio, 0.0, 1.0, "横方向比")
        grid = wafer.grid
        target_ids = _resolve_targets(self.targets)
        r = wafer.um_to_vox(self.depth_um)
        # 露出面から 1 ボクセルずつエッチ前線を伝播させる。
        # 縦方向(z)は毎回進め、横方向(x,y)は lateral_ratio に比例した
        # 頻度でのみ進めることで、アンダーカット量を縦エッチの
        # lateral_ratio 倍に制御する（等方〜異方の連続調整）。
        # 注: 距離変換による一括計算は障壁材料を貫通してしまい物理が崩れる
        # ため、ここでは前線伝播（反復ダイレーション）を意図的に用いる。
        full6 = ndimage.generate_binary_structure(3, 1)  # 6 近傍（等方）
        vert = np.zeros((3, 3, 3), dtype=bool)  # 縦方向(z±1)のみ
        vert[1, 1, 1] = True
        vert[0, 1, 1] = True
        vert[2, 1, 1] = True
        lr = float(np.clip(self.lateral_ratio, 0.0, 1.0))
        lat_budget = 0.0
        for _ in range(r):
            air = grid == materials.AIR
            lat_budget += lr
            if lat_budget >= 1.0:
                front = ndimage.binary_dilation(air, structure=full6)
                lat_budget -= 1.0
            else:
                front = ndimage.binary_dilation(air, structure=vert)
            cur_target = np.isin(grid, target_ids)
            remove = front & cur_target
            # 基板最下層は薬液で削り切らない（ウェハ貫通を防ぐ物理的下限）。
            remove[0, :, :] = False
            if not remove.any():
                break
            grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "targets": list(self.targets),
            "depth_um": self.depth_um,
            "lateral_ratio": self.lateral_ratio,
        }

    @classmethod
    def _from_params(cls, d: dict) -> WetEtch:
        return cls(
            targets=list(d.get("targets", [])),
            depth_um=float(d.get("depth_um", 0.5)),
            lateral_ratio=float(d.get("lateral_ratio", 1.0)),
        )


# === DIFFUSION（拡散/ドーピング）===========================================
@register
@dataclass
class Diffusion(Process):
    """露出シリコンに不純物を拡散し、拡散層へ変化させる。"""

    type = "DIFFUSION"
    label = "拡散"

    dopant: str = "doped_n"  # doped_n / doped_p
    depth_um: float = 0.6
    # 横方向広がり = 縦方向拡散深さ × lateral_factor（拡散の等方性の簡易係数）。
    # 既定 1/3: 等方拡散では横拡散が縦拡散の約 0.5〜0.8 倍だが、マスク端での
    # 二次元拡散の実測（横/縦 ≈ 0.3）に合わせた経験値。
    lateral_factor: float = 1.0 / 3.0

    def summary(self) -> str:
        m = materials.get(self.dopant)
        return f"DIFFUSION  {m.label}  深さ{self.depth_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "拡散深さ")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        dop_id = materials.get(self.dopant).id
        depth = wafer.um_to_vox(self.depth_um)

        # 露出シリコン（最上面がシリコン）の列のみ拡散
        _, top_id = _top_material(wafer)
        eligible = top_id == si_id  # (ny, nx)

        converted = np.zeros(grid.shape, dtype=bool)
        # 各列で最上シリコンから depth 分を変換
        work_top = wafer.top_surface_z().copy()
        for _ in range(depth):
            ys, xs = np.nonzero(eligible & (work_top >= 0))
            if ys.size == 0:
                break
            zt = work_top[ys, xs]
            is_si = grid[zt, ys, xs] == si_id
            sel_y = ys[is_si]
            sel_x = xs[is_si]
            sel_z = zt[is_si]
            grid[sel_z, sel_y, sel_x] = dop_id
            converted[sel_z, sel_y, sel_x] = True
            work_top[sel_y, sel_x] -= 1

        # 横方向の広がり（拡散の等方性を簡易表現）。
        # 縦方向拡散深さに lateral_factor を掛けた量を横方向広がりとする。
        lat = max(0, int(round(depth * self.lateral_factor)))
        if lat > 0 and converted.any():
            grown = _isotropic_dilate(converted, lat)
            spread = grown & (grid == si_id)
            grid[spread] = dop_id

    def params_dict(self) -> dict:
        return {
            "dopant": self.dopant,
            "depth_um": self.depth_um,
            "lateral_factor": self.lateral_factor,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Diffusion:
        return cls(
            dopant=d.get("dopant", "doped_n"),
            depth_um=float(d.get("depth_um", 0.6)),
            lateral_factor=float(d.get("lateral_factor", 1.0 / 3.0)),
        )


# === STRIP（除去/剥離）=====================================================
@register
@dataclass
class Strip(Process):
    """指定材料をすべて除去する（レジスト剥離など）。"""

    type = "STRIP"
    label = "剥離"

    material: str = "photoresist"

    def summary(self) -> str:
        m = materials.get(self.material)
        return f"STRIP  {m.label} を除去"

    def apply(self, wafer: Wafer) -> None:
        grid = wafer.grid
        mat_id = materials.get(self.material).id
        grid[grid == mat_id] = materials.AIR

    def params_dict(self) -> dict:
        return {"material": self.material}

    @classmethod
    def _from_params(cls, d: dict) -> Strip:
        return cls(material=d.get("material", "photoresist"))


# === CMP（化学機械研磨 / 平坦化）===========================================
@register
@dataclass
class CMP(Process):
    """上面を研磨して平坦化する。最も高い点から指定量だけ削り水平にする。

    stop_material を指定すると、その材料（研磨ストップ層）の最高点より下は
    削らない。STI の窒化膜ストップのような選択研磨を再現する。
    dishing_um と soft_material を指定すると、平坦化後に軟らかい材料
    （Cu など）を追加で凹ませるディッシングを再現する（ダマシン研磨）。
    """

    type = "CMP"
    label = "CMP平坦化"

    remove_um: float = 0.5
    stop_material: str = ""
    soft_material: str = ""
    dishing_um: float = 0.0
    # パターン密度依存エロージョン: 軟材料（Cu 等）が密集する領域ほど
    # 余分に削れる。erosion_um は密度=1 のときの追加除去量、
    # density_radius_um は密度を平均する近傍半径。
    erosion_um: float = 0.0
    density_radius_um: float = 1.0

    def summary(self) -> str:
        stop = f"  停止層={self.stop_material}" if self.stop_material else ""
        dish = ""
        if self.soft_material and self.dishing_um > 0:
            dish = f"  ディッシング{self.dishing_um:.2f}µm({self.soft_material})"
        ero = ""
        if self.soft_material and self.erosion_um > 0:
            ero = f"  エロージョン{self.erosion_um:.2f}µm"
        return f"CMP  上面から{self.remove_um:.2f}µm研磨し平坦化{stop}{dish}{ero}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.remove_um, "研磨量")
        _require_non_negative(self.dishing_um, "ディッシング量")
        _require_non_negative(self.erosion_um, "エロージョン量")
        grid = wafer.grid
        nz = grid.shape[0]
        z_top = wafer.top_surface_z()
        if int(z_top.max()) < 0:
            return
        max_top = int(z_top.max())
        cut = max_top - wafer.um_to_vox(self.remove_um)
        # 基板は研磨で消えない（CMP は表面層を平坦化する工程）。
        # 研磨量が大きすぎても基板上面より下は削らないよう下限を設ける。
        substrate_top = max(0, wafer.um_to_vox(wafer.config.substrate_um) - 1)
        cut = max(substrate_top, min(nz - 1, cut))
        # 研磨ストップ層が指定されていれば、その最高点より下は削らない。
        if self.stop_material:
            stop_id = materials.get(self.stop_material).id
            zs = np.nonzero(np.any(grid == stop_id, axis=(1, 2)))[0]
            if zs.size:
                cut = max(cut, int(zs.max()))
                cut = min(nz - 1, cut)
        # cut より上の全ボクセルを空気にして上面を平坦化
        if cut + 1 < nz:
            grid[cut + 1:, :, :] = materials.AIR

        # ディッシング: 軟らかい材料（Cu など）は研磨で余計に凹む。
        # 平坦化後の上面から dishing 分だけ、対象材料が露出する列を追加除去。
        if self.soft_material and self.dishing_um > 0:
            soft_id = materials.get(self.soft_material).id
            dish = wafer.um_to_vox(self.dishing_um)
            for _ in range(dish):
                z_top2 = wafer.top_surface_z()
                ys, xs = np.nonzero(z_top2 >= 0)
                if ys.size == 0:
                    break
                zt = z_top2[ys, xs]
                is_soft = grid[zt, ys, xs] == soft_id
                grid[zt[is_soft], ys[is_soft], xs[is_soft]] = materials.AIR

        # パターン密度依存エロージョン: 軟材料が密集する領域ほど余計に削れる。
        # 近傍の軟材料存在率（局所密度）に比例して追加除去する。
        if self.soft_material and self.erosion_um > 0:
            soft_id = materials.get(self.soft_material).id
            soft2d = np.any(grid == soft_id, axis=0).astype(np.float32)
            if soft2d.any():
                rad = max(1, wafer.um_to_vox(self.density_radius_um))
                density = ndimage.uniform_filter(
                    soft2d, size=2 * rad + 1, mode="nearest"
                )
                ero = wafer.um_to_vox(self.erosion_um)
                depth_map = np.round(ero * density).astype(int)  # (ny, nx)
                max_d = int(depth_map.max())
                for _ in range(max_d):
                    z_top2 = wafer.top_surface_z()
                    ys, xs = np.nonzero((z_top2 >= 0) & (depth_map > 0))
                    if ys.size == 0:
                        break
                    zt = z_top2[ys, xs]
                    is_soft = grid[zt, ys, xs] == soft_id
                    sel_y, sel_x, sel_z = ys[is_soft], xs[is_soft], zt[is_soft]
                    grid[sel_z, sel_y, sel_x] = materials.AIR
                    depth_map[sel_y, sel_x] -= 1

    def params_dict(self) -> dict:
        return {
            "remove_um": self.remove_um,
            "stop_material": self.stop_material,
            "soft_material": self.soft_material,
            "dishing_um": self.dishing_um,
            "erosion_um": self.erosion_um,
            "density_radius_um": self.density_radius_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> CMP:
        return cls(
            remove_um=float(d.get("remove_um", 0.5)),
            stop_material=d.get("stop_material", ""),
            soft_material=d.get("soft_material", ""),
            dishing_um=float(d.get("dishing_um", 0.0)),
            erosion_um=float(d.get("erosion_um", 0.0)),
            density_radius_um=float(d.get("density_radius_um", 1.0)),
        )


# === OXIDE（熱酸化）========================================================
@register
@dataclass
class Oxidation(Process):
    """露出シリコンを熱酸化して酸化膜に変える（Si を消費しつつ上方へ成長）。"""

    type = "OXIDE"
    label = "熱酸化"

    thickness_um: float = 0.3
    # 生成 SiO2 厚に対し消費される Si の割合（Deal-Grove の体積膨張比に由来、約 0.45）。
    # 熱酸化では SiO2 の分子体積が元の Si の約 2.27 倍に膨張するため、厚さ tox の
    # SiO2 を作るのに約 0.44〜0.46×tox の Si が消費される（1/2.27 ≈ 0.44）。
    consume_fraction: float = 0.45
    # LOCOS バーズビーク: マスク（窒化膜）端の下へ酸化膜が横方向に侵入し、
    # Si 表面でテーパ状に食い込む。beak_fraction は横侵入距離を酸化膜厚に対する
    # 比で与える（0=無効、典型 0.5〜1.0）。
    beak_fraction: float = 0.0

    def summary(self) -> str:
        bk = "" if self.beak_fraction <= 0 else "  +バーズビーク"
        return f"OXIDE  酸化膜{self.thickness_um:.2f}µm成長{bk}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "酸化膜厚")
        _require_range(self.consume_fraction, 0.0, 0.95, "消費比")
        _require_non_negative(self.beak_fraction, "バーズビーク比")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        ox_id = materials.BY_NAME["oxide"].id
        # ドープされたシリコンも熱酸化される（物理的に正しい）。
        si_like = [si_id, materials.BY_NAME["doped_n"].id, materials.BY_NAME["doped_p"].id]
        total = wafer.um_to_vox(self.thickness_um)
        # 熱酸化の体積則: 生成 SiO2 厚の consume_fraction 分の Si が消費され、
        # 残りが元の Si 表面より上方へ成長する（Deal-Grove の体積膨張比に由来）。
        frac = float(np.clip(self.consume_fraction, 0.0, 0.95))
        consume = max(0, int(round(total * frac)))
        grow = max(1, total - consume)

        # 露出シリコン（ドープ含む）列のみ酸化
        _, top_id = _top_material(wafer)
        eligible = np.isin(top_id, si_like)  # (ny, nx)

        # 1) シリコンを上から consume 分だけ酸化膜に変換
        work_top = wafer.top_surface_z().copy()
        for _ in range(consume):
            ys, xs = np.nonzero(eligible & (work_top >= 0))
            if ys.size == 0:
                break
            zt = work_top[ys, xs]
            is_si = np.isin(grid[zt, ys, xs], si_like)
            grid[zt[is_si], ys[is_si], xs[is_si]] = ox_id
            work_top[ys[is_si], xs[is_si]] -= 1

        # 2) 上方へ grow 分だけ酸化膜を成長（空気を充填）
        z_top = wafer.top_surface_z()
        z_idx = np.arange(nz)[:, None, None]
        lo = z_top[None, :, :]
        deposit = (
            (z_idx > lo)
            & (z_idx <= lo + grow)
            & (grid == materials.AIR)
            & eligible[None, :, :]
        )
        grid[deposit] = ox_id

        # 3) バーズビーク: マスク端の下へ Si 表面酸化を横方向にテーパ侵入させる。
        if self.beak_fraction > 0 and consume > 0:
            beak_len = max(1, int(round(self.beak_fraction * total)))
            # 非露出（マスク下）列について露出領域からの距離を測る。
            dist = ndimage.distance_transform_edt(~eligible)
            masked = (~eligible) & (dist > 0) & (dist <= beak_len)
            if masked.any():
                # 各マスク列で最上の Si を見つけ、テーパ深さぶん酸化膜へ変換。
                taper = 1.0 - (dist - 1.0) / float(beak_len)  # 端で1→先端で~0
                depth_cap = np.maximum(
                    0, np.round(consume * np.clip(taper, 0.0, 1.0)).astype(int)
                )
                ys, xs = np.nonzero(masked)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    d = int(depth_cap[y, x])
                    if d <= 0:
                        continue
                    col = grid[:, y, x]
                    si_z = np.nonzero(np.isin(col, si_like))[0]
                    if si_z.size == 0:
                        continue
                    top_si = int(si_z.max())
                    z0 = max(0, top_si - d + 1)
                    seg = col[z0 : top_si + 1]
                    seg[np.isin(seg, si_like)] = ox_id

    def params_dict(self) -> dict:
        return {
            "thickness_um": self.thickness_um,
            "consume_fraction": self.consume_fraction,
            "beak_fraction": self.beak_fraction,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Oxidation:
        return cls(
            thickness_um=float(d.get("thickness_um", 0.3)),
            consume_fraction=float(d.get("consume_fraction", 0.45)),
            beak_fraction=float(d.get("beak_fraction", 0.0)),
        )


# === SALICIDE（自己整合シリサイド形成）=====================================
@register
@dataclass
class Silicidation(Process):
    """自己整合シリサイド（SALICIDE）形成。

    金属（Ni/Co/Ti 等）を全面成膜してアニールすると、露出シリコン／ポリ
    シリコンと接した部分のみが反応してシリサイド（低抵抗）になり、酸化膜・
    窒化膜上の未反応金属は選択エッチで除去される。本モデルでは露出 Si／
    ポリの上面から thickness_um 分をシリサイドへ変換する（自己整合）。
    """

    type = "SALICIDE"
    label = "シリサイド形成"

    thickness_um: float = 0.05
    react_poly: bool = True  # ゲートポリも反応させるか（ゲートシリサイド）

    def summary(self) -> str:
        pl = "" if self.react_poly else "  (ポリ非反応)"
        return f"SALICIDE  シリサイド{self.thickness_um:.3f}µm形成{pl}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "シリサイド厚")
        grid = wafer.grid
        sil_id = materials.BY_NAME["silicide"].id
        si_like = [
            materials.BY_NAME["silicon"].id,
            materials.BY_NAME["doped_n"].id,
            materials.BY_NAME["doped_p"].id,
        ]
        if self.react_poly:
            si_like.append(materials.BY_NAME["poly"].id)
        t = wafer.um_to_vox(self.thickness_um)
        # 露出した Si／ポリ列のみ反応（自己整合）。
        _, top_id = _top_material(wafer)
        eligible = np.isin(top_id, si_like)  # (ny, nx)
        work_top = wafer.top_surface_z().copy()
        for _ in range(t):
            ys, xs = np.nonzero(eligible & (work_top >= 0))
            if ys.size == 0:
                break
            zt = work_top[ys, xs]
            react = np.isin(grid[zt, ys, xs], si_like)
            grid[zt[react], ys[react], xs[react]] = sil_id
            work_top[ys[react], xs[react]] -= 1

    def params_dict(self) -> dict:
        return {
            "thickness_um": self.thickness_um,
            "react_poly": self.react_poly,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Silicidation:
        return cls(
            thickness_um=float(d.get("thickness_um", 0.05)),
            react_poly=bool(d.get("react_poly", True)),
        )
# === IMPLANT（イオン注入）==================================================
@register
@dataclass
class Implant(Process):
    """イオン注入。投影飛程 Rp を中心としたガウス的な深さ分布で埋込ドープする。

    range_um (Rp): 表面からの投影飛程。
    straggle_um (ΔRp): 飛程のばらつき（縦方向標準偏差）。
    lateral_straggle_um: 横方向のばらつき（マスク端での横散乱）。
    threshold: ピーク濃度に対する変換しきい値（0–1）。既定 0.3247 は
        縦方向ガウスの ±1.5σ 等高線に相当する。
    ガウス濃度プロファイル C(z) = exp(-(depth-Rp)^2 / 2σ^2) を計算し、
    マスク端では横方向ガウスでにじませた被覆率を掛ける。C >= threshold の
    シリコン（既ドープ含む）ボクセルをドーパントへ変換する。
    レジストで覆われた列はイオンが止められ、下地は保護される。
    """

    type = "IMPLANT"
    label = "イオン注入"

    dopant: str = "doped_n"
    range_um: float = 0.4
    straggle_um: float = 0.1
    lateral_straggle_um: float = 0.0
    threshold: float = 0.3247  # exp(-1.5^2 / 2): ±1.5σ 等高線
    # チャネリングテール: 結晶軸（<100>等）に沿って深く潜るイオンが作る
    # ガウスより深い指数裾。channeling_fraction はガウスピークに対する裾の
    # 相対振幅（0〜1）、tail_decay_um は裾の減衰長（0以下なら Rp×0.5 を採用）。
    channeling_fraction: float = 0.0
    tail_decay_um: float = 0.0
    # 注入チルト角（度, 垂直から +x 方向への傾き）。0=垂直注入。
    # 背の高いマスク/ゲートの +x 側に影（シャドーイング）を作り、注入領域を
    # +x へずらす（ハロー/ポケット注入や LDD の非対称分布を模擬）。
    tilt_deg: float = 0.0

    def summary(self) -> str:
        m = materials.get(self.dopant)
        ch = "" if self.channeling_fraction <= 0 else "  +チャネリング裾"
        tl = "" if self.tilt_deg <= 0 else f"  チルト{self.tilt_deg:.0f}°"
        return (
            f"IMPLANT  {m.label}"
            f"  Rp{self.range_um:.2f}±{self.straggle_um:.2f}µm{ch}{tl}"
        )

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.range_um, "投影飛程")
        _require_non_negative(self.straggle_um, "ストラグル")
        _require_non_negative(self.lateral_straggle_um, "横ストラグル")
        _require_range(self.threshold, 0.0, 1.0, "しきい値")
        _require_range(self.channeling_fraction, 0.0, 1.0, "チャネリング比")
        _require_non_negative(self.tail_decay_um, "チャネリング減衰長")
        _require_range(self.tilt_deg, 0.0, 60.0, "チルト角")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        dn_id = materials.BY_NAME["doped_n"].id
        dp_id = materials.BY_NAME["doped_p"].id
        dop_id = materials.get(self.dopant).id
        rp = wafer.um_to_vox(self.range_um)
        sigma = max(1.0, float(wafer.um_to_vox(max(self.straggle_um, 1e-6))))
        thr = float(np.clip(self.threshold, 1e-6, 1.0))

        si_like = [si_id, dn_id, dp_id]
        sil_mask = np.isin(grid, si_like)
        any_si = sil_mask.any(axis=0)
        # シリコン系の上面 z（レジスト等が上に乗っていてもその下のシリコン表面）
        sil_surface = (nz - 1) - np.argmax(sil_mask[::-1, :, :], axis=0)
        sil_surface = np.where(any_si, sil_surface, -1)

        z_top, top_id = _top_material(wafer)
        resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
        # 列ごとの被覆率（1=露出してイオンが入射, 0=レジスト等で遮蔽）
        cover = ((z_top >= 0) & ~np.isin(top_id, resist_ids)).astype(float)

        # チルト注入のシャドーイング: 背の高い表面形状（マスク/ゲート）が
        # 傾いたビームを遮り、+x 側の隣接列に影を落とす。
        tan_t = float(np.tan(np.deg2rad(np.clip(self.tilt_deg, 0.0, 60.0))))
        if tan_t > 0:
            zt = np.where(z_top >= 0, z_top, 0).astype(float)
            shadowed = np.zeros((ny, nx), dtype=bool)
            span = float(zt.max() - zt.min())
            maxk = int(min(nx - 1, np.ceil(span * tan_t)))
            for k in range(1, maxk + 1):
                shifted = np.full((ny, nx), -1.0)
                shifted[:, k:] = z_top[:, :-k]  # -x 側 k 列の上面高さ
                advantage = shifted - z_top
                # ビームは -x へ k 進むと k/tan_t だけ高くなる。その高さを
                # 超える形状があれば遮蔽される。
                shadowed |= (shifted >= 0) & (advantage >= k / tan_t)
            cover[shadowed] = 0.0

        # 縦方向ガウス濃度（シリコン表面基準）。露出列のみ線源を持つ。
        z_idx = np.arange(nz)[:, None, None]
        depth_si = sil_surface[None, :, :] - z_idx  # シリコン表面からの深さ vox
        vert = np.exp(-((depth_si - rp) ** 2) / (2.0 * sigma * sigma))
        vert[depth_si < 0] = 0.0
        vert[np.broadcast_to(sil_surface[None, :, :] < 0, vert.shape)] = 0.0

        # チャネリングテール: Rp より深い領域に指数裾を加える（結晶軸チャネリング）。
        if self.channeling_fraction > 0:
            decay_um = (
                self.tail_decay_um if self.tail_decay_um > 0 else self.range_um * 0.5
            )
            decay = max(1.0, float(wafer.um_to_vox(decay_um)))
            tail = np.exp(-(depth_si - rp) / decay)
            tail[depth_si < rp] = 0.0  # ピークより浅い側には裾を付けない
            tail[np.broadcast_to(sil_surface[None, :, :] < 0, tail.shape)] = 0.0
            vert = vert + self.channeling_fraction * tail

        conc = vert * cover[None, :, :]  # 規格化ガウス濃度（ピーク=1）

        # チルトによる横方向オフセット: ビームが深く進むほど +x へずれるため、
        # 注入領域を Rp 相当だけ横シフトする（マスク端下への回り込み）。
        if tan_t > 0:
            shift = int(round(rp * tan_t))
            if shift > 0:
                conc = np.roll(conc, shift, axis=2)
                conc[:, :, :shift] = 0.0

        # 横方向ストラグル: 面内ガウスで濃度をにじませ、マスク端の下へ回り込む
        sig_lat = float(wafer.um_to_vox(max(self.lateral_straggle_um, 0.0)))
        if sig_lat > 0:
            conc = ndimage.gaussian_filter(
                conc, sigma=(0.0, sig_lat, sig_lat), mode="nearest"
            )

        # シリコン（および既ドープ）を濃度しきい値で変換
        convertible = np.isin(grid, si_like)
        implant = (conc >= thr) & convertible
        grid[implant] = dop_id

    def params_dict(self) -> dict:
        return {
            "dopant": self.dopant,
            "range_um": self.range_um,
            "straggle_um": self.straggle_um,
            "lateral_straggle_um": self.lateral_straggle_um,
            "threshold": self.threshold,
            "channeling_fraction": self.channeling_fraction,
            "tail_decay_um": self.tail_decay_um,
            "tilt_deg": self.tilt_deg,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Implant:
        return cls(
            dopant=d.get("dopant", "doped_n"),
            range_um=float(d.get("range_um", 0.4)),
            straggle_um=float(d.get("straggle_um", 0.1)),
            lateral_straggle_um=float(d.get("lateral_straggle_um", 0.0)),
            threshold=float(d.get("threshold", 0.3247)),
            channeling_fraction=float(d.get("channeling_fraction", 0.0)),
            tail_decay_um=float(d.get("tail_decay_um", 0.0)),
            tilt_deg=float(d.get("tilt_deg", 0.0)),
        )


# === ANNEAL（アニール / ドライブイン）======================================
@register
@dataclass
class Anneal(Process):
    """熱処理によりドーパントをドライブイン（等方的に再分布）させる。

    既存の拡散層を周囲のシリコンへ depth_um 分だけ広げる。
    """

    type = "ANNEAL"
    label = "アニール"

    depth_um: float = 0.3

    def summary(self) -> str:
        return f"ANNEAL  ドライブイン{self.depth_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "ドライブイン量")
        grid = wafer.grid
        si_id = materials.BY_NAME["silicon"].id
        lat = wafer.um_to_vox(self.depth_um)
        # n / p それぞれを独立に膨張させ、隣接シリコンを同型に変換する
        for name in ("doped_n", "doped_p"):
            dop_id = materials.get(name).id
            region = grid == dop_id
            if not region.any():
                continue
            grown = _isotropic_dilate(region, lat)
            spread = grown & (grid == si_id)
            grid[spread] = dop_id

    def params_dict(self) -> dict:
        return {"depth_um": self.depth_um}

    @classmethod
    def _from_params(cls, d: dict) -> Anneal:
        return cls(depth_um=float(d.get("depth_um", 0.3)))


# === RTP（急速熱処理 / スパイクアニール）====================================
@register
@dataclass
class RapidThermalAnneal(Process):
    """急速熱処理（RTP/スパイクアニール）。浅く、横方向拡散を抑えてドライブイン。

    炉アニール（等方的に丸く広がる）と異なり、短時間高温で活性化のみを狙うため
    横方向の広がりが小さい。lateral_factor（0〜1）で横/縦の拡散比を制御し、
    0 に近いほど純垂直、1 で等方（炉アニール相当）になる。
    """

    type = "RTP"
    label = "急速熱処理"

    depth_um: float = 0.15
    lateral_factor: float = 0.3  # 横拡散 / 縦拡散 の比（既定 0.3、炉より小さい）

    def summary(self) -> str:
        return (
            f"RTP  ドライブイン{self.depth_um:.2f}µm  横比{self.lateral_factor:.2f}"
        )

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "ドライブイン量")
        _require_range(self.lateral_factor, 0.0, 1.0, "横拡散比")
        grid = wafer.grid
        si_id = materials.BY_NAME["silicon"].id
        lat = wafer.um_to_vox(self.depth_um)
        if lat <= 0:
            return
        # 異方拡散: 縦は lat、横は lat×lateral_factor の到達距離。
        # distance_transform_edt の sampling で横方向の距離コストを増やし、
        # 横の広がりを抑える。lateral_factor=0 は実質純垂直。
        lf = self.lateral_factor
        big = 1e6  # 横拡散を実質ゼロにするための大コスト
        lateral_cost = (1.0 / lf) if lf > 0 else big
        sampling = (1.0, lateral_cost, lateral_cost)
        for name in ("doped_n", "doped_p"):
            dop_id = materials.get(name).id
            region = grid == dop_id
            if not region.any():
                continue
            dist = ndimage.distance_transform_edt(~region, sampling=sampling)
            grown = dist <= lat
            spread = grown & (grid == si_id)
            grid[spread] = dop_id

    def params_dict(self) -> dict:
        return {"depth_um": self.depth_um, "lateral_factor": self.lateral_factor}

    @classmethod
    def _from_params(cls, d: dict) -> RapidThermalAnneal:
        return cls(
            depth_um=float(d.get("depth_um", 0.15)),
            lateral_factor=float(d.get("lateral_factor", 0.3)),
        )


# === EPI（選択エピタキシャル成長）==========================================
@register
@dataclass
class Epitaxy(Process):
    """露出シリコン上のみに単結晶を選択的に成長させる（酸化膜上には付かない）。"""

    type = "EPI"
    label = "エピ成長"

    material: str = "epi_si"
    thickness_um: float = 0.5
    facet_angle_deg: float = 0.0

    def summary(self) -> str:
        m = materials.get(self.material)
        facet = f"  ファセット{self.facet_angle_deg:.0f}°" if self.facet_angle_deg > 0 else ""
        return f"EPI  {m.label}  厚{self.thickness_um:.2f}µm{facet}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "エピ厚")
        _require_non_negative(self.facet_angle_deg, "ファセット角")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        mat_id = materials.get(self.material).id
        t = wafer.um_to_vox(self.thickness_um)
        # 露出が結晶シリコン系（silicon / epi / 拡散層）の列のみ成長
        seed_ids = [
            materials.BY_NAME["silicon"].id,
            materials.BY_NAME["epi_si"].id,
            materials.BY_NAME["doped_n"].id,
            materials.BY_NAME["doped_p"].id,
        ]
        z_top, top_id = _top_material(wafer)
        eligible = (z_top >= 0) & np.isin(top_id, seed_ids)
        if not eligible.any():
            return

        if self.facet_angle_deg <= 0:
            # 等方的（コンフォーマル）成長
            z_idx = np.arange(nz)[:, None, None]
            lo = z_top[None, :, :]
            deposit = (
                (z_idx > lo)
                & (z_idx <= lo + t)
                & (grid == materials.AIR)
                & eligible[None, :, :]
            )
            grid[deposit] = mat_id
            return

        # ファセット成長: 高さとともに {111} 面に沿って footprint が内側へ収束し
        # 台形/三角形のキャップを形成する（選択エピの自己整合ファセット）。
        angle = float(np.clip(self.facet_angle_deg, 5.0, 89.9))
        tan_a = np.tan(np.deg2rad(angle))
        for d in range(1, t + 1):
            inset = int(round((d - 1) / tan_a))  # 高さに比例して内側へ後退
            if inset > 0:
                layer = ndimage.binary_erosion(eligible, iterations=inset)
            else:
                layer = eligible
            if not layer.any():
                break
            zlayer = z_top + d
            sel = layer & (zlayer < nz)
            ys, xs = np.nonzero(sel)
            zz = zlayer[ys, xs]
            is_air = grid[zz, ys, xs] == materials.AIR
            grid[zz[is_air], ys[is_air], xs[is_air]] = mat_id

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "facet_angle_deg": self.facet_angle_deg,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Epitaxy:
        return cls(
            material=d.get("material", "epi_si"),
            thickness_um=float(d.get("thickness_um", 0.5)),
            facet_angle_deg=float(d.get("facet_angle_deg", 0.0)),
        )


# === KOH（結晶異方性ウェットエッチ）========================================
@register
@dataclass
class AnisoWetEtch(Process):
    """結晶面に沿った異方性ウェットエッチ。斜め側壁の V 溝/台形を形成する。

    (100) Si の KOH エッチでは側壁が約 54.7° になる。深さとともに開口が
    内側へ狭まり、十分深いと V 溝の底で閉じる。
    """

    type = "KOH"
    label = "異方性ウェット"

    target: str = "silicon"
    depth_um: float = 1.0
    sidewall_angle_deg: float = 54.7

    def summary(self) -> str:
        return f"KOH  {materials.get(self.target).label}  深さ{self.depth_um:.2f}µm  {self.sidewall_angle_deg:.0f}°"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "エッチ深さ")
        _require_range(self.sidewall_angle_deg, 5.0, 89.9, "側壁角")
        grid = wafer.grid
        target_id = materials.get(self.target).id
        depth = wafer.um_to_vox(self.depth_um)
        angle = float(np.clip(self.sidewall_angle_deg, 5.0, 89.9))
        tan_a = np.tan(np.deg2rad(angle))

        z_top, top_id = _top_material(wafer)
        opening = (z_top >= 0) & (top_id == target_id)  # 露出ターゲット開口
        if not opening.any():
            return

        for d in range(depth):
            inset = int(round(d / tan_a))  # 深さに比例して内側へ後退
            if inset > 0:
                layer = ndimage.binary_erosion(opening, iterations=inset)
            else:
                layer = opening
            if not layer.any():
                break
            zlayer = z_top - d
            sel = layer & (zlayer >= 0)
            ys, xs = np.nonzero(sel)
            zz = zlayer[ys, xs]
            is_t = grid[zz, ys, xs] == target_id
            grid[zz[is_t], ys[is_t], xs[is_t]] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "target": self.target,
            "depth_um": self.depth_um,
            "sidewall_angle_deg": self.sidewall_angle_deg,
        }

    @classmethod
    def _from_params(cls, d: dict) -> AnisoWetEtch:
        return cls(
            target=d.get("target", "silicon"),
            depth_um=float(d.get("depth_um", 1.0)),
            sidewall_angle_deg=float(d.get("sidewall_angle_deg", 54.7)),
        )


# === FILL（ダマシン / ボトムアップ埋込）=====================================
@register
@dataclass
class Fill(Process):
    """開口・トレンチ・ビアを金属でボトムアップ充填する（電解めっき/ECD 相当）。

    各列で現表面から、最も高い既存表面 + overfill_um の高さまで空気を充填する。
    充填後に CMP を行うとダマシン配線になる。
    """

    type = "FILL"
    label = "埋込(ダマシン)"

    material: str = "metal_cu"
    overfill_um: float = 0.1
    # アスペクト比依存のキーホール空隙。void_ar>0 で、深さ/幅 が void_ar を
    # 超える狭いトレンチはコンフォーマル成長が上部で先に塞がり（ピンチオフ）、
    # 中央に縦長の空隙（シーム/ボイド）を残す。0 で完全充填（従来動作）。
    void_ar: float = 0.0

    def summary(self) -> str:
        m = materials.get(self.material)
        vd = "" if self.void_ar <= 0 else f"  ボイドAR>{self.void_ar:.1f}"
        return f"FILL  {m.label}  +{self.overfill_um:.2f}µm{vd}"

    def apply(self, wafer: Wafer) -> None:
        if self.overfill_um < 0:
            raise ValueError("オーバーフィル量は 0 以上である必要があります。")
        _require_non_negative(self.void_ar, "ボイドARしきい値")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        mat_id = materials.get(self.material).id
        z_top = wafer.top_surface_z()
        if int(z_top.max()) < 0:
            return
        fill_to = min(nz - 1, int(z_top.max()) + wafer.um_to_vox(self.overfill_um))
        has_solid = z_top >= 0
        z_idx = np.arange(nz)[:, None, None]
        deposit = (
            (z_idx > z_top[None, :, :])
            & (z_idx <= fill_to)
            & (grid == materials.AIR)
            & has_solid[None, :, :]
        )
        grid[deposit] = mat_id

        # キーホール空隙: 狭いトレンチで上部ピンチオフ → 中央に縦空隙を残す。
        if self.void_ar > 0:
            field_level = int(z_top.max())
            # 周囲フィールドより低い（=リセス/トレンチ）列のみ対象
            recess = has_solid & (z_top < field_level)
            if recess.any():
                hw = ndimage.distance_transform_edt(recess)  # 列ごとのハーフ幅
                labels, n = ndimage.label(recess)
                if n > 0:
                    feat = ndimage.maximum(hw, labels, index=range(1, n + 1))
                    lut = np.concatenate(([0.0], np.asarray(feat, dtype=float)))
                    hw_feat = lut[labels]  # トレンチごとの代表ハーフ幅
                    depth = np.where(recess, field_level - z_top, 0.0).astype(float)
                    width = np.maximum(2.0 * hw_feat, 1.0)
                    # AR が閾値を超える狭いリセスの中心線（ハーフ幅が最大の列）
                    narrow = recess & (depth > self.void_ar * width)
                    centerline = hw >= (hw_feat - 0.5)
                    cand = narrow & centerline
                    lo = z_top.astype(float) + hw_feat  # ボトムアップ充填上端
                    hi = float(field_level) - hw_feat  # トップピンチオフ下端
                    void = (
                        cand[None, :, :]
                        & (z_idx > lo[None, :, :])
                        & (z_idx <= hi[None, :, :])
                        & (grid == mat_id)
                    )
                    grid[void] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "overfill_um": self.overfill_um,
            "void_ar": self.void_ar,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Fill:
        return cls(
            material=d.get("material", "metal_cu"),
            overfill_um=float(d.get("overfill_um", 0.1)),
            void_ar=float(d.get("void_ar", 0.0)),
        )


# === SPINON（スピンオン平坦化膜）===========================================
@register
@dataclass
class SpinCoat(Process):
    """スピンオン塗布による自己平坦化膜（SOG/SOD）。

    液状材料を塗布・硬化させるため、窪みを厚く・凸部を薄く埋めて上面を
    平坦化する。最も高い表面 + cap_um の高さまで一様に空気を充填し、
    結果として平坦な上面を得る。CVD（コンフォーマル）とは対照的。
    """

    type = "SPINON"
    label = "スピンオン平坦化"

    material: str = "low_k"
    cap_um: float = 0.3
    # 平坦化度 DOP（0〜1）。1=完全平坦（最高点で一律）、0=コンフォーマル
    # （表面形状に追従）。実際の SOG/SOD は中間で、広い窪みほど平坦化が
    # 不完全になる現象の一次近似。
    planarization: float = 1.0

    def summary(self) -> str:
        m = materials.get(self.material)
        dp = "" if self.planarization >= 0.999 else f"  DOP{self.planarization:.2f}"
        return f"SPINON  {m.label}  上面+{self.cap_um:.2f}µm平坦化{dp}"

    def apply(self, wafer: Wafer) -> None:
        _require_non_negative(self.cap_um, "キャップ厚")
        _require_range(self.planarization, 0.0, 1.0, "平坦化度")
        grid = wafer.grid
        nz = grid.shape[0]
        mat_id = materials.get(self.material).id
        z_top = wafer.top_surface_z()
        if int(z_top.max()) < 0:
            return
        cap = wafer.um_to_vox(self.cap_um)
        dop = float(np.clip(self.planarization, 0.0, 1.0))
        # 完全平坦レベル（最高点 + cap）とコンフォーマル上面（各列 + cap）を
        # DOP で線形補間して、列ごとの充填上面を決める。
        flat_level = min(nz - 1, int(z_top.max()) + cap)
        conformal_top = z_top + cap  # 各列の追従上面
        fill_top = np.round(dop * flat_level + (1.0 - dop) * conformal_top)
        fill_top = np.clip(fill_top, 0, nz - 1).astype(int)
        z_idx = np.arange(nz)[:, None, None]
        deposit = (
            (z_idx > z_top[None, :, :])
            & (z_idx <= fill_top[None, :, :])
            & (grid == materials.AIR)
        )
        grid[deposit] = mat_id

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "cap_um": self.cap_um,
            "planarization": self.planarization,
        }

    @classmethod
    def _from_params(cls, d: dict) -> SpinCoat:
        return cls(
            material=d.get("material", "low_k"),
            cap_um=float(d.get("cap_um", 0.3)),
            planarization=float(d.get("planarization", 1.0)),
        )


# === LIFTOFF（リフトオフ）==================================================
@register
@dataclass
class LiftOff(Process):
    """レジストとその上に乗った膜を一括除去する（リフトオフ法）。

    レジストが存在する列で、レジスト底面以上の全ボクセルを除去する。
    レジスト下に直接堆積した膜は残る。
    """

    type = "LIFTOFF"
    label = "リフトオフ"

    def summary(self) -> str:
        return "LIFTOFF  レジスト＋上層を除去"

    def apply(self, wafer: Wafer) -> None:
        grid = wafer.grid
        nz, ny, nx = grid.shape
        resist = wafer.resist_mask()  # (z, y, x)
        has_resist = resist.any(axis=0)  # (y, x)
        if not has_resist.any():
            return
        # 各列のレジスト最下面 z（底から最初に True になる位置）
        resist_bottom = np.argmax(resist, axis=0)  # (y, x)
        z_idx = np.arange(nz)[:, None, None]
        remove = (z_idx >= resist_bottom[None, :, :]) & has_resist[None, :, :]
        grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {}

    @classmethod
    def _from_params(cls, d: dict) -> LiftOff:
        return cls()


# === DRIE（深掘り反応性イオンエッチ / Bosch）===============================
@register
@dataclass
class DRIE(Process):
    """高アスペクト比の深掘り垂直エッチ。任意でスキャロップ（側壁の波形）を付与。

    レジストで覆われた列は保護される。scallop_um>0 で Bosch プロセス特有の
    周期的な側壁の凹凸を簡易再現する。
    """

    type = "DRIE"
    label = "深掘りエッチ"

    target: str = "silicon"
    depth_um: float = 2.0
    scallop_um: float = 0.0
    scallop_pitch_um: float = 0.5
    # RIE ラグ / ARDE（0〜1）。開口が狭いほどエッチが浅くなる現象を再現する。
    # 0=幅に依存せず全開口が同深さ。1=最狭開口の到達深さがほぼ 0 になる。
    lag: float = 0.0
    # 再付着 / 側壁パシベーション（Bosch プロセス）。エッチ生成物が
    # トレンチ側壁に再堆積してトレンチを狭める厚み（µm、0=無効）。
    redeposit_um: float = 0.0
    # マイクロトレンチング（µm、0=無効）。側壁で反射したイオンがトレンチ
    # 底の隅（フット）に集中し、開口周縁が局所的に深く掘れる現象を再現する。
    microtrench_um: float = 0.0

    def summary(self) -> str:
        sc = "" if self.scallop_um <= 0 else f"  scallop{self.scallop_um:.2f}µm"
        lg = "" if self.lag <= 0 else f"  RIEラグ{self.lag:.0%}"
        rd = "" if self.redeposit_um <= 0 else f"  再付着{self.redeposit_um:.2f}µm"
        mt = "" if self.microtrench_um <= 0 else f"  μトレンチ{self.microtrench_um:.2f}µm"
        return (
            f"DRIE  {materials.get(self.target).label}"
            f"  深さ{self.depth_um:.2f}µm{sc}{lg}{rd}{mt}"
        )

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "深さ")
        _require_range(self.lag, 0.0, 1.0, "RIEラグ")
        _require_non_negative(self.redeposit_um, "再付着厚")
        _require_non_negative(self.microtrench_um, "マイクロトレンチ深さ")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        target_id = materials.get(self.target).id
        depth = wafer.um_to_vox(self.depth_um)

        # 初期に露出しているターゲット列のみ（レジスト等が無い箇所）
        z_top0, top_id0 = _top_material(wafer)
        opening = (z_top0 >= 0) & (top_id0 == target_id)
        if not opening.any():
            return

        # RIE ラグ: 各開口（連結成分）の幅に応じて到達深さを制限する。
        # 開口内の最大ハーフ幅を特徴幅とし、最広開口で正規化。狭い開口ほど
        # depth_cap を小さくして浅く止める（ARDE / アスペクト比依存エッチ）。
        if self.lag > 0:
            halfw = ndimage.distance_transform_edt(opening)
            labels, n = ndimage.label(opening)
            depth_cap = np.full((ny, nx), float(depth))
            if n > 0:
                feat_max = ndimage.maximum(halfw, labels, index=range(1, n + 1))
                gmax = max(float(np.max(feat_max)), 1e-9)
                lut = np.concatenate(([0.0], np.asarray(feat_max, dtype=float)))
                w_norm = lut[labels] / gmax  # 開口ごとの正規化幅 (0〜1)
                cap = depth * (1.0 - self.lag * (1.0 - w_norm))
                depth_cap = np.where(opening, np.maximum(cap, 0.0), float(depth))
        else:
            depth_cap = np.full((ny, nx), float(depth))
        etched = np.zeros((ny, nx), dtype=float)

        amp = wafer.um_to_vox(self.scallop_um) if self.scallop_um > 0 else 0
        pitch = max(2, wafer.um_to_vox(self.scallop_pitch_um))
        struct = ndimage.generate_binary_structure(3, 1)

        for d in range(depth):
            z_top, top_id = _top_material(wafer)
            do = opening & (z_top >= 0) & (top_id == target_id) & (etched < depth_cap)
            if not do.any():
                break
            ys, xs = np.nonzero(do)
            zz = z_top[ys, xs]
            grid[zz, ys, xs] = materials.AIR
            etched[do] += 1.0
            # スキャロップ: 周期の中央付近でその層だけ側壁を横へ膨らませる。
            # 直近に掘った 1 層分のみを横方向に拡張するため、深さに比例した
            # 累積的なテーパは生じず、垂直側壁に局所的な凹凸が付く。
            if amp > 0 and (d % pitch) == (pitch // 2):
                layer = np.zeros(grid.shape, dtype=bool)
                layer[zz, ys, xs] = True
                ring = ndimage.binary_dilation(layer, structure=struct, iterations=amp)
                bulge = ring & (grid == target_id)
                grid[bulge] = materials.AIR

        # マイクロトレンチング: 側壁で反射したイオンが開口周縁（フット）に
        # 集中し、その列だけ局所的に深く掘れる。開口の外周列を microtrench
        # 分だけ余分にエッチする。
        mt = wafer.um_to_vox(self.microtrench_um) if self.microtrench_um > 0 else 0
        if mt > 0:
            perim = opening & ~ndimage.binary_erosion(opening, border_value=0)
            z_top, top_id = _top_material(wafer)
            cols = perim & (z_top >= 0) & (top_id == target_id)
            ys, xs = np.nonzero(cols)
            for y, x in zip(ys.tolist(), xs.tolist()):
                zt = int(z_top[y, x])
                zlo = max(1, zt - mt + 1)  # 基板最下層は保護
                for z in range(zt, zlo - 1, -1):
                    if grid[z, y, x] == target_id:
                        grid[z, y, x] = materials.AIR
                    else:
                        break

        # 再付着 / 側壁パシベーション: エッチ生成物がトレンチ側壁に
        # 再堆積してトレンチを狭める。底面より側壁を優先するため、
        # 水平（x,y）方向だけの距離で壁近傍の空気をターゲット材で埋める。
        rd = wafer.um_to_vox(self.redeposit_um) if self.redeposit_um > 0 else 0
        if rd > 0:
            surface_level = int(z_top0[opening].max())
            target_mask = grid == target_id
            # z 方向の距離を実質無限大にして水平距離のみを評価。
            lat_dist = ndimage.distance_transform_edt(
                ~target_mask, sampling=(1e6, 1.0, 1.0)
            )
            z_idx = np.arange(nz)[:, None, None]
            coat = (
                (grid == materials.AIR)
                & (lat_dist > 0)
                & (lat_dist <= rd)
                & (z_idx <= surface_level)
            )
            grid[coat] = target_id

    def params_dict(self) -> dict:
        return {
            "target": self.target,
            "depth_um": self.depth_um,
            "scallop_um": self.scallop_um,
            "scallop_pitch_um": self.scallop_pitch_um,
            "lag": self.lag,
            "redeposit_um": self.redeposit_um,
            "microtrench_um": self.microtrench_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> DRIE:
        return cls(
            target=d.get("target", "silicon"),
            depth_um=float(d.get("depth_um", 2.0)),
            scallop_um=float(d.get("scallop_um", 0.0)),
            scallop_pitch_um=float(d.get("scallop_pitch_um", 0.5)),
            lag=float(d.get("lag", 0.0)),
            redeposit_um=float(d.get("redeposit_um", 0.0)),
            microtrench_um=float(d.get("microtrench_um", 0.0)),
        )


# === SPUTTER（スパッタエッチ / イオンミリング）============================
@register
@dataclass
class SputterEtch(Process):
    """物理スパッタによる非選択の指向性エッチ（イオンミリング）。

    材料種を問わず上方から物理的に削る（レジストは保護膜として残す）。
    isotropic（0..1）で横方向成分を持たせ、側壁のアンダーカット/ファセットを
    簡易再現する。
    """

    type = "SPUTTER"
    label = "スパッタエッチ"

    depth_um: float = 0.3
    isotropic: float = 0.0
    # ファセッティング（0..1）。イオンミリングは入射角依存スパッタ率により
    # 鋭い凸角を優先的に削り、角を斜めのファセット（面取り）にする。
    faceting: float = 0.0

    def summary(self) -> str:
        iso = "" if self.isotropic <= 0 else f"  等方{self.isotropic:.0%}"
        fac = "" if self.faceting <= 0 else f"  面取り{self.faceting:.0%}"
        return f"SPUTTER  深さ{self.depth_um:.2f}µm{iso}{fac}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "エッチ量")
        _require_range(self.isotropic, 0.0, 1.0, "等方成分")
        _require_range(self.faceting, 0.0, 1.0, "面取り")
        grid = wafer.grid
        depth = wafer.um_to_vox(self.depth_um)
        resist_ids = [m.id for m in materials.all_materials() if m.is_resist]

        # 露出している非レジスト固体の列のみミリング対象
        z_top0, top_id0 = _top_material(wafer)
        eligible = (z_top0 >= 0) & ~np.isin(top_id0, resist_ids)

        for _ in range(depth):
            z_top, top_id = _top_material(wafer)
            do = eligible & (z_top >= 0) & ~np.isin(top_id, resist_ids)
            do &= top_id != materials.AIR
            # 基板最下層は物理的に残す（ウェハ全体を削り切らない）
            do &= z_top > 0
            if not do.any():
                break
            ys, xs = np.nonzero(do)
            grid[z_top[ys, xs], ys, xs] = materials.AIR

        # 横方向成分: 生成した空気から側壁へ等方的に削る（アンダーカット）
        lat = int(round(depth * self.isotropic))
        if lat > 0:
            air = grid == materials.AIR
            grown = ndimage.distance_transform_edt(~air) <= lat
            erode = grown & ~np.isin(grid, resist_ids) & (grid != materials.AIR)
            # 基板底面は残す（最下層は削らない）
            erode[0, :, :] = False
            grid[erode] = materials.AIR

        # ファセッティング: 露出した凸角を優先的に削り面取りする。各反復で
        # 「空気に接し、かつ固体隣接が少ない（=凸の角/稜）」ボクセルを除去
        # する。これを繰り返すと角に約45°のファセットが形成される。
        fac = int(round(depth * self.faceting)) if self.faceting > 0 else 0
        if fac > 0:
            k = np.zeros((3, 3, 3), dtype=np.int8)
            k[1, 1, 0] = k[1, 1, 2] = k[1, 0, 1] = k[1, 2, 1] = 1
            k[0, 1, 1] = k[2, 1, 1] = 1
            struct = ndimage.generate_binary_structure(3, 1)
            for _ in range(fac):
                solid = (grid != materials.AIR) & ~np.isin(grid, resist_ids)
                exposed = solid & ndimage.binary_dilation(
                    grid == materials.AIR, structure=struct
                )
                neigh = ndimage.convolve(
                    solid.astype(np.int8), k, mode="constant", cval=0
                )
                # 凸角/稜: 6 近傍中 4 個以下が固体（平坦面は 5）
                convex = exposed & (neigh <= 4)
                convex[0, :, :] = False
                convex[:, 0, :] = convex[:, -1, :] = False
                convex[:, :, 0] = convex[:, :, -1] = False
                if not convex.any():
                    break
                grid[convex] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "depth_um": self.depth_um,
            "isotropic": self.isotropic,
            "faceting": self.faceting,
        }

    @classmethod
    def _from_params(cls, d: dict) -> SputterEtch:
        return cls(
            depth_um=float(d.get("depth_um", 0.3)),
            isotropic=float(d.get("isotropic", 0.0)),
            faceting=float(d.get("faceting", 0.0)),
        )


# === CLEAN（プラズマクリーン / デスカム）===================================
@register
@dataclass
class PlasmaClean(Process):
    """露出表面を薄く等方的に除去する軽いクリーニング（デスカム/残渣除去）。

    対象材料を指定でき（既定はレジスト残渣）、表面から thickness_um 分だけ
    等方的に削る。下地は保護される。
    """

    type = "CLEAN"
    label = "プラズマクリーン"

    target: str = "photoresist"
    thickness_um: float = 0.05

    def summary(self) -> str:
        return f"CLEAN  {materials.get(self.target).label}  {self.thickness_um:.3f}µm除去"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "除去量")
        grid = wafer.grid
        target_id = materials.get(self.target).id
        t = wafer.um_to_vox(self.thickness_um)
        # 空気に接する対象材料の表面から t 以内を除去（等方デスカム）
        target = grid == target_id
        if not target.any():
            return
        air = grid == materials.AIR
        dist_from_air = ndimage.distance_transform_edt(~air)
        remove = target & (dist_from_air <= t)
        grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {"target": self.target, "thickness_um": self.thickness_um}

    @classmethod
    def _from_params(cls, d: dict) -> PlasmaClean:
        return cls(
            target=d.get("target", "photoresist"),
            thickness_um=float(d.get("thickness_um", 0.05)),
        )


# === REFLOW（熱リフロー / 表面平滑化）======================================
@register
@dataclass
class Reflow(Process):
    """対象材料を熱リフローし、表面張力で角を丸めて平滑化する。

    モルフォロジのクロージング＋オープニングで凹凸を丸める。radius_um が
    大きいほど強く平滑化される。リフロー金属やレジストのリフローに使う。
    """

    type = "REFLOW"
    label = "熱リフロー"

    target: str = "photoresist"
    radius_um: float = 0.2

    def summary(self) -> str:
        return f"REFLOW  {materials.get(self.target).label}  r{self.radius_um:.2f}µm平滑化"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.radius_um, "平滑化半径")
        grid = wafer.grid
        target_id = materials.get(self.target).id
        r = wafer.um_to_vox(self.radius_um)
        mask = grid == target_id
        if not mask.any():
            return
        # クロージング(凹を埋める)→オープニング(凸を削る)で角を丸める
        closed = ndimage.binary_dilation(mask, iterations=r)
        closed = ndimage.binary_erosion(closed, iterations=r, border_value=1)
        smoothed = ndimage.binary_erosion(closed, iterations=r, border_value=1)
        smoothed = ndimage.binary_dilation(smoothed, iterations=r)
        # 増えた分は空気のみを埋める（他材料は侵さない）
        gained = smoothed & (grid == materials.AIR)
        # 減った分（凸の出っ張り）は空気へ戻す
        lost = mask & ~smoothed
        grid[gained] = target_id
        grid[lost] = materials.AIR

    def params_dict(self) -> dict:
        return {"target": self.target, "radius_um": self.radius_um}

    @classmethod
    def _from_params(cls, d: dict) -> Reflow:
        return cls(
            target=d.get("target", "photoresist"),
            radius_um=float(d.get("radius_um", 0.2)),
        )


def available_types() -> list[tuple[str, str]]:
    """(type, label) のリストを表示順で返す。"""
    order = [
        "PHOTO", "CVD", "ALD", "PVD", "EPI",
        "DRY", "WET", "KOH", "DRIE", "SPUTTER",
        "DIFFUSION", "IMPLANT", "ANNEAL", "RTP", "OXIDE",
        "SALICIDE", "FILL", "SPINON", "CMP", "REFLOW", "CLEAN", "LIFTOFF", "STRIP",
    ]
    out = []
    for t in order:
        cls = _REGISTRY.get(t)
        if cls is not None:
            out.append((t, cls.label))
    return out
