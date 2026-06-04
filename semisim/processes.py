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


def deal_grove_thickness_um(
    time_min: float, temperature_c: float, ambient: str = "dry"
) -> float:
    """Deal-Grove モデルで熱酸化膜厚 (µm) を求める。

    x² + A·x = B·(t+τ) を解く（初期酸化 τ は無視, x0=0）。
    線形速度定数 B/A と放物線速度定数 B は Arrhenius 形（教科書値, <100> Si）。
    ambient="dry"（乾燥 O2）または "wet"（水蒸気）。time_min は分。

    A = B / (B/A) より x = (A/2)(√(1 + 4B·t/A²) − 1)。
    """
    _require_non_negative(time_min, "酸化時間")
    if time_min <= 0:
        return 0.0
    k = 8.617e-5  # ボルツマン定数 [eV/K]
    t_k = float(temperature_c) + 273.15
    if t_k <= 0:
        raise ValueError("温度は絶対零度より高い必要があります。")
    amb = str(ambient).lower()
    if amb == "wet":
        # 水蒸気酸化（µm²/hr, µm/hr）
        b = 3.86e2 * math.exp(-0.78 / (k * t_k))
        b_over_a = 1.63e8 * math.exp(-2.05 / (k * t_k))
    elif amb == "dry":
        # 乾燥酸素酸化
        b = 7.72e2 * math.exp(-1.23 / (k * t_k))
        b_over_a = 6.23e6 * math.exp(-2.0 / (k * t_k))
    else:
        raise ValueError(f"ambient は 'dry' または 'wet'（指定値: {ambient}）。")
    t_hr = time_min / 60.0
    a = b / b_over_a  # µm
    x = (a / 2.0) * (math.sqrt(1.0 + 4.0 * b * t_hr / (a * a)) - 1.0)
    return float(x)


# ドーパント拡散の Arrhenius 定数（<111>/intrinsic Si, D0[cm²/s], Ea[eV]）。
# 教科書値（Sze, Plummer 等）。ドライブインの拡散長 L=√(Dt) 計算に使う。
_DIFFUSIVITY = {
    "boron": (0.76, 3.46),
    "phosphorus": (3.85, 3.66),
    "arsenic": (0.066, 3.44),
    "antimony": (0.214, 3.65),
}
# 材料名 → 代表ドーパント種（doped_p=ホウ素, doped_n=リン）
_DOPANT_SPECIES = {"doped_p": "boron", "doped_n": "phosphorus"}


def diffusion_length_um(time_min: float, temperature_c: float, dopant: str) -> float:
    """ドーパントの熱拡散長 L=√(D·t) を µm で返す。

    D = D0·exp(-Ea/kT)（Arrhenius）。dopant は "boron"/"phosphorus"/"arsenic"
    /"antimony"、または材料名 "doped_p"/"doped_n"。ドライブイン（Anneal）の
    到達深さの物理的目安に使う。time_min は分。
    """
    _require_non_negative(time_min, "拡散時間")
    if time_min <= 0:
        return 0.0
    species = _DOPANT_SPECIES.get(str(dopant), str(dopant))
    if species not in _DIFFUSIVITY:
        known = ", ".join(sorted(set(_DIFFUSIVITY) | set(_DOPANT_SPECIES)))
        raise ValueError(f"未知のドーパント: {dopant!r}（利用可能: {known}）")
    d0, ea = _DIFFUSIVITY[species]
    k = 8.617e-5  # eV/K
    t_k = float(temperature_c) + 273.15
    if t_k <= 0:
        raise ValueError("温度は絶対零度より高い必要があります。")
    d_cm2_s = d0 * math.exp(-ea / (k * t_k))  # cm²/s
    t_s = time_min * 60.0
    l_cm = math.sqrt(d_cm2_s * t_s)
    return float(l_cm * 1.0e4)  # cm → µm


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


# === ALE（原子層エッチ）====================================================
@register
@dataclass
class AtomicLayerEtch(Process):
    """原子層エッチ（ALE）。サイクル数×1サイクル除去量で nm 精度の自己制限エッチ。

    ALD の対となる工程。対象材料のみを露出面から精密・自己制限的に除去する。
    `anisotropy` で除去の指向性を切替: 0=等方コンフォーマル（側壁も均一に後退）、
    1=純垂直（指向性 ALE、底のみ後退）。除去量は cycles×etch_per_cycle_nm で
    厳密に決まり（過エッチ無し）、対象以外の材料では完全停止（高選択比）。
    最先端ノードの高精度・高選択エッチ（FinFET/GAA のリセス制御等）を模擬する。
    """

    type = "ALE"
    label = "ALE(原子層エッチ)"

    targets: list[str] = field(default_factory=list)
    cycles: int = 30
    etch_per_cycle_nm: float = 1.0
    anisotropy: float = 0.0  # 0=等方コンフォーマル / 1=純垂直（指向性）

    @property
    def depth_um(self) -> float:
        return self.cycles * self.etch_per_cycle_nm / 1000.0

    def summary(self) -> str:
        tgt = "/".join(self.targets) if self.targets else "露出材料"
        mode = "等方" if self.anisotropy < 0.5 else "指向性"
        return (
            f"ALE  {tgt}  {self.cycles}cyc×{self.etch_per_cycle_nm:.1f}nm"
            f"={self.depth_um * 1000:.0f}nm  {mode}"
        )

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.cycles, "サイクル数")
        _require_positive(self.etch_per_cycle_nm, "1サイクル除去量")
        _require_range(self.anisotropy, 0.0, 1.0, "指向性")
        grid = wafer.grid
        target_ids = _resolve_targets(self.targets)
        r = wafer.um_to_vox(self.depth_um)
        # 露出面から 1 ボクセルずつ精密に前線を伝播（自己制限）。
        # anisotropy=0 で等方（横比 1）、=1 で純垂直（横比 0）。
        full6 = ndimage.generate_binary_structure(3, 1)
        vert = np.zeros((3, 3, 3), dtype=bool)
        vert[1, 1, 1] = True
        vert[0, 1, 1] = True
        vert[2, 1, 1] = True
        lr = float(np.clip(1.0 - self.anisotropy, 0.0, 1.0))
        lat_budget = 0.0
        for _ in range(r):
            air = grid == materials.AIR
            lat_budget += lr
            if lat_budget >= 1.0:
                front = ndimage.binary_dilation(air, structure=full6)
                lat_budget -= 1.0
            else:
                front = ndimage.binary_dilation(air, structure=vert)
            remove = front & np.isin(grid, target_ids)
            remove[0, :, :] = False  # 基板最下層は残す（貫通防止）
            if not remove.any():
                break
            grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "targets": list(self.targets),
            "cycles": self.cycles,
            "etch_per_cycle_nm": self.etch_per_cycle_nm,
            "anisotropy": self.anisotropy,
        }

    @classmethod
    def _from_params(cls, d: dict) -> AtomicLayerEtch:
        return cls(
            targets=list(d.get("targets", [])),
            cycles=int(d.get("cycles", 30)),
            etch_per_cycle_nm=float(d.get("etch_per_cycle_nm", 1.0)),
            anisotropy=float(d.get("anisotropy", 0.0)),
        )


# === Spacer（サイドウォールスペーサ）======================================
@register
@dataclass
class Spacer(Process):
    """サイドウォールスペーサ形成。

    段差（ゲート等）にコンフォーマル成膜したのち異方性エッチバックを行い、
    水平面の膜を除去して垂直側壁にのみ材料を残す。LDD/ゲートスペーサ等で
    自己整合的に微細な側壁構造を作る代表的プロセス。
    """

    type = "SPACER"
    label = "スペーサ形成"

    material: str = "nitride"
    thickness_um: float = 0.05
    overetch_um: float = 0.0

    def summary(self) -> str:
        m = materials.get(self.material)
        oe = "" if self.overetch_um <= 0 else f" +OE{self.overetch_um * 1000:.0f}nm"
        return f"SPACER  {m.label}側壁スペーサ {self.thickness_um * 1000:.0f}nm{oe}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "膜厚")
        _require_non_negative(self.overetch_um, "オーバーエッチ量")
        grid = wafer.grid
        mat_id = materials.get(self.material).id
        t = wafer.um_to_vox(self.thickness_um)
        # 1) コンフォーマル成膜: 既存固体表面から距離 t 以内の空気に堆積
        air = grid == materials.AIR
        dist = ndimage.distance_transform_edt(air)
        coat = air & (dist <= t)
        if not coat.any():
            return
        grid[coat] = mat_id
        # 2) 異方性エッチバック: 各列のスペーサ縦連続ラン高さがしきい値以下
        #    （＝水平膜）なら除去し、高いラン（側壁）は残す。
        thresh = t + (wafer.um_to_vox(self.overetch_um) if self.overetch_um > 0 else 0)
        remove = np.zeros_like(coat)
        cols = np.argwhere(coat.any(axis=0))
        for y, x in cols:
            colz = np.flatnonzero(coat[:, y, x])
            breaks = np.where(np.diff(colz) > 1)[0] + 1
            for run in np.split(colz, breaks):
                if run.size <= thresh:
                    remove[run, y, x] = True
        grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "overetch_um": self.overetch_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Spacer:
        return cls(
            material=d.get("material", "nitride"),
            thickness_um=float(d.get("thickness_um", 0.05)),
            overetch_um=float(d.get("overetch_um", 0.0)),
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
    # 斜め蒸着（指向性）の入射角（度, 鉛直から）。0=真上から。>0 で +x 方向
    # から斜めにフラックスが入り、背の高い構造の風下(-x)側に影ができて膜が
    # 付かない。電子ビーム蒸着の指向性シャドーイング/リフトオフに対応。
    tilt_deg: float = 0.0

    def summary(self) -> str:
        m = materials.get(self.material)
        sc = "" if self.step_coverage >= 0.999 else f"  被覆{self.step_coverage:.0%}"
        oh = "" if self.overhang <= 0 else f"  庇{self.overhang:.1f}"
        tl = "" if self.tilt_deg <= 0 else f"  傾斜{self.tilt_deg:.0f}°"
        return f"PVD  {m.label}  厚{self.thickness_um:.2f}µm{sc}{oh}{tl}"

    @staticmethod
    def _open_to_top(air: np.ndarray) -> np.ndarray:
        """上面（z 最大の平面）まで空気で連結している領域を True で返す。

        フラックスは開口から入るため、最上面と空気でつながった部分にしか
        堆積しない。開口が膜で塞がれた瞬間に内部の空気は連結が切れて
        「到達不能」となり、以降は成長せずキーホールボイドとして凍結する。
        """
        struct = ndimage.generate_binary_structure(3, 1)  # 6 近傍
        lbl, n = ndimage.label(air, structure=struct)
        if n == 0:
            return np.zeros_like(air)
        top_labels = np.unique(lbl[-1])
        top_labels = top_labels[top_labels != 0]
        if top_labels.size == 0:
            return np.zeros_like(air)
        return np.isin(lbl, top_labels)

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "膜厚")
        _require_range(self.step_coverage, 0.0, 1.0, "ステップカバレッジ")
        _require_non_negative(self.overhang, "オーバーハング")
        _require_range(self.tilt_deg, 0.0, 89.0, "入射角")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        mat_id = materials.get(self.material).id
        t = wafer.um_to_vox(self.thickness_um)
        if t <= 0:
            return
        sc = float(np.clip(self.step_coverage, 0.0, 1.0))
        oh = float(self.overhang)

        # 成膜前から存在する閉空気（上面に連結しない空洞）。庇なし成膜では
        # 新たに空洞を封止しないことを保証するため、後段で「成膜後に新たに
        # 封じた空気」だけを偽像として除く際の基準に使う。
        pre_air = grid == materials.AIR
        pre_sealed = pre_air & ~self._open_to_top(pre_air)

        # 斜め蒸着シャドーイング: +x からの斜め入射で風下(-x)側の列が背の高い
        # 構造に隠れて膜が付かない列を求める。鉛直から角 θ の入射は水平距離 d
        # 進むのに高さ d/tan(θ) を要するので、x+d 列が z_top[x]+d/tan(θ) 以上で遮蔽。
        col_shadow = np.zeros((ny, nx), dtype=bool)
        if self.tilt_deg > 0:
            z_top = wafer.top_surface_z()
            valid = z_top >= 0
            tan_t = math.tan(math.radians(self.tilt_deg))
            ztf = np.where(valid, z_top, -(10**9)).astype(np.float64)
            zspan = int(z_top[valid].max() - z_top[valid].min()) if valid.any() else 0
            for d in range(1, nx):
                needed = math.ceil(d / tan_t)
                if needed > zspan:
                    break
                shifted = np.full((ny, nx), -(10**9), dtype=np.float64)
                shifted[:, : nx - d] = ztf[:, d:]
                col_shadow |= shifted >= (ztf + needed)

        # 横方向(側壁/庇)の成長を司る 3x3x3 構造要素（z 方向は含めない）。
        lat = np.zeros((3, 3, 3), dtype=bool)
        lat[1, 1, 1] = lat[1, 1, 0] = lat[1, 1, 2] = lat[1, 0, 1] = lat[1, 2, 1] = True

        # 垂直堆積のシャドーイング: 深い窪みの列ほど上方からのフラックスが
        # 届きにくく、底に積もる膜が薄くなる。各列の窪み深さ(周囲最大との差)
        # から減衰係数 atten∈[sc,1] を求め、列ごとに堆積頻度を下げる。平坦な
        # フィールド(atten=1)は毎反復堆積し、深い底は sc 倍の頻度に落ちる。
        z_top0 = wafer.top_surface_z()
        valid0 = z_top0 >= 0
        zt0 = np.where(valid0, z_top0, 0).astype(np.float32)
        # フィールド基準高さ（上面高さの 75 パーセンタイル）からの窪み深さで
        # 評価する。狭く深いトレンチ底でも局所最大ではなくフィールド基準と
        # 比較するため、深さ＝アスペクト比相当のシャドーイングを捉えられる。
        if valid0.any():
            field_ref = float(np.percentile(zt0[valid0], 75))
        else:
            field_ref = 0.0
        recess = np.clip(field_ref - zt0, 0, None)
        atten = 1.0 - (1.0 - sc) * np.clip(recess / max(t, 1), 0.0, 1.0)
        # 列ごとの垂直堆積バジェット（ボクセル数）。フィールドは t、深い窪み底は
        # 約 sc·t に減る。フラックスが届く露出面には最低 1 ボクセルは積もるので
        # 0 にはしない（被覆が悪くても底に薄膜が付く実挙動）。
        v_budget = np.maximum(np.rint(atten * t).astype(np.int64), 1)
        v_done = np.zeros((ny, nx), dtype=np.int64)

        # 側壁被覆の深さ依存テーパ: 指向性フラックスは開口から入るため、側壁
        # 上端ほど空(開口)を広く見込んで厚く付き、深い底ほど壁に遮蔽されて
        # 薄くなる。底のコンキャブ隅が最も遮蔽されて最薄になる実挙動を表す。
        # フィールド面からの深さ(ボクセル)で各 z の堆積露出度を与え、z ごとに
        # 側壁成長頻度を変える（深い z ほど稀にしか成長せず薄膜にとどまる）。
        z_idx_s = np.arange(nz)
        depth_below = np.clip(field_ref - z_idx_s, 0.0, None)
        if valid0.any():
            h_shadow = max(field_ref - float(zt0[valid0].min()), 1.0)
        else:
            h_shadow = 1.0
        # 上端=1.0 → 底=0.08 へ線形にテーパ（底でも薄膜は付くので 0 にしない）。
        side_expo = np.clip(1.0 - depth_below / h_shadow, 0.08, 1.0)
        # 各 z の側壁被覆の最大横厚（ボクセル）。上端で約 sc·t、底で薄い。
        # 元の壁からの横距離がこのキャップを超えて成長しないよう抑え、底隅に
        # 金属が三角形に充填されるフィレット偽像を防ぐ。
        side_cap = sc * t * side_expo
        # 成膜前の壁/床からの横距離場（横方向のみ評価, 縦の空気は無視）。
        init_air = grid == materials.AIR
        wall_dist = ndimage.distance_transform_edt(
            init_air, sampling=(1e6, 1.0, 1.0)
        )

        # 反復堆積: 1 反復で膜厚 1 ボクセル相当を成長させる。
        #  - 垂直成長: 水平面（直下が固体の開口空気）に積もる。窪み底は
        #             バジェット(≈sc·t)で頭打ちになり薄い膜にとどまる。
        #  - 側壁成長: 垂直面（横が固体の開口空気）に被覆率 sc×露出度 の頻度で
        #             堆積。深い側壁ほど稀になり、底隅に向かって薄くテーパする。
        #  - 庇成長 : 開口上端の膜先端（真下が空気の膜＝庇）から横へ張り出し、
        #             overhang に比例した速さで開口を塞ぐ。塞がると内部は
        #             _open_to_top の連結が切れてボイドとして凍結する。
        side_acc = np.zeros(nz, dtype=np.float64)
        oh_acc = 0.0
        for _ in range(t):
            air = grid == materials.AIR
            open_air = self._open_to_top(air)
            if not open_air.any():
                break
            solid = ~air

            # 垂直成長フロント（直下が固体の開口空気）を列ごとバジェット内で
            below_solid = np.zeros_like(air)
            below_solid[1:] = solid[:-1]
            can_v = v_done < v_budget
            grow_v = open_air & below_solid & can_v[None, :, :]
            if self.tilt_deg > 0:
                grow_v &= ~col_shadow[None, :, :]
            v_done += grow_v.any(axis=0)

            grow = grow_v.copy()

            # 側壁成長フロント（横が固体の開口空気）を z ごとの被覆頻度
            # sc×side_expo[z] で堆積。深い z ほど稀にしか tick せず薄くなる。
            # 元の壁からの横距離が side_cap[z] 以内に限り、底隅のフィレット
            # （三角形充填）偽像を防ぐ。
            side_acc += sc * side_expo
            side_tick = side_acc >= 1.0
            if side_tick.any():
                side_acc[side_tick] -= 1.0
                side_solid = ndimage.binary_dilation(solid, structure=lat) & air
                grow |= (
                    open_air
                    & side_solid
                    & side_tick[:, None, None]
                    & (wall_dist <= side_cap[:, None, None])
                )

            grid[grow] = mat_id

            # 庇（ブレッドローフィング）: 開口上端付近の側壁ほどフラックスが
            # 集中して内側へ速く張り出す。口元バンドを余分に内側成長させると
            # 上部が下部より先に塞がり、内部にキーホールボイドが封じ込められる。
            # overhang を蓄積し、tick した反復だけ口元を 1 段内側へ進める
            # （率制御）。基本等角充填が庇封止を上回るほど壁が厚くなり、
            # 残る空洞は上部に向かって細るティアドロップ状になる。
            oh_acc += oh
            if oh > 0 and oh_acc >= 1.0:
                oh_acc -= 1.0
                air2 = grid == materials.AIR
                open2 = self._open_to_top(air2)
                solid2 = ~air2
                side2 = ndimage.binary_dilation(solid2, structure=lat) & air2
                front = open2 & side2  # 側壁成長フロント
                if front.any():
                    zmax = int(np.where(front.any(axis=(1, 2)))[0].max())
                    zrange = np.arange(nz)[:, None, None]
                    mouth = front & (zrange >= zmax - 1)  # 口元 1 バンド
                    nxt = ndimage.binary_dilation(mouth, structure=lat)
                    nxt &= grid == materials.AIR
                    nxt &= self._open_to_top(grid == materials.AIR)
                    grid[nxt] = mat_id

        # === 凸トップコーナーのカスプ ===
        # 指向性フラックスはトレンチ／段差の凸角（肩）に広い立体角で入射する
        # ため、膜が盛り上がってカスプ状の膨らみを作り、フィールド膜と側壁膜を
        # 連続的に橋渡しする。これが無いとフィールド膜上面と一段低い側壁膜
        # 上面の間に段差が生じ、肩の「凹み」偽像になる。各凸コーナーから、
        # フィールド面と側壁面の段差を四半円で丸めて連続化する。
        metal = grid == mat_id
        col_has = metal.any(axis=0)
        if col_has.any():
            mt = np.where(
                col_has, (nz - 1) - np.argmax(metal[::-1], axis=0), -1
            )  # (y, x) 各列の金属上面 z
            air_now = grid == materials.AIR
            open_now = self._open_to_top(air_now)
            for yy in range(ny):
                row_top = mt[yy]
                if row_top.max() < 0:
                    continue
                field_z = int(row_top.max())  # フィールド膜上面（最も高い列）
                edges = []  # (コーナー列 x, トレンチ方向 sgn)
                for x in range(nx):
                    if row_top[x] != field_z:
                        continue
                    if x + 1 < nx and 0 <= row_top[x + 1] < field_z - 1:
                        edges.append((x, +1))
                    if x - 1 >= 0 and 0 <= row_top[x - 1] < field_z - 1:
                        edges.append((x, -1))
                for cx, sgn in edges:
                    # カスプ半径＝段差の高さ。ただし膜厚 t で頭打ちにして、
                    # 深い窪みで庇が伸びすぎて開口を塞ぐのを防ぐ（カスプは
                    # 堆積膜厚スケールの局所効果）。
                    nb = cx + sgn
                    rad = min(field_z - int(mt[yy, nb]), t)
                    if rad <= 1:
                        continue
                    for d in range(1, rad + 1):
                        xc = cx + sgn * d
                        if not (0 <= xc < nx):
                            break
                        # 四半円: コーナーで field_z、離れるほど低くなる。
                        cz = int(round(
                            field_z - rad + math.sqrt(max(rad * rad - d * d, 0))
                        ))
                        top_xc = int(mt[yy, xc])
                        if cz <= top_xc:
                            continue
                        lo = top_xc + 1 if top_xc >= 0 else 0
                        for z in range(lo, cz + 1):
                            if air_now[z, yy, xc] and open_now[z, yy, xc]:
                                grid[z, yy, xc] = mat_id

        # === 封止偽像の除去 ===
        # 庇(overhang)を指定しない指向性 PVD は、内部に空気を封じ込めない
        # （開口を塞ぐのは庇/ブレッドローフィングの効果）。側壁被覆や肩の
        # カスプ処理の結果、肩部に閉じた空気クラック（中空の庇状偽像）が残る
        # ことがあるため、成膜後に新たに封止された空気を金属で充填して連続膜
        # にする。overhang>0 の場合は意図したキーホールボイドなので保持する。
        if oh <= 0:
            air_final = grid == materials.AIR
            sealed = air_final & ~self._open_to_top(air_final)
            new_sealed = sealed & ~pre_sealed
            if new_sealed.any():
                grid[new_sealed] = mat_id

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "step_coverage": self.step_coverage,
            "overhang": self.overhang,
            "tilt_deg": self.tilt_deg,
        }

    @classmethod
    def _from_params(cls, d: dict) -> PVD:
        return cls(
            material=d.get("material", "metal_al"),
            thickness_um=float(d.get("thickness_um", 0.5)),
            step_coverage=float(d.get("step_coverage", 1.0)),
            overhang=float(d.get("overhang", 0.0)),
            tilt_deg=float(d.get("tilt_deg", 0.0)),
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
    # ARDE / RIE ラグ（µm, 0=無効）。アスペクト比依存エッチング: 狭い開口は
    # イオン/中性種の供給律速で削れにくく、浅くなる。各列の到達深さに
    # 係数 f = W/(W+arde_lag_um) を掛ける（W=局所開口幅）。狭いほど f→0、
    # 広いほど f→1。値が大きいほどラグが強い。
    arde_lag_um: float = 0.0

    def summary(self) -> str:
        tgt = "/".join(self.targets) if self.targets else "露出材料"
        oe = "" if self.overetch_pct <= 0 else f"  +OE{self.overetch_pct:.0f}%"
        lat = "" if self.lateral_um <= 0 else f"  横{self.lateral_um:.2f}µm"
        sel = "  選択比あり" if self.selectivity else ""
        tp = "" if self.taper_deg <= 0 else f"  テーパ{self.taper_deg:.0f}°"
        nt = "" if self.notch_um <= 0 else f"  ノッチ{self.notch_um:.2f}µm"
        ar = "" if self.arde_lag_um <= 0 else f"  ARDE{self.arde_lag_um:.2f}µm"
        return f"DRY  {tgt}  深さ{self.depth_um:.2f}µm{oe}{lat}{sel}{tp}{nt}{ar}"

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
        _require_non_negative(self.arde_lag_um, "ARDEラグ量")
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

        # ARDE / RIE ラグ: 局所開口幅 W が狭い列ほど到達深さを減らす。
        # 各連結開口領域の代表幅を「最大内接円直径 = 2×max(EDT)」で見積もり、
        # 係数 f = W/(W + L)（L=arde_lag のボクセル換算）を depth_cap に掛ける。
        if self.arde_lag_um > 0 and eligible.any():
            edt = ndimage.distance_transform_edt(eligible)
            labels, nlab = ndimage.label(eligible)
            width = np.zeros((ny, nx), dtype=float)
            if nlab > 0:
                # 領域ごとの最大 EDT（=内接半径）→ 幅 = 2×半径
                maxr = ndimage.maximum(edt, labels, index=np.arange(1, nlab + 1))
                maxr = np.atleast_1d(maxr)
                region_w = 2.0 * maxr  # ボクセル
                for li in range(1, nlab + 1):
                    width[labels == li] = region_w[li - 1]
            ll = max(1.0, float(wafer.um_to_vox(self.arde_lag_um)))
            with np.errstate(invalid="ignore"):
                factor = np.where(width > 0, width / (width + ll), 1.0)
            depth_cap = depth_cap * factor
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
            "arde_lag_um": self.arde_lag_um,
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
            arde_lag_um=float(d.get("arde_lag_um", 0.0)),
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
    # ディッシングの幅依存性を決める特性幅[µm]。軟材料領域の縁からの横方向
    # 距離がこの値を超えると最大ディッシング量に漸近する。狭い配線は浅く、
    # 広いパッドほど深く凹む（実機の幅依存ディッシングを再現）。
    dishing_width_um: float = 0.3
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
        # 実機のディッシングは平底ではなく「中央ほど深く、縁（バリア/絶縁膜
        # との境界）ほど浅い」凹面（皿状）になり、配線幅が広いほど深くなる。
        # 露出した軟材料領域の縁からの横方向距離（距離変換）でプロファイルを
        # 決め、距離が特性幅 dishing_width_um を超えると最大量に漸近させる。
        if self.soft_material and self.dishing_um > 0:
            soft_id = materials.get(self.soft_material).id
            z_top2 = wafer.top_surface_z()
            ys0, xs0 = np.nonzero(z_top2 >= 0)
            exposed = np.zeros(z_top2.shape, dtype=bool)
            if ys0.size:
                sel0 = grid[z_top2[ys0, xs0], ys0, xs0] == soft_id
                exposed[ys0[sel0], xs0[sel0]] = True
            if exposed.any():
                # 縁からの横方向距離[vox]。中央ほど大きい。
                edt = ndimage.distance_transform_edt(exposed)
                char_vox = max(1.0, float(wafer.um_to_vox(self.dishing_width_um)))
                dish_max = wafer.um_to_vox(self.dishing_um)
                # 縁で 0、中央で dish_max に漸近する皿状（凹面）プロファイル。
                profile = dish_max * (edt / (edt + char_vox))
                depth_map = np.round(profile).astype(int)
                depth_map[~exposed] = 0
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
            "dishing_width_um": self.dishing_width_um,
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
            dishing_width_um=float(d.get("dishing_width_um", 0.3)),
            erosion_um=float(d.get("erosion_um", 0.0)),
            density_radius_um=float(d.get("density_radius_um", 1.0)),
        )


# === BACKGRIND（裏面研削 / ウェハ薄化）=====================================
@register
@dataclass
class Backgrind(Process):
    """ウェハ裏面（底）を研削して基板を薄くする（3D-IC/パッケージ向け）。

    基板シリコンを底面から thin_um だけ除去し、全構造を下方へシフトする。
    表面のデバイス層は保護され、最下の基板シリコンのみが削られる（研削が
    デバイスに到達しないよう、最低 1 ボクセルの基板を残す）。config の
    substrate_um も実際に削れた分だけ更新する。
    """

    type = "BACKGRIND"
    label = "裏面研削"

    thin_um: float = 1.0

    def summary(self) -> str:
        return f"BACKGRIND  裏面から{self.thin_um:.2f}µm研削しウェハ薄化"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thin_um, "研削量")
        grid = wafer.grid
        nz = grid.shape[0]
        si_id = materials.BY_NAME["silicon"].id
        # 各列で底から連続するシリコン（基板）の厚さを求める。
        is_si = grid == si_id
        contig = np.cumprod(is_si, axis=0)  # 最初の非シリコンで 0 になる
        bottom_si = contig.sum(axis=0)  # (ny, nx) 底基板の厚さ[vox]
        min_sub = int(bottom_si.min())
        # デバイスに到達しないよう、最低 1 ボクセルの基板を残す。
        t = wafer.um_to_vox(self.thin_um)
        t = min(t, max(0, min_sub - 1))
        if t <= 0:
            return
        # 全構造を t ボクセルだけ下へシフト（底の基板を除去）。
        grid[: nz - t, :, :] = grid[t:, :, :]
        grid[nz - t :, :, :] = materials.AIR
        # 実際に削れた分だけ基板厚を更新。
        removed_um = t * wafer.config.pitch_um
        wafer.config.substrate_um = max(0.0, wafer.config.substrate_um - removed_um)

    def params_dict(self) -> dict:
        return {"thin_um": self.thin_um}

    @classmethod
    def _from_params(cls, d: dict) -> Backgrind:
        return cls(thin_um=float(d.get("thin_um", 1.0)))


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
    # Deal-Grove モード: time_min>0 のとき thickness_um は無視し、酸化時間と
    # 温度・雰囲気から物理的に膜厚を計算する（x²+Ax=B(t+τ)）。
    time_min: float = 0.0
    temperature_c: float = 1000.0
    ambient: str = "dry"

    def _effective_thickness_um(self) -> float:
        """Deal-Grove モードなら計算膜厚、そうでなければ指定膜厚を返す。"""
        if self.time_min > 0:
            return deal_grove_thickness_um(self.time_min, self.temperature_c, self.ambient)
        return self.thickness_um

    def summary(self) -> str:
        bk = "" if self.beak_fraction <= 0 else "  +バーズビーク"
        tox = self._effective_thickness_um()
        if self.time_min > 0:
            return (
                f"OXIDE  酸化膜{tox:.3f}µm成長"
                f"({self.ambient} {self.temperature_c:.0f}℃ {self.time_min:.0f}分){bk}"
            )
        return f"OXIDE  酸化膜{tox:.2f}µm成長{bk}"

    def apply(self, wafer: Wafer) -> None:
        thickness_um = self._effective_thickness_um()
        _require_positive(thickness_um, "酸化膜厚")
        _require_range(self.consume_fraction, 0.0, 0.95, "消費比")
        _require_non_negative(self.beak_fraction, "バーズビーク比")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        ox_id = materials.BY_NAME["oxide"].id
        # ドープされたシリコンも熱酸化される（物理的に正しい）。
        si_like = [si_id, materials.BY_NAME["doped_n"].id, materials.BY_NAME["doped_p"].id]
        total = wafer.um_to_vox(thickness_um)
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
            "time_min": self.time_min,
            "temperature_c": self.temperature_c,
            "ambient": self.ambient,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Oxidation:
        return cls(
            thickness_um=float(d.get("thickness_um", 0.3)),
            consume_fraction=float(d.get("consume_fraction", 0.45)),
            beak_fraction=float(d.get("beak_fraction", 0.0)),
            time_min=float(d.get("time_min", 0.0)),
            temperature_c=float(d.get("temperature_c", 1000.0)),
            ambient=str(d.get("ambient", "dry")),
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
    # 時間/温度モード: time_min>0 のとき depth_um は無視し、ドーパント拡散長
    # L=√(D·t) からドライブイン量を物理計算する（炉アニールの等方拡散）。
    time_min: float = 0.0
    temperature_c: float = 1000.0

    def _effective_depth_um(self, dopant: str) -> float:
        """時間/温度モードなら拡散長、そうでなければ指定 depth_um を返す。"""
        if self.time_min > 0:
            return diffusion_length_um(self.time_min, self.temperature_c, dopant)
        return self.depth_um

    def summary(self) -> str:
        if self.time_min > 0:
            return (
                f"ANNEAL  ドライブイン({self.temperature_c:.0f}℃ "
                f"{self.time_min:.0f}分, L=√Dt)"
            )
        return f"ANNEAL  ドライブイン{self.depth_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        if self.time_min <= 0:
            _require_positive(self.depth_um, "ドライブイン量")
        _require_non_negative(self.time_min, "アニール時間")
        grid = wafer.grid
        si_id = materials.BY_NAME["silicon"].id
        # n / p それぞれを独立に膨張させ、隣接シリコンを同型に変換する。
        # 時間/温度モードではドーパント種ごとに拡散長が異なる。
        for name in ("doped_n", "doped_p"):
            dop_id = materials.get(name).id
            region = grid == dop_id
            if not region.any():
                continue
            lat = wafer.um_to_vox(self._effective_depth_um(name))
            if lat <= 0:
                continue
            grown = _isotropic_dilate(region, lat)
            spread = grown & (grid == si_id)
            grid[spread] = dop_id

    def params_dict(self) -> dict:
        return {
            "depth_um": self.depth_um,
            "time_min": self.time_min,
            "temperature_c": self.temperature_c,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Anneal:
        return cls(
            depth_um=float(d.get("depth_um", 0.3)),
            time_min=float(d.get("time_min", 0.0)),
            temperature_c=float(d.get("temperature_c", 1000.0)),
        )


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

        # 各開口列について、最も近い開口縁までの水平距離を求める。結晶異方性
        # エッチでは側壁が角度 angle で内側へ傾くため、縁から距離 r の列は
        # 深さ r·tan(angle) まで掘れて V 溝/逆ピラミッド底で自己停止する。
        # （帯状開口は幅、正方形開口は対角で律速。binary_erosion を使うと
        #  y 方向にも侵食して薄い格子では深さが ny に律速される不具合を回避）
        dist = ndimage.distance_transform_edt(opening)  # 縁からの距離[vox]
        max_d_col = np.minimum(dist * tan_a, float(depth))  # 列ごと到達深さ[vox]

        z_idx = np.arange(grid.shape[0])[:, None, None]
        # 各列で z_top-d (d=1..max_d_col) のターゲット材を除去する。
        etch_mask = (
            opening[None, :, :]
            & (z_idx <= z_top[None, :, :])
            & (z_idx > (z_top - max_d_col.astype(int))[None, :, :])
            & (grid == target_id)
        )
        grid[etch_mask] = materials.AIR

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
        # ECD/めっきは空洞をボトムアップで完全充填する。各セルの「真下に固体が
        # あるか」で判定することで、テーパしたバリア側壁の庇下に取り残された
        # エア隙間（シーム偽像）も埋める。z_top（列の最上端固体）基準だと、
        # 庇より下の空気は z_idx>z_top を満たさず埋め残されてしまう。
        solid = grid != materials.AIR
        solid_below = np.zeros_like(solid)
        solid_below[1:] = np.cumsum(solid, axis=0)[:-1] > 0
        deposit = (
            (z_idx <= fill_to)
            & (grid == materials.AIR)
            & solid_below
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
        resist_cols = (z_idx >= resist_bottom[None, :, :]) & has_resist[None, :, :]
        # レジスト列（レジスト＋その上に乗った膜）を一括除去する。
        grid[resist_cols] = materials.AIR

        # フェンス（ラビットイヤー）除去: 指向性成膜はレジスト側壁にも膜を
        # 付けるが、リフトオフ用レジストはアンダーカット形状のため側壁膜は
        # 基板上の膜と切れて一緒に剥がれる。残ると開口端に三角形の突起
        # （偽像）になるので、平坦膜の上面より高くレジスト跡に連結する固体を
        # 連結的に除去し、基板上の所望パターンだけを残す。
        solid = grid != materials.AIR
        if solid.any():
            col_has = solid.any(axis=0)
            top_z = np.where(
                col_has, (nz - 1) - np.argmax(solid[::-1], axis=0), -1
            )
            flat = top_z[top_z >= 0]
            if flat.size:
                # 平坦膜の上面（最頻値）。突起はこれより上にだけ立つ。
                flat_top = int(np.bincount(flat).argmax())
                lat = np.zeros((3, 3, 3), dtype=bool)
                lat[1, 1, 1] = lat[1, 1, 0] = lat[1, 1, 2] = True
                lat[1, 0, 1] = lat[1, 2, 1] = True
                above = z_idx > flat_top
                fence = ndimage.binary_dilation(
                    resist_cols, structure=lat
                ) & solid & above
                while True:
                    grown = (
                        ndimage.binary_dilation(fence, structure=lat)
                        & solid & above
                    )
                    if int(grown.sum()) == int(fence.sum()):
                        break
                    fence = grown
                grid[fence] = materials.AIR

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

        for _ in range(depth):
            z_top, top_id = _top_material(wafer)
            do = opening & (z_top >= 0) & (top_id == target_id) & (etched < depth_cap)
            if not do.any():
                break
            ys, xs = np.nonzero(do)
            zz = z_top[ys, xs]
            grid[zz, ys, xs] = materials.AIR
            etched[do] += 1.0

        # スキャロップ（Bosch プロセスの側壁波形）: エッチ後のトレンチ側壁を
        # サイクル位相に応じた半円プロファイルで横方向へ追加除去する。各サイクル
        # 中央で最も膨らみ境界で 0 に絞れるため、解像度に依らず丸みのある周期的な
        # 凹凸になる（旧実装の 1 ボクセル線にならない）。横方向(x,y)距離だけで
        # 評価するので深さは増えない（垂直方向は実質無限大の sampling）。
        amp = wafer.um_to_vox(self.scallop_um) if self.scallop_um > 0 else 0
        if amp > 0:
            s_pitch = max(2, wafer.um_to_vox(self.scallop_pitch_um))
            target_mask = grid == target_id
            lat_dist = ndimage.distance_transform_edt(
                target_mask, sampling=(1e6, 1.0, 1.0)
            )
            surf = int(z_top0[opening].max())  # フィールド上面（深さ位相の基準）
            z_idx = np.arange(nz)
            phase = (surf - z_idx) % s_pitch  # 上面からの深さ方向のサイクル位相
            radius_z = amp * np.sin(np.pi * phase / s_pitch)  # 半円: 中央で最大
            radius = radius_z[:, None, None]
            carve = (
                target_mask
                & (lat_dist > 0.0)
                & (lat_dist <= radius)
                & (z_idx[:, None, None] <= surf)
            )
            grid[carve] = materials.AIR

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


# 各工程の 1 行説明（GUI のダイアログ上部バナー等で使用, GUI 非依存）
_PROCESS_HELP = {
    "PHOTO": "レジストを塗布・露光・現像してマスクパターンを転写します。後続のエッチ／注入のマスクになります。",
    "CVD": "化学気相成長。段差をほぼ一様な膜厚で覆う（コンフォーマル）成膜です。",
    "PVD": "スパッタ／蒸着。指向性が強く、被覆が悪いとオーバーハングや窪み底の薄化が起きます。",
    "ALD": "原子層堆積。1 サイクル 1 原子層で nm 精度・高コンフォーマル。high-k／バリア膜に使います。",
    "EPI": "エピタキシャル成長。露出シリコン上にのみ単結晶を選択成長します（SiGe ソース／ドレイン等）。",
    "DRY": "プラズマ異方性エッチ。ほぼ垂直な側壁の溝／ホールを形成します。",
    "WET": "薬液等方性エッチ。横方向にも削れ、マスク下にアンダーカットが生じます。",
    "ALE": "原子層エッチ。1 サイクルで原子層レベルを自己制限的に除去し、nm 精度に深さを制御します。",
    "KOH": "結晶異方性ウェット。Si(111) が現れ 54.7° テーパの V 溝／逆ピラミッドを作ります（MEMS）。",
    "DRIE": "Bosch プロセス深掘りエッチ。側壁にスキャロップが残る高アスペクト比加工（TSV）です。",
    "SPUTTER": "スパッタエッチ。物理的に表面を削り、角に面取り（ファセット）が生じます。",
    "DIFFUSION": "高温拡散。ドーパントを濃度勾配に従って拡散させます（ドライブイン）。",
    "IMPLANT": "イオン注入。飛程 Rp を中心にガウス分布で埋め込みドープ層を形成します。",
    "ANNEAL": "アニール。注入損傷を回復し、ドーパントを活性化・拡散させます。",
    "RTP": "急速熱処理（短時間アニール）。浅い接合を保ったまま活性化します。",
    "OXIDE": "熱酸化。露出 Si を SiO2 に変換します。LOCOS のバーズビークも再現します。",
    "SALICIDE": "自己整合シリサイド。露出 Si／ポリ上のみ金属と反応させ低抵抗化します。",
    "FILL": "ダマシン充填。トレンチ／ビアを金属（Cu 等）で埋めます。",
    "SPINON": "スピンオン塗布。流動材料を回転塗布し、凹凸を平坦化します。",
    "CMP": "化学機械研磨。上面を平坦化します。ディッシング／エロージョンも評価できます。",
    "BACKGRIND": "裏面研削。ウェハを薄化します（3D 実装／TSV 露出）。",
    "REFLOW": "熱リフロー。はんだ／ガラスを溶融して丸め・平坦化します。",
    "CLEAN": "プラズマ洗浄。残渣・有機物を除去します。",
    "LIFTOFF": "リフトオフ。レジスト上の金属ごと剥離し、開口部の金属だけを残します。",
    "STRIP": "ストリップ。レジスト等を除去します。",
}


def process_help(proc_type: str) -> str:
    """工程タイプの 1 行説明を返す（未登録は空文字）。"""
    return _PROCESS_HELP.get(proc_type, "")


# 工程の分類（GUI の「工程を追加」メニューをカテゴリ分けする, GUI 非依存）
_CATEGORY_ORDER = [
    "リソグラフィ", "成膜", "エッチング", "ドーピング・熱処理", "平坦化・仕上げ",
]
_PROCESS_CATEGORY = {
    "PHOTO": "リソグラフィ",
    "CVD": "成膜", "ALD": "成膜", "PVD": "成膜", "EPI": "成膜",
    "SPINON": "成膜", "FILL": "成膜", "SALICIDE": "成膜",
    "DRY": "エッチング", "WET": "エッチング", "ALE": "エッチング", "KOH": "エッチング",
    "DRIE": "エッチング", "SPUTTER": "エッチング", "LIFTOFF": "エッチング",
    "DIFFUSION": "ドーピング・熱処理", "IMPLANT": "ドーピング・熱処理",
    "ANNEAL": "ドーピング・熱処理", "RTP": "ドーピング・熱処理", "OXIDE": "ドーピング・熱処理",
    "CMP": "平坦化・仕上げ", "BACKGRIND": "平坦化・仕上げ", "REFLOW": "平坦化・仕上げ",
    "CLEAN": "平坦化・仕上げ", "STRIP": "平坦化・仕上げ",
}


def process_category(proc_type: str) -> str:
    """工程タイプのカテゴリ名を返す（未登録は 'その他'）。"""
    return _PROCESS_CATEGORY.get(proc_type, "その他")


# カテゴリの識別色（GUI のレシピ一覧で工程種別を色分けする, GUI 非依存）
_CATEGORY_COLOR = {
    "リソグラフィ": "#8e44ad",
    "成膜": "#2d6cdf",
    "エッチング": "#e67e22",
    "ドーピング・熱処理": "#c0392b",
    "平坦化・仕上げ": "#16a085",
    "その他": "#7f8c8d",
}


def category_color(category: str) -> str:
    """カテゴリ名の識別色（#rrggbb）を返す（未登録はグレー）。"""
    return _CATEGORY_COLOR.get(category, "#7f8c8d")


def process_color(proc_type: str) -> str:
    """工程タイプの識別色（カテゴリ色）を返す。"""
    return category_color(process_category(proc_type))


def categorized_types() -> list[tuple[str, list[tuple[str, str]]]]:
    """カテゴリ別に (カテゴリ名, [(type, label), ...]) を表示順で返す。

    「工程を追加」メニューをカテゴリのサブメニューに分けるために使う。
    既知カテゴリを _CATEGORY_ORDER 順に、未知は末尾にまとめる。
    """
    groups: dict[str, list[tuple[str, str]]] = {c: [] for c in _CATEGORY_ORDER}
    for t, label in available_types():
        groups.setdefault(process_category(t), []).append((t, label))
    ordered = [(c, groups[c]) for c in _CATEGORY_ORDER if groups.get(c)]
    extra = [(c, v) for c, v in groups.items() if c not in _CATEGORY_ORDER and v]
    return ordered + extra


def available_types() -> list[tuple[str, str]]:
    """(type, label) のリストを表示順で返す。"""
    order = [
        "PHOTO", "CVD", "ALD", "PVD", "EPI",
        "DRY", "WET", "ALE", "KOH", "DRIE", "SPUTTER",
        "DIFFUSION", "IMPLANT", "ANNEAL", "RTP", "OXIDE",
        "SALICIDE", "FILL", "SPINON", "CMP", "BACKGRIND", "REFLOW", "CLEAN", "LIFTOFF", "STRIP",
    ]
    out = []
    for t in order:
        cls = _REGISTRY.get(t)
        if cls is not None:
            out.append((t, cls.label))
    return out
