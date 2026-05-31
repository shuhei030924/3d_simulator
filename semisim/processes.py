"""半導体プロセス工程の定義と適用ロジック。

各工程は `apply(wafer)` でボクセルグリッドを変更し、`to_dict`/`from_dict`
で JSON シリアライズできる。レシピは工程の順序付きリストで、毎回初期状態
から順に再適用（リプレイ）して決定論的に断面を再現する。

すべての工程は充填ボリュームを操作するため、断面が空洞になることはない。
"""
from __future__ import annotations

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

    def summary(self) -> str:
        pol = "ポジ" if self.polarity == "positive" else "ネガ"
        return f"PHOTO  厚{self.thickness_um:.2f}µm  {pol}  図形{len(self.mask.shapes)}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "レジスト厚")
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
        }

    @classmethod
    def _from_params(cls, d: dict) -> Photo:
        return cls(
            mask=Mask.from_dict(d.get("mask")),
            thickness_um=float(d.get("thickness_um", 1.0)),
            polarity=d.get("polarity", "positive"),
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

    def summary(self) -> str:
        m = materials.get(self.material)
        return f"CVD  {m.label}  厚{self.thickness_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "膜厚")
        grid = wafer.grid
        mat_id = materials.get(self.material).id
        t = wafer.um_to_vox(self.thickness_um)
        air = grid == materials.AIR
        # 固体表面からの距離 t 以内の空気に堆積（コンフォーマル）
        dist = ndimage.distance_transform_edt(air)
        deposit = air & (dist <= t)
        grid[deposit] = mat_id

    def params_dict(self) -> dict:
        return {"material": self.material, "thickness_um": self.thickness_um}

    @classmethod
    def _from_params(cls, d: dict) -> CVD:
        return cls(
            material=d.get("material", "oxide"),
            thickness_um=float(d.get("thickness_um", 0.5)),
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

    def summary(self) -> str:
        m = materials.get(self.material)
        sc = "" if self.step_coverage >= 0.999 else f"  被覆{self.step_coverage:.0%}"
        return f"PVD  {m.label}  厚{self.thickness_um:.2f}µm{sc}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "膜厚")
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

    def params_dict(self) -> dict:
        return {
            "material": self.material,
            "thickness_um": self.thickness_um,
            "step_coverage": self.step_coverage,
        }

    @classmethod
    def _from_params(cls, d: dict) -> PVD:
        return cls(
            material=d.get("material", "metal_al"),
            thickness_um=float(d.get("thickness_um", 0.5)),
            step_coverage=float(d.get("step_coverage", 1.0)),
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
    """上方から垂直に削る異方性エッチング。レジスト等で保護される。"""

    type = "DRY"
    label = "ドライエッチ"

    targets: list[str] = field(default_factory=list)  # 空 = 自動
    depth_um: float = 0.5
    overetch_pct: float = 0.0  # ターゲット枯渇後に下層も削る割合(%)

    def summary(self) -> str:
        tgt = "/".join(self.targets) if self.targets else "露出材料"
        oe = "" if self.overetch_pct <= 0 else f"  +OE{self.overetch_pct:.0f}%"
        return f"DRY  {tgt}  深さ{self.depth_um:.2f}µm{oe}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "エッチ量")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        target_ids = set(_resolve_targets(self.targets))
        depth = wafer.um_to_vox(self.depth_um)

        # 初期の最上面材料がターゲットの列のみエッチ対象（マスク効果）
        _, top_id = _top_material(wafer)
        eligible = np.isin(top_id, list(target_ids))  # (ny, nx)

        for _ in range(depth):
            z_top, top_id = _top_material(wafer)
            top_is_target = np.isin(top_id, list(target_ids))
            do = eligible & top_is_target & (z_top >= 0)
            if not do.any():
                break
            ys, xs = np.nonzero(do)
            grid[z_top[ys, xs], ys, xs] = materials.AIR

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

    def params_dict(self) -> dict:
        return {
            "targets": list(self.targets),
            "depth_um": self.depth_um,
            "overetch_pct": self.overetch_pct,
        }

    @classmethod
    def _from_params(cls, d: dict) -> DryEtch:
        return cls(
            targets=list(d.get("targets", [])),
            depth_um=float(d.get("depth_um", 0.5)),
            overetch_pct=float(d.get("overetch_pct", 0.0)),
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

    def summary(self) -> str:
        tgt = "/".join(self.targets) if self.targets else "露出材料"
        return f"WET  {tgt}  深さ{self.depth_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "エッチ量")
        grid = wafer.grid
        target_ids = _resolve_targets(self.targets)
        r = wafer.um_to_vox(self.depth_um)
        # 露出面から 1 ボクセルずつ等方的に後退させる。
        # 空気に隣接するターゲットのみを毎回除去するため、間に別材料が
        # あれば貫通せず、マスク下のアンダーカットも自然に再現される。
        # 注: 距離変換による一括計算は障壁材料を貫通してしまい物理が崩れる
        # ため、ここでは前線伝播（反復ダイレーション）を意図的に用いる。
        struct = ndimage.generate_binary_structure(3, 1)  # 6 近傍
        for _ in range(r):
            air = grid == materials.AIR
            front = ndimage.binary_dilation(air, structure=struct)
            cur_target = np.isin(grid, target_ids)
            remove = front & cur_target
            if not remove.any():
                break
            grid[remove] = materials.AIR

    def params_dict(self) -> dict:
        return {"targets": list(self.targets), "depth_um": self.depth_um}

    @classmethod
    def _from_params(cls, d: dict) -> WetEtch:
        return cls(
            targets=list(d.get("targets", [])),
            depth_um=float(d.get("depth_um", 0.5)),
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
        # 経験的に縦方向拡散の約 1/3 を横方向広がりとする（簡易モデル）。
        lat = max(0, depth // 3)
        if lat > 0 and converted.any():
            grown = _isotropic_dilate(converted, lat)
            spread = grown & (grid == si_id)
            grid[spread] = dop_id

    def params_dict(self) -> dict:
        return {"dopant": self.dopant, "depth_um": self.depth_um}

    @classmethod
    def _from_params(cls, d: dict) -> Diffusion:
        return cls(
            dopant=d.get("dopant", "doped_n"),
            depth_um=float(d.get("depth_um", 0.6)),
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
    """上面を研磨して平坦化する。最も高い点から指定量だけ削り水平にする。"""

    type = "CMP"
    label = "CMP平坦化"

    remove_um: float = 0.5

    def summary(self) -> str:
        return f"CMP  上面から{self.remove_um:.2f}µm研磨し平坦化"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.remove_um, "研磨量")
        grid = wafer.grid
        nz = grid.shape[0]
        z_top = wafer.top_surface_z()
        if int(z_top.max()) < 0:
            return
        max_top = int(z_top.max())
        cut = max_top - wafer.um_to_vox(self.remove_um)
        cut = max(-1, min(nz - 1, cut))
        # cut より上の全ボクセルを空気にして上面を平坦化
        if cut + 1 < nz:
            grid[cut + 1:, :, :] = materials.AIR

    def params_dict(self) -> dict:
        return {"remove_um": self.remove_um}

    @classmethod
    def _from_params(cls, d: dict) -> CMP:
        return cls(remove_um=float(d.get("remove_um", 0.5)))


# === OXIDE（熱酸化）========================================================
@register
@dataclass
class Oxidation(Process):
    """露出シリコンを熱酸化して酸化膜に変える（Si を消費しつつ上方へ成長）。"""

    type = "OXIDE"
    label = "熱酸化"

    thickness_um: float = 0.3

    def summary(self) -> str:
        return f"OXIDE  酸化膜{self.thickness_um:.2f}µm成長"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "酸化膜厚")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        ox_id = materials.BY_NAME["oxide"].id
        # ドープされたシリコンも熱酸化される（物理的に正しい）。
        si_like = [si_id, materials.BY_NAME["doped_n"].id, materials.BY_NAME["doped_p"].id]
        total = wafer.um_to_vox(self.thickness_um)
        # 熱酸化の体積則: 生成 SiO2 厚の約 45% 分の Si が消費され、残り約 55%
        # が元の Si 表面より上方へ成長する（Deal-Grove の体積膨張比に由来）。
        consume = max(0, int(round(total * 0.45)))
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

    def params_dict(self) -> dict:
        return {"thickness_um": self.thickness_um}

    @classmethod
    def _from_params(cls, d: dict) -> Oxidation:
        return cls(thickness_um=float(d.get("thickness_um", 0.3)))


# === IMPLANT（イオン注入）==================================================
@register
@dataclass
class Implant(Process):
    """イオン注入。投影飛程 Rp を中心としたガウス的な深さ分布で埋込ドープする。

    range_um (Rp): 表面からの投影飛程。
    straggle_um (ΔRp): 飛程のばらつき（標準偏差）。±1.5σ の帯を注入領域とする。
    レジストで覆われた列はイオンが止められ、下地は保護される。
    """

    type = "IMPLANT"
    label = "イオン注入"

    dopant: str = "doped_n"
    range_um: float = 0.4
    straggle_um: float = 0.1

    def summary(self) -> str:
        m = materials.get(self.dopant)
        return f"IMPLANT  {m.label}  Rp{self.range_um:.2f}±{self.straggle_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.range_um, "投影飛程")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        dop_id = materials.get(self.dopant).id
        rp = wafer.um_to_vox(self.range_um)
        sigma = max(1, wafer.um_to_vox(max(self.straggle_um, 1e-6)))
        band = int(round(1.5 * sigma))  # ±1.5σ の注入帯
        lo_d = max(0, rp - band)
        hi_d = rp + band

        z_top, top_id = _top_material(wafer)
        resist_ids = [m.id for m in materials.all_materials() if m.is_resist]
        eligible = (z_top >= 0) & ~np.isin(top_id, resist_ids)  # (ny, nx)

        z_idx = np.arange(nz)[:, None, None]
        depth_below = z_top[None, :, :] - z_idx  # 表面からの深さ(vox)
        in_band = (depth_below >= lo_d) & (depth_below <= hi_d)
        # シリコン（および既ドープ）を注入帯の範囲で変換する
        convertible = (grid == si_id) | (grid == materials.BY_NAME["doped_n"].id) | (
            grid == materials.BY_NAME["doped_p"].id
        )
        implant = in_band & convertible & eligible[None, :, :]
        grid[implant] = dop_id

    def params_dict(self) -> dict:
        return {
            "dopant": self.dopant,
            "range_um": self.range_um,
            "straggle_um": self.straggle_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> Implant:
        return cls(
            dopant=d.get("dopant", "doped_n"),
            range_um=float(d.get("range_um", 0.4)),
            straggle_um=float(d.get("straggle_um", 0.1)),
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


# === EPI（選択エピタキシャル成長）==========================================
@register
@dataclass
class Epitaxy(Process):
    """露出シリコン上のみに単結晶を選択的に成長させる（酸化膜上には付かない）。"""

    type = "EPI"
    label = "エピ成長"

    material: str = "epi_si"
    thickness_um: float = 0.5

    def summary(self) -> str:
        m = materials.get(self.material)
        return f"EPI  {m.label}  厚{self.thickness_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.thickness_um, "エピ厚")
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

        z_idx = np.arange(nz)[:, None, None]
        lo = z_top[None, :, :]
        deposit = (
            (z_idx > lo)
            & (z_idx <= lo + t)
            & (grid == materials.AIR)
            & eligible[None, :, :]
        )
        grid[deposit] = mat_id

    def params_dict(self) -> dict:
        return {"material": self.material, "thickness_um": self.thickness_um}

    @classmethod
    def _from_params(cls, d: dict) -> Epitaxy:
        return cls(
            material=d.get("material", "epi_si"),
            thickness_um=float(d.get("thickness_um", 0.5)),
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

    def summary(self) -> str:
        m = materials.get(self.material)
        return f"FILL  {m.label}  +{self.overfill_um:.2f}µm"

    def apply(self, wafer: Wafer) -> None:
        if self.overfill_um < 0:
            raise ValueError("オーバーフィル量は 0 以上である必要があります。")
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

    def params_dict(self) -> dict:
        return {"material": self.material, "overfill_um": self.overfill_um}

    @classmethod
    def _from_params(cls, d: dict) -> Fill:
        return cls(
            material=d.get("material", "metal_cu"),
            overfill_um=float(d.get("overfill_um", 0.1)),
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

    def summary(self) -> str:
        sc = "" if self.scallop_um <= 0 else f"  scallop{self.scallop_um:.2f}µm"
        return f"DRIE  {materials.get(self.target).label}  深さ{self.depth_um:.2f}µm{sc}"

    def apply(self, wafer: Wafer) -> None:
        _require_positive(self.depth_um, "深さ")
        grid = wafer.grid
        nz, ny, nx = grid.shape
        target_id = materials.get(self.target).id
        depth = wafer.um_to_vox(self.depth_um)

        # 初期に露出しているターゲット列のみ（レジスト等が無い箇所）
        z_top0, top_id0 = _top_material(wafer)
        opening = (z_top0 >= 0) & (top_id0 == target_id)
        if not opening.any():
            return

        amp = wafer.um_to_vox(self.scallop_um) if self.scallop_um > 0 else 0
        pitch = max(2, wafer.um_to_vox(self.scallop_pitch_um))
        struct = ndimage.generate_binary_structure(3, 1)

        for d in range(depth):
            z_top, top_id = _top_material(wafer)
            do = opening & (z_top >= 0) & (top_id == target_id)
            if not do.any():
                break
            ys, xs = np.nonzero(do)
            zz = z_top[ys, xs]
            grid[zz, ys, xs] = materials.AIR
            # スキャロップ: 周期の中央付近でその層だけ側壁を横へ膨らませる。
            # 直近に掘った 1 層分のみを横方向に拡張するため、深さに比例した
            # 累積的なテーパは生じず、垂直側壁に局所的な凹凸が付く。
            if amp > 0 and (d % pitch) == (pitch // 2):
                layer = np.zeros(grid.shape, dtype=bool)
                layer[zz, ys, xs] = True
                ring = ndimage.binary_dilation(layer, structure=struct, iterations=amp)
                bulge = ring & (grid == target_id)
                grid[bulge] = materials.AIR

    def params_dict(self) -> dict:
        return {
            "target": self.target,
            "depth_um": self.depth_um,
            "scallop_um": self.scallop_um,
            "scallop_pitch_um": self.scallop_pitch_um,
        }

    @classmethod
    def _from_params(cls, d: dict) -> DRIE:
        return cls(
            target=d.get("target", "silicon"),
            depth_um=float(d.get("depth_um", 2.0)),
            scallop_um=float(d.get("scallop_um", 0.0)),
            scallop_pitch_um=float(d.get("scallop_pitch_um", 0.5)),
        )


def available_types() -> list[tuple[str, str]]:
    """(type, label) のリストを表示順で返す。"""
    order = [
        "PHOTO", "CVD", "PVD", "EPI",
        "DRY", "WET", "KOH", "DRIE",
        "DIFFUSION", "IMPLANT", "ANNEAL", "OXIDE",
        "FILL", "CMP", "LIFTOFF", "STRIP",
    ]
    out = []
    for t in order:
        cls = _REGISTRY.get(t)
        if cls is not None:
            out.append((t, cls.label))
    return out
