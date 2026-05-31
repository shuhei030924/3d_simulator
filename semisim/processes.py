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
_REGISTRY: dict[str, type["Process"]] = {}


def register(cls: type["Process"]) -> type["Process"]:
    _REGISTRY[cls.type] = cls
    return cls


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
    def from_dict(d: dict) -> "Process":
        t = d.get("type")
        cls = _REGISTRY.get(t)
        if cls is None:
            raise ValueError(f"未知の工程タイプ: {t}")
        return cls._from_params(d)

    @classmethod
    def _from_params(cls, d: dict) -> "Process":  # pragma: no cover - 抽象
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
    def _from_params(cls, d: dict) -> "Photo":
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
    def _from_params(cls, d: dict) -> "CVD":
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
            # 窪みが深いほど被覆率を sc に向けて低下させる
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
    def _from_params(cls, d: dict) -> "PVD":
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
    def _from_params(cls, d: dict) -> "DryEtch":
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
        grid = wafer.grid
        target_ids = _resolve_targets(self.targets)
        r = wafer.um_to_vox(self.depth_um)
        # 露出面から 1 ボクセルずつ等方的に後退させる。
        # 空気に隣接するターゲットのみを毎回除去するため、間に別材料が
        # あれば貫通せず、マスク下のアンダーカットも自然に再現される。
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
    def _from_params(cls, d: dict) -> "WetEtch":
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

        # 横方向の広がり（拡散の等方性を簡易表現）
        lat = max(0, depth // 3)
        if lat > 0 and converted.any():
            grown = ndimage.binary_dilation(converted, iterations=lat)
            spread = grown & (grid == si_id)
            grid[spread] = dop_id

    def params_dict(self) -> dict:
        return {"dopant": self.dopant, "depth_um": self.depth_um}

    @classmethod
    def _from_params(cls, d: dict) -> "Diffusion":
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
    def _from_params(cls, d: dict) -> "Strip":
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
    def _from_params(cls, d: dict) -> "CMP":
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
        grid = wafer.grid
        nz, ny, nx = grid.shape
        si_id = materials.BY_NAME["silicon"].id
        ox_id = materials.BY_NAME["oxide"].id
        total = wafer.um_to_vox(self.thickness_um)
        # 熱酸化は約 45% がシリコン側へ、55% が上方へ成長する。
        consume = max(0, int(round(total * 0.45)))
        grow = max(1, total - consume)

        # 露出シリコン列のみ酸化
        _, top_id = _top_material(wafer)
        eligible = top_id == si_id  # (ny, nx)

        # 1) シリコンを上から consume 分だけ酸化膜に変換
        work_top = wafer.top_surface_z().copy()
        for _ in range(consume):
            ys, xs = np.nonzero(eligible & (work_top >= 0))
            if ys.size == 0:
                break
            zt = work_top[ys, xs]
            is_si = grid[zt, ys, xs] == si_id
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
    def _from_params(cls, d: dict) -> "Oxidation":
        return cls(thickness_um=float(d.get("thickness_um", 0.3)))


def available_types() -> list[tuple[str, str]]:
    """(type, label) のリストを表示順で返す。"""
    order = ["PHOTO", "CVD", "PVD", "DRY", "WET", "DIFFUSION", "OXIDE", "CMP", "STRIP"]
    out = []
    for t in order:
        cls = _REGISTRY.get(t)
        if cls is not None:
            out.append((t, cls.label))
    return out
