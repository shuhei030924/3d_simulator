"""材料（マテリアル）の定義。

各材料は一意の整数 ID を持ち、ボクセルグリッドに格納されます。
ID 0 は必ず「空気/真空（air）」を表します。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """1 つの材料を表す。

    Attributes:
        id: グリッドに格納される整数 ID（0 は air 固定）。
        name: 内部識別名（レシピ JSON でも使用）。
        label: 表示用の名称。
        color: RGB (0-1) の表示色。
        opacity: 表示時の不透明度（0-1）。
        etchable: エッチングで除去可能か。
        is_resist: フォトレジストかどうか（マスクとして機能）。
    """

    id: int
    name: str
    label: str
    color: tuple[float, float, float]
    opacity: float = 1.0
    etchable: bool = True
    is_resist: bool = False


# --- 標準材料テーブル -------------------------------------------------------
# 半導体プロセスで頻出する材料を定義。色は断面で識別しやすいものを選択。
_MATERIALS: list[Material] = [
    Material(0, "air", "空気/真空", (1.0, 1.0, 1.0), opacity=0.0, etchable=False),
    Material(1, "silicon", "シリコン基板", (0.45, 0.45, 0.50), etchable=False),
    Material(2, "oxide", "酸化膜 (SiO2)", (0.40, 0.78, 0.95)),
    Material(3, "poly", "ポリシリコン", (0.80, 0.25, 0.25)),
    Material(4, "nitride", "窒化膜 (Si3N4)", (0.30, 0.72, 0.40)),
    Material(5, "photoresist", "フォトレジスト", (0.95, 0.82, 0.25), opacity=0.85, is_resist=True),
    Material(6, "metal_al", "金属 (Al)", (0.82, 0.82, 0.88)),
    Material(7, "metal_cu", "金属 (Cu)", (0.85, 0.52, 0.25)),
    Material(8, "tungsten", "タングステン (W)", (0.55, 0.55, 0.60)),
    Material(9, "doped_n", "n型拡散層", (0.30, 0.45, 0.85)),
    Material(10, "doped_p", "p型拡散層", (0.85, 0.35, 0.55)),
    Material(11, "tin", "バリア (TiN)", (0.65, 0.62, 0.45)),
    Material(12, "low_k", "Low-k 絶縁膜", (0.55, 0.80, 0.78)),
    Material(13, "epi_si", "エピ層 (Si)", (0.55, 0.55, 0.62), etchable=False),
    Material(14, "hafnia", "High-k (HfO2)", (0.72, 0.45, 0.80)),
    Material(15, "tan", "バリア (TaN)", (0.50, 0.48, 0.55)),
    Material(16, "silicide", "シリサイド (NiSi)", (0.78, 0.70, 0.30)),
]

# 名前 / ID での高速参照
BY_NAME: dict[str, Material] = {m.name: m for m in _MATERIALS}
BY_ID: dict[int, Material] = {m.id: m for m in _MATERIALS}

AIR = 0


def all_materials() -> list[Material]:
    """全材料のリストを返す。"""
    return list(_MATERIALS)


def deposit_materials() -> list[Material]:
    """成膜（CVD/PVD）で選択可能な材料を返す。"""
    skip = {"air", "silicon", "doped_n", "doped_p", "epi_si"}
    return [m for m in _MATERIALS if m.name not in skip]


def get(name_or_id) -> Material:
    """名前または ID から Material を取得する。

    未知の名前 / ID の場合は、原因が分かる ValueError を投げる。
    """
    if isinstance(name_or_id, str):
        try:
            return BY_NAME[name_or_id]
        except KeyError as exc:
            known = ", ".join(sorted(BY_NAME))
            raise ValueError(
                f"未知の材料名: {name_or_id!r}（利用可能: {known}）"
            ) from exc
    try:
        return BY_ID[int(name_or_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"未知の材料 ID: {name_or_id!r}") from exc


def color_lookup() -> tuple[list[tuple[float, float, float]], list[float]]:
    """ID 順に並んだ (色リスト, 不透明度リスト) を返す。

    ID は連番ではない可能性があるため、最大 ID までを埋める。
    """
    max_id = max(BY_ID)
    colors: list[tuple[float, float, float]] = []
    opacities: list[float] = []
    for i in range(max_id + 1):
        m = BY_ID.get(i)
        if m is None:
            colors.append((0.0, 0.0, 0.0))
            opacities.append(0.0)
        else:
            colors.append(m.color)
            opacities.append(m.opacity)
    return colors, opacities
