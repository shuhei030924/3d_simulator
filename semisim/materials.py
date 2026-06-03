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
        stress_mpa: 残留膜応力（MPa, 引張+ / 圧縮-）。反り計算に使用。
        resistivity_ohm_um: 電気抵抗率（Ω·µm）。配線抵抗推定に使用。0=非導体扱い。
        rel_permittivity: 比誘電率 εr（無次元）。寄生容量/RC 推定に使用。
            導体は 0（誘電体として扱わない）。空気/真空は 1.0。0=未設定（容量計算で 1.0 扱い）。
        em_jmax_a_cm2: エレクトロマイグレーション許容電流密度（A/cm²）。配線の
            EM 信頼性判定に使用。導体のみ意味を持つ。0=未設定（判定対象外）。
        breakdown_field_mv_cm: 絶縁破壊電界（MV/cm）。誘電体の絶縁破壊判定に使用。
            0=未設定（破壊判定対象外）。
        thermal_conductivity_w_mk: 熱伝導率（W/m·K）。スタック熱抵抗の計算に使用。
            0=未設定（熱抵抗計算で空気相当 0.026 扱い）。
        tcr_per_k: 抵抗温度係数（1/K）。ρ(T)=ρ0·(1+TCR·(T−T0)) の温度依存抵抗に使用。
            金属は正（~0.004/K）。0=未設定（温度依存なし）。
        volumetric_heat_capacity_j_m3k: 体積熱容量 ρ·c_p（J/m³·K）。熱時定数
            τ_th=R_th·C_th の過渡熱応答に使用。0=未設定（熱容量計算で空気相当
            ~1.2e3 扱い）。
        cte_ppm_k: 線膨張係数 CTE（ppm/K）。CTE 不整合による熱応力
            σ=E/(1−ν)·Δα·ΔT の評価に使用。0=未設定。
        youngs_modulus_gpa: ヤング率（GPa）。熱応力・機械応力の評価に使用。
            0=未設定（熱応力対象外）。
        refractive_index_n: 屈折率の実部 n（~633nm 目安）。薄膜の垂直入射反射率
            （TMM）に使用。0=未設定（空気 n=1.0 扱い）。
        extinction_k: 消光係数 k（屈折率の虚部）。金属/吸収膜の反射率に使用。
            0=透明（誘電体）。
    """

    id: int
    name: str
    label: str
    color: tuple[float, float, float]
    opacity: float = 1.0
    etchable: bool = True
    is_resist: bool = False
    stress_mpa: float = 0.0
    resistivity_ohm_um: float = 0.0
    rel_permittivity: float = 0.0
    em_jmax_a_cm2: float = 0.0
    breakdown_field_mv_cm: float = 0.0
    thermal_conductivity_w_mk: float = 0.0
    tcr_per_k: float = 0.0
    volumetric_heat_capacity_j_m3k: float = 0.0
    cte_ppm_k: float = 0.0
    youngs_modulus_gpa: float = 0.0
    refractive_index_n: float = 0.0
    extinction_k: float = 0.0


# --- 標準材料テーブル -------------------------------------------------------
# 半導体プロセスで頻出する材料を定義。色は断面で識別しやすいものを選択。
_MATERIALS: list[Material] = [
    Material(0, "air", "空気/真空", (1.0, 1.0, 1.0), opacity=0.0, etchable=False,
             rel_permittivity=1.0, breakdown_field_mv_cm=0.03,
             thermal_conductivity_w_mk=0.026, volumetric_heat_capacity_j_m3k=1.2e3),
    Material(1, "silicon", "シリコン基板", (0.45, 0.45, 0.50), etchable=False,
             rel_permittivity=11.7, thermal_conductivity_w_mk=150.0,
             volumetric_heat_capacity_j_m3k=1.63e6, cte_ppm_k=2.6,
             youngs_modulus_gpa=165.0, refractive_index_n=3.88, extinction_k=0.02),
    Material(2, "oxide", "酸化膜 (SiO2)", (0.40, 0.78, 0.95), stress_mpa=-300.0,
             rel_permittivity=3.9, breakdown_field_mv_cm=10.0,
             thermal_conductivity_w_mk=1.4, volumetric_heat_capacity_j_m3k=1.62e6,
             cte_ppm_k=0.5, youngs_modulus_gpa=70.0, refractive_index_n=1.46),
    Material(3, "poly", "ポリシリコン", (0.80, 0.25, 0.25), stress_mpa=-200.0,
             thermal_conductivity_w_mk=30.0, volumetric_heat_capacity_j_m3k=1.63e6,
             cte_ppm_k=2.6, youngs_modulus_gpa=160.0, refractive_index_n=3.88,
             extinction_k=0.02),
    Material(4, "nitride", "窒化膜 (Si3N4)", (0.30, 0.72, 0.40), stress_mpa=1000.0,
             rel_permittivity=7.5, breakdown_field_mv_cm=10.0,
             thermal_conductivity_w_mk=30.0, volumetric_heat_capacity_j_m3k=2.1e6,
             cte_ppm_k=3.3, youngs_modulus_gpa=250.0, refractive_index_n=2.0),
    Material(5, "photoresist", "フォトレジスト", (0.95, 0.82, 0.25), opacity=0.85, is_resist=True,
             rel_permittivity=3.0, thermal_conductivity_w_mk=0.2,
             volumetric_heat_capacity_j_m3k=1.5e6, cte_ppm_k=50.0,
             youngs_modulus_gpa=5.0, refractive_index_n=1.7),
    Material(6, "metal_al", "金属 (Al)", (0.82, 0.82, 0.88), stress_mpa=100.0,
             resistivity_ohm_um=0.0265, em_jmax_a_cm2=2.0e5,
             thermal_conductivity_w_mk=237.0, tcr_per_k=0.0043,
             volumetric_heat_capacity_j_m3k=2.42e6, cte_ppm_k=23.1,
             youngs_modulus_gpa=70.0, refractive_index_n=1.37, extinction_k=7.62),
    Material(7, "metal_cu", "金属 (Cu)", (0.85, 0.52, 0.25), stress_mpa=200.0,
             resistivity_ohm_um=0.0168, em_jmax_a_cm2=2.0e6,
             thermal_conductivity_w_mk=400.0, tcr_per_k=0.0039,
             volumetric_heat_capacity_j_m3k=3.45e6, cte_ppm_k=16.5,
             youngs_modulus_gpa=130.0, refractive_index_n=0.62, extinction_k=2.82),
    Material(8, "tungsten", "タングステン (W)", (0.55, 0.55, 0.60), stress_mpa=1200.0,
             resistivity_ohm_um=0.056, em_jmax_a_cm2=1.0e7,
             thermal_conductivity_w_mk=170.0, tcr_per_k=0.0045,
             volumetric_heat_capacity_j_m3k=2.58e6, cte_ppm_k=4.5,
             youngs_modulus_gpa=410.0, refractive_index_n=3.6, extinction_k=2.8),
    Material(9, "doped_n", "n型拡散層", (0.30, 0.45, 0.85), resistivity_ohm_um=1000.0,
             thermal_conductivity_w_mk=100.0, volumetric_heat_capacity_j_m3k=1.63e6,
             cte_ppm_k=2.6, youngs_modulus_gpa=165.0, refractive_index_n=3.88,
             extinction_k=0.02),
    Material(10, "doped_p", "p型拡散層", (0.85, 0.35, 0.55), resistivity_ohm_um=2000.0,
             thermal_conductivity_w_mk=100.0, volumetric_heat_capacity_j_m3k=1.63e6,
             cte_ppm_k=2.6, youngs_modulus_gpa=165.0, refractive_index_n=3.88,
             extinction_k=0.02),
    Material(11, "tin", "バリア (TiN)", (0.65, 0.62, 0.45), stress_mpa=-500.0,
             resistivity_ohm_um=0.25, thermal_conductivity_w_mk=30.0,
             volumetric_heat_capacity_j_m3k=3.2e6, cte_ppm_k=9.4,
             youngs_modulus_gpa=600.0, refractive_index_n=1.5, extinction_k=2.6),
    Material(12, "low_k", "Low-k 絶縁膜", (0.55, 0.80, 0.78), stress_mpa=-60.0,
             rel_permittivity=2.5, breakdown_field_mv_cm=4.0,
             thermal_conductivity_w_mk=0.3, volumetric_heat_capacity_j_m3k=1.5e6,
             cte_ppm_k=20.0, youngs_modulus_gpa=10.0, refractive_index_n=1.4),
    Material(13, "epi_si", "エピ層 (Si)", (0.55, 0.55, 0.62), etchable=False,
             rel_permittivity=11.7, thermal_conductivity_w_mk=150.0,
             volumetric_heat_capacity_j_m3k=1.63e6, cte_ppm_k=2.6,
             youngs_modulus_gpa=165.0, refractive_index_n=3.88, extinction_k=0.02),
    Material(14, "hafnia", "High-k (HfO2)", (0.72, 0.45, 0.80), stress_mpa=500.0,
             rel_permittivity=25.0, breakdown_field_mv_cm=5.0,
             thermal_conductivity_w_mk=23.0, volumetric_heat_capacity_j_m3k=2.2e6,
             cte_ppm_k=5.3, youngs_modulus_gpa=220.0, refractive_index_n=2.07),
    Material(15, "tan", "バリア (TaN)", (0.50, 0.48, 0.55), stress_mpa=-1000.0,
             resistivity_ohm_um=2.5, thermal_conductivity_w_mk=12.0,
             volumetric_heat_capacity_j_m3k=2.7e6, cte_ppm_k=3.6,
             youngs_modulus_gpa=300.0, refractive_index_n=2.0, extinction_k=1.5),
    Material(16, "silicide", "シリサイド (NiSi)", (0.78, 0.70, 0.30), stress_mpa=500.0,
             resistivity_ohm_um=0.15, thermal_conductivity_w_mk=50.0,
             volumetric_heat_capacity_j_m3k=3.5e6, cte_ppm_k=12.0,
             youngs_modulus_gpa=130.0, refractive_index_n=3.0, extinction_k=3.5),
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
