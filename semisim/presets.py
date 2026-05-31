"""組み込みレシピのプリセット集（レシピライブラリ）。

GUI のメニューから即座に読み込める代表的な半導体プロセスフロー。
各プリセットは関数で Recipe を生成する。GUI 非依存なのでテスト可能。

ギャラリー(tools/render_gallery.py)は高解像度版を別途持つが、ここでは
対話操作で軽快に動く中解像度の設定を既定にする。
"""
from __future__ import annotations

from .grid import WaferConfig
from .masks import Mask, Shape
from .processes import (
    ALD,
    CMP,
    CVD,
    DRIE,
    PVD,
    AnisoWetEtch,
    Anneal,
    Backgrind,
    Diffusion,
    DryEtch,
    Epitaxy,
    Fill,
    Implant,
    LiftOff,
    Oxidation,
    Photo,
    Silicidation,
    Spacer,
    Strip,
)
from .recipe import Recipe


def _default_cfg() -> WaferConfig:
    """対話操作向けの中解像度設定。"""
    return WaferConfig(nx=120, ny=120, nz=140, pitch_um=0.06, substrate_um=3.0)


def _center_mask(w: float = 0.4) -> Mask:
    a = (1.0 - w) / 2.0
    b = a + w
    return Mask(shapes=[Shape("rect", {"x0": a, "y0": a, "x1": b, "y1": b})])


def _stripe_mask(period: float = 0.3, width: float = 0.15) -> Mask:
    return Mask(shapes=[Shape("grating", {"angle": 90.0, "period": period, "width": width})])


def implant_buried_layer() -> Recipe:
    """イオン注入による埋込ドープ層。"""
    r = Recipe(config=_default_cfg())
    r.add(Photo(mask=_center_mask(0.4), thickness_um=1.0, polarity="negative"))
    r.add(Implant(dopant="doped_n", range_um=0.8, straggle_um=0.2))
    r.add(Strip(material="photoresist"))
    return r


def anneal_drivein() -> Recipe:
    """拡散 → アニールによる等方ドライブイン。"""
    r = Recipe(config=_default_cfg())
    r.add(Photo(mask=_center_mask(0.3), thickness_um=1.0, polarity="positive"))
    r.add(Diffusion(dopant="doped_p", depth_um=0.6))
    r.add(Strip(material="photoresist"))
    r.add(Anneal(depth_um=0.6))
    return r


def selective_epitaxy() -> Recipe:
    """酸化膜開口部の露出 Si 上のみ選択エピ成長。"""
    r = Recipe(config=_default_cfg())
    r.add(CVD(material="oxide", thickness_um=0.4))
    r.add(Photo(mask=_center_mask(0.4), thickness_um=1.0, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.5))
    r.add(Strip(material="photoresist"))
    r.add(Epitaxy(material="epi_si", thickness_um=0.8))
    return r


def koh_vgroove() -> Recipe:
    """KOH 異方性エッチによる V 溝（54.7°側壁）。"""
    r = Recipe(config=_default_cfg())
    r.add(CVD(material="nitride", thickness_um=0.3))
    r.add(Photo(mask=_center_mask(0.5), thickness_um=1.0, polarity="positive"))
    r.add(DryEtch(targets=["nitride"], depth_um=0.4))
    r.add(Strip(material="photoresist"))
    r.add(AnisoWetEtch(target="silicon", depth_um=2.0, sidewall_angle_deg=54.7))
    return r


def drie_deep_trench() -> Recipe:
    """Bosch プロセス（DRIE）による高アスペクト比の深掘り。"""
    r = Recipe(config=_default_cfg())
    r.add(Photo(mask=_stripe_mask(0.3, 0.15), thickness_um=1.2, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=3.0, scallop_um=0.15, scallop_pitch_um=0.4))
    r.add(Strip(material="photoresist"))
    return r


def cu_damascene() -> Recipe:
    """Cu デュアルダマシン配線（バリア＋充填＋CMP）。"""
    r = Recipe(config=_default_cfg())
    r.add(CVD(material="low_k", thickness_um=1.2))
    r.add(Photo(mask=_stripe_mask(0.35, 0.18), thickness_um=1.2, polarity="positive"))
    r.add(DryEtch(targets=["low_k"], depth_um=1.0))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="tin", thickness_um=0.1))
    r.add(Fill(material="metal_cu", overfill_um=0.3))
    r.add(CMP(remove_um=0.5))
    return r


def metal_liftoff() -> Recipe:
    """リフトオフによる金属ラインのパターニング。"""
    r = Recipe(config=_default_cfg())
    r.add(Photo(mask=_stripe_mask(0.35, 0.18), thickness_um=0.8, polarity="negative"))
    r.add(PVD(material="metal_al", thickness_um=0.4, step_coverage=0.7))
    r.add(LiftOff())
    return r


def mosfet_flow() -> Recipe:
    """簡易 MOSFET 風フロー（ゲート積層＋自己整合 S/D）。"""
    r = Recipe(config=_default_cfg())
    r.add(CVD(material="oxide", thickness_um=0.2))
    r.add(CVD(material="metal_al", thickness_um=0.4))
    r.add(Photo(mask=_center_mask(0.3), thickness_um=1.0, polarity="negative"))
    r.add(DryEtch(targets=["metal_al", "oxide"], depth_um=0.7))
    r.add(Strip(material="photoresist"))
    r.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.12))
    r.add(Anneal(depth_um=0.3))
    return r


def tsv_flow() -> Recipe:
    """TSV（シリコン貫通ビア）: 深掘り→ライナー→バリア→Cu充填→CMP。"""
    r = Recipe(config=WaferConfig(nx=120, ny=120, nz=160, pitch_um=0.06, substrate_um=4.0))
    r.add(Photo(mask=_center_mask(0.25), thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=3.0, scallop_um=0.1, scallop_pitch_um=0.4))
    r.add(Strip(material="photoresist"))
    r.add(CVD(material="oxide", thickness_um=0.15))  # 絶縁ライナー
    r.add(ALD(material="tan", cycles=80, growth_per_cycle_nm=1.0))  # バリア
    r.add(Fill(material="metal_cu", overfill_um=0.4))
    r.add(CMP(remove_um=0.6, stop_material="oxide"))
    return r


def multilevel_interconnect() -> Recipe:
    """多層 Cu 配線（M1/Via/M2）: ダマシンを 3 段繰り返す先端ノード相当フロー。

    各層は 層間絶縁膜(low_k) 成膜 → パターニング → バリア(TaN) →
    Cu 充填 → CMP 平坦化（Cu ディッシング込み）で構成する。
    DryEtch には酸化膜ストップ層相当の選択比を持たせ、下層を保護する。
    """
    r = Recipe(config=WaferConfig(nx=120, ny=120, nz=180, pitch_um=0.06, substrate_um=2.0))
    r.name = "多層 Cu 配線 (M1/Via/M2)"
    r.description = "ダマシンを3段繰り返す多層配線フロー"
    masks = (_stripe_mask(0.40, 0.20), _center_mask(0.25), _stripe_mask(0.40, 0.20))
    for mask in masks:
        r.add(CVD(material="oxide", thickness_um=0.1))  # エッチストップ層（下地）
        r.add(CVD(material="low_k", thickness_um=0.8))  # 層間絶縁膜
        r.add(Photo(mask=mask, thickness_um=0.8, polarity="positive"))
        # low_k を選択的に削り、下地 oxide で速度を落として（選択比0.1）ストップ
        r.add(DryEtch(
            targets=["low_k"], depth_um=0.8, overetch_pct=20.0,
            selectivity={"oxide": 0.1},
        ))
        r.add(Strip(material="photoresist"))
        r.add(ALD(material="tan", cycles=60, growth_per_cycle_nm=1.0))  # バリア
        r.add(Fill(material="metal_cu", overfill_um=0.3))
        r.add(CMP(remove_um=0.4, soft_material="metal_cu", dishing_um=0.05))
    return r


def salicide_gate_flow() -> Recipe:
    """自己整合シリサイド付きゲート: ゲート酸化→ポリゲート→S/D注入→SALICIDE。

    ゲートポリと露出ソース/ドレインのシリコン上にのみシリサイドが自己整合で
    形成され、ゲート酸化膜で覆われた領域には付かない様子を再現する。
    """
    r = Recipe(config=_default_cfg())
    r.name = "サリサイド ゲート"
    r.description = "自己整合シリサイド形成を含むゲートフロー"
    r.add(Oxidation(thickness_um=0.15))  # ゲート酸化膜
    r.add(CVD(material="poly", thickness_um=0.4))  # ゲートポリ
    r.add(Photo(mask=_center_mask(0.3), thickness_um=1.0, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=0.4))  # ゲート整形
    r.add(Strip(material="photoresist"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.15))  # S/D 上のゲート酸化膜除去→Si露出
    r.add(Implant(dopant="doped_n", range_um=0.3, straggle_um=0.1))  # S/D 注入
    r.add(Anneal(depth_um=0.2))  # 活性化
    r.add(Silicidation(thickness_um=0.06))  # 自己整合シリサイド
    return r


def thinned_3dic_flow() -> Recipe:
    """3D-IC 向け薄化フロー: 配線形成→裏面研削でウェハ薄化。

    表面に金属配線を作った後、裏面研削で基板を大きく薄くする。デバイス層は
    保護され、基板のみが薄くなる（積層実装/TSV 露出の前工程相当）。
    """
    r = Recipe(config=WaferConfig(nx=120, ny=120, nz=160, pitch_um=0.06, substrate_um=6.0))
    r.name = "薄化 3D-IC"
    r.description = "配線形成後に裏面研削でウェハを薄化する 3D-IC フロー"
    r.add(CVD(material="oxide", thickness_um=0.3))
    r.add(PVD(material="metal_al", thickness_um=0.5))
    r.add(Photo(mask=_stripe_mask(0.4, 0.2), thickness_um=1.0, polarity="positive"))
    r.add(DryEtch(targets=["metal_al"], depth_um=0.5))
    r.add(Strip(material="photoresist"))
    r.add(CVD(material="oxide", thickness_um=0.4))  # パッシベーション
    r.add(Backgrind(thin_um=4.0))  # 裏面研削でウェハ薄化
    return r


def ldd_mosfet_flow() -> Recipe:
    """LDD MOSFET: ゲート→LDD注入→側壁スペーサ→S/D注入→サリサイド。

    ゲート形成後に浅い LDD（低濃度ドレイン）を自己整合注入し、窒化膜の
    サイドウォールスペーサを形成。スペーサをマスクに深い S/D を注入する
    ことで、ゲート端のホットキャリア劣化を抑える現代 CMOS の代表フロー。
    """
    r = Recipe(config=_default_cfg())
    r.name = "LDD MOSFET"
    r.description = "サイドウォールスペーサを用いた LDD MOSFET フロー"
    r.add(Oxidation(thickness_um=0.12))  # ゲート酸化膜
    r.add(CVD(material="poly", thickness_um=0.4))  # ゲートポリ
    r.add(Photo(mask=_center_mask(0.25), thickness_um=1.0, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=0.4))  # ゲート整形
    r.add(Strip(material="photoresist"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.12))  # S/D 上の酸化膜除去→Si露出
    r.add(Implant(dopant="doped_n", range_um=0.12, straggle_um=0.05))  # 浅い LDD
    r.add(Spacer(material="nitride", thickness_um=0.1))  # 側壁スペーサ
    r.add(Implant(dopant="doped_n", range_um=0.35, straggle_um=0.12))  # 深い S/D
    r.add(Anneal(depth_um=0.2))  # 活性化
    r.add(Silicidation(thickness_um=0.05))  # 自己整合シリサイド
    return r


# 表示ラベル -> 生成関数（GUI メニュー/テストで使用）
PRESETS: dict[str, callable] = {
    "イオン注入 埋込層": implant_buried_layer,
    "拡散＋アニール": anneal_drivein,
    "選択エピ成長": selective_epitaxy,
    "KOH V溝": koh_vgroove,
    "DRIE 深掘り": drie_deep_trench,
    "Cu ダマシン配線": cu_damascene,
    "金属リフトオフ": metal_liftoff,
    "MOSFET フロー": mosfet_flow,
    "サリサイド ゲート": salicide_gate_flow,
    "LDD MOSFET": ldd_mosfet_flow,
    "TSV 貫通ビア": tsv_flow,
    "多層 Cu 配線": multilevel_interconnect,
    "薄化 3D-IC": thinned_3dic_flow,
}


def available() -> list[str]:
    """利用可能なプリセット名の一覧。"""
    return list(PRESETS.keys())


def build(name: str) -> Recipe:
    """プリセット名から Recipe を生成する。"""
    if name not in PRESETS:
        raise KeyError(f"未知のプリセット: {name}")
    return PRESETS[name]()
