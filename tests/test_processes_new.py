"""新規工程（IMPLANT/ANNEAL/EPI/KOH/DRIE/FILL/LIFTOFF）のテスト。"""
from __future__ import annotations

from semisim import materials
from semisim.masks import Mask, Shape
from semisim.processes import (
    DRIE,
    PVD,
    AnisoWetEtch,
    Anneal,
    Epitaxy,
    Fill,
    Implant,
    LiftOff,
    Photo,
)


def test_implant_buried_layer(wafer):
    Implant(dopant="doped_n", range_um=0.4, straggle_um=0.1).apply(wafer)
    dop_id = materials.get("doped_n").id
    assert (wafer.grid == dop_id).any()
    # 埋込層は表面より下にある（最上面はシリコンのまま）
    z_top = wafer.top_surface_z()
    cx = wafer.config.nx // 2
    cy = wafer.config.ny // 2
    top_id = wafer.grid[z_top[cy, cx], cy, cx]
    assert top_id == materials.get("silicon").id


def test_implant_blocked_by_resist(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    # negative: 選択部にレジストが残る → 中央を保護
    Photo(mask=mask, thickness_um=0.6, polarity="negative").apply(wafer)
    Implant(dopant="doped_n", range_um=0.3).apply(wafer)
    dop = wafer.grid == materials.get("doped_n").id
    # レジスト直下（中央列）にはドープされない
    assert dop[:, 20, 20].sum() == 0
    # レジストの無い周辺はドープされる
    assert dop[:, 2, 2].sum() > 0


def test_anneal_spreads_dopant(wafer):
    from semisim.processes import Diffusion

    Diffusion(dopant="doped_n", depth_um=0.3).apply(wafer)
    before = int((wafer.grid == materials.get("doped_n").id).sum())
    Anneal(depth_um=0.3).apply(wafer)
    after = int((wafer.grid == materials.get("doped_n").id).sum())
    assert after > before


def test_epitaxy_grows_on_silicon(wafer):
    epi_id = materials.get("epi_si").id
    before = int((wafer.grid == epi_id).sum())
    Epitaxy(material="epi_si", thickness_um=0.4).apply(wafer)
    after = int((wafer.grid == epi_id).sum())
    assert after > before


def test_epitaxy_selective_no_growth_on_oxide(wafer):
    from semisim.processes import CVD

    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    Epitaxy(material="epi_si", thickness_um=0.4).apply(wafer)
    # 酸化膜で覆われているのでエピは成長しない
    assert int((wafer.grid == materials.get("epi_si").id).sum()) == 0


def test_koh_anisotropic_etch_makes_groove(wafer):
    target_before = int((wafer.grid == materials.get("silicon").id).sum())
    AnisoWetEtch(target="silicon", depth_um=0.8, sidewall_angle_deg=54.7).apply(wafer)
    target_after = int((wafer.grid == materials.get("silicon").id).sum())
    assert target_after < target_before
    # 斜め側壁の溝が形成されている（空気ボクセルが増える）
    air = wafer.grid == materials.AIR
    assert air.any()


def test_drie_deep_vertical(wafer):
    before = int((wafer.grid == materials.get("silicon").id).sum())
    DRIE(target="silicon", depth_um=1.5, scallop_um=0.0).apply(wafer)
    after = int((wafer.grid == materials.get("silicon").id).sum())
    assert after < before


def test_drie_with_scallop(wafer):
    DRIE(target="silicon", depth_um=1.5, scallop_um=0.1, scallop_pitch_um=0.3).apply(wafer)
    # スキャロップ有りでもエラーなくエッチされる
    assert (wafer.grid == materials.AIR).any()


def test_fill_bottom_up(wafer):
    # トレンチを掘る
    DRIE(target="silicon", depth_um=1.0).apply(wafer)
    cu_id = materials.get("metal_cu").id
    Fill(material="metal_cu", overfill_um=0.1).apply(wafer)
    assert (wafer.grid == cu_id).any()


def test_liftoff_removes_resist_and_overlayer(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    Photo(mask=mask, thickness_um=0.6, polarity="negative").apply(wafer)
    PVD(material="metal_al", thickness_um=0.3).apply(wafer)
    LiftOff().apply(wafer)
    # レジストは消える
    assert not (wafer.grid == materials.get("photoresist").id).any()
    # 基板上に直接堆積した金属は残る
    assert (wafer.grid == materials.get("metal_al").id).any()
