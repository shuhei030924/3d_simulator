"""基本工程と材料保存則のテスト。"""
from __future__ import annotations

from semisim import materials
from semisim.grid import Wafer
from semisim.masks import Mask, Shape
from semisim.processes import (
    CMP,
    CVD,
    PVD,
    Diffusion,
    DryEtch,
    Oxidation,
    Photo,
    Strip,
    WetEtch,
)


def solid_count(w: Wafer) -> int:
    return int((w.grid != materials.AIR).sum())


def test_cvd_adds_conformal_film(wafer):
    before = solid_count(wafer)
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    after = solid_count(wafer)
    assert after > before
    # 酸化膜が上面に乗っている
    assert (wafer.grid == materials.get("oxide").id).any()


def test_pvd_step_coverage_reduces_sidewall(wafer):
    PVD(material="metal_al", thickness_um=0.3, step_coverage=0.5).apply(wafer)
    assert (wafer.grid == materials.get("metal_al").id).any()


def test_photo_positive_opening(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    Photo(mask=mask, thickness_um=0.6, polarity="positive").apply(wafer)
    resist_id = materials.get("photoresist").id
    cols = (wafer.grid == resist_id).any(axis=0)  # (ny, nx)
    # 開口部（中央）にはレジストが無く、周辺にはある
    assert not cols[20, 20]
    assert cols[2, 2]


def test_dry_etch_removes_target(wafer):
    CVD(material="oxide", thickness_um=0.4).apply(wafer)
    before = int((wafer.grid == materials.get("oxide").id).sum())
    DryEtch(targets=["oxide"], depth_um=0.4).apply(wafer)
    after = int((wafer.grid == materials.get("oxide").id).sum())
    assert after < before


def test_wet_etch_removes_target(wafer):
    PVD(material="metal_al", thickness_um=0.4).apply(wafer)
    before = int((wafer.grid == materials.get("metal_al").id).sum())
    WetEtch(targets=["metal_al"], depth_um=0.2).apply(wafer)
    after = int((wafer.grid == materials.get("metal_al").id).sum())
    assert after < before


def test_strip_removes_resist(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    Photo(mask=mask, thickness_um=0.6, polarity="positive").apply(wafer)
    assert (wafer.grid == materials.get("photoresist").id).any()
    Strip(material="photoresist").apply(wafer)
    assert not (wafer.grid == materials.get("photoresist").id).any()


def test_diffusion_creates_doped_region(wafer):
    Diffusion(dopant="doped_n", depth_um=0.5).apply(wafer)
    assert (wafer.grid == materials.get("doped_n").id).any()


def test_diffusion_lateral_factor_widens(wafer):
    from semisim.recipe import Recipe

    base = Recipe(config=wafer.config)
    base.add(Diffusion(dopant="doped_n", depth_um=0.5, lateral_factor=0.0))
    narrow = base.simulate()
    wide_r = Recipe(config=wafer.config)
    wide_r.add(Diffusion(dopant="doped_n", depth_um=0.5, lateral_factor=1.0))
    wide = wide_r.simulate()
    dop = materials.get("doped_n").id
    # 横方向係数を上げると拡散層のボクセル数が増える
    assert int((wide.grid == dop).sum()) > int((narrow.grid == dop).sum())


def test_diffusion_lateral_factor_roundtrip():
    d = Diffusion(dopant="doped_p", depth_um=0.4, lateral_factor=0.5)
    restored = Diffusion._from_params(d.params_dict())
    assert restored.lateral_factor == 0.5
    assert restored.dopant == "doped_p"


def test_oxidation_consumes_silicon(wafer):
    si_before = int((wafer.grid == materials.get("silicon").id).sum())
    Oxidation(thickness_um=0.3).apply(wafer)
    si_after = int((wafer.grid == materials.get("silicon").id).sum())
    # 熱酸化はシリコンを消費する
    assert si_after < si_before
    assert (wafer.grid == materials.get("oxide").id).any()


def test_cmp_flattens_top(wafer):
    # 段差を作る
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    Photo(mask=mask, thickness_um=1.0, polarity="negative").apply(wafer)
    CVD(material="oxide", thickness_um=0.5).apply(wafer)
    z_before = wafer.top_surface_z()
    step_before = int(z_before.max() - z_before[z_before >= 0].min())
    # 段差より大きく研磨すれば全面が同一平面まで削られて平坦になる
    CMP(remove_um=2.0).apply(wafer)
    z_top = wafer.top_surface_z()
    flat = z_top[z_top >= 0]
    assert int(flat.max() - flat.min()) < step_before
    assert int(flat.max() - flat.min()) <= 1
