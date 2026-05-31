"""新プロセス（SPUTTER/CLEAN/REFLOW）と入力検証のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import (
    CMP,
    CVD,
    PVD,
    AnisoWetEtch,
    Anneal,
    DryEtch,
    Implant,
    Oxidation,
    Photo,
    PlasmaClean,
    Reflow,
    SputterEtch,
)
from semisim.recipe import Recipe


# --- SPUTTER ---------------------------------------------------------------
def test_sputter_removes_top_material(wafer):
    CVD(material="oxide", thickness_um=0.5).apply(wafer)
    before = int((wafer.grid == materials.get("oxide").id).sum())
    SputterEtch(depth_um=0.3).apply(wafer)
    after = int((wafer.grid == materials.get("oxide").id).sum())
    assert after < before


def test_sputter_isotropic_widens_removal(wafer):
    cfg = wafer.config
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])

    def remaining(iso):
        r = Recipe(config=cfg)
        r.add(CVD(material="oxide", thickness_um=0.6))
        r.add(Photo(mask=mask, thickness_um=0.5, polarity="negative"))
        r.add(SputterEtch(depth_um=0.4, isotropic=iso))
        w = r.simulate()
        return int((w.grid == materials.get("oxide").id).sum())

    # 等方成分が大きいほど多く削れる
    assert remaining(0.8) < remaining(0.0)


def test_sputter_keeps_substrate_bottom(wafer):
    # 大きく削っても最下層（基板底）は残る
    SputterEtch(depth_um=50.0).apply(wafer)
    assert (wafer.grid[0] != materials.AIR).any()


def test_sputter_invalid_isotropic_raises(wafer):
    with pytest.raises(ValueError):
        SputterEtch(depth_um=0.3, isotropic=1.5).apply(wafer)


# --- CLEAN -----------------------------------------------------------------
def test_plasma_clean_thins_resist(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    Photo(mask=mask, thickness_um=0.5, polarity="negative").apply(wafer)
    before = int((wafer.grid == materials.get("photoresist").id).sum())
    PlasmaClean(target="photoresist", thickness_um=0.1).apply(wafer)
    after = int((wafer.grid == materials.get("photoresist").id).sum())
    assert 0 < after < before


def test_plasma_clean_no_target_is_noop(wafer):
    before = wafer.grid.copy()
    PlasmaClean(target="oxide", thickness_um=0.1).apply(wafer)
    assert np.array_equal(before, wafer.grid)


# --- REFLOW ----------------------------------------------------------------
def test_reflow_smooths_corners(wafer):
    # 角張ったレジストブロックをリフローして表面粗さを下げる
    from semisim import metrology

    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    Photo(mask=mask, thickness_um=1.0, polarity="negative").apply(wafer)
    rough_before = metrology.surface_roughness_um(wafer)
    Reflow(target="photoresist", radius_um=0.3).apply(wafer)
    rough_after = metrology.surface_roughness_um(wafer)
    # リフローで表面が滑らかになる（粗さが増えない）
    assert rough_after <= rough_before + 1e-9


def test_reflow_no_target_is_noop(wafer):
    before = wafer.grid.copy()
    Reflow(target="metal_cu", radius_um=0.2).apply(wafer)
    assert np.array_equal(before, wafer.grid)


def test_reflow_roundtrip():
    r = Reflow(target="low_k", radius_um=0.25)
    restored = Reflow._from_params(r.params_dict())
    assert restored.target == "low_k"
    assert restored.radius_um == 0.25


# --- 入力検証 --------------------------------------------------------------
def test_pvd_invalid_step_coverage_raises(wafer):
    with pytest.raises(ValueError):
        PVD(material="metal_al", thickness_um=0.3, step_coverage=2.0).apply(wafer)


def test_dryetch_negative_overetch_raises(wafer):
    with pytest.raises(ValueError):
        DryEtch(depth_um=0.3, overetch_pct=-5.0).apply(wafer)


def test_implant_invalid_threshold_raises(wafer):
    with pytest.raises(ValueError):
        Implant(range_um=0.3, threshold=1.5).apply(wafer)


def test_oxide_invalid_consume_fraction_raises(wafer):
    with pytest.raises(ValueError):
        Oxidation(thickness_um=0.3, consume_fraction=1.5).apply(wafer)


def test_koh_invalid_angle_raises(wafer):
    with pytest.raises(ValueError):
        AnisoWetEtch(target="silicon", depth_um=0.5, sidewall_angle_deg=120.0).apply(wafer)


def test_unknown_material_raises():
    with pytest.raises(ValueError):
        materials.get("unobtanium")


# --- WaferConfig 検証 ------------------------------------------------------
def test_wafer_config_rejects_zero_dim():
    with pytest.raises(ValueError):
        WaferConfig(nx=0, ny=10, nz=10)


def test_wafer_config_rejects_zero_pitch():
    with pytest.raises(ValueError):
        WaferConfig(pitch_um=0.0)


def test_wafer_config_rejects_negative_substrate():
    with pytest.raises(ValueError):
        WaferConfig(substrate_um=-1.0)


# --- CMP 基板保護 ----------------------------------------------------------
def test_cmp_does_not_remove_substrate(wafer):
    # 過大な研磨量でも基板は残る
    CMP(remove_um=100.0).apply(wafer)
    assert (wafer.grid == materials.get("silicon").id).any()


# --- Anneal 空グリッド -----------------------------------------------------
def test_anneal_without_dopant_is_safe(wafer):
    before = wafer.grid.copy()
    Anneal(depth_um=0.3).apply(wafer)
    # 拡散層が無ければ何も変化しない
    assert np.array_equal(before, wafer.grid)
