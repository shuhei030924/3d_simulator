"""サブステップ補間（物理スケール）と輪郭スムージングのテスト。

中間フレームは工程の量パラメータを比例配分した工程を実エンジンで適用して
生成する（モーフィング禁止）。よって各フレームは物理的に到達可能な状態で
あり、成膜は単調増加・エッチは単調減少になるはず。
"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import animation, materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import ALD, CVD, DryEtch, Oxidation, Photo, Strip
from semisim.recipe import Recipe


def _cfg() -> WaferConfig:
    return WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=1.5)


# --- scaled_process -----------------------------------------------------------
def test_scaled_cvd_half_thickness():
    p = animation.scaled_process(CVD(material="oxide", thickness_um=0.8), 0.5)
    assert isinstance(p, CVD)
    assert p.thickness_um == pytest.approx(0.4)
    assert p.material == "oxide"


def test_scaled_ald_cycles_rounded_min1():
    p = animation.scaled_process(ALD(material="hafnia", cycles=12), 0.25)
    assert p.cycles == 3  # round(12*0.25)
    p2 = animation.scaled_process(ALD(material="hafnia", cycles=2), 0.1)
    assert p2.cycles == 1  # 最低 1 サイクル


def test_scaled_oxidation_time_mode_scales_time():
    """時間指定の熱酸化は時間を比例配分（Deal-Grove で厚さ∝√t を保つ）。"""
    p = animation.scaled_process(
        Oxidation(thickness_um=0.3, time_min=60.0, temperature_c=1000.0), 0.5
    )
    assert p.time_min == pytest.approx(30.0)
    assert p.thickness_um == pytest.approx(0.3)  # 厚さ指定値は触らない


def test_scaled_instant_process_returns_none():
    assert animation.scaled_process(Photo(), 0.5) is None
    assert animation.scaled_process(Strip(), 0.5) is None


# --- step_slices substeps -----------------------------------------------------
def _deposition_recipe() -> Recipe:
    r = Recipe(config=_cfg())
    r.add(CVD(material="nitride", thickness_um=0.8))
    return r


def test_substeps_monotone_deposition():
    """成膜のサブステップは膜量が単調増加し、最終フレームは正規結果と一致。"""
    r = _deposition_recipe()
    slices = animation.step_slices(r, substeps=4)
    assert len(slices) == 1 + 4  # 初期 + 25/50/75/100%
    nid = materials.get("nitride").id
    counts = [int(np.count_nonzero(p == nid)) for _t, p, _w, _h in slices]
    assert counts[0] == 0
    assert all(a < b for a, b in zip(counts[1:], counts[2:]))  # 単調増加
    # 最終フレーム = 通常シミュレーションの断面と完全一致
    ref = animation.step_slices(r, substeps=1)[-1][1]
    assert np.array_equal(slices[-1][1], ref)


def test_substeps_monotone_etch():
    """エッチのサブステップは除去量が単調増加（材料が単調減少）。"""
    r = Recipe(config=_cfg())
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="positive"))
    r.add(DryEtch(targets=["silicon"], depth_um=0.8))
    slices = animation.step_slices(r, substeps=4)
    si = materials.get("silicon").id
    # PHOTO は瞬時 1 フレーム、DRY は 4 フレーム → 初期+1+4
    assert len(slices) == 6
    si_counts = [int(np.count_nonzero(p == si)) for _t, p, _w, _h in slices[2:]]
    assert all(a > b for a, b in zip(si_counts, si_counts[1:]))  # 単調減少


def test_substep_titles_show_progress():
    slices = animation.step_slices(_deposition_recipe(), substeps=2)
    assert "（50%）" in slices[1][0]
    assert not slices[-1][0].endswith("%）")  # 確定フレームに % は付かない


def test_substeps_invalid():
    with pytest.raises(ValueError):
        animation.step_slices(_deposition_recipe(), substeps=0)


# --- smooth_plane -------------------------------------------------------------
def test_smooth_plane_preserves_ids_and_shape():
    """平滑化は形状を factor 倍し、存在しない材料 ID を発明しない。"""
    rng = np.random.default_rng(0)
    plane = rng.choice([0, 1, 2], size=(20, 24)).astype(np.uint8)
    out = animation.smooth_plane(plane, factor=3)
    assert out.shape == (60, 72)
    assert set(np.unique(out)) <= set(np.unique(plane))


def test_smooth_plane_keeps_bulk_regions():
    """大きな一様領域の内部は平滑化で変化しない（境界±半ボクセルのみ）。"""
    plane = np.zeros((20, 20), dtype=np.uint8)
    plane[:10] = 1  # 下半分が材料 1
    out = animation.smooth_plane(plane, factor=3)
    assert (out[:24] == 1).all()  # 境界から十分離れた内部
    assert (out[36:] == 0).all()
