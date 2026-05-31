"""metrology.conformality_pct のテスト。"""
from __future__ import annotations

from semisim import metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DRIE, PVD, Photo, Strip
from semisim.recipe import Recipe


def _trench_then_film(deposit):
    """トレンチを掘ってから deposit プロセスで成膜する。"""
    cfg = WaferConfig(nx=50, ny=16, nz=70, pitch_um=0.05, substrate_um=2.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.0, "x1": 0.65, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=1.2))
    r.add(Strip())
    r.add(deposit)
    return r.simulate()


def test_conformal_cvd_high_coverage():
    """コンフォーマルな CVD は段差被覆率が高い。"""
    w = _trench_then_film(CVD(material="oxide", thickness_um=0.3))
    res = metrology.conformality_pct(w, "oxide")
    assert res["step_coverage_pct"] > 60.0


def test_directional_pvd_low_coverage():
    """指向性 PVD（低被覆）は段差被覆率が低い。"""
    w = _trench_then_film(
        PVD(material="metal_al", thickness_um=0.3, step_coverage=0.1)
    )
    res = metrology.conformality_pct(w, "metal_al")
    cvd = metrology.conformality_pct(
        _trench_then_film(CVD(material="oxide", thickness_um=0.3)), "oxide"
    )
    assert res["step_coverage_pct"] < cvd["step_coverage_pct"]


def test_no_film_returns_zero():
    """膜が無ければ 0 を返す。"""
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    res = metrology.conformality_pct(w, "metal_cu")
    assert res["step_coverage_pct"] == 0.0
    assert res["field_thickness_um"] == 0.0


def test_keys_present():
    """戻り値のキーが揃っている。"""
    w = _trench_then_film(CVD(material="oxide", thickness_um=0.2))
    res = metrology.conformality_pct(w, "oxide")
    assert set(res) == {
        "field_thickness_um",
        "min_thickness_um",
        "step_coverage_pct",
    }
