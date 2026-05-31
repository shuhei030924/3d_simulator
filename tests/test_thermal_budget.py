"""サーマルバジェット（熱履歴）集計のテスト。"""
from __future__ import annotations

import math

from semisim import metrology
from semisim.processes import CVD, Anneal, Diffusion, RapidThermalAnneal


def test_thermal_budget_quadrature_sum():
    """合計 Dt は depth² の和、実効拡散長は二乗和平方根になる。"""
    steps = [
        Diffusion(dopant="doped_n", depth_um=0.3),
        Anneal(depth_um=0.4),
        CVD(material="oxide", thickness_um=0.5),  # 非熱工程 → 無視
    ]
    tb = metrology.thermal_budget(steps)
    assert math.isclose(tb["total_dt_um2"], 0.3**2 + 0.4**2, rel_tol=1e-9)
    assert math.isclose(tb["effective_length_um"], 0.5, rel_tol=1e-9)


def test_thermal_budget_ignores_non_thermal():
    steps = [CVD(material="oxide", thickness_um=1.0)]
    tb = metrology.thermal_budget(steps)
    assert tb["total_dt_um2"] == 0.0
    assert tb["steps"] == []


def test_thermal_budget_by_type():
    steps = [
        Diffusion(dopant="doped_n", depth_um=0.2),
        RapidThermalAnneal(depth_um=0.1),
    ]
    tb = metrology.thermal_budget(steps)
    assert "DIFFUSION" in tb["by_type"]
    assert "RTP" in tb["by_type"]
    assert len(tb["steps"]) == 2


def test_thermal_budget_empty():
    tb = metrology.thermal_budget([])
    assert tb["total_dt_um2"] == 0.0
    assert tb["effective_length_um"] == 0.0
