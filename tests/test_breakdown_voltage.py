"""pn 接合アバランシェ降伏電圧（Sze 経験式）のテスト。"""
import math

import pytest

from semisim import metrology as M


def test_bv_reference_value_at_1e16():
    """N=1e16 cm⁻³ で BV≈60V（Sze 経験式の基準点）。"""
    r = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e16)
    assert r["bv_v"] == pytest.approx(60.0 * (1.12 / 1.1) ** 1.5, rel=1e-9)
    assert r["n_light_cm3"] == 1e16


def test_bv_scales_as_n_power_minus_three_quarters():
    """BV ∝ N^(−3/4)（10 倍ドーピングで BV×10^−0.75）。"""
    lo = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e16)["bv_v"]
    hi = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e17)["bv_v"]
    assert hi / lo == pytest.approx(10 ** -0.75, rel=1e-9)


def test_higher_doping_lower_bv():
    """高ドープほど BV は低い。"""
    bvs = [M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=n)["bv_v"]
           for n in (1e15, 1e16, 1e17, 1e18)]
    assert bvs[0] > bvs[1] > bvs[2] > bvs[3]


def test_lighter_side_dominates():
    """軽ドープ側が BV を決める（高ドープ側に非依存）。"""
    a = M.junction_breakdown_voltage(na_cm3=1e16, nd_cm3=1e19)["bv_v"]
    b = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e16)["bv_v"]
    c = M.junction_breakdown_voltage(na_cm3=5e19, nd_cm3=1e16)["bv_v"]
    assert a == pytest.approx(b, rel=1e-12)
    assert b == pytest.approx(c, rel=1e-12)


def test_max_field_consistent_with_depletion():
    """E_crit = 2·BV/W_BD（降伏時の三角電界分布）。"""
    r = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e16)
    e_from_geom = 2.0 * r["bv_v"] / (r["w_bd_um"] * 1e-6) / 1e8  # MV/cm
    assert r["ecrit_mv_cm"] == pytest.approx(e_from_geom, rel=1e-9)


def test_ecrit_increases_with_doping():
    """臨界電界は高ドープほど高い。"""
    lo = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e15)["ecrit_mv_cm"]
    hi = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e18)["ecrit_mv_cm"]
    assert hi > lo


def test_bandgap_dependence():
    """広バンドギャップほど BV が高い。"""
    si = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e16, eg_ev=1.12)["bv_v"]
    wide = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e16, eg_ev=3.3)["bv_v"]
    assert wide > si


def test_invalid_doping_raises():
    """ドーピングが非正ならエラー。"""
    with pytest.raises(ValueError):
        M.junction_breakdown_voltage(na_cm3=0.0, nd_cm3=1e16)


def test_depletion_width_positive():
    """降伏時空乏層幅は正で、低ドープほど広い。"""
    wide = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e15)["w_bd_um"]
    narrow = M.junction_breakdown_voltage(na_cm3=1e19, nd_cm3=1e18)["w_bd_um"]
    assert wide > narrow > 0
    assert not math.isinf(wide)
