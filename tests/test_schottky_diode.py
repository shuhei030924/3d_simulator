"""ショットキーバリアダイオード（熱電子放出, Richardson 式）のテスト。"""
import math

import pytest

from semisim import metrology as M


def test_saturation_current_richardson_formula():
    """Js=A*·T²·exp(−Φ_B/kT) の定義に一致。"""
    phi, T, a = 0.6, 300.0, 110.0
    s = M.schottky_saturation_current(1.0, barrier_ev=phi, temp_k=T,
                                      richardson_a_cm2_k2=a)
    expected = a * T ** 2 * math.exp(-phi / (M._K_BOLTZMANN_EV * T))
    assert s["j_sat_a_cm2"] == pytest.approx(expected, rel=1e-9)


def test_saturation_current_area_scaling():
    """Is=Js·面積（1µm²=1e−8 cm²）。"""
    s1 = M.schottky_saturation_current(1.0, barrier_ev=0.6)
    s100 = M.schottky_saturation_current(100.0, barrier_ev=0.6)
    assert s100["i_sat_a"] == pytest.approx(100.0 * s1["i_sat_a"], rel=1e-9)
    assert s1["i_sat_a"] == pytest.approx(s1["j_sat_a_cm2"] * 1e-8, rel=1e-9)


def test_higher_barrier_lowers_saturation_current():
    """障壁が高いほど Js は指数的に小さい。"""
    lo = M.schottky_saturation_current(1.0, barrier_ev=0.5)["j_sat_a_cm2"]
    hi = M.schottky_saturation_current(1.0, barrier_ev=0.7)["j_sat_a_cm2"]
    assert lo > hi


def test_temperature_activation():
    """温度上昇で Js は急増（T²·exp 律速）。"""
    j300 = M.schottky_saturation_current(1.0, temp_k=300.0)["j_sat_a_cm2"]
    j350 = M.schottky_saturation_current(1.0, temp_k=350.0)["j_sat_a_cm2"]
    assert j350 > 10.0 * j300


def test_forward_exponential():
    """順方向電流は電圧とともに指数的に増加。"""
    i1 = M.schottky_diode_current(100.0, 0.1)
    i2 = M.schottky_diode_current(100.0, 0.2)
    assert i2 > i1 > 0.0
    assert i2 / i1 > 10.0


def test_reverse_saturates_to_minus_is():
    """逆方向は −Is に飽和。"""
    is_a = M.schottky_saturation_current(100.0, barrier_ev=0.6)["i_sat_a"]
    i_rev = M.schottky_diode_current(100.0, -0.5, barrier_ev=0.6)
    assert i_rev == pytest.approx(-is_a, rel=1e-6)


def test_zero_bias_zero_current():
    """V=0 で I=0。"""
    assert M.schottky_diode_current(100.0, 0.0) == pytest.approx(0.0, abs=1e-30)


def test_schottky_higher_is_than_pn():
    """ショットキー Is は pn 接合（Shockley, Is~1e−15）より桁違いに大きい。"""
    is_sch = M.schottky_saturation_current(100.0, barrier_ev=0.6)["i_sat_a"]
    assert is_sch > 1e3 * 1.0e-15


def test_series_resistance_limits_current():
    """直列抵抗 Rs>0 は高電流を制限。"""
    no_r = M.schottky_diode_current(100.0, 0.5, series_r_ohm=0.0)
    with_r = M.schottky_diode_current(100.0, 0.5, series_r_ohm=100.0)
    assert with_r < no_r


def test_invalid_inputs_raise():
    """負面積・非正温度はエラー。"""
    with pytest.raises(ValueError):
        M.schottky_saturation_current(-1.0)
    with pytest.raises(ValueError):
        M.schottky_saturation_current(1.0, temp_k=0.0)
