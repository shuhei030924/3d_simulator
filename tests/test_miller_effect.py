"""ミラー効果（実効入力/出力容量・入力極帯域）のテスト。"""
import math

import pytest

from semisim import metrology as M


def test_input_capacitance_multiplication():
    """C_in=Cf·(1+|Av|) のミラー増倍に一致。"""
    r = M.miller_effect(2.0, 50.0)
    assert r["cin_ff"] == pytest.approx(2.0 * (1.0 + 50.0), rel=1e-12)


def test_output_capacitance():
    """C_out=Cf·(1+1/|Av|)。"""
    r = M.miller_effect(2.0, 10.0)
    assert r["cout_ff"] == pytest.approx(2.0 * (1.0 + 1.0 / 10.0), rel=1e-12)


def test_output_cap_approaches_cf_for_high_gain():
    """高利得では C_out→Cf。"""
    r = M.miller_effect(2.0, 1000.0)
    assert r["cout_ff"] == pytest.approx(2.0, rel=0.01)


def test_sign_of_gain_irrelevant():
    """利得の符号に依らず |Av| で決まる。"""
    pos = M.miller_effect(2.0, 20.0)["cin_ff"]
    neg = M.miller_effect(2.0, -20.0)["cin_ff"]
    assert pos == pytest.approx(neg, rel=1e-12)


def test_zero_gain_returns_cf():
    """|Av|=0 では C_in=C_out=Cf。"""
    r = M.miller_effect(2.0, 0.0)
    assert r["cin_ff"] == pytest.approx(2.0) and r["cout_ff"] == pytest.approx(2.0)


def test_input_cap_increases_with_gain():
    """利得が大きいほど入力容量が増える。"""
    assert M.miller_effect(2.0, 100.0)["cin_ff"] > M.miller_effect(2.0, 10.0)["cin_ff"]


def test_input_pole_formula():
    """f_in=1/(2π·Rs·C_in) に一致。"""
    r = M.miller_effect(2.0, 100.0, source_resistance_ohm=1000.0)
    cin_f = r["cin_ff"] * 1e-15
    assert r["input_pole_hz"] == pytest.approx(
        1.0 / (2.0 * math.pi * 1000.0 * cin_f), rel=1e-9)


def test_higher_gain_lowers_bandwidth():
    """利得が高いほど入力極帯域は下がる（ミラー帯域制限）。"""
    lo_gain = M.miller_effect(2.0, 10.0, source_resistance_ohm=1e3)["input_pole_hz"]
    hi_gain = M.miller_effect(2.0, 100.0, source_resistance_ohm=1e3)["input_pole_hz"]
    assert hi_gain < lo_gain


def test_no_rs_omits_pole():
    """Rs 未指定では input_pole_hz を返さない。"""
    assert "input_pole_hz" not in M.miller_effect(2.0, 10.0)


def test_invalid_inputs_raise():
    """負の帰還容量・負の源抵抗はエラー。"""
    with pytest.raises(ValueError):
        M.miller_effect(-1.0, 10.0)
    with pytest.raises(ValueError):
        M.miller_effect(2.0, 10.0, source_resistance_ohm=-100.0)
