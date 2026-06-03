"""バイポーラトランジスタ(BJT) Ebers-Moll/Gummel 順活性モデルのテスト。"""
import math

import pytest

from semisim import metrology as M


def test_collector_current_exponential():
    """Ic は Vbe に対し指数的（60mV で約 10 倍）。"""
    i1 = M.bjt_currents(0.60)["ic_a"]
    i2 = M.bjt_currents(0.66)["ic_a"]
    assert i2 / i1 == pytest.approx(10.0, rel=0.05)


def test_beta_is_ic_over_ib():
    """β=Ic/Ib。"""
    r = M.bjt_currents(0.7)
    assert r["beta"] == pytest.approx(r["ic_a"] / r["ib_a"], rel=1e-9)


def test_beta_equals_betaf_at_zero_vce():
    """Vce=0 では β=βF（アーリー項=1）。"""
    assert M.bjt_currents(0.7, vce=0.0, beta_f=100.0)["beta"] == \
        pytest.approx(100.0, rel=1e-6)


def test_early_effect_increases_ic_and_beta():
    """Vce を上げると Ic・β が増える（アーリー効果）。"""
    lo = M.bjt_currents(0.7, vce=0.5)
    hi = M.bjt_currents(0.7, vce=5.0)
    assert hi["ic_a"] > lo["ic_a"]
    assert hi["beta"] > lo["beta"]


def test_beta_early_formula():
    """β=βF·(1+Vce/VA)。"""
    r = M.bjt_currents(0.7, vce=2.0, beta_f=100.0, early_v=50.0)
    assert r["beta"] == pytest.approx(100.0 * (1.0 + 2.0 / 50.0), rel=1e-6)


def test_transconductance_ic_over_vt():
    """gm=Ic/Vt。"""
    r = M.bjt_currents(0.7)
    assert r["gm_s"] == pytest.approx(r["ic_a"] / r["vt_v"], rel=1e-12)


def test_output_resistance_va_over_ic():
    """ro=VA/Ic。"""
    r = M.bjt_currents(0.7, vce=1.0, early_v=50.0)
    assert r["ro_ohm"] == pytest.approx(50.0 / r["ic_a"], rel=1e-9)


def test_gummel_lines_parallel():
    """Gummel: log Ic と log Ib の差は一定（=ln βF, Vbe に依らず平行）。"""
    d1 = M.bjt_currents(0.5, vce=0.0)
    d2 = M.bjt_currents(0.8, vce=0.0)
    sep1 = math.log(d1["ic_a"] / d1["ib_a"])
    sep2 = math.log(d2["ic_a"] / d2["ib_a"])
    assert sep1 == pytest.approx(sep2, rel=1e-9)
    assert sep1 == pytest.approx(math.log(100.0), rel=1e-6)


def test_higher_betaf_lowers_base_current():
    """βF が大きいほどベース電流は小さい（同じ Ic 駆動）。"""
    low_beta = M.bjt_currents(0.7, beta_f=50.0)["ib_a"]
    high_beta = M.bjt_currents(0.7, beta_f=200.0)["ib_a"]
    assert high_beta < low_beta


def test_invalid_inputs_raise():
    """非正パラメータ・温度はエラー。"""
    with pytest.raises(ValueError):
        M.bjt_currents(0.7, is_a=0.0)
    with pytest.raises(ValueError):
        M.bjt_currents(0.7, beta_f=-1.0)
    with pytest.raises(ValueError):
        M.bjt_currents(0.7, temp_k=0.0)
