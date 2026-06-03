"""真性キャリア濃度 ni(T) と Varshni バンドギャップ Eg(T) のテスト。"""
import pytest

from semisim import metrology as M


def test_bandgap_at_300k_matches_code_default():
    """Eg(300K)≈1.12eV（コード内の既定値と一致）。"""
    assert M.bandgap_ev(300.0) == pytest.approx(1.12, abs=0.005)


def test_bandgap_decreases_with_temperature():
    """バンドギャップは温度上昇で単調減少（Varshni）。"""
    egs = [M.bandgap_ev(t) for t in (100, 200, 300, 400, 500)]
    assert all(egs[i] > egs[i + 1] for i in range(len(egs) - 1))


def test_bandgap_zero_temp_is_eg0():
    """T≤0 では Eg=Eg(0)=1.166eV。"""
    assert M.bandgap_ev(0.0) == pytest.approx(M._EG0_SI_EV, rel=1e-9)


def test_ni_at_300k_is_1e10():
    """ni(300K)=1×10¹⁰ cm⁻³（_NI_SI_M3 と整合, 規格化点）。"""
    assert M.intrinsic_carrier_concentration(300.0)["ni_cm3"] == \
        pytest.approx(1.0e10, rel=1e-6)


def test_ni_increases_with_temperature():
    """ni は温度とともに指数的に増加。"""
    nis = [M.intrinsic_carrier_concentration(t)["ni_cm3"]
           for t in (250, 300, 350, 400, 450)]
    assert all(nis[i] < nis[i + 1] for i in range(len(nis) - 1))


def test_ni_roughly_doubles_every_8k():
    """室温付近で ni は約 8K ごとに倍増する。"""
    ratio = (M.intrinsic_carrier_concentration(308.0)["ni_cm3"]
             / M.intrinsic_carrier_concentration(300.0)["ni_cm3"])
    assert ratio == pytest.approx(2.0, rel=0.1)


def test_ni_matches_literature_order_of_magnitude():
    """文献値オーダーに一致（ni(400K)~5×10¹² cm⁻³）。"""
    ni400 = M.intrinsic_carrier_concentration(400.0)["ni_cm3"]
    assert 1e12 < ni400 < 1e13


def test_ni_formula_consistency():
    """ni(T)=1e10·(T/300)^1.5·exp(Eg(300)/2k·300−Eg(T)/2kT) に一致。"""
    t = 360.0
    eg_t = M.bandgap_ev(t)
    eg_0 = M.bandgap_ev(300.0)
    k = M._K_BOLTZMANN_EV
    import math
    expected = 1.0e10 * (t / 300.0) ** 1.5 * math.exp(
        eg_0 / (2 * k * 300.0) - eg_t / (2 * k * t))
    assert M.intrinsic_carrier_concentration(t)["ni_cm3"] == \
        pytest.approx(expected, rel=1e-9)


def test_ni_zero_temp_edge_case():
    """T≤0 では ni=0。"""
    assert M.intrinsic_carrier_concentration(0.0)["ni_cm3"] == 0.0
