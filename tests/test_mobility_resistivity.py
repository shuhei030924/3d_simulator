"""ドーピング依存移動度（Caughey–Thomas）と体積抵抗率（Irvin 曲線）のテスト。"""
import math

import pytest

from semisim import metrology as M

_Q = 1.602176634e-19


def test_low_doping_approaches_mu_max():
    """低ドープで µ→µ_max（格子散乱律速）。"""
    for c in ("electron", "hole"):
        p = M._CT_MOBILITY[c]
        mu = M.carrier_mobility(1e13, carrier=c)["mobility_cm2_vs"]
        assert mu == pytest.approx(p["mu_max"], rel=0.01)


def test_high_doping_approaches_mu_min():
    """高ドープで µ→µ_min（不純物散乱律速）。"""
    for c in ("electron", "hole"):
        p = M._CT_MOBILITY[c]
        mu = M.carrier_mobility(1e21, carrier=c)["mobility_cm2_vs"]
        assert mu == pytest.approx(p["mu_min"], rel=0.02)


def test_mobility_at_nref_is_midpoint():
    """N=N_ref で µ=(µ_max+µ_min)/2。"""
    for c in ("electron", "hole"):
        p = M._CT_MOBILITY[c]
        mu = M.carrier_mobility(p["n_ref"], carrier=c)["mobility_cm2_vs"]
        assert mu == pytest.approx(0.5 * (p["mu_max"] + p["mu_min"]), rel=1e-9)


def test_mobility_monotonic_decreasing():
    """移動度はドーピング濃度とともに単調減少。"""
    mus = [M.carrier_mobility(10.0 ** e, carrier="electron")["mobility_cm2_vs"]
           for e in range(13, 21)]
    assert all(mus[i] >= mus[i + 1] for i in range(len(mus) - 1))


def test_electron_higher_mobility_than_hole():
    """電子移動度は正孔移動度より大きい（同一ドーピング）。"""
    n = M.carrier_mobility(1e16, carrier="electron")["mobility_cm2_vs"]
    p = M.carrier_mobility(1e16, carrier="hole")["mobility_cm2_vs"]
    assert n > p


def test_resistivity_matches_irvin_curve():
    """n 型 Si の体積抵抗率が Irvin 曲線に一致（N=1e16→約 0.5 Ω·cm）。"""
    r = M.bulk_resistivity_ohm_cm(1e16, carrier="electron")
    assert r["resistivity_ohm_cm"] == pytest.approx(0.5, rel=0.1)
    r2 = M.bulk_resistivity_ohm_cm(1e18, carrier="electron")
    assert r2["resistivity_ohm_cm"] == pytest.approx(0.024, rel=0.15)


def test_resistivity_definition():
    """ρ=1/(q·N·µ) の定義に一致し、σ=1/ρ。"""
    N = 5e16
    r = M.bulk_resistivity_ohm_cm(N, carrier="hole")
    mu = r["mobility_cm2_vs"]
    expected = 1.0 / (_Q * N * mu)
    assert r["resistivity_ohm_cm"] == pytest.approx(expected, rel=1e-9)
    assert r["conductivity_s_cm"] == pytest.approx(1.0 / expected, rel=1e-9)


def test_higher_doping_lowers_resistivity():
    """ドーピングを上げると（移動度低下を含めても）抵抗率は下がる。"""
    lo = M.bulk_resistivity_ohm_cm(1e15, carrier="electron")["resistivity_ohm_cm"]
    hi = M.bulk_resistivity_ohm_cm(1e19, carrier="electron")["resistivity_ohm_cm"]
    assert hi < lo


def test_zero_doping_edge_cases():
    """N≤0 では µ=µ_max・ρ=inf。"""
    assert M.carrier_mobility(0.0, carrier="electron")["mobility_cm2_vs"] == \
        pytest.approx(M._CT_MOBILITY["electron"]["mu_max"])
    assert math.isinf(M.bulk_resistivity_ohm_cm(0.0)["resistivity_ohm_cm"])


def test_invalid_carrier_raises():
    """未知の carrier はエラー。"""
    with pytest.raises(ValueError):
        M.carrier_mobility(1e16, carrier="proton")
