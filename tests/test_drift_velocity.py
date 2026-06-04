"""高電界ドリフト速度（速度飽和, Caughey–Thomas/Canali）の検証テスト。"""
import pytest

from semisim import metrology as M


def test_low_field_is_ohmic():
    """低電界では v≈µ₀·E（実効移動度が低電界移動度に一致）。"""
    r = M.drift_velocity(100.0, carrier="electron")
    mu0 = r["low_field_mobility_cm2_vs"]
    assert r["effective_mobility_cm2_vs"] == pytest.approx(mu0, rel=1e-3)
    assert r["drift_velocity_cm_s"] == pytest.approx(mu0 * 100.0, rel=1e-3)


def test_high_field_saturates():
    """高電界ではドリフト速度が飽和速度に漸近する（ただし超えない）。"""
    vsat = M._VSAT["electron"]["v_sat_cm_s"]
    r = M.drift_velocity(1.0e6, carrier="electron")
    assert r["drift_velocity_cm_s"] < vsat
    assert r["drift_velocity_cm_s"] == pytest.approx(vsat, rel=0.05)


def test_velocity_bounded_by_vsat():
    """どの電界でもドリフト速度は飽和速度を超えない。"""
    vsat = M._VSAT["hole"]["v_sat_cm_s"]
    for e in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
        v = M.drift_velocity(e, carrier="hole")["drift_velocity_cm_s"]
        assert v < vsat


def test_monotonic_increasing():
    """ドリフト速度は電界に対し単調増加。"""
    fields = [1e2, 1e3, 1e4, 1e5, 1e6]
    vs = [M.drift_velocity(e)["drift_velocity_cm_s"] for e in fields]
    assert all(b > a for a, b in zip(vs, vs[1:]))


def test_zero_field():
    """電界 0 で速度 0、実効移動度は低電界移動度に等しい。"""
    r = M.drift_velocity(0.0, carrier="electron")
    assert r["drift_velocity_cm_s"] == 0.0
    assert r["effective_mobility_cm2_vs"] == r["low_field_mobility_cm2_vs"]


def test_doping_lowers_low_field_mobility():
    """高ドープでは低電界移動度が下がる（不純物散乱）。"""
    undoped = M.drift_velocity(100.0, doping_cm3=0.0)["low_field_mobility_cm2_vs"]
    doped = M.drift_velocity(100.0, doping_cm3=1e18)["low_field_mobility_cm2_vs"]
    assert doped < undoped


def test_electron_vsat_above_hole():
    """電子の飽和速度は正孔より高い。"""
    ve = M.drift_velocity(1e7, carrier="electron")["saturation_velocity_cm_s"]
    vh = M.drift_velocity(1e7, carrier="hole")["saturation_velocity_cm_s"]
    assert ve > vh


def test_invalid_carrier_raises():
    with pytest.raises(ValueError):
        M.drift_velocity(1e4, carrier="muon")
