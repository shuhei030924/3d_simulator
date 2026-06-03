"""ホール効果（ホール係数・ホール電圧・ホール移動度）のテスト。"""
import pytest

from semisim import metrology as M

_Q = 1.602176634e-19


def test_hall_coefficient_inverse_qn():
    """|R_H|=1/(q·n)（n=1e16 cm⁻³→約 625 cm³/C）。"""
    r = M.hall_effect(1e16, carrier="electron")
    expected_cm3 = 1.0 / (_Q * 1e16)  # cm³/C（n を cm⁻³ で扱う）
    assert abs(r["hall_coefficient_cm3_c"]) == pytest.approx(expected_cm3, rel=1e-6)


def test_carrier_sign():
    """n 型は R_H<0、p 型は R_H>0。"""
    assert M.hall_effect(1e16, carrier="electron")["hall_coefficient_cm3_c"] < 0
    assert M.hall_effect(1e16, carrier="hole")["hall_coefficient_cm3_c"] > 0


def test_hall_voltage_proportional_to_field():
    """V_H ∝ B。"""
    v1 = M.hall_effect(1e16, b_field_t=0.5)["hall_voltage_v"]
    v2 = M.hall_effect(1e16, b_field_t=1.0)["hall_voltage_v"]
    assert v2 / v1 == pytest.approx(2.0, rel=1e-9)


def test_hall_voltage_proportional_to_current():
    """V_H ∝ I。"""
    v1 = M.hall_effect(1e16, current_a=1e-3)["hall_voltage_v"]
    v2 = M.hall_effect(1e16, current_a=3e-3)["hall_voltage_v"]
    assert v2 / v1 == pytest.approx(3.0, rel=1e-9)


def test_hall_voltage_inverse_thickness():
    """V_H ∝ 1/t。"""
    v1 = M.hall_effect(1e16, thickness_um=1.0)["hall_voltage_v"]
    v2 = M.hall_effect(1e16, thickness_um=2.0)["hall_voltage_v"]
    assert v1 / v2 == pytest.approx(2.0, rel=1e-9)


def test_hall_voltage_inverse_density():
    """V_H ∝ 1/n（ドーピング 10 倍で 1/10）。"""
    lo = M.hall_effect(1e16)["hall_voltage_v"]
    hi = M.hall_effect(1e17)["hall_voltage_v"]
    assert lo / hi == pytest.approx(10.0, rel=1e-6)


def test_hall_mobility_matches_drift_mobility():
    """ホール散乱係数 r_H=1 ではホール移動度=ドリフト移動度。"""
    mu_h = M.hall_effect(1e16, hall_factor=1.0)["hall_mobility_cm2_vs"]
    mu_d = M.carrier_mobility(1e16)["mobility_cm2_vs"]
    assert mu_h == pytest.approx(mu_d, rel=1e-6)


def test_hall_factor_scales_mobility():
    """ホール移動度はホール散乱係数 r_H に比例。"""
    base = M.hall_effect(1e16, hall_factor=1.0)["hall_mobility_cm2_vs"]
    scaled = M.hall_effect(1e16, hall_factor=1.15)["hall_mobility_cm2_vs"]
    assert scaled == pytest.approx(1.15 * base, rel=1e-6)


def test_sheet_density():
    """シートキャリア密度 = n·t。"""
    r = M.hall_effect(1e16, thickness_um=1.0)
    assert r["sheet_density_cm2"] == pytest.approx(1e16 * 1.0 * 1e-4, rel=1e-9)


def test_invalid_inputs_raise():
    """未知 carrier・非正試料厚はエラー。"""
    with pytest.raises(ValueError):
        M.hall_effect(1e16, carrier="ion")
    with pytest.raises(ValueError):
        M.hall_effect(1e16, thickness_um=0.0)
