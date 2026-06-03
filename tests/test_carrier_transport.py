"""キャリア輸送の基礎量（拡散係数・デバイ長・拡散長）のテスト。"""
import math

import pytest

from semisim import metrology as M


def test_einstein_relation_definition():
    """D=µ·(kT/q) のアインシュタイン関係に厳密一致。"""
    d = M.diffusion_coefficient(1e16, carrier="electron")
    assert d["diffusion_cm2_s"] == pytest.approx(
        d["mobility_cm2_vs"] * d["thermal_voltage_v"], rel=1e-12)


def test_electron_diffusion_textbook_value():
    """低ドープ電子の D≈35 cm²/s（教科書 ~36）。"""
    d = M.diffusion_coefficient(1e14, carrier="electron")["diffusion_cm2_s"]
    assert d == pytest.approx(35.0, rel=0.1)


def test_electron_diffusion_exceeds_hole():
    """電子拡散係数は正孔より大きい（移動度比を反映）。"""
    dn = M.diffusion_coefficient(1e16, carrier="electron")["diffusion_cm2_s"]
    dp = M.diffusion_coefficient(1e16, carrier="hole")["diffusion_cm2_s"]
    assert dn > dp


def test_diffusion_scales_with_temperature():
    """Vt=kT/q ∝ T により D は温度に比例して増える（µ 固定）。"""
    d300 = M.diffusion_coefficient(1e16, temp_k=300.0)["diffusion_cm2_s"]
    d400 = M.diffusion_coefficient(1e16, temp_k=400.0)["diffusion_cm2_s"]
    assert d400 / d300 == pytest.approx(400.0 / 300.0, rel=1e-9)


def test_debye_length_textbook_value():
    """n 型 N=1e16 でデバイ長 L_D≈40nm。"""
    ld = M.debye_length(1e16)["debye_length_nm"]
    assert ld == pytest.approx(40.0, rel=0.1)


def test_debye_length_scales_inverse_sqrt_n():
    """L_D ∝ 1/√N（ドーピング 100 倍で 1/10）。"""
    lo = M.debye_length(1e16)["debye_length_nm"]
    hi = M.debye_length(1e18)["debye_length_nm"]
    assert lo / hi == pytest.approx(10.0, rel=1e-6)


def test_debye_length_zero_doping_infinite():
    """N≤0 では L_D=inf。"""
    assert math.isinf(M.debye_length(0.0)["debye_length_um"])


def test_diffusion_length_definition():
    """L=√(D·τ) に一致（D は cm²/s, L は µm）。"""
    res = M.diffusion_length(1e15, 1e-6, carrier="electron")
    d = res["diffusion_coefficient_cm2_s"]
    expected_um = math.sqrt(d * 1e-6) * 1e4
    assert res["diffusion_length_um"] == pytest.approx(expected_um, rel=1e-12)


def test_diffusion_length_scales_sqrt_tau():
    """L ∝ √τ（寿命 100 倍で 10 倍）。"""
    short = M.diffusion_length(1e15, 1e-7)["diffusion_length_um"]
    long = M.diffusion_length(1e15, 1e-5)["diffusion_length_um"]
    assert long / short == pytest.approx(10.0, rel=1e-9)


def test_diffusion_length_zero_lifetime():
    """τ=0 では L=0。"""
    assert M.diffusion_length(1e16, 0.0)["diffusion_length_um"] == 0.0


def test_invalid_inputs_raise():
    """温度≤0・負の寿命はエラー。"""
    with pytest.raises(ValueError):
        M.diffusion_coefficient(1e16, temp_k=0.0)
    with pytest.raises(ValueError):
        M.debye_length(1e16, temp_k=-10.0)
    with pytest.raises(ValueError):
        M.diffusion_length(1e16, -1e-6)
