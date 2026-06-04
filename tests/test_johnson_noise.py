"""抵抗の熱雑音（Johnson–Nyquist 雑音）の検証テスト。"""
import math

import pytest

from semisim import metrology as M

_K_B = 1.380649e-23


def test_voltage_density_matches_textbook():
    """T=300K・R=1kΩ で電圧雑音密度 ≈4.07 nV/√Hz。"""
    r = M.johnson_nyquist_noise(1000.0, temp_k=300.0)
    assert r["voltage_noise_density_nv_sqrthz"] == pytest.approx(4.07, abs=0.05)


def test_psd_formula():
    """電圧 PSD は 4kTR に厳密一致。"""
    R, T = 5000.0, 300.0
    r = M.johnson_nyquist_noise(R, temp_k=T)
    assert r["voltage_psd_v2_hz"] == pytest.approx(4.0 * _K_B * T * R, rel=1e-9)


def test_rms_scales_with_bandwidth():
    """RMS 電圧は帯域の平方根に比例する。"""
    a = M.johnson_nyquist_noise(1000.0, bandwidth_hz=1.0)["voltage_rms_v"]
    b = M.johnson_nyquist_noise(1000.0, bandwidth_hz=4.0)["voltage_rms_v"]
    assert b == pytest.approx(2.0 * a, rel=1e-9)


def test_available_power_independent_of_r():
    """利用可能雑音電力 kTB は抵抗値に依らない。"""
    p1 = M.johnson_nyquist_noise(100.0, bandwidth_hz=1e6)["available_power_w"]
    p2 = M.johnson_nyquist_noise(1e6, bandwidth_hz=1e6)["available_power_w"]
    assert p1 == pytest.approx(p2, rel=1e-9)
    assert p1 == pytest.approx(_K_B * 300.0 * 1e6, rel=1e-9)


def test_voltage_rises_with_resistance_current_falls():
    """抵抗を上げると電圧雑音は増え、電流雑音は減る。"""
    lo = M.johnson_nyquist_noise(100.0, bandwidth_hz=1.0)
    hi = M.johnson_nyquist_noise(10000.0, bandwidth_hz=1.0)
    assert hi["voltage_rms_v"] > lo["voltage_rms_v"]
    assert hi["current_rms_a"] < lo["current_rms_a"]


def test_temperature_scaling():
    """電圧 PSD は絶対温度に比例する。"""
    a = M.johnson_nyquist_noise(1000.0, temp_k=150.0)["voltage_psd_v2_hz"]
    b = M.johnson_nyquist_noise(1000.0, temp_k=300.0)["voltage_psd_v2_hz"]
    assert b == pytest.approx(2.0 * a, rel=1e-9)


def test_nonpositive_returns_zero():
    """R≤0 / T≤0 では 0 を返す。"""
    assert M.johnson_nyquist_noise(0.0)["voltage_rms_v"] == 0.0
    assert M.johnson_nyquist_noise(1000.0, temp_k=0.0)["voltage_psd_v2_hz"] == 0.0


def test_current_psd_consistency():
    """i_rms と v_rms が R を介して整合（v=i·R）。"""
    r = M.johnson_nyquist_noise(2000.0, bandwidth_hz=1e3)
    assert r["voltage_rms_v"] == pytest.approx(
        r["current_rms_a"] * r["resistance_ohm"], rel=1e-9)
    assert math.isfinite(r["voltage_rms_v"])
