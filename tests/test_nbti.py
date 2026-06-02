"""NBTI しきい値電圧シフト（経時劣化）のテスト。"""
import numpy as np
import pytest

from semisim import metrology as M


def test_nbti_time_power_law():
    """ΔVth は時間のべき乗 tⁿ（n=0.16 既定）に従う。"""
    d1 = M.nbti_vth_shift(1.5, 125, 10.0)
    d2 = M.nbti_vth_shift(1.5, 125, 100.0)
    assert d2 / d1 == pytest.approx(10 ** 0.16, rel=1e-3)


def test_nbti_voltage_acceleration():
    """ストレス電圧が高いほど ΔVth は指数的に増える。"""
    lo = M.nbti_vth_shift(1.2, 125, 1e3)
    hi = M.nbti_vth_shift(1.8, 125, 1e3)
    assert hi / lo == pytest.approx(np.exp(1.5 * 0.6), rel=1e-3)


def test_nbti_temperature_acceleration():
    """温度が高いほど ΔVth は大きい（Arrhenius）。"""
    assert M.nbti_vth_shift(1.5, 125, 1e3) > M.nbti_vth_shift(1.5, 85, 1e3)


def test_nbti_zero_time():
    assert M.nbti_vth_shift(1.5, 125, 0.0) == 0.0


def test_nbti_negative_time_raises():
    with pytest.raises(ValueError):
        M.nbti_vth_shift(1.5, 125, -1.0)


def test_nbti_realistic_magnitude():
    """10 年・1.5V・125°C の ΔVth が現実的な範囲（数〜数十 mV）。"""
    dv = M.nbti_vth_shift(1.5, 125, 10 * 365 * 24 * 3600)
    assert 0.001 < dv < 0.1
