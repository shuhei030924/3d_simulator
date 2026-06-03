"""アバランシェ増倍係数 M=1/(1−(V/BV)^n)（Miller 式）のテスト。"""
import math

import pytest

from semisim import metrology as M


def test_unity_at_zero_bias():
    """V=0 では M=1（増倍なし）。"""
    assert M.avalanche_multiplication(0.0, 60.0)["multiplication"] == \
        pytest.approx(1.0, rel=1e-12)


def test_miller_formula():
    """M=1/(1−(V/BV)^n) の定義に一致。"""
    v, bv, n = 45.0, 60.0, 4.0
    r = M.avalanche_multiplication(v, bv, miller_exponent=n)
    assert r["multiplication"] == pytest.approx(1.0 / (1.0 - (v / bv) ** n), rel=1e-12)


def test_monotonic_increasing_with_bias():
    """逆バイアスが BV に近づくほど M は単調増加。"""
    ms = [M.avalanche_multiplication(v, 60.0)["multiplication"]
          for v in (0, 15, 30, 45, 55, 59)]
    assert all(ms[i] < ms[i + 1] for i in range(len(ms) - 1))


def test_diverges_at_breakdown():
    """V=BV で M=∞（降伏）。"""
    assert math.isinf(M.avalanche_multiplication(60.0, 60.0)["multiplication"])


def test_infinite_above_breakdown():
    """V>BV でも M=∞。"""
    assert math.isinf(M.avalanche_multiplication(70.0, 60.0)["multiplication"])


def test_realistic_apd_gain():
    """0.95·BV 付近で APD 利得は数倍オーダー。"""
    g = M.avalanche_multiplication(57.0, 60.0)["multiplication"]
    assert 3.0 < g < 10.0


def test_ratio_field():
    """v_over_bv=V/BV を返す。"""
    r = M.avalanche_multiplication(30.0, 60.0)
    assert r["v_over_bv"] == pytest.approx(0.5, rel=1e-12)


def test_invalid_inputs_raise():
    """BV≤0・負バイアスはエラー。"""
    with pytest.raises(ValueError):
        M.avalanche_multiplication(10.0, 0.0)
    with pytest.raises(ValueError):
        M.avalanche_multiplication(-5.0, 60.0)


def test_integration_with_breakdown_voltage():
    """junction_breakdown_voltage の BV と連携して利得を算出できる。"""
    bv = M.junction_breakdown_voltage(1e19, 1e16)["bv_v"]
    r = M.avalanche_multiplication(0.9 * bv, bv)
    assert r["multiplication"] > 1.0 and math.isfinite(r["multiplication"])
