"""トンネルダイオード I-V（負性抵抗・ピーク/谷・PVCR）のテスト。"""
import numpy as np
import pytest

from semisim import metrology as M


def test_zero_current_at_zero_bias():
    """V=0 で I=0（全成分が原点で消える）。"""
    r = M.tunnel_diode_iv()
    assert r["current_a"][0] == pytest.approx(0.0, abs=1e-15)


def test_has_negative_differential_resistance():
    """負性微分抵抗（dI/dV<0）の領域が存在する。"""
    r = M.tunnel_diode_iv()
    assert r["has_ndr"] is True
    di = np.diff(r["current_a"])
    assert np.any(di < 0)


def test_peak_near_peak_voltage():
    """ピークは設定したピーク電圧 Vp 付近。"""
    r = M.tunnel_diode_iv(peak_voltage_v=0.065)
    assert r["peak_voltage_v"] == pytest.approx(0.065, abs=0.02)
    assert r["peak_current_a"] == pytest.approx(1.0e-3, rel=0.1)


def test_valley_after_peak():
    """谷はピークより高電圧側・低電流。"""
    r = M.tunnel_diode_iv()
    assert r["valley_voltage_v"] > r["peak_voltage_v"]
    assert r["valley_current_a"] < r["peak_current_a"]


def test_ndr_range_between_peak_and_valley():
    """NDR 区間はピーク〜谷の電圧範囲。"""
    r = M.tunnel_diode_iv()
    assert r["ndr_v_range"][0] == pytest.approx(r["peak_voltage_v"])
    assert r["ndr_v_range"][1] == pytest.approx(r["valley_voltage_v"])


def test_pvcr_definition():
    """ピーク谷電流比 PVCR=Ip/Iv。"""
    r = M.tunnel_diode_iv()
    assert r["pvcr"] == pytest.approx(r["peak_current_a"] / r["valley_current_a"], rel=1e-9)
    assert r["pvcr"] > 1.0


def test_current_rises_after_valley():
    """谷の後は熱拡散電流で電流が再上昇（N 字特性）。"""
    r = M.tunnel_diode_iv()
    assert r["current_a"][-1] > r["valley_current_a"]


def test_higher_peak_current_raises_pvcr():
    """ピーク電流が大きいほど PVCR が上がる。"""
    lo = M.tunnel_diode_iv(peak_current_a=1e-3)["pvcr"]
    hi = M.tunnel_diode_iv(peak_current_a=5e-3)["pvcr"]
    assert hi > lo


def test_invalid_inputs_raise():
    """非正ピーク・谷電圧≤ピーク電圧はエラー。"""
    with pytest.raises(ValueError):
        M.tunnel_diode_iv(peak_current_a=0.0)
    with pytest.raises(ValueError):
        M.tunnel_diode_iv(peak_voltage_v=0.3, valley_voltage_v=0.2)
