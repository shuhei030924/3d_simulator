"""ロジックゲート遅延(CV/I)・HCI寿命のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _mos() -> Wafer:
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:16, :, :] = materials.get("metal_al").id
    return w


def test_gate_delay_proportional_to_load_cap():
    """遅延は負荷容量に比例（CV/I）。"""
    w = _mos()
    d1 = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=1.0, vdd=1.2)
    d5 = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=5.0, vdd=1.2)
    assert d5["delay_ps"] == pytest.approx(5 * d1["delay_ps"], rel=1e-6)


def test_gate_delay_faster_with_more_drive():
    """W/L が大きいほど駆動電流が増え遅延が短い。"""
    w = _mos()
    slow = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=5.0, vdd=1.2, w_over_l=5)
    fast = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=5.0, vdd=1.2, w_over_l=20)
    assert fast["delay_ps"] < slow["delay_ps"]
    assert fast["drive_current_a"] > slow["drive_current_a"]


def test_gate_delay_huge_below_threshold():
    """Vdd が Vth を大きく下回ると弱反転リークのみで遅延が桁違いに大きい。"""
    w = _mos()
    vth = M.threshold_voltage_v(w, "metal_al", doping_cm3=1e17)["vth_v"]
    sub = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=1.0, vdd=vth * 0.3)
    on = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=1.0, vdd=vth + 0.5)
    assert sub["delay_ps"] > 1e3 * on["delay_ps"]  # サブスレショルドは桁違いに遅い


def test_hci_lifetime_decreases_with_vds():
    """HCI 寿命はドレイン電圧が高いほど指数的に短い。"""
    assert M.hci_lifetime(1.0) > M.hci_lifetime(1.5) > M.hci_lifetime(2.0)


def test_hci_voltage_acceleration_form():
    """TTF = A·exp(B/Vds) の形（比が exp(B(1/V1−1/V2))）。"""
    import numpy as np
    r = M.hci_lifetime(1.0, b_volt=30) / M.hci_lifetime(2.0, b_volt=30)
    assert r == pytest.approx(np.exp(30 * (1 / 1.0 - 1 / 2.0)), rel=1e-6)


def test_hci_zero_vds_inf():
    assert M.hci_lifetime(0.0) == float("inf")
