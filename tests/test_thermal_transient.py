"""熱過渡応答（熱時定数 τ_th=R_th·C_th）のテスト。"""
import math

import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _stack() -> Wafer:
    """Si 基板 10 層 + Cu 配線 4 層。"""
    w = Wafer(WaferConfig(nx=10, ny=10, nz=20, pitch_um=1.0, substrate_um=0.0))
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:14, :, :] = materials.get("metal_cu").id
    return w


def test_thermal_capacitance_sum_of_cv_volume():
    """C_th = Σ C_v·ΔV（固体ボクセル）。"""
    w = _stack()
    cth = M.thermal_capacitance_j_k(w)
    dv = (1.0e-6) ** 3
    expected = (1000 * 1.63e6 + 400 * 3.45e6) * dv
    assert cth == pytest.approx(expected, rel=1e-9)


def test_tau_equals_rth_times_cth():
    """τ_th = R_th·C_th。"""
    w = _stack()
    tc = M.thermal_time_constant_s(w)
    assert tc["tau_s"] == pytest.approx(tc["rth_k_w"] * tc["cth_j_k"], rel=1e-12)


def test_transient_reaches_63pct_at_tau():
    """t=τ で定常の約 63.2%（1−1/e）。"""
    w = _stack()
    tau = M.thermal_time_constant_s(w)["tau_s"]
    p = 1e-3
    dt_final = M.temperature_rise_k(w, p)
    dt_tau = M.transient_temperature_rise_k(w, p, tau)
    assert dt_tau / dt_final == pytest.approx(1.0 - 1.0 / math.e, rel=1e-6)


def test_transient_converges_to_steady_state():
    """t≫τ で定常 ΔT=P·R_th に収束。"""
    w = _stack()
    tau = M.thermal_time_constant_s(w)["tau_s"]
    p = 2e-3
    dt_final = M.temperature_rise_k(w, p)
    dt_big = M.transient_temperature_rise_k(w, p, tau * 30)
    assert dt_big == pytest.approx(dt_final, rel=1e-6)


def test_transient_zero_time_is_zero():
    """t=0 では温度上昇 0。"""
    w = _stack()
    assert M.transient_temperature_rise_k(w, 1e-3, 0.0) == pytest.approx(0.0, abs=1e-18)


def test_more_thermal_mass_longer_tau():
    """熱容量が大きい（厚い）ほど τ が長い。"""
    thin = Wafer(WaferConfig(nx=10, ny=10, nz=20, pitch_um=1.0, substrate_um=0.0))
    thin.grid[:5, :, :] = materials.get("silicon").id
    thick = Wafer(WaferConfig(nx=10, ny=10, nz=20, pitch_um=1.0, substrate_um=0.0))
    thick.grid[:10, :, :] = materials.get("silicon").id
    # 同じ材料・断面なら R_th は厚いほど大、C_th も厚いほど大 → τ は厚いほど長い
    assert (M.thermal_time_constant_s(thick)["tau_s"]
            > M.thermal_time_constant_s(thin)["tau_s"])


def test_no_solid_infinite_tau():
    """固体が無ければ R_th=inf・τ=inf。"""
    w = Wafer(WaferConfig(nx=5, ny=5, nz=10, pitch_um=1.0, substrate_um=0.0))
    w.grid[:, :, :] = materials.AIR  # 既定で残る基板1層も air にする
    tc = M.thermal_time_constant_s(w)
    assert math.isinf(tc["tau_s"])
    assert math.isinf(M.transient_temperature_rise_k(w, 1e-3, 1e-6))
