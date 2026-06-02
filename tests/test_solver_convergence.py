"""数値ソルバ（容量・熱拡散）のメッシュ収束検証。

既存ソルバが格子細分で解析解へ収束する（=実装が正しい）ことを定量的に裏付ける。
"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

EPS0_FF = 8.854e-3


def _plate_cap_error(pitch: float) -> float:
    """物理寸法固定の全面平行平板で field ソルバ容量の解析解相対誤差を返す。"""
    nx = round(2.0 / pitch)
    ny = round(1.0 / pitch)
    nz = round(2.0 / pitch)
    cfg = WaferConfig(nx=nx, ny=ny, nz=nz, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    gv = round(0.4 / pitch)
    pt = round(0.3 / pitch)
    z0 = nz // 3
    g[z0:z0 + pt, :, :] = materials.get("metal_al").id
    g[z0 + pt + gv:z0 + pt + gv + pt, :, :] = materials.get("metal_cu").id
    cf = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    analytic = EPS0_FF * 3.9 * (2.0 * 1.0) / 0.4  # ε0·εr·A/gap（全面=フリンジ無視）
    return abs(cf - analytic) / analytic


def _thermal_error(pitch: float) -> float:
    """均一スラブ全面発熱で ΔT_max が 1D 解 P·R_th に一致する相対誤差。"""
    n = round(2.0 / pitch)
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get("silicon").id
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, :, :] = True
    p = 0.05
    ref = p * M.thermal_resistance_k_w(w)
    tmax = M.peak_temperature_rise_k(w, src, p)
    return abs(tmax - ref) / ref


def test_capacitance_solver_converges():
    """容量ソルバの誤差が細分で単調減少し、1 次精度（order≳0.8）。"""
    pitches = [0.1, 0.05, 0.025]
    errors = [_plate_cap_error(p) for p in pitches]
    assert errors[0] > errors[1] > errors[2]            # 単調減少
    order = M.estimate_convergence_order(pitches, errors)
    assert order > 0.8


def test_thermal_solver_converges():
    """熱拡散ソルバの誤差が細分で単調減少し、1 次精度（order≳0.8）。"""
    pitches = [0.2, 0.1, 0.05]
    errors = [_thermal_error(p) for p in pitches]
    assert errors[0] > errors[1] > errors[2]
    order = M.estimate_convergence_order(pitches, errors)
    assert order > 0.8


def test_convergence_order_recovers_known_rates():
    """収束次数推定が既知の O(h)・O(h²) を回復する。"""
    h = np.array([0.1, 0.05, 0.025])
    assert M.estimate_convergence_order(h, 0.5 * h) == pytest.approx(1.0, abs=1e-6)
    assert M.estimate_convergence_order(h, 0.5 * h ** 2) == pytest.approx(2.0, abs=1e-6)


def test_convergence_order_validates_input():
    with pytest.raises(ValueError):
        M.estimate_convergence_order([0.1], [0.1])
    with pytest.raises(ValueError):
        M.estimate_convergence_order([0.1, 0.05], [0.0, 0.1])
