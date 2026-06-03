"""MOS チャネル熱雑音（S_id・入力換算電圧雑音）のテスト。"""
import math

import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

_K_B = 1.380649e-23


def _mos() -> Wafer:
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:16, :, :] = materials.get("metal_al").id
    return w


def _vth(w):
    return M.threshold_voltage_v(w, "metal_al", doping_cm3=1e17)["vth_v"]


def test_sid_equals_4ktgamma_gm():
    """ドレイン電流雑音 S_id = 4kT·γ·gm。"""
    w = _mos()
    vth = _vth(w)
    gamma = 2.0 / 3.0
    r = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.6, vd=1.5, gamma=gamma)
    expected = 4 * _K_B * 300.0 * gamma * r["gm_s"]
    assert r["sid_a2_hz"] == pytest.approx(expected, rel=1e-9)


def test_svg_equals_sid_over_gm2():
    """入力換算電圧雑音 S_vg = S_id/gm²。"""
    w = _mos()
    vth = _vth(w)
    r = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.6, vd=1.5)
    assert r["svg_v2_hz"] == pytest.approx(r["sid_a2_hz"] / r["gm_s"] ** 2, rel=1e-9)


def test_input_noise_decreases_with_overdrive():
    """過剰電圧（gm 増）で入力換算電圧雑音は下がる（低雑音化）。"""
    w = _mos()
    vth = _vth(w)
    lo = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.2, vd=1.5)["vn_input_nv_sqrthz"]
    hi = M.mos_thermal_noise(w, "metal_al", vg=vth + 1.0, vd=1.5)["vn_input_nv_sqrthz"]
    assert hi < lo


def test_sid_scales_with_temperature():
    """S_id は温度に比例（4kT）。"""
    w = _mos()
    vth = _vth(w)
    cold = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.6, vd=1.5, temp_k=150.0)
    hot = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.6, vd=1.5, temp_k=300.0)
    # gm は温度非依存（モデル上）なので S_id 比 = 温度比
    assert hot["sid_a2_hz"] == pytest.approx(2.0 * cold["sid_a2_hz"], rel=1e-6)


def test_in_rms_scales_with_sqrt_bandwidth():
    """電流雑音実効値は √帯域に比例。"""
    w = _mos()
    vth = _vth(w)
    b1 = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.6, vd=1.5, bandwidth_hz=1e6)
    b4 = M.mos_thermal_noise(w, "metal_al", vg=vth + 0.6, vd=1.5, bandwidth_hz=4e6)
    assert b4["in_rms_a"] == pytest.approx(2.0 * b1["in_rms_a"], rel=1e-9)


def test_no_gate_zero_noise_inf_input():
    """ゲート無しでは gm=0・S_id=0・入力換算 inf。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.mos_thermal_noise(w, "metal_al", vg=1.0, vd=1.5)
    assert r["sid_a2_hz"] == 0.0
    assert math.isinf(r["vn_input_nv_sqrthz"])
