"""アナログ出力段のスルーレート・全電力帯域のテスト。"""
import math

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


def test_slew_rate_is_current_over_cap():
    """SR = I_drive/C_load。"""
    w = _mos()
    r = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0)
    assert r["slew_rate_v_per_us"] * 1e6 == pytest.approx(
        r["drive_current_a"] / (10e-15), rel=1e-9)


def test_full_power_bw_formula():
    """f_FP = SR/(2π·V_peak)。"""
    w = _mos()
    r = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0, v_peak=0.5)
    sr_v_s = r["slew_rate_v_per_us"] * 1e6
    assert r["full_power_bw_hz"] == pytest.approx(sr_v_s / (2 * math.pi * 0.5), rel=1e-9)


def test_inverse_with_load_cap():
    """負荷容量を 10 倍にすると SR は 1/10。"""
    w = _mos()
    light = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=1.0)["slew_rate_v_per_us"]
    heavy = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0)["slew_rate_v_per_us"]
    assert heavy == pytest.approx(light / 10.0, rel=1e-9)


def test_default_vpeak_is_half_vdd():
    """V_peak 既定は Vdd/2。"""
    w = _mos()
    r = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0)
    assert r["v_peak_v"] == pytest.approx(0.9, rel=1e-12)


def test_smaller_vpeak_higher_fpb():
    """小振幅ほど全電力帯域は高い。"""
    w = _mos()
    big = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0,
                      v_peak=1.0)["full_power_bw_hz"]
    small = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0,
                        v_peak=0.2)["full_power_bw_hz"]
    assert small > big


def test_invalid_inputs_raise():
    """負荷容量・V_peak が非正ならエラー。"""
    w = _mos()
    with pytest.raises(ValueError):
        M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=0.0)
    with pytest.raises(ValueError):
        M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0, v_peak=-1.0)


def test_no_drive_zero_slew():
    """駆動電流 0（ゲート無し）では SR=0・f_FP=0。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.slew_rate(w, "metal_al", vdd=1.8, load_cap_ff=10.0)
    assert r["slew_rate_v_per_us"] == 0.0
    assert r["full_power_bw_hz"] == 0.0
