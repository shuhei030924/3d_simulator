"""MOS 小信号特性（gm・gds・真性利得）のテスト。"""
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


def _vth(w):
    return M.threshold_voltage_v(w, "metal_al", doping_cm3=1e17)["vth_v"]


def test_gm_positive_and_increases_with_overdrive():
    """トランスコンダクタンス gm は正で、過剰電圧とともに増える。"""
    w = _mos()
    vth = _vth(w)
    g_lo = M.mos_small_signal(w, "metal_al", vg=vth + 0.3, vd=1.5)["gm_s"]
    g_hi = M.mos_small_signal(w, "metal_al", vg=vth + 0.9, vd=1.5)["gm_s"]
    assert 0 < g_lo < g_hi


def test_gain_equals_gm_over_gds():
    """真性利得 = gm/gds。"""
    w = _mos()
    vth = _vth(w)
    s = M.mos_small_signal(w, "metal_al", vg=vth + 0.6, vd=1.5, lambda_per_v=0.1)
    assert s["intrinsic_gain"] == pytest.approx(s["gm_s"] / s["gds_s"], rel=1e-6)


def test_saturation_higher_gain_than_triode():
    """飽和域は三極管域より gds が小さく利得が高い。"""
    w = _mos()
    vth = _vth(w)
    triode = M.mos_small_signal(w, "metal_al", vg=vth + 0.6, vd=0.1, lambda_per_v=0.1)
    sat = M.mos_small_signal(w, "metal_al", vg=vth + 0.6, vd=1.5, lambda_per_v=0.1)
    assert sat["gds_s"] < triode["gds_s"]
    assert sat["intrinsic_gain"] > triode["intrinsic_gain"]


def test_realistic_gain_with_channel_length_modulation():
    """λ>0 で真性利得が現実的な範囲（数十〜数百）になる。"""
    w = _mos()
    vth = _vth(w)
    s = M.mos_small_signal(w, "metal_al", vg=vth + 0.6, vd=1.5, lambda_per_v=0.1)
    assert 10 < s["intrinsic_gain"] < 2000
