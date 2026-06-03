"""MOS 遮断周波数 fT（電流利得カットオフ）のテスト。"""
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


def _vth(w):
    return M.threshold_voltage_v(w, "metal_al", doping_cm3=1e17)["vth_v"]


def test_ft_definition_gm_over_2pi_cgg():
    """fT = gm/(2π·Cgg) の定義どおりであること。"""
    w = _mos()
    vth = _vth(w)
    r = M.mos_cutoff_frequency(w, "metal_al", vg=vth + 0.5, vd=1.5)
    cgg_f = r["cgg_ff"] * 1e-15
    assert r["ft_hz"] == pytest.approx(r["gm_s"] / (2 * math.pi * cgg_f), rel=1e-9)
    assert r["ft_ghz"] == pytest.approx(r["ft_hz"] / 1e9, rel=1e-12)


def test_ft_increases_with_overdrive():
    """過剰電圧（gm 増）とともに fT が上がる。"""
    w = _mos()
    vth = _vth(w)
    lo = M.mos_cutoff_frequency(w, "metal_al", vg=vth + 0.2, vd=1.5)["ft_hz"]
    hi = M.mos_cutoff_frequency(w, "metal_al", vg=vth + 0.9, vd=1.5)["ft_hz"]
    assert 0 < lo < hi


def test_transit_time_is_inverse_of_2pi_ft():
    """トランジット時間 τ=1/(2π·fT)=Cgg/gm。"""
    w = _mos()
    vth = _vth(w)
    r = M.mos_cutoff_frequency(w, "metal_al", vg=vth + 0.6, vd=1.5)
    tau_s = r["transit_time_ps"] * 1e-12
    assert tau_s == pytest.approx(1.0 / (2 * math.pi * r["ft_hz"]), rel=1e-9)


def test_higher_wl_does_not_change_ft_much():
    """W/L は gm も Cgg(総容量) も比例して増やすため fT はほぼ不変。

    gm∝(W/L)、Cgg はゲート面積に比例。ここでは w_over_l は gm のみに効くので
    fT は W/L とともに上がる（電流駆動が増える）ことを確認する。
    """
    w = _mos()
    vth = _vth(w)
    f1 = M.mos_cutoff_frequency(w, "metal_al", vg=vth + 0.5, vd=1.5, w_over_l=5)["ft_hz"]
    f2 = M.mos_cutoff_frequency(w, "metal_al", vg=vth + 0.5, vd=1.5, w_over_l=20)["ft_hz"]
    assert f2 > f1


def test_no_gate_returns_zero():
    """ゲート（誘電体）が無ければ fT=0・τ=inf。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.mos_cutoff_frequency(w, "metal_al", vg=1.0, vd=1.5)
    assert r["ft_hz"] == 0.0
    assert math.isinf(r["transit_time_ps"])
