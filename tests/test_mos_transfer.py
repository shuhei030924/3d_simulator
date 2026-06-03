"""MOS 伝達特性（SS・Ion・Ioff・Ion/Ioff）のテスト。"""
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


def test_ss_matches_n_times_thermal_limit():
    """サブスレショルドスイング SS ≈ n·(kT/q)·ln10。"""
    w = _mos()
    n = 1.3
    r = M.mos_transfer_characteristics(w, "metal_al", vd=1.0, vdd=1.8,
                                       subthreshold_n=n)
    ss_ideal = n * 0.025852 * math.log(10) * 1e3  # mV/dec
    assert r["ss_mv_dec"] == pytest.approx(ss_ideal, rel=0.02)


def test_ss_scales_with_subthreshold_n():
    """理想係数 n が大きいほど SS が大きい（傾きが緩む）。"""
    w = _mos()
    ss_lo = M.mos_transfer_characteristics(w, "metal_al", vdd=1.8,
                                           subthreshold_n=1.0)["ss_mv_dec"]
    ss_hi = M.mos_transfer_characteristics(w, "metal_al", vdd=1.8,
                                           subthreshold_n=1.6)["ss_mv_dec"]
    assert ss_lo < ss_hi
    assert ss_lo == pytest.approx(60.0, abs=2.0)  # 理想 60mV/dec 限界


def test_ion_greater_than_ioff():
    """Ion ≫ Ioff、Ion/Ioff 比は大きい。"""
    w = _mos()
    r = M.mos_transfer_characteristics(w, "metal_al", vd=1.0, vdd=1.8)
    assert r["ion_a"] > r["ioff_a"] >= 0
    assert r["on_off_ratio"] > 1e6


def test_ion_equals_drain_current_at_vdd():
    """Ion は Vg=Vdd でのドレイン電流に一致。"""
    w = _mos()
    r = M.mos_transfer_characteristics(w, "metal_al", vd=1.0, vdd=1.8)
    direct = M.mos_drain_current(w, "metal_al", vg=1.8, vd=1.0)
    assert r["ion_a"] == pytest.approx(direct, rel=1e-9)


def test_higher_vdd_increases_on_off_ratio():
    """Vdd を上げると Ion が増え Ion/Ioff 比が向上。"""
    w = _mos()
    lo = M.mos_transfer_characteristics(w, "metal_al", vdd=1.2)["on_off_ratio"]
    hi = M.mos_transfer_characteristics(w, "metal_al", vdd=2.0)["on_off_ratio"]
    assert hi > lo


def test_no_gate_returns_inf_ss():
    """ゲート（誘電体）が無ければ Ion=0・SS=inf・比=0。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.mos_transfer_characteristics(w, "metal_al", vdd=1.8)
    assert r["ion_a"] == 0.0
    assert math.isinf(r["ss_mv_dec"])
    assert r["on_off_ratio"] == 0.0
