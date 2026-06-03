"""CMOS インバータ VTC・反転しきい値 VM・雑音マージンのテスト。"""
import numpy as np
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


def test_symmetric_vm_is_half_vdd():
    """対称設計（βp=βn, |Vthp|=Vthn）では VM=Vdd/2。"""
    w = _mos()
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=1.2)
    assert r["beta_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert r["vm_v"] == pytest.approx(0.6, abs=1e-3)


def test_symmetric_noise_margins_equal():
    """対称インバータでは NMH=NML。"""
    w = _mos()
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0)
    assert r["nmh_v"] == pytest.approx(r["nml_v"], rel=1e-6)
    assert r["nmh_v"] > 0.0 and r["nml_v"] > 0.0


def test_vtc_monotonic_decreasing():
    """VTC は単調減少（入力↑で出力↓）。"""
    w = _mos()
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0)
    assert np.all(np.diff(r["vout"]) <= 1e-9)


def test_output_swings_rail_to_rail():
    """低入力で出力≈Vdd、高入力で出力≈0（フルスイング）。"""
    w = _mos()
    vdd = 1.0
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=vdd)
    assert r["voh_v"] > 0.9 * vdd
    assert r["vol_v"] < 0.1 * vdd
    assert r["vout"][0] == pytest.approx(vdd, abs=1e-3)
    assert r["vout"][-1] == pytest.approx(0.0, abs=1e-3)


def test_high_gain_at_trip_point():
    """遷移点近傍で電圧利得 |dVout/dVin| が大きい（>10）。"""
    w = _mos()
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0)
    assert r["max_gain"] > 10.0


def test_unity_gain_points_bracket_vm():
    """単位利得点 VIL<VM<VIH、かつ VOL<VOH。"""
    w = _mos()
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0)
    assert r["vil_v"] < r["vm_v"] < r["vih_v"]
    assert r["vol_v"] < r["voh_v"]


def test_stronger_pmos_raises_vm():
    """pMOS を強くする（βp/βn↑）と VM は Vdd 側へ動く。"""
    w = _mos()
    strong = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0, w_over_l_p=100.0)
    weak = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0, w_over_l_p=2.0)
    assert strong["beta_ratio"] > 1.0 > weak["beta_ratio"]
    assert strong["vm_v"] > 0.5 > weak["vm_v"]


def test_skewed_threshold_shifts_vm():
    """|Vthp| を下げる（pMOS が早く入る）と VM は Vdd 側へ動く。"""
    w = _mos()
    base = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0)
    low_vthp = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0,
                                   vthp_mag=base["vthn_v"] - 0.1)
    assert low_vthp["vm_v"] > base["vm_v"]


def test_no_gate_returns_none():
    """ゲート（ゲート導体）が無ければ全電圧 None・VTC 空。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.cmos_inverter_vtc(w, "metal_al", vdd=1.0)
    assert r["vm_v"] is None
    assert r["vin"].size == 0


def test_invalid_args_raise():
    """vdd≤0・n_points<11 はエラー。"""
    w = _mos()
    with pytest.raises(ValueError):
        M.cmos_inverter_vtc(w, "metal_al", vdd=0.0)
    with pytest.raises(ValueError):
        M.cmos_inverter_vtc(w, "metal_al", vdd=1.0, n_points=5)
