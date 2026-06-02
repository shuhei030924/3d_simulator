"""MOS C-V 特性・しきい値電圧のテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _mos(t_ox: int = 2, pitch: float = 0.001) -> Wafer:
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:10 + t_ox, :, :] = materials.get("oxide").id
    g[10 + t_ox:14 + t_ox, :, :] = materials.get("metal_al").id
    return w


def test_vth_increases_with_doping():
    """しきい値電圧は基板ドーピングが高いほど大きい。"""
    w = _mos()
    v_lo = M.threshold_voltage_v(w, "metal_al", doping_cm3=1e16)["vth_v"]
    v_hi = M.threshold_voltage_v(w, "metal_al", doping_cm3=1e18)["vth_v"]
    assert v_hi > v_lo > 0


def test_depletion_width_decreases_with_doping():
    """最大空乏層幅はドーピングが高いほど薄い。"""
    w = _mos()
    wmax_lo = M.threshold_voltage_v(w, "metal_al", doping_cm3=1e16)["w_max_um"]
    wmax_hi = M.threshold_voltage_v(w, "metal_al", doping_cm3=1e18)["w_max_um"]
    assert wmax_hi < wmax_lo


def test_cv_accumulation_equals_cox():
    """蓄積（V≤Vfb）の容量は Cox に等しい。"""
    cv = M.mos_cv_curve(_mos(), "metal_al", doping_cm3=1e17, v_min=-2, v_max=2)
    assert cv["c_ff_per_um2"][0] == pytest.approx(cv["cox_ff_per_um2"], rel=1e-6)
    assert cv["c_over_cox"].max() == pytest.approx(1.0, rel=1e-6)


def test_cv_inversion_is_cmin_below_cox():
    """反転（V≥Vth）の高周波容量は Cmin < Cox。"""
    cv = M.mos_cv_curve(_mos(), "metal_al", doping_cm3=1e17, v_min=-2, v_max=3)
    assert cv["c_ff_per_um2"][-1] == pytest.approx(cv["cmin_ff_per_um2"], rel=1e-6)
    assert cv["cmin_ff_per_um2"] < cv["cox_ff_per_um2"]


def test_cv_monotonic_non_increasing():
    """C-V は電圧増加に対し単調非増加（蓄積→空乏→反転）。"""
    cv = M.mos_cv_curve(_mos(), "metal_al", doping_cm3=1e17, v_min=-2, v_max=3)
    c = cv["c_ff_per_um2"]
    assert np.all(np.diff(c) <= 1e-9)


def test_cv_no_gate_empty():
    """ゲート構造が無ければ空の結果。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id  # 誘電体/ゲート無し
    cv = M.mos_cv_curve(w, "metal_al")
    assert cv["v"].size == 0 and cv["vth_v"] is None
