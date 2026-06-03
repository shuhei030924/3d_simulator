"""MOS ボディ効果（基板バイアスによる Vth シフト）のテスト。"""
import math

import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

_Q = 1.602176634e-19
_EPS_SI = 11.7 * 8.854e-12


def _mos() -> Wafer:
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:16, :, :] = materials.get("metal_al").id
    return w


def test_zero_vsb_gives_vth0():
    """Vsb=0 では Vth=Vth0（シフト無し）。"""
    w = _mos()
    r = M.body_effect(w, "metal_al", vsb=0.0)
    assert r["dvth_v"] == pytest.approx(0.0, abs=1e-12)
    assert r["vth_v"] == pytest.approx(r["vth0_v"], rel=1e-12)


def test_vth_increases_with_vsb():
    """逆バイアス Vsb が増えると Vth が上がる。"""
    w = _mos()
    lo = M.body_effect(w, "metal_al", vsb=0.5)["vth_v"]
    hi = M.body_effect(w, "metal_al", vsb=2.0)["vth_v"]
    assert hi > lo > M.body_effect(w, "metal_al", vsb=0.0)["vth_v"]


def test_gamma_formula():
    """ボディ係数 γ=√(2·εs·q·Na)/Cox。"""
    w = _mos()
    na_cm3 = 1e17
    r = M.body_effect(w, "metal_al", vsb=1.0, doping_cm3=na_cm3)
    th = M.threshold_voltage_v(w, "metal_al", doping_cm3=na_cm3)
    na = na_cm3 * 1e6
    expected = math.sqrt(2 * _EPS_SI * _Q * na) / th["cox_f_m2"]
    assert r["gamma_sqrt_v"] == pytest.approx(expected, rel=1e-9)


def test_gamma_scales_with_sqrt_doping():
    """γ∝√Na（4 倍ドーピングで γ 2 倍）。"""
    w = _mos()
    g1 = M.body_effect(w, "metal_al", vsb=1.0, doping_cm3=1e17)["gamma_sqrt_v"]
    g4 = M.body_effect(w, "metal_al", vsb=1.0, doping_cm3=4e17)["gamma_sqrt_v"]
    assert g4 == pytest.approx(2.0 * g1, rel=1e-6)


def test_dvth_matches_formula():
    """ΔVth=γ(√(2φF+Vsb)−√(2φF))。"""
    w = _mos()
    r = M.body_effect(w, "metal_al", vsb=1.5)
    expected = r["gamma_sqrt_v"] * (
        math.sqrt(2 * r["phi_f_v"] + 1.5) - math.sqrt(2 * r["phi_f_v"]))
    assert r["dvth_v"] == pytest.approx(expected, rel=1e-9)


def test_negative_vsb_raises():
    """負の Vsb はエラー。"""
    w = _mos()
    with pytest.raises(ValueError):
        M.body_effect(w, "metal_al", vsb=-0.5)


def test_no_gate_returns_none():
    """ゲート無しでは Vth=None。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.body_effect(w, "metal_al", vsb=1.0)
    assert r["vth_v"] is None


def test_dvth_concave_in_vsb():
    """ΔVth は Vsb に対して凹（増分が逓減, √ 依存）。"""
    w = _mos()
    d = [M.body_effect(w, "metal_al", vsb=float(v))["dvth_v"]
         for v in (0.0, 0.5, 1.0, 1.5, 2.0)]
    diffs = np.diff(d)
    assert np.all(diffs > 0)
    assert np.all(np.diff(diffs) < 0)  # 増分が単調減少（凹）
