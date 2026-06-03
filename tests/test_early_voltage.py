"""MOS アーリー電圧 VA=Id/gds のテスト。"""
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


def test_va_equals_id_over_gds():
    """VA = Id/gds。"""
    w = _mos()
    vth = _vth(w)
    r = M.early_voltage(w, "metal_al", vg=vth + 0.5, vd=1.5, lambda_per_v=0.1)
    assert r["early_voltage_v"] == pytest.approx(r["id_a"] / r["gds_s"], rel=1e-9)


def test_gain_decomposition():
    """真性利得 Av = (gm/Id)·VA。"""
    w = _mos()
    vth = _vth(w)
    r = M.early_voltage(w, "metal_al", vg=vth + 0.5, vd=1.5, lambda_per_v=0.1)
    assert r["intrinsic_gain"] == pytest.approx(
        r["gm_id_per_v"] * r["early_voltage_v"], rel=1e-9)


def test_va_approx_inverse_lambda():
    """VA ≈ 1/λ + Vd（チャネル長変調モデル）。"""
    w = _mos()
    vth = _vth(w)
    vd = 1.5
    for lam in (0.05, 0.1, 0.2):
        r = M.early_voltage(w, "metal_al", vg=vth + 0.5, vd=vd, lambda_per_v=lam)
        assert r["early_voltage_v"] == pytest.approx(1.0 / lam + vd, rel=0.02)


def test_smaller_lambda_larger_va():
    """λ が小さい（長チャネル）ほど VA が大きい。"""
    w = _mos()
    vth = _vth(w)
    small_l = M.early_voltage(w, "metal_al", vg=vth + 0.5, vd=1.5,
                              lambda_per_v=0.05)["early_voltage_v"]
    large_l = M.early_voltage(w, "metal_al", vg=vth + 0.5, vd=1.5,
                              lambda_per_v=0.2)["early_voltage_v"]
    assert small_l > large_l


def test_no_channel_modulation_huge_va():
    """λ=0（理想）では gds≈0・VA が非常に大きい。"""
    w = _mos()
    vth = _vth(w)
    r = M.early_voltage(w, "metal_al", vg=vth + 0.5, vd=1.5, lambda_per_v=0.0)
    assert r["early_voltage_v"] > 100.0 or math.isinf(r["early_voltage_v"])
