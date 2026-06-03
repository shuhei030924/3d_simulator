"""MOS マッチング（Pelgrom 則 σ(ΔVth)=AVT/√WL）のテスト。"""
import math

import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _mos(nx=20, ny=20, pitch=0.1) -> Wafer:
    cfg = WaferConfig(nx=nx, ny=ny, nz=30, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:16, :, :] = materials.get("metal_al").id
    return w


def test_sigma_vth_equals_avt_over_sqrt_area():
    """σ(ΔVth) = A_VT/√(W·L)。"""
    w = _mos()
    r = M.mos_mismatch(w, "metal_al", avt_mv_um=3.0)
    expected = 3.0 / math.sqrt(r["gate_area_um2"])
    assert r["sigma_vth_mv"] == pytest.approx(expected, rel=1e-9)


def test_sigma_halves_when_area_quadruples():
    """面積 4 倍で σ は半分（1/√面積）。"""
    small = M.mos_mismatch(_mos(20, 20, 0.1), "metal_al", avt_mv_um=3.0)
    big = M.mos_mismatch(_mos(40, 40, 0.1), "metal_al", avt_mv_um=3.0)
    assert big["gate_area_um2"] == pytest.approx(4 * small["gate_area_um2"], rel=1e-9)
    assert big["sigma_vth_mv"] == pytest.approx(0.5 * small["sigma_vth_mv"], rel=1e-9)


def test_nsigma_is_n_times_sigma():
    """nσ オフセット = n_sigma·σ。"""
    w = _mos()
    r = M.mos_mismatch(w, "metal_al", avt_mv_um=3.0, n_sigma=3.0)
    assert r["nsigma_vth_mv"] == pytest.approx(3.0 * r["sigma_vth_mv"], rel=1e-12)


def test_beta_mismatch_scales_with_area():
    """電流係数ミスマッチも 1/√面積。"""
    w = _mos()
    r = M.mos_mismatch(w, "metal_al", abeta_pct_um=1.0)
    assert r["sigma_beta_pct"] == pytest.approx(1.0 / math.sqrt(r["gate_area_um2"]),
                                                rel=1e-9)


def test_negative_coefficient_raises():
    """負のマッチング係数はエラー。"""
    w = _mos()
    with pytest.raises(ValueError):
        M.mos_mismatch(w, "metal_al", avt_mv_um=-1.0)


def test_no_gate_returns_inf():
    """ゲート面積 0 では σ=inf。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.1, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.mos_mismatch(w, "metal_al")
    assert math.isinf(r["sigma_vth_mv"])
    assert r["gate_area_um2"] == 0.0
