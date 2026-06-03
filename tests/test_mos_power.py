"""CMOS 消費電力（動的＋静的リーク）のテスト。"""
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


def test_dynamic_power_formula():
    """P_dyn = α·C·Vdd²·f。"""
    w = _mos()
    r = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=1e9,
                                load_cap_ff=10.0, activity=0.15)
    expected = 0.15 * 10e-15 * 1.0 ** 2 * 1e9
    assert r["p_dynamic_w"] == pytest.approx(expected, rel=1e-9)


def test_static_power_is_ioff_times_vdd():
    """P_static = Ioff·Vdd。"""
    w = _mos()
    r = M.mos_power_dissipation(w, "metal_al", vdd=1.2, freq_hz=1e9,
                                load_cap_ff=10.0)
    assert r["p_static_w"] == pytest.approx(r["ioff_a"] * 1.2, rel=1e-9)


def test_total_is_sum():
    """P_total = P_dyn + P_static。"""
    w = _mos()
    r = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=1e9,
                                load_cap_ff=10.0)
    assert r["p_total_w"] == pytest.approx(r["p_dynamic_w"] + r["p_static_w"], rel=1e-12)


def test_dynamic_scales_linearly_with_frequency():
    """動的電力は周波数に比例。"""
    w = _mos()
    lo = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=1e9,
                                 load_cap_ff=10.0)["p_dynamic_w"]
    hi = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=3e9,
                                 load_cap_ff=10.0)["p_dynamic_w"]
    assert hi == pytest.approx(3.0 * lo, rel=1e-9)


def test_static_fraction_rises_at_low_frequency():
    """周波数を下げると静的電力の比率が上がる。"""
    w = _mos()
    fast = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=1e10,
                                   load_cap_ff=10.0)["static_fraction"]
    slow = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=1e2,
                                   load_cap_ff=10.0)["static_fraction"]
    assert slow > fast >= 0


def test_invalid_activity_raises():
    """活性率が範囲外ならエラー。"""
    w = _mos()
    with pytest.raises(ValueError):
        M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=1e9,
                                load_cap_ff=10.0, activity=1.5)


def test_zero_frequency_only_static():
    """周波数 0 では動的電力 0・全電力＝静的電力。"""
    w = _mos()
    r = M.mos_power_dissipation(w, "metal_al", vdd=1.0, freq_hz=0.0,
                                load_cap_ff=10.0)
    assert r["p_dynamic_w"] == 0.0
    assert r["p_total_w"] == pytest.approx(r["p_static_w"], rel=1e-12)
    assert r["static_fraction"] == pytest.approx(1.0, rel=1e-9)
