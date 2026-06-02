"""分布RC Elmore遅延・EM Blech不死条件のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _line(nx=80) -> Wafer:
    cfg = WaferConfig(nx=nx, ny=20, nz=20, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[3:6, :, :] = materials.get("metal_al").id        # 基準面
    g[12:15, 8:12, :] = materials.get("metal_cu").id    # 配線(x方向)
    return w


def test_elmore_about_half_lumped_for_uniform_line():
    """一様配線では Elmore 遅延 ≈ ½·集中定数 RC。"""
    e = M.elmore_delay_ps(_line(), "metal_cu", "metal_al", "x")
    assert e["elmore_delay_ps"] == pytest.approx(0.5 * e["lumped_rc_ps"], rel=0.05)


def test_elmore_less_than_lumped():
    """Elmore（分布）は集中定数 RC より小さい。"""
    e = M.elmore_delay_ps(_line(), "metal_cu", "metal_al", "x")
    assert 0 < e["elmore_delay_ps"] < e["lumped_rc_ps"]


def test_elmore_open_is_inf():
    """断線配線では Elmore 遅延 inf。"""
    w = _line()
    w.grid[12:15, 8:12, 38:42] = materials.get("oxide").id  # 中央分断
    e = M.elmore_delay_ps(w, "metal_cu", "metal_al", "x")
    assert e["elmore_delay_ps"] == float("inf")


def _blech_line(nx: int) -> Wafer:
    cfg = WaferConfig(nx=nx, ny=10, nz=10, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[3:6, 4:6, :] = materials.get("metal_cu").id
    return w


def test_blech_jl_proportional_to_length():
    """j·L 積は配線長に比例する。"""
    b1 = M.blech_immortal(_blech_line(40), "metal_cu", 0.5, "x")
    b2 = M.blech_immortal(_blech_line(80), "metal_cu", 0.5, "x")
    assert b2["jl_product_a_cm"] == pytest.approx(2 * b1["jl_product_a_cm"], rel=0.05)


def test_blech_short_immortal_long_fails():
    """短い配線は EM 不死、長い配線は故障（jL が閾値超）。"""
    short = M.blech_immortal(_blech_line(20), "metal_cu", 0.5, "x", jl_threshold_a_cm=4000.0)
    long = M.blech_immortal(_blech_line(300), "metal_cu", 0.5, "x", jl_threshold_a_cm=4000.0)
    assert short["immortal"]
    assert not long["immortal"]


def test_blech_jl_proportional_to_current():
    """j·L 積は電流に比例する。"""
    b1 = M.blech_immortal(_blech_line(80), "metal_cu", 0.5, "x")
    b2 = M.blech_immortal(_blech_line(80), "metal_cu", 1.0, "x")
    assert b2["jl_product_a_cm"] == pytest.approx(2 * b1["jl_product_a_cm"], rel=1e-6)
