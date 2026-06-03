"""リングオシレータ発振周波数 f_osc=1/(2·N·τ_pd) のテスト。"""
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


def test_fosc_equals_one_over_2n_tau():
    """f_osc = 1/(2·N·τ_pd)。"""
    w = _mos()
    d = M.gate_switching_delay_ps(w, "metal_al", load_cap_ff=1.0, vdd=1.5)["delay_ps"]
    r = M.ring_oscillator_frequency(w, "metal_al", n_stages=5,
                                    load_cap_ff=1.0, vdd=1.5)
    assert r["f_osc_hz"] == pytest.approx(1.0 / (2 * 5 * d * 1e-12), rel=1e-9)
    assert r["stage_delay_ps"] == pytest.approx(d, rel=1e-12)


def test_frequency_inversely_proportional_to_stages():
    """発振周波数は段数に反比例。"""
    w = _mos()
    f5 = M.ring_oscillator_frequency(w, "metal_al", n_stages=5,
                                     load_cap_ff=1.0, vdd=1.5)["f_osc_hz"]
    f15 = M.ring_oscillator_frequency(w, "metal_al", n_stages=15,
                                      load_cap_ff=1.0, vdd=1.5)["f_osc_hz"]
    assert f5 == pytest.approx(3.0 * f15, rel=1e-9)


def test_higher_load_lowers_frequency():
    """負荷容量が大きいほど段遅延が増え発振周波数が下がる。"""
    w = _mos()
    light = M.ring_oscillator_frequency(w, "metal_al", n_stages=11,
                                        load_cap_ff=0.5, vdd=1.5)["f_osc_hz"]
    heavy = M.ring_oscillator_frequency(w, "metal_al", n_stages=11,
                                        load_cap_ff=2.0, vdd=1.5)["f_osc_hz"]
    assert heavy < light


def test_period_is_2n_times_stage_delay():
    """発振周期 T = 2·N·τ_pd。"""
    w = _mos()
    r = M.ring_oscillator_frequency(w, "metal_al", n_stages=7,
                                    load_cap_ff=1.0, vdd=1.5)
    assert r["period_ps"] == pytest.approx(2 * 7 * r["stage_delay_ps"], rel=1e-12)


def test_even_or_too_few_stages_raise():
    """偶数段・3 段未満はエラー。"""
    w = _mos()
    with pytest.raises(ValueError):
        M.ring_oscillator_frequency(w, "metal_al", n_stages=4, load_cap_ff=1.0)
    with pytest.raises(ValueError):
        M.ring_oscillator_frequency(w, "metal_al", n_stages=1, load_cap_ff=1.0)


def test_no_drive_zero_frequency():
    """駆動電流 0（ゲート無し）では f_osc=0。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.ring_oscillator_frequency(w, "metal_al", n_stages=5,
                                    load_cap_ff=1.0, vdd=1.5)
    assert r["f_osc_hz"] == 0.0
    assert math.isinf(r["period_ps"])
