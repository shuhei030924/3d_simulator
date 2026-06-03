"""MOS フリッカ（1/f）雑音とノイズコーナーのテスト。"""
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


def test_flicker_inversely_proportional_to_frequency():
    """1/f 雑音は周波数に反比例（f 倍増で半減）。"""
    w = _mos()
    vth = _vth(w)
    r1 = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=1e3)
    r2 = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=2e3)
    assert r2["svg_flicker_v2_hz"] == pytest.approx(
        0.5 * r1["svg_flicker_v2_hz"], rel=1e-9)


def test_flicker_equals_thermal_at_corner():
    """コーナー周波数で 1/f 雑音 = 熱雑音。"""
    w = _mos()
    vth = _vth(w)
    fc = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5,
                             freq_hz=1e3)["corner_freq_hz"]
    r = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=fc)
    assert r["svg_flicker_v2_hz"] == pytest.approx(r["svg_thermal_v2_hz"], rel=1e-9)


def test_total_is_sum_of_flicker_and_thermal():
    """総雑音 = 1/f + 熱。"""
    w = _mos()
    vth = _vth(w)
    r = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=1e4)
    assert r["svg_total_v2_hz"] == pytest.approx(
        r["svg_flicker_v2_hz"] + r["svg_thermal_v2_hz"], rel=1e-12)


def test_larger_kf_higher_corner():
    """Kf が大きいほどコーナー周波数が高い。"""
    w = _mos()
    vth = _vth(w)
    lo = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5,
                             freq_hz=1e3, kf_v2f=1e-25)["corner_freq_hz"]
    hi = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5,
                             freq_hz=1e3, kf_v2f=1e-24)["corner_freq_hz"]
    assert hi == pytest.approx(10.0 * lo, rel=1e-6)


def test_flicker_dominates_below_corner():
    """コーナー以下では 1/f 雑音が支配的、以上では熱雑音が支配的。"""
    w = _mos()
    vth = _vth(w)
    fc = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5,
                             freq_hz=1e3)["corner_freq_hz"]
    low = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=fc / 100)
    high = M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=fc * 100)
    assert low["svg_flicker_v2_hz"] > low["svg_thermal_v2_hz"]
    assert high["svg_flicker_v2_hz"] < high["svg_thermal_v2_hz"]


def test_invalid_frequency_raises():
    """周波数が非正ならエラー。"""
    w = _mos()
    vth = _vth(w)
    with pytest.raises(ValueError):
        M.mos_flicker_noise(w, "metal_al", vg=vth + 0.5, vd=1.5, freq_hz=0.0)


def test_no_gate_zero_corner():
    """ゲート無し（C_ox=0）ではコーナー周波数 0。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.mos_flicker_noise(w, "metal_al", vg=1.0, vd=1.5, freq_hz=1e3)
    assert r["corner_freq_hz"] == 0.0
    assert math.isinf(r["svg_flicker_v2_hz"])
