"""ダイオード I-V（Shockley）・自己発熱結合 EM 寿命のテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def test_diode_reverse_saturation():
    """逆方向では電流が −Is に飽和する。"""
    assert M.diode_current(-0.5, i_sat_a=1e-15) == pytest.approx(-1e-15, rel=1e-6)


def test_diode_forward_ideality_slope():
    """順方向 log(I)-V の傾斜は n·(kT/q)·ln10（n=1 で ~60mV/dec）。"""
    for n in (1.0, 2.0):
        i1 = M.diode_current(0.3, ideality=n)
        i2 = M.diode_current(0.4, ideality=n)
        ss = 0.1 / (np.log10(i2) - np.log10(i1)) * 1000
        assert ss == pytest.approx(n * 0.025852 * np.log(10) * 1000, rel=0.02)


def test_diode_series_resistance_limits_current():
    """直列抵抗があると高電圧で電流が制限される。"""
    i_ideal = M.diode_current(0.8, i_sat_a=1e-12, series_r_ohm=0.0)
    i_rs = M.diode_current(0.8, i_sat_a=1e-12, series_r_ohm=10.0)
    assert i_rs < i_ideal
    # Rs により電圧降下後の接合電圧と整合（I·Rs < V）
    assert i_rs * 10.0 < 0.8


def test_diode_iv_curve_monotonic():
    """ダイオード I-V は電圧に対し単調増加。"""
    cv = M.diode_iv_curve(v_min=-0.5, v_max=0.7, n_points=40, i_sat_a=1e-14)
    assert np.all(np.diff(cv["i"]) >= -1e-30)
    assert cv["i"][0] < 0 < cv["i"][-1]


def _wire() -> Wafer:
    cfg = WaferConfig(nx=60, ny=20, nz=15, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[3:6, 8:12, :] = materials.get("metal_cu").id
    return w


def test_self_heating_reduces_em_lifetime():
    """自己発熱で接合温度が上がり、EM 寿命が等温時より短くなる。"""
    r = M.em_lifetime_self_heated(_wire(), "metal_cu", 3.0, ambient_c=85.0)
    assert r["delta_t_k"] > 0
    assert r["junction_temp_c"] > 85.0
    assert r["mttf"] < r["mttf_isothermal"]


def test_self_heating_higher_current_more_heating():
    """電流が大きいほど自己発熱 ΔT が大きい（P=I²R）。"""
    lo = M.em_lifetime_self_heated(_wire(), "metal_cu", 1.0, ambient_c=85.0)
    hi = M.em_lifetime_self_heated(_wire(), "metal_cu", 3.0, ambient_c=85.0)
    assert hi["delta_t_k"] > lo["delta_t_k"]
