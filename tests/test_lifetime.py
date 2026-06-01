"""経時劣化寿命（Black EM-MTTF・TDDB）のテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

K_EV = 8.617333e-5


def test_em_mttf_current_exponent():
    """Black の式: 電流密度 2 倍で MTTF は 1/2^n（n=2）。"""
    m1 = M.electromigration_mttf(1e6, 110, n=2.0)
    m2 = M.electromigration_mttf(2e6, 110, n=2.0)
    assert m1 / m2 == pytest.approx(4.0, rel=1e-6)


def test_em_mttf_arrhenius_temperature():
    """温度が高いほど MTTF は短い（Arrhenius）。"""
    hot = M.electromigration_mttf(1e6, 125, ea_ev=0.9)
    cool = M.electromigration_mttf(1e6, 85, ea_ev=0.9)
    assert cool > hot
    expect = np.exp(0.9 / K_EV / (85 + 273.15)) / np.exp(0.9 / K_EV / (125 + 273.15))
    assert cool / hot == pytest.approx(expect, rel=1e-6)


def test_em_mttf_zero_current_inf():
    assert M.electromigration_mttf(0.0, 125) == float("inf")


def test_tddb_field_acceleration():
    """E モデル: 電界が高いほど TDDB 寿命が指数的に短い。"""
    low = M.tddb_lifetime(5.0, 125, gamma=4.0)
    high = M.tddb_lifetime(8.0, 125, gamma=4.0)
    assert low > high
    assert low / high == pytest.approx(np.exp(-4.0 * 5.0) / np.exp(-4.0 * 8.0), rel=1e-6)


def test_tddb_temperature_and_zero_field():
    """温度が高いほど TDDB 寿命は短い。E=0 は inf。"""
    assert M.tddb_lifetime(6.0, 85) > M.tddb_lifetime(6.0, 125)
    assert M.tddb_lifetime(0.0, 125) == float("inf")


def test_em_lifetime_wafer_couples_current_density():
    """配線結合: 細い配線（高 J）ほど MTTF が短い。"""
    def wire(half_w):
        cfg = WaferConfig(nx=40, ny=40, nz=20, pitch_um=0.05, substrate_um=0.0)
        w = Wafer(cfg)
        w.grid[:] = materials.get("oxide").id
        w.grid[8:12, 20 - half_w:20 + half_w, :] = materials.get("metal_cu").id
        return w
    narrow = M.em_lifetime_wafer(wire(3), "metal_cu", 1.0, 110)
    wide = M.em_lifetime_wafer(wire(9), "metal_cu", 1.0, 110)
    assert narrow["j_max_a_cm2"] > wide["j_max_a_cm2"]
    assert narrow["mttf"] < wide["mttf"]


def test_em_lifetime_wafer_open_zero():
    """断線配線は mttf=0（j_max=inf）。"""
    cfg = WaferConfig(nx=40, ny=20, nz=20, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get("oxide").id
    w.grid[8:12, 8:12, :15] = materials.get("metal_cu").id
    w.grid[8:12, 8:12, 25:] = materials.get("metal_cu").id  # 分断
    res = M.em_lifetime_wafer(w, "metal_cu", 1.0, 110)
    assert res["mttf"] == 0.0
