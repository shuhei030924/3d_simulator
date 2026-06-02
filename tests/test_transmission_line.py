"""伝送線路パラメータ（Z0・インダクタンス・信号遅延）のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

C0 = 2.99792458e8  # 光速 m/s


def _microstrip(diel: str) -> Wafer:
    cfg = WaferConfig(nx=60, ny=20, nz=30, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get(diel).id
    g[5:8, :, :] = materials.get("metal_al").id        # 基準面
    g[15:18, 8:12, :] = materials.get("metal_cu").id    # 信号線（x 方向）
    return w


def test_eps_eff_matches_dielectric():
    """一様誘電体では実効比誘電率が εr に一致する。"""
    for diel in ("oxide", "low_k", "air"):
        r = M.transmission_line_params(_microstrip(diel), "metal_cu", "metal_al", "x")
        assert r["eps_eff"] == pytest.approx(materials.get(diel).rel_permittivity, rel=0.02)


def test_velocity_is_c_over_sqrt_eps():
    """信号速度 v = c/√εr_eff（真空で c）。"""
    import numpy as np
    r = M.transmission_line_params(_microstrip("oxide"), "metal_cu", "metal_al", "x")
    assert r["signal_velocity_m_s"] == pytest.approx(C0 / np.sqrt(3.9), rel=0.02)
    r_air = M.transmission_line_params(_microstrip("air"), "metal_cu", "metal_al", "x")
    assert r_air["signal_velocity_m_s"] == pytest.approx(C0, rel=0.02)


def test_inductance_independent_of_dielectric():
    """インダクタンスは幾何のみで決まり誘電体に依らない（TEM の性質）。"""
    l_ox = M.transmission_line_params(_microstrip("oxide"), "metal_cu", "metal_al", "x")
    l_air = M.transmission_line_params(_microstrip("air"), "metal_cu", "metal_al", "x")
    assert l_ox["inductance_ph_per_um"] == pytest.approx(
        l_air["inductance_ph_per_um"], rel=0.02)


def test_z0_lower_for_higher_permittivity():
    """高 εr ほど容量が増え特性インピーダンス Z0 が下がる。"""
    z_ox = M.transmission_line_params(_microstrip("oxide"), "metal_cu", "metal_al", "x")["z0_ohm"]
    z_air = M.transmission_line_params(_microstrip("air"), "metal_cu", "metal_al", "x")["z0_ohm"]
    assert z_air > z_ox > 0


def test_z0_equals_sqrt_l_over_c():
    """Z0 = √(L'/C') の関係が成り立つ。"""
    r = M.transmission_line_params(_microstrip("oxide"), "metal_cu", "metal_al", "x")
    length_m = r["length_um"] * 1e-6
    l_prime = r["inductance_ph_per_um"] * 1e-12 / 1e-6  # pH/µm → H/m
    c_prime = (r["capacitance_ff"] * 1e-15) / length_m
    assert r["z0_ohm"] == pytest.approx((l_prime / c_prime) ** 0.5, rel=1e-3)


def test_absent_conductor_zero():
    w = _microstrip("oxide")
    w.grid[w.grid == materials.get("metal_cu").id] = materials.get("oxide").id
    r = M.transmission_line_params(w, "metal_cu", "metal_al", "x")
    assert r["z0_ohm"] == 0.0
