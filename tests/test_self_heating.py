"""自己発熱（温度上昇）メトロロジのテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _slab(mat: str, n: int = 20, pitch: float = 0.1) -> Wafer:
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get(mat).id
    return w


def test_temperature_rise_linear_in_power():
    """ΔT = P·R_th（電力に比例）。"""
    w = _slab("silicon")
    rth = M.thermal_resistance_k_w(w)
    assert M.temperature_rise_k(w, 1.0) == pytest.approx(rth)
    assert M.temperature_rise_k(w, 2.0) == pytest.approx(2 * rth)
    assert M.temperature_rise_k(w, 0.0) == 0.0


def test_low_k_stack_hotter():
    """低 k 膜を積むと同電力での温度上昇が大きい（放熱が悪い）。"""
    w = _slab("silicon")
    dt_si = M.temperature_rise_k(w, 1.0)
    w.grid[10:, :, :] = materials.get("low_k").id
    assert M.temperature_rise_k(w, 1.0) > dt_si


def test_temperature_rise_negative_power_raises():
    with pytest.raises(ValueError):
        M.temperature_rise_k(_slab("silicon"), -1.0)


def test_joule_self_heating():
    """ジュール発熱 P=I²R と ΔT=P·R_th を返す。"""
    cfg = WaferConfig(nx=40, ny=40, nz=40, pitch_um=0.1, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get("oxide").id
    w.grid[10:13, 10:14, :] = materials.get("metal_al").id
    jh = M.joule_self_heating_k(w, "metal_al", 5.0, "x")
    r = jh["resistance_ohm"]
    assert r > 0
    assert jh["power_w"] == pytest.approx((5e-3) ** 2 * r)
    assert jh["delta_t_k"] == pytest.approx(M.temperature_rise_k(w, jh["power_w"]))


def test_joule_open_is_inf():
    """断線配線では発熱・温度上昇が inf。"""
    cfg = WaferConfig(nx=40, ny=20, nz=30, pitch_um=0.1, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get("oxide").id
    w.grid[10:13, 8:12, :15] = materials.get("metal_al").id
    w.grid[10:13, 8:12, 25:] = materials.get("metal_al").id  # 間が途切れ
    jh = M.joule_self_heating_k(w, "metal_al", 1.0, "x")
    assert jh["delta_t_k"] == float("inf")
