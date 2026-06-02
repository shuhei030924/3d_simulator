"""温度依存抵抗（TCR）のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _wire(mat: str) -> Wafer:
    cfg = WaferConfig(nx=40, ny=20, nz=20, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[8:11, 8:12, :] = materials.get(mat).id
    return w


def test_resistance_increases_with_temperature():
    """金属（正 TCR）は高温で抵抗が増える。"""
    w = _wire("metal_cu")
    r20 = M.resistance_at_temperature(w, "metal_cu", 20.0)["r_t_ohm"]
    r125 = M.resistance_at_temperature(w, "metal_cu", 125.0)["r_t_ohm"]
    assert r125 > r20


def test_tcr_ratio_exact():
    """R(T)/R₀ = 1+TCR·(T−T₀) に厳密一致。"""
    w = _wire("metal_cu")
    res = M.resistance_at_temperature(w, "metal_cu", 125.0, ref_temp_c=20.0)
    tcr = materials.get("metal_cu").tcr_per_k
    assert res["ratio"] == pytest.approx(1 + tcr * (125 - 20), rel=1e-9)
    assert res["r_t_ohm"] == pytest.approx(res["r_ref_ohm"] * res["ratio"], rel=1e-9)


def test_at_reference_temp_ratio_one():
    """基準温度では ratio=1。"""
    w = _wire("metal_al")
    res = M.resistance_at_temperature(w, "metal_al", 20.0, ref_temp_c=20.0)
    assert res["ratio"] == pytest.approx(1.0, rel=1e-9)


def test_open_line_inf():
    """断線配線は inf。"""
    w = _wire("metal_cu")
    w.grid[8:11, 8:12, 18:22] = materials.get("oxide").id  # 分断
    assert M.resistance_at_temperature(w, "metal_cu", 100.0)["r_t_ohm"] == float("inf")


def test_tungsten_higher_tcr_than_copper():
    """W の TCR は Cu より大きい（同 ΔT で抵抗増加率が大）。"""
    tcr_w = materials.get("tungsten").tcr_per_k
    tcr_cu = materials.get("metal_cu").tcr_per_k
    assert tcr_w > tcr_cu > 0
