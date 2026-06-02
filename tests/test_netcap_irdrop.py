"""対全導体総容量・IRドロップのテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _three_conductor() -> Wafer:
    cfg = WaferConfig(nx=24, ny=8, nz=30, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[14:17, :, :] = materials.get("metal_al").id   # A（対象）
    g[6:9, :, :] = materials.get("metal_cu").id       # 下 Cu
    g[22:25, :, :] = materials.get("tungsten").id     # 上 W
    return w


def test_total_net_cap_exceeds_single_pair():
    """総容量（全導体接地）は単一ペア容量より大きい。"""
    w = _three_conductor()
    c_single = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    c_total = M.total_net_capacitance_ff(w, "metal_al")
    assert c_total > c_single > 0


def test_total_net_cap_zero_without_others():
    """他に導体が無ければ総容量 0。"""
    cfg = WaferConfig(nx=20, ny=8, nz=20, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[8:11, :, :] = materials.get("metal_al").id  # A のみ
    assert M.total_net_capacitance_ff(w, "metal_al") == 0.0


def _power_rail(nx=80) -> Wafer:
    cfg = WaferConfig(nx=nx, ny=20, nz=15, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[3:6, 8:12, :] = materials.get("metal_cu").id
    return w


def test_ir_drop_proportional_to_current():
    """IR ドロップは電流に比例する。"""
    w = _power_rail()
    d1 = M.ir_drop_v(w, "metal_cu", 1.0)
    d5 = M.ir_drop_v(w, "metal_cu", 5.0)
    assert d5["ir_drop_v"] == pytest.approx(5 * d1["ir_drop_v"], rel=1e-9)


def test_ir_drop_higher_for_longer_rail():
    """長い電源配線ほど抵抗が増え IR ドロップが大きい。"""
    short = M.ir_drop_v(_power_rail(40), "metal_cu", 5.0)
    long = M.ir_drop_v(_power_rail(160), "metal_cu", 5.0)
    assert long["ir_drop_v"] > short["ir_drop_v"]


def test_ir_drop_open_inf():
    """断線配線では IR ドロップ inf。"""
    w = _power_rail()
    w.grid[3:6, 8:12, 38:42] = materials.get("oxide").id  # 分断
    assert M.ir_drop_v(w, "metal_cu", 1.0)["ir_drop_v"] == float("inf")
