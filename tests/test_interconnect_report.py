"""配線総合特性レポート（interconnect_report）のテスト。"""
import json

import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _line() -> Wafer:
    cfg = WaferConfig(nx=60, ny=20, nz=30, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[5:8, :, :] = materials.get("metal_al").id        # 基準面
    g[15:18, 8:12, :] = materials.get("metal_cu").id    # 信号線
    return w


def test_report_has_all_keys_and_consistent():
    """総合レポートが全項目を含み、個別関数と整合する。"""
    w = _line()
    rep = M.interconnect_report(w, "metal_cu", "metal_al", current_ma=2.0)
    for key in ("resistance_ohm", "capacitance_ff", "rc_delay_ps", "elmore_delay_ps",
                "inductance_ph_per_um", "z0_ohm", "propagation_delay_ps",
                "current_density_a_cm2", "em_fail", "ir_drop_v", "open"):
        assert key in rep
    # 個別関数との整合
    assert rep["resistance_ohm"] == pytest.approx(M.line_resistance_ohm(w, "metal_cu", "x"))
    assert rep["z0_ohm"] == pytest.approx(
        M.transmission_line_params(w, "metal_cu", "metal_al", axis="y")["z0_ohm"])
    assert not rep["open"]


def test_report_is_json_serializable():
    """レポートは JSON シリアライズ可能（inf は None 化）。"""
    rep = M.interconnect_report(_line(), "metal_cu", "metal_al")
    s = json.dumps(rep)  # 例外が出なければ OK
    assert isinstance(s, str)


def test_report_open_line():
    """断線配線は open=True、抵抗・RC が None。"""
    w = _line()
    w.grid[15:18, 8:12, 28:32] = materials.get("oxide").id  # 分断
    rep = M.interconnect_report(w, "metal_cu", "metal_al")
    assert rep["open"]
    assert rep["resistance_ohm"] is None
    assert rep["rc_delay_ps"] is None


def test_report_em_fail_high_current():
    """過大電流では EM 判定が fail になる。"""
    w = _line()
    safe = M.interconnect_report(w, "metal_cu", "metal_al", current_ma=0.01)
    hot = M.interconnect_report(w, "metal_cu", "metal_al", current_ma=100.0)
    assert not safe["em_fail"]
    assert hot["em_fail"]


def test_report_ir_drop_scales_with_current():
    """IR ドロップは電流に比例する。"""
    w = _line()
    r1 = M.interconnect_report(w, "metal_cu", "metal_al", current_ma=1.0)
    r5 = M.interconnect_report(w, "metal_cu", "metal_al", current_ma=5.0)
    assert r5["ir_drop_v"] == pytest.approx(5 * r1["ir_drop_v"], rel=1e-9)
