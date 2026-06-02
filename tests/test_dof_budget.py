"""平坦化 DOF バジェット検証のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _flat() -> Wafer:
    cfg = WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.05, substrate_um=0.5)
    w = Wafer(cfg)
    w.grid[:15, :, :] = materials.get("oxide").id
    return w


def _ramp() -> Wafer:
    """x に沿って階段状に高くなる傾斜トポグラフィ。"""
    cfg = WaferConfig(nx=40, ny=20, nz=60, pitch_um=0.05, substrate_um=0.5)
    w = Wafer(cfg)
    g = w.grid
    g[:15, :, :] = materials.get("oxide").id
    for x in range(40):
        extra = x // 4  # 0..9 ボクセル
        if extra:
            g[15:15 + extra, :, x] = materials.get("oxide").id
    return w


def test_flat_within_dof():
    """平坦面は焦点内（範囲0, 焦点外れ0）。"""
    r = M.planarization_dof_check(_flat(), 0.2)
    assert r["surface_range_um"] == 0.0
    assert r["out_of_focus_fraction"] == 0.0
    assert r["within_dof"]


def test_step_exceeds_dof():
    """高低差が DOF を超えると within_dof=False, 焦点外れ>0。"""
    w = _flat()
    w.grid[15:25, :, :20] = materials.get("oxide").id  # 半分だけ 0.5µm 高い
    r = M.planarization_dof_check(w, 0.2)
    assert r["surface_range_um"] == pytest.approx(0.5)
    assert not r["within_dof"]
    assert r["out_of_focus_fraction"] > 0.0


def test_out_of_focus_decreases_with_larger_dof():
    """DOF を広げると焦点外れ領域が減る（単調）。"""
    w = _ramp()
    f_small = M.planarization_dof_check(w, 0.1)["out_of_focus_fraction"]
    f_large = M.planarization_dof_check(w, 0.4)["out_of_focus_fraction"]
    assert f_large <= f_small
    assert 0.0 < f_small <= 1.0


def test_ramp_partial_out_of_focus():
    """傾斜トポグラフィでは焦点外れが部分的（0〜1 の中間）。"""
    r = M.planarization_dof_check(_ramp(), 0.15)
    assert 0.0 < r["out_of_focus_fraction"] < 1.0


def test_invalid_dof_raises():
    with pytest.raises(ValueError):
        M.planarization_dof_check(_flat(), 0.0)


def test_empty_wafer():
    cfg = WaferConfig(nx=10, ny=10, nz=10, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.AIR
    r = M.planarization_dof_check(w, 0.2)
    assert r["within_dof"] and r["out_of_focus_fraction"] == 0.0
