"""局所応力集中マップのテスト。"""
import numpy as np

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _wafer() -> Wafer:
    cfg = WaferConfig(nx=30, ny=30, nz=30, pitch_um=0.05, substrate_um=0.0)
    return Wafer(cfg)


def test_uniform_film_interior_zero():
    """均一膜の内部はミスマッチ 0（応力集中なし）。"""
    w = _wafer()
    w.grid[5:15, :, :] = materials.get("oxide").id
    c = M.stress_concentration_map(w)
    assert c[10, 15, 15] == 0.0  # 膜中央（全周同材料）
    assert (c == 0.0)[w.grid == materials.AIR].all()  # 空気は 0


def test_interface_mismatch_concentration():
    """高応力 Si3N4(+1000) と SiO2(-300) の界面で Δσ=1300 の集中。"""
    w = _wafer()
    w.grid[5:10, :, :] = materials.get("oxide").id
    w.grid[10:15, :, :] = materials.get("nitride").id
    c = M.stress_concentration_map(w)
    mismatch = abs(1000.0 - (-300.0))
    # 界面ボクセルは異材 1 面 → Kt=1.5
    assert c.max() == np.float64(mismatch * 1.5)


def test_convex_corner_higher_than_flat():
    """同じ高応力膜でも凸角（露出面多）の方が平坦面より集中が高い。"""
    w = _wafer()
    # 平坦な厚膜
    w.grid[5:15, :, :] = materials.get("nitride").id
    flat_max = M.stress_concentration_map(w).max()
    # 小ブロック（凸角あり）
    w2 = _wafer()
    w2.grid[5:8, 13:17, 13:17] = materials.get("nitride").id
    corner_max = M.stress_concentration_map(w2).max()
    assert corner_max > flat_max


def test_max_stress_concentration_dict():
    """最大応力集中の値・位置・材料を返す。"""
    w = _wafer()
    w.grid[5:10, :, :] = materials.get("oxide").id
    w.grid[10:15, :, :] = materials.get("nitride").id
    mx = M.max_stress_concentration(w)
    assert mx["value_mpa"] > 0
    assert mx["location"] is not None
    assert mx["material"] in ("oxide", "nitride")


def test_empty_wafer_zero():
    """固体の無いウェハは集中 0。"""
    w = _wafer()
    w.grid[:] = materials.AIR
    assert M.max_stress_concentration(w)["value_mpa"] == 0.0
