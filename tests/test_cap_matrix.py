"""Maxwell 容量行列抽出のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _three() -> Wafer:
    """Al(中央) / Cu(下) / W(上) を酸化膜中に積む。"""
    cfg = WaferConfig(nx=24, ny=6, nz=30, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[14:17, :, :] = materials.get("metal_al").id
    g[6:9, :, :] = materials.get("metal_cu").id
    g[22:25, :, :] = materials.get("tungsten").id
    return w


def test_matrix_shape_and_symmetry():
    """容量行列は n×n で対称（C_ik = C_ki）。"""
    r = M.capacitance_matrix_ff(_three(), ["metal_al", "metal_cu", "tungsten"])
    assert r["conductors"] == ["metal_al", "metal_cu", "tungsten"]
    mat = r["matrix_ff"]
    assert len(mat) == 3 and all(len(row) == 3 for row in mat)
    for i in range(3):
        for k in range(3):
            assert mat[i][k] == pytest.approx(mat[k][i], rel=0.02, abs=1e-4)


def test_diagonal_equals_sum_of_couplings():
    """接地系では自己容量 C_kk = Σ|C_ik|（対基板無し構成）。"""
    r = M.capacitance_matrix_ff(_three(), ["metal_al", "metal_cu", "tungsten"])
    cpl = r["coupling_ff"]
    al_couplings = cpl["metal_cu-metal_al"] + cpl["tungsten-metal_al"]
    assert r["self_ff"]["metal_al"] == pytest.approx(al_couplings, rel=0.02)


def test_middle_conductor_shields_outer_pair():
    """中央導体（Al）が上下（Cu/W）の直接結合を遮蔽し Cu-W 結合は小さい。"""
    r = M.capacitance_matrix_ff(_three(), ["metal_al", "metal_cu", "tungsten"])
    assert r["coupling_ff"]["tungsten-metal_cu"] < 0.1 * r["coupling_ff"]["metal_cu-metal_al"]


def test_matrix_coupling_matches_pairwise_field():
    """行列の隣接結合は pairwise 静電界容量と整合する。"""
    w = _three()
    r = M.capacitance_matrix_ff(w, ["metal_al", "metal_cu", "tungsten"])
    c_pair = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    assert r["coupling_ff"]["metal_cu-metal_al"] == pytest.approx(c_pair, rel=0.1)


def test_matrix_empty_for_single_conductor():
    """導体が 2 未満なら空の結果。"""
    cfg = WaferConfig(nx=20, ny=6, nz=20, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get("oxide").id
    w.grid[8:11, :, :] = materials.get("metal_al").id
    r = M.capacitance_matrix_ff(w, ["metal_al", "metal_cu"])
    assert r["conductors"] == [] and r["matrix_ff"] == []
