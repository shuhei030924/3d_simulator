"""熱伝導率・スタック熱抵抗メトロロジのテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _slab(mat: str, n: int = 20, pitch: float = 0.1) -> Wafer:
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get(mat).id
    return w


def test_thermal_conductivity_values():
    """熱伝導率が物理的に妥当な順序（金属≫半導体≫酸化膜≫low-k）。"""
    k = lambda n: materials.get(n).thermal_conductivity_w_mk  # noqa: E731
    assert k("metal_cu") > k("silicon") > k("oxide") > k("low_k")
    assert k("metal_cu") > k("metal_al")  # Cu の方が高い


def test_thermal_resistance_matches_analytic():
    """均一スラブで R = L/(k·A) の解析値に厳密一致。"""
    n, pitch = 20, 0.1
    for mat in ("silicon", "oxide"):
        w = _slab(mat, n, pitch)
        r = M.thermal_resistance_k_w(w)
        k = materials.get(mat).thermal_conductivity_w_mk
        length = n * pitch * 1e-6
        area = (n * pitch * 1e-6) ** 2
        assert r == pytest.approx(length / (k * area), rel=1e-6)


def test_oxide_more_resistive_than_silicon():
    """酸化膜スラブは Si スラブより熱抵抗が k 比（150/1.4）倍高い。"""
    r_si = M.thermal_resistance_k_w(_slab("silicon"))
    r_ox = M.thermal_resistance_k_w(_slab("oxide"))
    ratio = materials.get("silicon").thermal_conductivity_w_mk \
        / materials.get("oxide").thermal_conductivity_w_mk
    assert r_ox == pytest.approx(r_si * ratio, rel=1e-6)


def test_low_k_layer_raises_stack_resistance():
    """低 k 膜を積むとスタック熱抵抗が増える。"""
    w = _slab("silicon")
    r_base = M.thermal_resistance_k_w(w)
    w.grid[10:, :, :] = materials.get("low_k").id  # 上半分を low-k
    assert M.thermal_resistance_k_w(w) > r_base


def test_thermal_resistance_map_hotspot():
    """局所的に低 k のピラーがある列の熱抵抗が高い（ホットスポット）。"""
    w = _slab("silicon")
    w.grid[:, 5, 5] = materials.get("low_k").id
    rmap = M.thermal_resistance_map(w)
    assert rmap[5, 5] > rmap[0, 0]
    assert np.isfinite(rmap).all()  # 全列に固体あり


def test_empty_column_is_inf():
    """固体の無い列は熱抵抗 inf、全空気ウェハは inf。"""
    cfg = WaferConfig(nx=10, ny=10, nz=10, pitch_um=0.1, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.AIR  # 全 air（基板も除去）
    assert M.thermal_resistance_k_w(w) == float("inf")
    rmap = M.thermal_resistance_map(w)
    assert np.isinf(rmap).all()
