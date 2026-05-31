"""CVD 表面ラフネスのテスト。"""
from __future__ import annotations

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.processes import CVD
from semisim.recipe import Recipe


def _blanket(roughness, seed=0):
    cfg = WaferConfig(nx=50, ny=50, nz=50, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.6, roughness_um=roughness, seed=seed))
    return r.simulate()


def test_roughness_increases_surface_roughness():
    """roughness_um>0 で計測される表面粗さが増える。"""
    smooth = metrology.surface_roughness_um(_blanket(0.0))
    rough = metrology.surface_roughness_um(_blanket(0.15))
    assert rough > smooth


def test_zero_roughness_is_smooth():
    """roughness_um=0 では平滑（粗さほぼ 0）。"""
    assert metrology.surface_roughness_um(_blanket(0.0)) < 0.02


def test_roughness_deterministic_with_seed():
    """同じシードなら結果が一致する（再現性）。"""
    a = _blanket(0.1, seed=42)
    b = _blanket(0.1, seed=42)
    assert (a.grid == b.grid).all()


def test_roughness_differs_with_seed():
    """異なるシードでは表面形状が変わる。"""
    a = _blanket(0.12, seed=1)
    b = _blanket(0.12, seed=2)
    assert not (a.grid == b.grid).all()


def test_roughness_keeps_material():
    """ラフネス付与後も酸化膜が存在する。"""
    w = _blanket(0.1)
    assert (w.grid == materials.get("oxide").id).any()


def test_roughness_roundtrip_params():
    """params_dict / _from_params で roughness_um と seed を保持。"""
    p = CVD(material="oxide", thickness_um=0.5, roughness_um=0.08, seed=7)
    d = p.params_dict()
    assert d["roughness_um"] == 0.08
    assert d["seed"] == 7
    p2 = CVD._from_params(d)
    assert p2.roughness_um == 0.08
    assert p2.seed == 7
