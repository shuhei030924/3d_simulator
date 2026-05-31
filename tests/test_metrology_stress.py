"""膜応力によるウェハ反り（Stoney 則）メトロロジのテスト。"""
from __future__ import annotations

from semisim import metrology
from semisim.grid import WaferConfig
from semisim.processes import CVD
from semisim.recipe import Recipe


def _bare():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    return Recipe(config=cfg).simulate()


def test_bare_wafer_no_bow():
    """膜が無い裸ウェハは反らない。"""
    assert abs(metrology.wafer_bow_um(_bare())) < 1e-9


def test_tensile_film_positive_bow():
    """引張膜（窒化膜）は凸（正）に反る。"""
    cfg = WaferConfig(nx=20, ny=20, nz=50, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="nitride", thickness_um=0.5))
    bow = metrology.wafer_bow_um(r.simulate())
    assert bow > 0


def test_compressive_film_negative_bow():
    """圧縮膜（酸化膜）は凹（負）に反る。"""
    cfg = WaferConfig(nx=20, ny=20, nz=50, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    bow = metrology.wafer_bow_um(r.simulate())
    assert bow < 0


def test_thicker_film_more_bow():
    """膜が厚いほど反りが大きい（線形）。"""
    def bow_for(t):
        cfg = WaferConfig(nx=20, ny=20, nz=60, pitch_um=0.1, substrate_um=2.0)
        r = Recipe(config=cfg)
        r.add(CVD(material="tungsten", thickness_um=t))
        return metrology.wafer_bow_um(r.simulate())

    assert bow_for(0.6) > bow_for(0.2) > 0


def test_film_stress_thickness_keys():
    cfg = WaferConfig(nx=20, ny=20, nz=50, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="nitride", thickness_um=0.4))
    st = metrology.film_stress_thickness(r.simulate())
    assert "nitride" in st["per_material"]
    assert st["per_material"]["nitride"]["stress_mpa"] == 1000.0
    assert st["net_N_per_m"] > 0


def test_thinner_substrate_more_bow():
    """基板が薄いほど同じ膜でも反りが大きい（t_s² に反比例）。"""
    cfg = WaferConfig(nx=20, ny=20, nz=50, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="nitride", thickness_um=0.5))
    w = r.simulate()
    thick = metrology.wafer_bow_um(w, substrate_thickness_um=775.0)
    thin = metrology.wafer_bow_um(w, substrate_thickness_um=400.0)
    assert thin > thick > 0
