"""Epitaxy ファセット成長のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Epitaxy, Photo, Strip
from semisim.recipe import Recipe


def _seed_window(facet_angle):
    """酸化膜に窓を開けて露出シリコンのシード窓を作り、その上にエピ成長。"""
    cfg = WaferConfig(nx=60, ny=60, nz=70, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.35, "x1": 0.65, "y1": 0.65})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.5))  # 窓底のシリコンを露出
    r.add(Strip())
    r.add(Epitaxy(material="epi_si", thickness_um=1.5, facet_angle_deg=facet_angle))
    return r.simulate()


def _epi_footprint_by_z(wafer):
    """各高さ z のエピ画素数（footprint）を返す。"""
    epi_id = materials.get("epi_si").id
    return (wafer.grid == epi_id).sum(axis=(1, 2))


def test_facet_default_is_conformal():
    """facet=0 ではエピは収束せず、ほぼ一定の footprint を保つ。"""
    w = _seed_window(0.0)
    counts = _epi_footprint_by_z(w)
    nz = counts[counts > 0]
    assert nz.size > 0
    # 等方成長: 上層でも footprint がほぼ維持される（収束しない）
    assert nz.min() >= nz.max() * 0.7


def test_facet_narrows_with_height():
    """ファセット角を付けると上に行くほど footprint が収束する。"""
    w = _seed_window(45.0)
    counts = _epi_footprint_by_z(w)
    nz_idx = np.flatnonzero(counts > 0)
    assert nz_idx.size >= 3
    bottom = counts[nz_idx[0]]
    top = counts[nz_idx[-1]]
    # 台形/三角キャップ: 頂部の footprint は底部より小さい
    assert top < bottom


def test_facet_only_on_silicon():
    """シードが無ければ成長しない。"""
    cfg = WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))  # 全面を酸化膜で覆う
    r.add(Epitaxy(material="epi_si", thickness_um=1.0, facet_angle_deg=45.0))
    w = r.simulate()
    epi_id = materials.get("epi_si").id
    assert not (w.grid == epi_id).any()


def test_facet_roundtrip_params():
    """params_dict / _from_params で facet_angle_deg を保持する。"""
    p = Epitaxy(material="epi_si", thickness_um=0.8, facet_angle_deg=54.7)
    d = p.params_dict()
    assert d["facet_angle_deg"] == 54.7
    p2 = Epitaxy._from_params(d)
    assert p2.facet_angle_deg == 54.7
