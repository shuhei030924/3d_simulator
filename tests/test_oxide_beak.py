"""Oxidation LOCOS バーズビークのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Oxidation, Photo, Strip
from semisim.recipe import Recipe


def _oxide_under_nitride(wafer):
    g = wafer.grid
    nit = materials.get("nitride").id
    ox = materials.get("oxide").id
    cnt = 0
    for y in range(g.shape[1]):
        for x in range(g.shape[2]):
            col = g[:, y, x]
            if (col == nit).any():
                cnt += int(np.count_nonzero(col == ox))
    return cnt


def _locos_recipe(beak_fraction):
    cfg = WaferConfig(nx=80, ny=20, nz=60, pitch_um=0.05, substrate_um=1.0)
    win = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="nitride", thickness_um=0.2))
    r.add(Photo(mask=win, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["nitride"], depth_um=0.2))
    r.add(Strip())
    r.add(Oxidation(thickness_um=0.5, beak_fraction=beak_fraction))
    return r


def test_beak_encroaches_under_mask():
    """beak_fraction>0 でマスク下に酸化膜が侵入する。"""
    assert _oxide_under_nitride(_locos_recipe(1.0).simulate()) > 0


def test_no_beak_when_disabled():
    """beak_fraction=0 ではマスク下に酸化膜は生じない。"""
    assert _oxide_under_nitride(_locos_recipe(0.0).simulate()) == 0


def test_roundtrip():
    p = Oxidation(thickness_um=0.4, consume_fraction=0.45, beak_fraction=0.8)
    d = p.params_dict()
    q = Oxidation._from_params(d)
    assert q.beak_fraction == 0.8


def test_negative_beak_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(Oxidation(thickness_um=0.3, beak_fraction=-0.5))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の beak_fraction で ValueError が出るべき")
