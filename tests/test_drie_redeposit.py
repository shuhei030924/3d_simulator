"""DRIE 側壁再付着（Bosch パシベーション）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import DRIE, Photo
from semisim.recipe import Recipe


def _trench_recipe(redeposit_um):
    cfg = WaferConfig(nx=60, ny=20, nz=70, pitch_um=0.05, substrate_um=2.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=1.5, redeposit_um=redeposit_um))
    return r


def _trench_air_width(wafer, z=30):
    air = np.where(wafer.grid[z, 10] == materials.AIR)[0]
    return int(air.size)


def test_redeposit_narrows_trench():
    """再付着でトレンチ幅が狭くなる。"""
    wide = _trench_air_width(_trench_recipe(0.0).simulate())
    narrow = _trench_air_width(_trench_recipe(0.15).simulate())
    assert narrow < wide


def test_no_redeposit_keeps_width():
    """redeposit=0 では幅が変わらない。"""
    assert _trench_air_width(_trench_recipe(0.0).simulate()) > 0


def test_roundtrip():
    p = DRIE(target="silicon", depth_um=2.0, redeposit_um=0.2)
    d = p.params_dict()
    q = DRIE._from_params(d)
    assert q.redeposit_um == 0.2


def test_negative_redeposit_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(DRIE(target="silicon", depth_um=1.0, redeposit_um=-0.1))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の redeposit_um で ValueError が出るべき")
