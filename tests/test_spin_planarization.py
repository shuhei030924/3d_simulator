"""SpinCoat 平坦化度（DOP）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, SpinCoat, Strip
from semisim.recipe import Recipe


def _topography_then_spin(dop):
    """段差地形を作ってからスピンオン塗布し、上面の平坦度を測る。"""
    cfg = WaferConfig(nx=80, ny=20, nz=60, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="poly", thickness_um=1.0))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=1.0))  # 段差を作る
    r.add(Strip())
    r.add(SpinCoat(material="low_k", cap_um=0.3, planarization=dop))
    return r.simulate()


def _top_height_std(wafer):
    """塗布後の上面高さの標準偏差（小さいほど平坦）。"""
    z_top = wafer.top_surface_z()
    return float(np.std(z_top[z_top >= 0]))


def test_full_planarization_is_flatter():
    """DOP=1 は DOP=0 より上面が平坦（高さばらつきが小さい）。"""
    flat = _top_height_std(_topography_then_spin(1.0))
    conf = _top_height_std(_topography_then_spin(0.0))
    assert flat < conf


def test_conformal_follows_topography():
    """DOP=0 は地形に追従し段差が残る。"""
    w = _topography_then_spin(0.0)
    assert _top_height_std(w) > 0.5


def test_low_k_deposited():
    """いずれの DOP でも low_k が堆積する。"""
    for dop in (0.0, 0.5, 1.0):
        w = _topography_then_spin(dop)
        assert (w.grid == materials.get("low_k").id).any()


def test_planarization_step_height_decreases():
    """DOP を上げるほど段差(step height)が減る。"""
    s0 = metrology.step_height_um(_topography_then_spin(0.0))
    s1 = metrology.step_height_um(_topography_then_spin(1.0))
    assert s1 <= s0


def test_spin_roundtrip_params():
    """params_dict / _from_params で planarization を保持。"""
    p = SpinCoat(material="low_k", cap_um=0.4, planarization=0.6)
    d = p.params_dict()
    assert d["planarization"] == 0.6
    assert SpinCoat._from_params(d).planarization == 0.6
