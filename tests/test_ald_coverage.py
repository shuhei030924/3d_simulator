"""ALD アスペクト比依存被覆（前駆体枯渇）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import ALD, CVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _deep_count(wafer, zlo=20, zhi=35):
    g = wafer.grid
    hf = materials.get("hafnia").id
    return int(np.count_nonzero(g[zlo:zhi] == hf))


def _trench_recipe(ar_coverage, ar_threshold=5.0):
    cfg = WaferConfig(nx=60, ny=20, nz=80, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.5))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.5))
    r.add(Strip())
    r.add(ALD(material="hafnia", cycles=200, growth_per_cycle_nm=1.0,
              ar_coverage=ar_coverage, ar_threshold=ar_threshold))
    return r


def test_low_coverage_thins_deep_film():
    """低い底被覆率で深部の膜が薄くなる。"""
    full = _deep_count(_trench_recipe(1.0).simulate())
    poor = _deep_count(_trench_recipe(0.2).simulate())
    assert poor < full


def test_full_coverage_is_conformal():
    """ar_coverage=1.0 は従来どおり完全コンフォーマル（既定動作）。"""
    cfg = WaferConfig(nx=40, ny=40, nz=50, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(ALD(material="hafnia", cycles=50, growth_per_cycle_nm=1.0))
    w = r.simulate()
    assert np.count_nonzero(w.grid == materials.get("hafnia").id) > 0


def test_roundtrip():
    p = ALD(material="hafnia", cycles=120, growth_per_cycle_nm=0.8,
            ar_coverage=0.3, ar_threshold=8.0)
    d = p.params_dict()
    q = ALD._from_params(d)
    assert q.ar_coverage == 0.3
    assert q.ar_threshold == 8.0


def test_invalid_coverage_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(ALD(material="hafnia", cycles=10, growth_per_cycle_nm=1.0,
              ar_coverage=1.5))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("被覆率>1 で ValueError が出るべき")
