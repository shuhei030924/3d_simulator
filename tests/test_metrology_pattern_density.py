"""metrology.pattern_density_map / pattern_density_stats のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _half_patterned():
    """左半分だけ金属メサを残す（疎密のあるパターン）。"""
    cfg = WaferConfig(nx=60, ny=20, nz=50, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="tungsten", thickness_um=0.4))
    # 左半分を残すマスク
    mask = Mask(shapes=[Shape("rect", {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="negative"))
    r.add(DryEtch(targets=["tungsten"], depth_um=0.5))
    r.add(Strip())
    return r.simulate(), cfg


def test_density_map_shape_and_range():
    """マップ形状が (ny, nx) で値域 0..1。"""
    w, cfg = _half_patterned()
    dens = metrology.pattern_density_map(w, "tungsten", radius_um=1.0)
    assert dens.shape == (cfg.ny, cfg.nx)
    assert dens.min() >= 0.0 and dens.max() <= 1.0


def test_dense_vs_sparse_region():
    """金属のある左側は密度が高く、右側は低い。"""
    w, cfg = _half_patterned()
    dens = metrology.pattern_density_map(w, "tungsten", radius_um=0.5)
    left = float(dens[:, : cfg.nx // 4].mean())
    right = float(dens[:, 3 * cfg.nx // 4 :].mean())
    assert left > right


def test_density_stats_keys_and_range():
    """統計のキーが揃い、疎密差 range>0。"""
    w, _ = _half_patterned()
    st = metrology.pattern_density_stats(w, "tungsten", radius_um=1.0)
    assert set(st) == {"min", "max", "mean", "range"}
    assert st["range"] > 0.0


def test_full_coverage_density_one():
    """全面成膜なら密度はほぼ 1。"""
    cfg = WaferConfig(nx=30, ny=20, nz=40, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.3))
    w = r.simulate()
    dens = metrology.pattern_density_map(w, "oxide", radius_um=1.0)
    assert np.isclose(dens.mean(), 1.0)
