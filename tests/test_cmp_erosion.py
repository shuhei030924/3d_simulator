"""CMP パターン密度依存エロージョンのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CMP, CVD, DryEtch, Fill, Photo, Strip
from semisim.recipe import Recipe


def _dense_iso_recipe(erosion_um):
    cfg = WaferConfig(nx=80, ny=20, nz=60, pitch_um=0.05, substrate_um=1.0)
    shapes = []
    for i in range(6):  # 左側に密集トレンチ配列
        x0 = 0.05 + i * 0.05
        shapes.append(Shape("rect", {"x0": x0, "y0": 0.0, "x1": x0 + 0.025, "y1": 1.0}))
    shapes.append(Shape("rect", {"x0": 0.8, "y0": 0.0, "x1": 0.85, "y1": 1.0}))  # 孤立
    mask = Mask(shapes=shapes)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.0))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.3))
    r.add(CMP(remove_um=0.3, soft_material="metal_cu",
              erosion_um=erosion_um, density_radius_um=0.3))
    return r


def _cu_top(wafer, xlo, xhi):
    g = wafer.grid
    cu = materials.get("metal_cu").id
    tops = []
    for x in range(xlo, xhi):
        col = g[:, 10, x]
        z = np.where(col == cu)[0]
        if z.size:
            tops.append(int(z.max()))
    return float(np.mean(tops)) if tops else -1.0


def test_dense_erodes_more_than_isolated():
    """密集領域は孤立領域より Cu 上面が低くなる。"""
    w = _dense_iso_recipe(0.4).simulate()
    dense = _cu_top(w, 2, 18)
    iso = _cu_top(w, 62, 70)
    assert dense < iso


def test_no_erosion_uniform():
    """erosion=0 では密集/孤立で差が出ない。"""
    w = _dense_iso_recipe(0.0).simulate()
    dense = _cu_top(w, 2, 18)
    iso = _cu_top(w, 62, 70)
    assert abs(dense - iso) < 1.0


def test_roundtrip():
    p = CMP(remove_um=0.4, soft_material="metal_cu", erosion_um=0.5,
            density_radius_um=2.0)
    d = p.params_dict()
    q = CMP._from_params(d)
    assert q.erosion_um == 0.5
    assert q.density_radius_um == 2.0


def test_negative_erosion_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(CMP(remove_um=0.2, soft_material="oxide", erosion_um=-0.3))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の erosion_um で ValueError が出るべき")
