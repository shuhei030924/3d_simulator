"""SpinCoat（スピンオン平坦化）のテスト。"""
from __future__ import annotations

import pytest

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import Photo, SpinCoat
from semisim.recipe import Recipe


def test_spincoat_planarizes_top():
    cfg = WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.1, substrate_um=2.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="negative"))
    rough_before = metrology.surface_roughness_um(r.simulate())
    r.add(SpinCoat(material="low_k", cap_um=0.3))
    w = r.simulate()
    # スピンオン後は上面が平坦化され粗さが下がる
    assert metrology.surface_roughness_um(w) < rough_before


def test_spincoat_covers_full_wafer():
    cfg = WaferConfig(nx=30, ny=30, nz=50, pitch_um=0.1, substrate_um=2.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="negative"))
    r.add(SpinCoat(material="low_k", cap_um=0.2))
    w = r.simulate()
    # 全列が同じ高さで覆われている（上面が水平）
    heights = w.top_surface_z()
    assert heights.min() == heights.max()


def test_spincoat_deposits_material(wafer):
    before = int((wafer.grid == materials.get("low_k").id).sum())
    SpinCoat(material="low_k", cap_um=0.3).apply(wafer)
    after = int((wafer.grid == materials.get("low_k").id).sum())
    assert after > before


def test_spincoat_negative_cap_raises(wafer):
    with pytest.raises(ValueError):
        SpinCoat(material="low_k", cap_um=-0.1).apply(wafer)


def test_spincoat_roundtrip():
    s = SpinCoat(material="oxide", cap_um=0.25)
    restored = SpinCoat._from_params(s.params_dict())
    assert restored.material == "oxide"
    assert restored.cap_um == 0.25
