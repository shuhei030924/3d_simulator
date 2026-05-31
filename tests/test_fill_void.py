"""Fill アスペクト比依存キーホール空隙のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Fill, Photo, Strip
from semisim.recipe import Recipe


def _buried_air(wafer) -> int:
    """金属に上下を挟まれた埋没空気（=ボイド）のボクセル数。"""
    cu = materials.get("metal_cu").id
    g = wafer.grid
    cnt = 0
    for y in range(g.shape[1]):
        for x in range(g.shape[2]):
            col = g[:, y, x]
            air = np.where(col == materials.AIR)[0]
            cu_z = np.where(col == cu)[0]
            if cu_z.size and air.size:
                cnt += int(np.count_nonzero((air > cu_z.min()) & (air < cu_z.max())))
    return cnt


def _narrow_trench_recipe(void_ar):
    cfg = WaferConfig(nx=60, ny=20, nz=70, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.5))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.5))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=void_ar))
    return r


def test_void_forms_in_narrow_trench():
    """高 AR の狭いトレンチでキーホール空隙が生じる。"""
    assert _buried_air(_narrow_trench_recipe(2.0).simulate()) > 0


def test_no_void_when_disabled():
    """void_ar=0 では空隙が生じない（完全充填）。"""
    assert _buried_air(_narrow_trench_recipe(0.0).simulate()) == 0


def test_no_void_in_wide_trench():
    """幅広（低 AR）トレンチでは閾値を超えず空隙が生じない。"""
    cfg = WaferConfig(nx=60, ny=20, nz=70, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.2, "y0": 0.0, "x1": 0.8, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.0))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=5.0))
    assert _buried_air(r.simulate()) == 0


def test_void_roundtrip():
    p = Fill(material="metal_cu", overfill_um=0.1, void_ar=8.0)
    d = p.params_dict()
    assert d["void_ar"] == 8.0
    assert Fill._from_params(d).void_ar == 8.0


def test_void_negative_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(Fill(material="metal_cu", overfill_um=0.1, void_ar=-1.0))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の void_ar で ValueError が出るべき")
