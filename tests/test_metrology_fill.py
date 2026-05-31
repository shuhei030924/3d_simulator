"""via_fill_quality / sidewall_bowing_um のテスト。"""
from __future__ import annotations

from semisim import metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Fill, Photo, Strip
from semisim.recipe import Recipe


def _narrow_trench(void_ar):
    cfg = WaferConfig(nx=60, ny=20, nz=70, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.5))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.5))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=void_ar))
    return r.simulate()


def test_via_fill_quality_perfect():
    """空隙なし充填では fill_fraction=1.0。"""
    w = _narrow_trench(0.0)
    q = metrology.via_fill_quality(w, "metal_cu")
    assert q["fill_fraction"] == 1.0
    assert q["void_count"] == 0


def test_via_fill_quality_with_void():
    """キーホール空隙ありでは fill_fraction<1 かつ void_count>0。"""
    w = _narrow_trench(2.0)
    q = metrology.via_fill_quality(w, "metal_cu")
    assert q["fill_fraction"] < 1.0
    assert q["void_count"] > 0
    assert q["void_volume_um3"] > 0.0


def test_via_fill_quality_no_material():
    """対象材料が無ければ完全充填扱い。"""
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    q = metrology.via_fill_quality(w, "metal_cu")
    assert q["fill_fraction"] == 1.0


def test_sidewall_bowing_flat_is_zero():
    """トレンチの無い平坦ウェハではボーイング 0。"""
    cfg = WaferConfig(nx=30, ny=30, nz=30, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    assert metrology.sidewall_bowing_um(w, 15) == 0.0


def test_sidewall_bowing_detects_bulge():
    """側壁が膨らんだトレンチで正のボーイング量を返す。"""

    from semisim import materials
    cfg = WaferConfig(nx=40, ny=10, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    g = w.grid
    si = materials.get("silicon").id
    # 人工的に樽型トレンチを彫る: 上は狭く中央は広い
    for z in range(5, 20):
        bulge = 2 if 9 <= z <= 14 else 0  # 中央で広げる
        x0, x1 = 17 - bulge, 23 + bulge
        g[z, :, x0:x1] = materials.AIR
    assert (g == si).any()
    assert metrology.sidewall_bowing_um(w, 5) > 0.0
