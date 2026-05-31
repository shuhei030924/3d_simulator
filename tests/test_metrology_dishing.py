"""dishing_depth_um（ダマシン CMP ディッシング深さ）メトロロジのテスト。"""
from __future__ import annotations

from semisim import metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CMP, CVD, DryEtch, Fill, Photo, Strip
from semisim.recipe import Recipe


def _damascene(dishing_um):
    """酸化膜にトレンチを掘り Cu 充填→CMP（任意でディッシング）。"""
    cfg = WaferConfig(nx=60, ny=20, nz=50, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.8))
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.8, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.6))
    r.add(Strip(material="photoresist"))
    r.add(Fill(material="metal_cu", overfill_um=0.3))
    r.add(CMP(remove_um=0.4, stop_material="oxide",
              soft_material="metal_cu", dishing_um=dishing_um))
    return r.simulate()


def test_no_dishing_is_flat():
    """ディッシングなしの CMP では Cu 凹みはほぼ 0。"""
    w = _damascene(0.0)
    assert metrology.dishing_depth_um(w, "metal_cu") <= 0.06


def test_dishing_measured():
    """ディッシング指定で Cu 上面が凹み、深さが計測できる。"""
    w = _damascene(0.2)
    dish = metrology.dishing_depth_um(w, "metal_cu")
    assert dish > 0.1


def test_more_dishing_deeper():
    """ディッシング量が大きいほど計測値も大きい。"""
    d1 = metrology.dishing_depth_um(_damascene(0.1), "metal_cu")
    d2 = metrology.dishing_depth_um(_damascene(0.3), "metal_cu")
    assert d2 > d1


def test_absent_material_returns_zero():
    """対象材料が無ければ 0。"""
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    assert metrology.dishing_depth_um(w, "metal_cu") == 0.0
