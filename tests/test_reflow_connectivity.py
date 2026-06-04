"""REFLOW の連結性（宙吊り防止）の回帰テスト。

リフローのモルフォロジ処理が、上に他材料が載っている target を取り除いて
上層を宙に浮かせる偽像を生まないことを保証する。
"""
import numpy as np
from scipy import ndimage

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Reflow, Strip
from semisim.recipe import Recipe

AIR = materials.AIR


def _cfg():
    return WaferConfig(nx=120, ny=16, nz=90, pitch_um=0.05, substrate_um=2.0)


def _cmask(w):
    lo = (1 - w) / 2
    return Mask(shapes=[Shape("rect", {"x0": lo, "y0": 0.0, "x1": 1 - lo, "y1": 1.0})])


def _floating_voxels(grid):
    solid = grid != AIR
    lbl, _ = ndimage.label(solid, structure=ndimage.generate_binary_structure(3, 1))
    bottom = set(np.unique(lbl[0])) - {0}
    grounded = np.isin(lbl, list(bottom)) if bottom else np.zeros_like(solid)
    return int((solid & ~grounded).sum())


def test_reflow_buried_layer_no_floating():
    """上にポリが載った薄い酸化膜をリフローしても上層が宙に浮かない。"""
    r = Recipe(config=_cfg())
    r.add(CVD(material="oxide", thickness_um=0.1))
    r.add(CVD(material="poly", thickness_um=0.5))
    r.add(Photo(mask=_cmask(0.25), thickness_um=0.8, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=0.5))
    r.add(Strip())
    r.add(Reflow(target="oxide", radius_um=0.2))
    w = r.simulate()
    assert _floating_voxels(w.grid) == 0


def test_reflow_surface_resist_still_smooths():
    """表面レジストのリフローは従来通り角を丸める（粗さが増えない）。"""
    r = Recipe(config=_cfg())
    r.add(Photo(mask=_cmask(0.4), thickness_um=0.6, polarity="negative"))
    w = r.simulate()
    before = metrology.surface_roughness_um(w)
    Reflow(target="photoresist", radius_um=0.2).apply(w)
    after = metrology.surface_roughness_um(w)
    assert after <= before + 1e-9
    assert _floating_voxels(w.grid) == 0


def test_reflow_preserves_overlying_material_count():
    """リフローで上層（ポリ）のボクセル数が減らない（消失しない）。"""
    r = Recipe(config=_cfg())
    r.add(CVD(material="oxide", thickness_um=0.1))
    r.add(CVD(material="poly", thickness_um=0.5))
    r.add(Photo(mask=_cmask(0.25), thickness_um=0.8, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=0.5))
    r.add(Strip())
    w = r.simulate()
    poly_before = int((w.grid == materials.get("poly").id).sum())
    Reflow(target="oxide", radius_um=0.2).apply(w)
    poly_after = int((w.grid == materials.get("poly").id).sum())
    assert poly_after == poly_before
