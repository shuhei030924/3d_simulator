"""PVD の肩部封止偽像（中空の庇状ボイド）に対する回帰テスト。

overhang=0 の指向性 PVD は内部に空気を封止しないはず。マニュアルの PVD
デモ（酸化膜トレンチ上に metal_al を step_coverage=0.4 で成膜）で、肩部に
閉じた空気クラックが残らないことを保証する。
"""
import numpy as np
from scipy import ndimage

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, PVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _center_mask(w):
    lo = (1 - w) / 2
    return Mask(shapes=[Shape("rect", {"x0": lo, "y0": 0.0, "x1": 1 - lo, "y1": 1.0})])


def _stepped_pvd(overhang=0.0, step_coverage=0.4):
    cfg = WaferConfig(nx=200, ny=20, nz=140, pitch_um=0.04, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.6))
    r.add(Photo(mask=_center_mask(0.3), thickness_um=0.8, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.6))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="metal_al", thickness_um=0.3,
              step_coverage=step_coverage, overhang=overhang))
    return r.simulate()


def _enclosed_air_voxels(grid):
    """上面に連結しない閉じた空気のボクセル数。"""
    air = grid == materials.AIR
    lbl, n = ndimage.label(air, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return 0
    top_labels = set(np.unique(lbl[-1])) - {0}
    enclosed = air & ~np.isin(lbl, list(top_labels))
    return int(enclosed.sum())


def test_no_sealed_void_on_trench_shoulder():
    """overhang=0 ではトレンチ肩部に閉じた空気ボイドが生じない。"""
    w = _stepped_pvd(overhang=0.0)
    assert _enclosed_air_voxels(w.grid) == 0
    assert metrology.void_volume_um3(w) == 0.0


def test_step_coverage_physics_preserved():
    """封止偽像の除去後も底/フィールドの膜厚比が step_coverage を反映する。"""
    w = _stepped_pvd(overhang=0.0, step_coverage=0.4)
    g = w.grid
    al = materials.get("metal_al").id
    y = g.shape[1] // 2
    field = int((g[:, y, 30] == al).sum())          # 左メサ平坦部
    bottom = int((g[:, y, 100] == al).sum())         # トレンチ底中央
    assert field > 0 and bottom > 0
    assert bottom < field                            # 底は薄い（被覆悪化）
    assert 0.2 < bottom / field < 0.7                # おおむね step_coverage 近傍


def test_overhang_still_creates_keyhole():
    """overhang>0 では意図したキーホールボイドが保持される（偽像除去で消えない）。"""
    cfg = WaferConfig(nx=40, ny=14, nz=70, pitch_um=0.05, substrate_um=2.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.42, "y0": 0.0, "x1": 0.58, "y1": 1.0})])
    from semisim.processes import DRIE
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=1.5))
    r.add(Strip())
    r.add(PVD(material="metal_al", thickness_um=0.4, overhang=2.0))
    w = r.simulate()
    assert metrology.void_volume_um3(w) > 0.0
