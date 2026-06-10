"""コンフォーマル成膜の帯域限定 EDT 最適化の厳密一致テスト。

CVD/ALD/Spacer は固体上面より膜厚を超えて上の空気に堆積し得ないため、
z 帯域を切ってから距離変換しても結果は全グリッド計算と厳密に一致する
はずである（_conformal_air_distance）。ここでは最適化前の素朴な全グリッド
計算を参照実装として比較する。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from semisim import materials
from semisim.grid import Wafer, WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import ALD, CVD, DryEtch, Photo, Spacer, Strip
from semisim.recipe import Recipe


def _trench_wafer(nz: int = 80) -> Wafer:
    """トレンチ付きの構造ウェハ（上方に空気の余裕が大きい）。"""
    cfg = WaferConfig(nx=40, ny=40, nz=nz, pitch_um=0.1, substrate_um=2.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.0, "x1": 0.65, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["silicon"], depth_um=1.0))
    r.add(Strip(material="photoresist"))
    return r.simulate()


def _reference_conformal(grid_before: np.ndarray, t: int, mat_id: int) -> np.ndarray:
    """最適化前の素朴な全グリッド EDT による参照実装。"""
    out = grid_before.copy()
    air = out == materials.AIR
    dist = ndimage.distance_transform_edt(air)
    out[air & (dist <= t)] = mat_id
    return out


def test_cvd_matches_full_grid_reference():
    w = _trench_wafer()
    before = w.grid.copy()
    t = w.um_to_vox(0.3)
    CVD(material="nitride", thickness_um=0.3).apply(w)
    ref = _reference_conformal(before, t, materials.get("nitride").id)
    assert np.array_equal(w.grid, ref)


def test_ald_matches_full_grid_reference():
    w = _trench_wafer()
    before = w.grid.copy()
    proc = ALD(material="hafnia", cycles=200, growth_per_cycle_nm=1.0)
    t = w.um_to_vox(proc.thickness_um)
    proc.apply(w)
    ref = _reference_conformal(before, t, materials.get("hafnia").id)
    assert np.array_equal(w.grid, ref)


def test_cvd_with_solid_near_grid_top():
    """固体がグリッド上端付近にあっても帯域クランプで正しく堆積する。"""
    cfg = WaferConfig(nx=20, ny=20, nz=24, pitch_um=0.1, substrate_um=2.0)
    w = Wafer(cfg)  # 基板 20 vox / 上空 4 vox
    before = w.grid.copy()
    t = w.um_to_vox(0.8)  # 8 vox > 上空 → クランプ発生
    CVD(material="oxide", thickness_um=0.8).apply(w)
    ref = _reference_conformal(before, t, materials.get("oxide").id)
    assert np.array_equal(w.grid, ref)
    # 上端まで膜が詰まる
    assert (w.grid[-1] == materials.get("oxide").id).all()


def test_spacer_sidewall_preserved():
    """Spacer の帯域化後も側壁のみ残るエッチバック動作が保たれる。"""
    w = _trench_wafer()
    Spacer(material="nitride", thickness_um=0.2).apply(w)
    nid = materials.get("nitride").id
    nit = w.grid == nid
    assert nit.any()
    # 平坦部（トレンチから遠い列）には水平膜が残らない
    assert not nit[:, 2, 2].any()


def test_empty_solid_no_crash():
    """固体が無い退化グリッドでも例外なく何も堆積しない。"""
    cfg = WaferConfig(nx=10, ny=10, nz=12, pitch_um=0.1, substrate_um=0.1)
    w = Wafer(cfg)
    w.grid[:] = materials.AIR  # 固体なし
    CVD(material="oxide", thickness_um=0.2).apply(w)
    assert (w.grid == materials.AIR).all()
