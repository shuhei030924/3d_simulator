"""DRIE マイクロトレンチングのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import DRIE, Photo, Strip
from semisim.recipe import Recipe


def _trench(microtrench):
    """中央開口のトレンチを DRIE で掘る。"""
    cfg = WaferConfig(nx=60, ny=20, nz=80, pitch_um=0.05, substrate_um=3.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=1.5, microtrench_um=microtrench))
    r.add(Strip())
    return r.simulate(), cfg


def _floor_profile(wafer, cfg):
    """Y 中央断面でトレンチ底（各 x のシリコン上面）の高さを返す。"""
    y = cfg.ny // 2
    si_id = materials.get("silicon").id
    s = wafer.grid[:, y, :]
    nz, nx = s.shape
    top = np.full(nx, -1)
    for x in range(nx):
        solid = np.flatnonzero(s[:, x] == si_id)
        if solid.size:
            top[x] = solid.max()
    return top


def test_microtrench_deepens_edges():
    """マイクロトレンチで開口周縁(底の隅)が中央より深く掘れる。"""
    w, cfg = _trench(0.6)
    prof = _floor_profile(w, cfg)
    # トレンチ底のある列のみ（開口範囲）
    trench_cols = np.flatnonzero((prof > 0) & (prof < prof.max()))
    assert trench_cols.size > 0
    # 開口の両端付近の底が中央より低い（=深い）
    left, right = trench_cols.min(), trench_cols.max()
    center = (left + right) // 2
    edge_depth = min(prof[left], prof[right])
    assert edge_depth < prof[center]


def test_no_microtrench_flat_floor():
    """microtrench=0 ではトレンチ底がほぼ平坦。"""
    w, cfg = _trench(0.0)
    prof = _floor_profile(w, cfg)
    trench_cols = np.flatnonzero((prof > 0) & (prof < prof.max()))
    floor = prof[trench_cols]
    # 底の高さばらつきが小さい（平坦）
    assert floor.max() - floor.min() <= 1


def test_microtrench_roundtrip_params():
    """params_dict / _from_params で microtrench_um を保持。"""
    p = DRIE(target="silicon", depth_um=2.0, microtrench_um=0.5)
    d = p.params_dict()
    assert d["microtrench_um"] == 0.5
    assert DRIE._from_params(d).microtrench_um == 0.5


def test_microtrench_rejects_negative():
    """負のマイクロトレンチ深さは弾く。"""
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        DRIE(target="silicon", depth_um=1.0, microtrench_um=-0.1).apply(w)
    except ValueError:
        return
    raise AssertionError("負の microtrench_um で ValueError が出るべき")
