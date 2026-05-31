"""SputterEtch ファセッティングのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, SputterEtch, Strip
from semisim.recipe import Recipe


def _mesa(faceting):
    """孤立メサ（凸構造）を作り SputterEtch でミリングする。"""
    cfg = WaferConfig(nx=44, ny=16, nz=60, pitch_um=0.05, substrate_um=1.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.0, "x1": 0.65, "y1": 1.0})])
    r = Recipe(config=cfg)
    # メサ: 金属を全面成膜→マスクでメサ部だけ残す
    r.add(CVD(material="tungsten", thickness_um=0.8))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="negative"))
    r.add(DryEtch(targets=["tungsten"], depth_um=0.9))
    r.add(Strip())
    r.add(SputterEtch(depth_um=0.4, faceting=faceting))
    return r.simulate(), cfg


def _top_corner_solid(w, cfg):
    """メサ上端の角付近にある固体ボクセル数（角が削れると減る）。"""
    wid = materials.get("tungsten").id
    y = cfg.ny // 2
    s = w.grid[:, y, :]
    solid = s == wid
    # メサ上端高さ
    zs = np.where(solid)[0]
    if zs.size == 0:
        return 0
    ztop = int(zs.max())
    # 上端2層の幅（角が面取りされると上端が細る）
    return int(solid[ztop].sum() + solid[max(0, ztop - 1)].sum())


def test_faceting_removes_top_corners():
    """ファセットでメサ上端の角が削れ上端幅が細る。"""
    w0, c0 = _mesa(0.0)
    w1, c1 = _mesa(1.0)
    assert _top_corner_solid(w1, c1) < _top_corner_solid(w0, c0)


def test_no_faceting_keeps_corners():
    """faceting=0 では従来通り（角が保たれる）。"""
    w0, c0 = _mesa(0.0)
    # 上端に十分な固体が残っている
    assert _top_corner_solid(w0, c0) > 0


def test_faceting_roundtrip_params():
    p = SputterEtch(depth_um=0.4, faceting=0.5)
    d = p.params_dict()
    assert d["faceting"] == 0.5
    assert SputterEtch._from_params(d).faceting == 0.5


def test_faceting_rejects_out_of_range():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        SputterEtch(depth_um=0.3, faceting=1.5).apply(w)
    except ValueError:
        return
    raise AssertionError("範囲外の faceting で ValueError が出るべき")
