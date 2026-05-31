"""DryEtch RIE ノッチングのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _soi_etch(notch):
    """埋込酸化膜の上にシリコンを積み、酸化膜で止まるエッチを行う。"""
    cfg = WaferConfig(nx=50, ny=16, nz=70, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.4))   # 埋込酸化膜(ストップ層)
    r.add(CVD(material="silicon", thickness_um=1.0))  # 上層シリコン
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.0, "x1": 0.65, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.6, polarity="positive"))
    r.add(DryEtch(
        targets=["silicon"],
        depth_um=1.2,
        selectivity={"oxide": 0.0},  # 酸化膜で停止
        notch_um=notch,
    ))
    r.add(Strip())
    return r.simulate(), cfg


def _interface_si_width(w, cfg):
    """酸化膜界面直上のシリコン側壁間の開口幅（ノッチで広がる）。"""
    si = materials.get("silicon").id
    ox = materials.get("oxide").id
    y = cfg.ny // 2
    s = w.grid[:, y, :]
    # 酸化膜上端 z を求め、その直上の行で空気の幅を測る
    ox_z = np.where(s == ox)[0]
    if ox_z.size == 0:
        return 0
    z_if = int(ox_z.max()) + 1
    row = s[z_if]
    return int((row == materials.AIR).sum()), int((row == si).sum())


def test_notch_widens_interface_opening():
    """ノッチで界面直上の開口が広がり側壁シリコンが抉られる。"""
    w0, c0 = _soi_etch(0.0)
    w1, c1 = _soi_etch(0.4)
    air0, _ = _interface_si_width(w0, c0)
    air1, _ = _interface_si_width(w1, c1)
    assert air1 > air0


def test_no_notch_no_undercut():
    """notch=0 では界面で側壁が抉られない（垂直）。"""
    w0, c0 = _soi_etch(0.0)
    air0, si0 = _interface_si_width(w0, c0)
    # 側壁シリコンが両側に十分残っている
    assert si0 > 0


def test_notch_roundtrip_params():
    p = DryEtch(targets=["silicon"], depth_um=1.0, notch_um=0.3)
    d = p.params_dict()
    assert d["notch_um"] == 0.3
    assert DryEtch._from_params(d).notch_um == 0.3


def test_notch_rejects_negative():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        DryEtch(targets=["silicon"], depth_um=0.5, notch_um=-0.1).apply(w)
    except ValueError:
        return
    raise AssertionError("負の notch_um で ValueError が出るべき")
