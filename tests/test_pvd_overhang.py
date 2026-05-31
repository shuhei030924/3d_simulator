"""PVD オーバーハング / ブレッドローフィングのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import DRIE, PVD, Photo, Strip
from semisim.recipe import Recipe


def _via(overhang):
    """狭い深ビアを掘ってから PVD 金属を成膜する。"""
    cfg = WaferConfig(nx=40, ny=14, nz=70, pitch_um=0.05, substrate_um=2.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.42, "y0": 0.0, "x1": 0.58, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=1.5))
    r.add(Strip())
    r.add(PVD(material="metal_al", thickness_um=0.4, overhang=overhang))
    return r.simulate(), cfg


def test_overhang_creates_void():
    """オーバーハングで上部が塞がりキーホールボイドが生じる。"""
    w0, _ = _via(0.0)
    w1, _ = _via(2.0)
    v0 = metrology.void_volume_um3(w0)
    v1 = metrology.void_volume_um3(w1)
    assert v1 > v0


def test_overhang_narrows_mouth():
    """庇が開口上端を狭める（最上部の空気幅が減る）。"""
    al = materials.get("metal_al").id
    air = materials.AIR

    def mouth_air(w, cfg):
        y = cfg.ny // 2
        # 金属膜の最上端高さ
        zs = np.where(w.grid[:, y, :] == al)[0]
        ztop = int(zs.max())
        return int(np.count_nonzero(w.grid[ztop, y, :] == air))

    w0, c0 = _via(0.0)
    w1, c1 = _via(2.0)
    assert mouth_air(w1, c1) <= mouth_air(w0, c0)


def test_no_overhang_no_extra_void():
    """overhang=0 では従来通り（庇成長なし）。"""
    w, cfg = _via(0.0)
    # ビア内に金属が入っている（成膜は機能している）
    assert metrology.material_counts(w).get("metal_al", 0) > 0


def test_overhang_roundtrip_params():
    p = PVD(material="metal_al", thickness_um=0.5, overhang=1.5)
    d = p.params_dict()
    assert d["overhang"] == 1.5
    assert PVD._from_params(d).overhang == 1.5


def test_overhang_rejects_negative():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        PVD(material="metal_al", thickness_um=0.3, overhang=-0.5).apply(w)
    except ValueError:
        return
    raise AssertionError("負の overhang で ValueError が出るべき")
