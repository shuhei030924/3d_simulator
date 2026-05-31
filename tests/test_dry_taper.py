"""DryEtch 側壁テーパ角のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _etch_width_at_depth(wafer, z_index) -> int:
    """指定 z での空気（エッチされた）ボクセル数を中央行で測る。"""
    air = materials.AIR
    row = wafer.grid[z_index, wafer.config.ny // 2, :]
    return int(np.count_nonzero(row == air))


def _stack():
    cfg = WaferConfig(nx=60, ny=60, nz=70, pitch_um=0.1, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=2.0))
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    return cfg, r


def test_taper_makes_trapezoid():
    """テーパ角ありで上部が下部より広い台形プロファイルになる。"""
    cfg, r = _stack()
    r.add(DryEtch(targets=["oxide"], depth_um=1.5, taper_deg=30.0))
    r.add(Strip())
    w = r.simulate()
    sub_top = w.um_to_vox(cfg.substrate_um)
    # 上部（浅い）と下部（深い）のエッチ幅を比較
    z_shallow = sub_top + w.um_to_vox(1.8)
    z_deep = sub_top + w.um_to_vox(0.7)
    w_shallow = _etch_width_at_depth(w, z_shallow)
    w_deep = _etch_width_at_depth(w, z_deep)
    assert w_shallow > w_deep > 0


def test_no_taper_is_vertical():
    """テーパ 0 では上下のエッチ幅がほぼ等しい（垂直）。"""
    cfg, r = _stack()
    r.add(DryEtch(targets=["oxide"], depth_um=1.5, taper_deg=0.0))
    r.add(Strip())
    w = r.simulate()
    sub_top = w.um_to_vox(cfg.substrate_um)
    z_shallow = sub_top + w.um_to_vox(1.8)
    z_deep = sub_top + w.um_to_vox(0.7)
    assert _etch_width_at_depth(w, z_shallow) == _etch_width_at_depth(w, z_deep)


def test_taper_roundtrip():
    p = DryEtch(targets=["oxide"], depth_um=1.0, taper_deg=15.0)
    d = p.params_dict()
    assert d["taper_deg"] == 15.0
    assert DryEtch._from_params(d).taper_deg == 15.0


def test_taper_out_of_range_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(DryEtch(targets=["oxide"], depth_um=0.5, taper_deg=90.0))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("範囲外テーパ角で ValueError が出るべき")
