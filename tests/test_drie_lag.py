"""DRIE の RIE ラグ（ARDE / アスペクト比依存エッチ）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import DRIE, Photo, Strip
from semisim.recipe import Recipe


def _trench_depth(wafer, x_center_um) -> int:
    """指定 x 中心列のエッチ深さ（空気ボクセル数）。"""
    air = materials.AIR
    xi = int(round(x_center_um / wafer.config.pitch_um))
    col = wafer.grid[:, wafer.config.ny // 2, xi]
    return int(np.count_nonzero(col == air))


def _wide_and_narrow():
    cfg = WaferConfig(nx=80, ny=40, nz=80, pitch_um=0.1, substrate_um=5.0)
    # 広い開口（左）と狭い開口（右）を 1 枚のマスクに用意
    mask = Mask(shapes=[
        Shape("rect", {"x0": 0.1, "y0": 0.0, "x1": 0.35, "y1": 1.0}),  # 広
        Shape("rect", {"x0": 0.7, "y0": 0.0, "x1": 0.75, "y1": 1.0}),  # 狭
    ])
    return cfg, mask


def test_rie_lag_narrow_shallower():
    """RIE ラグありで狭い開口が広い開口より浅くなる。"""
    cfg, mask = _wide_and_narrow()
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=3.0, lag=0.8))
    r.add(Strip())
    w = r.simulate()
    wide = _trench_depth(w, 0.22 * cfg.nx * cfg.pitch_um)
    narrow = _trench_depth(w, 0.72 * cfg.nx * cfg.pitch_um)
    assert wide > narrow > 0


def test_no_lag_equal_depth():
    """ラグ 0 では広い開口・狭い開口とも同深さになる。"""
    cfg, mask = _wide_and_narrow()
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=3.0, lag=0.0))
    r.add(Strip())
    w = r.simulate()
    wide = _trench_depth(w, 0.22 * cfg.nx * cfg.pitch_um)
    narrow = _trench_depth(w, 0.72 * cfg.nx * cfg.pitch_um)
    assert wide == narrow


def test_lag_roundtrip():
    p = DRIE(target="silicon", depth_um=2.0, lag=0.6)
    d = p.params_dict()
    assert d["lag"] == 0.6
    assert DRIE._from_params(d).lag == 0.6


def test_lag_out_of_range_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(DRIE(target="silicon", depth_um=1.0, lag=1.5))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("範囲外 lag で ValueError が出るべき")
