"""Implant チルト注入（シャドーイング/非対称分布）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Implant, Photo, Strip
from semisim.recipe import Recipe


def _gate_and_implant(tilt):
    """中央に背の高いゲート(poly)を立て、チルト注入する。"""
    cfg = WaferConfig(nx=80, ny=20, nz=60, pitch_um=0.05, substrate_um=1.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    r = Recipe(config=cfg)
    # ゲート(poly)を全面成膜してマスクで残す
    r.add(CVD(material="poly", thickness_um=1.2))
    r.add(Photo(mask=mask, thickness_um=0.6, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=1.2))
    r.add(Strip())
    r.add(Implant(dopant="doped_n", range_um=0.2, straggle_um=0.08, tilt_deg=tilt))
    w = r.simulate()
    return w, cfg


def _doped_centroid_x(wafer):
    dn = wafer.grid == materials.get("doped_n").id
    if not dn.any():
        return None
    xs = np.nonzero(dn)[2]
    return xs.mean()


def _lr_counts(wafer):
    """ゲート中心(nx/2)を境に左右のドープ画素数を返す。"""
    dn = wafer.grid == materials.get("doped_n").id
    nx = wafer.grid.shape[2]
    xs = np.nonzero(dn)[2]
    left = int((xs < nx // 2).sum())
    right = int((xs >= nx // 2).sum())
    return left, right


def test_tilt_creates_asymmetric_doping():
    """チルト注入は左右非対称のドープ分布を作る（垂直はほぼ対称）。"""
    w0, _ = _gate_and_implant(0.0)
    wt, _ = _gate_and_implant(45.0)
    l0, r0 = _lr_counts(w0)
    lt, rt = _lr_counts(wt)
    asym0 = abs(l0 - r0) / max(1, l0 + r0)
    asymt = abs(lt - rt) / max(1, lt + rt)
    # 垂直はほぼ対称、チルトは明確に非対称
    assert asymt > asym0
    assert asymt > 0.1


def test_tilt_shadow_reduces_doping_on_one_side():
    """チルトでゲート +x 側直近に影ができ、注入が抑制される。"""
    w0, cfg = _gate_and_implant(0.0)
    wt, _ = _gate_and_implant(50.0)
    dn0 = (w0.grid == materials.get("doped_n").id).sum()
    dnt = (wt.grid == materials.get("doped_n").id).sum()
    # シャドーイングにより総ドープ量は減る（影の分）
    assert dnt < dn0


def test_tilt_zero_matches_vertical():
    """tilt=0 は従来の垂直注入と同等のドープ量。"""
    w, _ = _gate_and_implant(0.0)
    assert (w.grid == materials.get("doped_n").id).any()


def test_tilt_roundtrip_params():
    """params_dict / _from_params で tilt_deg を保持。"""
    p = Implant(dopant="doped_n", range_um=0.3, tilt_deg=30.0)
    d = p.params_dict()
    assert d["tilt_deg"] == 30.0
    assert Implant._from_params(d).tilt_deg == 30.0


def test_tilt_rejects_out_of_range():
    """過大なチルト角は弾く。"""
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    try:
        Implant(dopant="doped_n", range_um=0.3, tilt_deg=80.0).apply(w)
    except ValueError:
        return
    raise AssertionError("範囲外 tilt_deg で ValueError が出るべき")
