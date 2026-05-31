"""PVD 斜め蒸着シャドーイング（tilt_deg）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, PVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _tall_feature_recipe(tilt):
    """中央に背の高いポリ柱を立て、その後 PVD（任意で斜め）を行う。"""
    cfg = WaferConfig(nx=60, ny=12, nz=70, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    # 背の高い柱を作る: ポリ厚塗り→中央だけ残して周囲を除去
    r.add(CVD(material="poly", thickness_um=1.0))
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=1.2, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=1.0))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="metal_al", thickness_um=0.2, tilt_deg=tilt))
    return r.simulate(), cfg


def test_tilt_creates_asymmetric_shadow():
    """斜め蒸着では柱の片側（風下）に膜が付かず左右非対称になる。"""
    w, cfg = _tall_feature_recipe(70.0)
    al = materials.get("metal_al").id
    # 基板上面付近の Al 被覆を左右で比較
    footprint = np.any(w.grid == al, axis=0)  # (ny, nx)
    cov_per_x = footprint.sum(axis=0)  # (nx,)
    left = int(cov_per_x[: cfg.nx // 2].sum())
    right = int(cov_per_x[cfg.nx // 2 :].sum())
    # 片側が明確に少ない（影）
    assert min(left, right) < 0.7 * max(left, right)


def test_no_tilt_symmetric():
    """傾斜0では左右対称（影なし）。"""
    w, cfg = _tall_feature_recipe(0.0)
    al = materials.get("metal_al").id
    footprint = np.any(w.grid == al, axis=0)
    cov_per_x = footprint.sum(axis=0)
    left = int(cov_per_x[: cfg.nx // 2].sum())
    right = int(cov_per_x[cfg.nx // 2 :].sum())
    assert abs(left - right) <= max(2, int(0.1 * max(left, right)))


def test_roundtrip_params():
    p = PVD(material="metal_cu", thickness_um=0.3, tilt_deg=45.0)
    d = p.params_dict()
    assert d["tilt_deg"] == 45.0
    assert PVD._from_params(d).tilt_deg == 45.0


def test_rejects_out_of_range_tilt():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        PVD(material="metal_al", thickness_um=0.2, tilt_deg=95.0).apply(w)
    except ValueError:
        return
    raise AssertionError("入射角95°で ValueError が出るべき")
