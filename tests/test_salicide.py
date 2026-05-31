"""SALICIDE（自己整合シリサイド）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Silicidation, Strip
from semisim.recipe import Recipe


def test_silicide_forms_on_exposed_silicon():
    """露出シリコン上面がシリサイドに変換される。"""
    cfg = WaferConfig(nx=30, ny=20, nz=50, pitch_um=0.05, substrate_um=1.5)
    r = Recipe(config=cfg)
    r.add(Silicidation(thickness_um=0.1))
    w = r.simulate()
    counts = metrology.material_counts(w)
    assert counts.get("silicide", 0) > 0


def test_self_aligned_not_on_oxide():
    """酸化膜で覆われた領域にはシリサイドが形成されない（自己整合）。"""
    cfg = WaferConfig(nx=40, ny=20, nz=55, pitch_um=0.05, substrate_um=1.5)
    r = Recipe(config=cfg)
    # 左半分に酸化膜キャップを残す
    r.add(CVD(material="oxide", thickness_um=0.3))
    mask = Mask(shapes=[Shape("rect", {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="negative"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.4))  # 右半分の酸化膜を除去→Si露出
    r.add(Strip())
    r.add(Silicidation(thickness_um=0.1))
    w = r.simulate()
    sil = materials.get("silicide").id
    # シリサイドは右半分（露出Si側）に偏在する
    footprint = np.any(w.grid == sil, axis=0)  # (ny, nx)
    left = int(footprint[:, : cfg.nx // 2].sum())
    right = int(footprint[:, cfg.nx // 2 :].sum())
    assert right > left


def test_consumes_silicon():
    """シリサイド化でシリコンが消費される（変換）。"""
    cfg = WaferConfig(nx=24, ny=16, nz=45, pitch_um=0.05, substrate_um=1.5)
    r0 = Recipe(config=cfg)
    si0 = metrology.material_counts(r0.simulate()).get("silicon", 0)
    r1 = Recipe(config=cfg)
    r1.add(Silicidation(thickness_um=0.15))
    si1 = metrology.material_counts(r1.simulate()).get("silicon", 0)
    assert si1 < si0


def test_react_poly_flag_roundtrip():
    p = Silicidation(thickness_um=0.08, react_poly=False)
    d = p.params_dict()
    assert d["thickness_um"] == 0.08
    assert d["react_poly"] is False
    back = Silicidation._from_params(d)
    assert back.react_poly is False


def test_rejects_non_positive_thickness():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        Silicidation(thickness_um=0.0).apply(w)
    except ValueError:
        return
    raise AssertionError("厚さ0で ValueError が出るべき")
