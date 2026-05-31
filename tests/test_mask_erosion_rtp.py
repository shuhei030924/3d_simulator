"""DryEtch マスク消耗と RTP（急速熱処理）のテスト。"""
from __future__ import annotations

import pytest

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import (
    CVD,
    DryEtch,
    Implant,
    Photo,
    RapidThermalAnneal,
    Strip,
)
from semisim.recipe import Recipe


# --- マスク消耗 ------------------------------------------------------------
def test_mask_erosion_thins_resist():
    cfg = WaferConfig(nx=30, ny=30, nz=50, pitch_um=0.1, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])

    def resist_voxels(erosion):
        r = Recipe(config=cfg)
        r.add(CVD(material="oxide", thickness_um=0.5))
        r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
        r.add(DryEtch(targets=["oxide"], depth_um=0.5, mask_erosion=erosion))
        w = r.simulate()
        return int((w.grid == materials.get("photoresist").id).sum())

    full = resist_voxels(0.0)
    eroded = resist_voxels(0.8)
    # マスク消耗ありの方がレジストが減る
    assert eroded < full


def test_mask_erosion_roundtrip():
    d = DryEtch(targets=["oxide"], depth_um=0.5, mask_erosion=0.4)
    restored = DryEtch._from_params(d.params_dict())
    assert restored.mask_erosion == 0.4


def test_mask_erosion_rejects_negative():
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    with pytest.raises(ValueError):
        DryEtch(targets=["oxide"], depth_um=0.5, mask_erosion=-0.1).apply(w)


# --- RTP -------------------------------------------------------------------
def _implant_wafer():
    cfg = WaferConfig(nx=40, ny=40, nz=50, pitch_um=0.1, substrate_um=2.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=0.8, polarity="positive"))
    r.add(Implant(dopant="doped_n", range_um=0.3, straggle_um=0.1))
    r.add(Strip(material="photoresist"))
    return cfg, r


def test_rtp_drives_dopant():
    cfg, r = _implant_wafer()
    before = r.simulate()
    n_before = int((before.grid == materials.get("doped_n").id).sum())
    r.add(RapidThermalAnneal(depth_um=0.2, lateral_factor=0.3))
    after = r.simulate()
    n_after = int((after.grid == materials.get("doped_n").id).sum())
    # ドライブインでドープ領域が増える
    assert n_after > n_before


def test_rtp_lateral_suppressed_vs_isotropic():
    # 横拡散比が小さいほど横方向の広がりが小さい
    def width(lf):
        cfg, r = _implant_wafer()
        r.add(RapidThermalAnneal(depth_um=0.3, lateral_factor=lf))
        w = r.simulate()
        dop = w.grid == materials.get("doped_n").id
        # 中央断面でドープが占める最大 x 幅
        ny = w.config.ny // 2
        cols = dop[:, ny, :].any(axis=0)
        return int(cols.sum())

    narrow = width(0.0)
    wide = width(1.0)
    assert narrow <= wide


def test_rtp_roundtrip():
    rtp = RapidThermalAnneal(depth_um=0.18, lateral_factor=0.25)
    restored = RapidThermalAnneal._from_params(rtp.params_dict())
    assert restored.depth_um == 0.18
    assert restored.lateral_factor == 0.25


def test_rtp_registered():
    from semisim.processes import available_types
    assert any(t == "RTP" for t, _ in available_types())
