"""BACKGRIND（裏面研削 / ウェハ薄化）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials, metrology
from semisim.grid import WaferConfig
from semisim.processes import CVD, Backgrind
from semisim.recipe import Recipe


def test_thins_substrate():
    """裏面研削で基板厚が減る。"""
    cfg = WaferConfig(nx=16, ny=16, nz=60, pitch_um=0.1, substrate_um=3.0)
    r = Recipe(config=cfg)
    si0 = metrology.material_counts(r.simulate()).get("silicon", 0)
    r2 = Recipe(config=cfg)
    r2.add(Backgrind(thin_um=1.0))
    w = r2.simulate()
    si1 = metrology.material_counts(w).get("silicon", 0)
    assert si1 < si0
    # config も更新される
    assert w.config.substrate_um < 3.0


def test_device_layer_preserved():
    """表面のデバイス膜は研削で削られず、底へシフトする。"""
    cfg = WaferConfig(nx=16, ny=16, nz=70, pitch_um=0.1, substrate_um=3.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="tungsten", thickness_um=0.5))
    r.add(Backgrind(thin_um=1.0))
    w = r.simulate()
    w_id = materials.get("tungsten").id
    # タングステンは残っている
    assert int(np.sum(w.grid == w_id)) > 0


def test_does_not_grind_into_devices():
    """基板より厚い研削指定でも最低 1 ボクセルの基板を残す。"""
    cfg = WaferConfig(nx=12, ny=12, nz=50, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(Backgrind(thin_um=5.0))  # 基板厚を超える要求
    w = r.simulate()
    si = metrology.material_counts(w).get("silicon", 0)
    assert si > 0  # 基板は残る


def test_roundtrip_params():
    p = Backgrind(thin_um=2.5)
    d = p.params_dict()
    assert d["thin_um"] == 2.5
    assert Backgrind._from_params(d).thin_um == 2.5


def test_rejects_non_positive():
    cfg = WaferConfig(nx=12, ny=12, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Recipe(config=cfg).simulate()
    try:
        Backgrind(thin_um=0.0).apply(w)
    except ValueError:
        return
    raise AssertionError("研削量0で ValueError が出るべき")
