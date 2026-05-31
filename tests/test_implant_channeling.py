"""Implant チャネリングテール（結晶軸チャネリング）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.processes import Implant
from semisim.recipe import Recipe


def _deepest_doped_z(wafer) -> int:
    """ドープ層の最深 z インデックス（小さいほど深い）。無ければ nz。"""
    dn = materials.BY_NAME["doped_n"].id
    dp = materials.BY_NAME["doped_p"].id
    doped = np.isin(wafer.grid, [dn, dp])
    zs = np.where(doped.any(axis=(1, 2)))[0]
    return int(zs.min()) if zs.size else wafer.grid.shape[0]


def test_channeling_extends_deeper():
    """チャネリング裾ありでドープ層がより深く（小さい z まで）達する。"""
    cfg = WaferConfig(nx=30, ny=30, nz=80, pitch_um=0.05, substrate_um=3.0)

    def deepest(cf):
        r = Recipe(config=cfg)
        r.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.08,
                      channeling_fraction=cf, tail_decay_um=0.6))
        return _deepest_doped_z(r.simulate())

    no_tail = deepest(0.0)
    with_tail = deepest(0.8)
    # 裾ありの方が深く（z が小さく）達する
    assert with_tail < no_tail


def test_no_channeling_unchanged():
    """channeling_fraction=0 なら従来の埋込深さと一致する。"""
    cfg = WaferConfig(nx=30, ny=30, nz=80, pitch_um=0.05, substrate_um=3.0)
    r1 = Recipe(config=cfg)
    r1.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.08))
    r2 = Recipe(config=cfg)
    r2.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.08,
                   channeling_fraction=0.0))
    assert _deepest_doped_z(r1.simulate()) == _deepest_doped_z(r2.simulate())


def test_channeling_roundtrip():
    p = Implant(dopant="doped_p", range_um=0.5, channeling_fraction=0.3,
                tail_decay_um=0.4)
    d = p.params_dict()
    assert d["channeling_fraction"] == 0.3
    assert d["tail_decay_um"] == 0.4
    p2 = Implant._from_params(d)
    assert p2.channeling_fraction == 0.3
    assert p2.tail_decay_um == 0.4


def test_channeling_out_of_range_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=50, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(Implant(dopant="doped_n", range_um=0.4, channeling_fraction=1.5))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("範囲外チャネリング比で ValueError が出るべき")
