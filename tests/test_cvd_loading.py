"""CVD 負荷効果（マクロローディング）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, Photo
from semisim.recipe import Recipe


def _oxide_thickness_vox(wafer) -> int:
    """平坦領域での酸化膜の積層厚（ボクセル）を中央列で測る。"""
    oxide = materials.get("oxide").id
    col = wafer.grid[:, wafer.config.ny // 2, wafer.config.nx // 2]
    return int(np.count_nonzero(col == oxide))


def test_loading_zero_matches_baseline():
    """loading=0 なら従来どおり一定厚で成膜される。"""
    cfg = WaferConfig(nx=30, ny=30, nz=60, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5, loading=0.0))
    w = r.simulate()
    assert _oxide_thickness_vox(w) == w.um_to_vox(0.5)


def test_loading_thins_film_on_dense_pattern():
    """密パターン上では負荷効果により膜が薄くなる。"""
    cfg = WaferConfig(nx=40, ny=40, nz=80, pitch_um=0.1, substrate_um=1.0)

    # 全面にレジストの厚い柱を残してパターン密度を高くする
    mask = Mask(shapes=[Shape("rect", {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0})])

    def thickness(loading):
        r = Recipe(config=cfg)
        r.add(Photo(mask=mask, thickness_um=1.0, polarity="negative"))
        r.add(CVD(material="oxide", thickness_um=0.5, loading=loading))
        return _oxide_thickness_vox(r.simulate())

    t_no = thickness(0.0)
    t_load = thickness(1.0)
    assert t_load < t_no


def test_loading_roundtrip():
    p = CVD(material="nitride", thickness_um=0.3, loading=0.4)
    d = p.params_dict()
    assert d["loading"] == 0.4
    p2 = CVD._from_params(d)
    assert p2.loading == 0.4
    assert p2.material == "nitride"


def test_loading_out_of_range_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5, loading=1.5))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("範囲外の loading で ValueError が出るべき")


def test_blanket_wafer_no_loading_effect():
    """平坦ウェハ（密度0）では loading を上げても厚みが変わらない。"""
    cfg = WaferConfig(nx=30, ny=30, nz=60, pitch_um=0.1, substrate_um=1.0)

    def thickness(loading):
        r = Recipe(config=cfg)
        r.add(CVD(material="oxide", thickness_um=0.5, loading=loading))
        return _oxide_thickness_vox(r.simulate())

    assert thickness(0.0) == thickness(1.0)
