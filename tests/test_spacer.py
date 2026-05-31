"""Spacer（サイドウォールスペーサ）プロセスのテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Spacer, Strip
from semisim.recipe import Recipe


def _gate_stack():
    """中央にポリのゲート段差を作ったウェハを返す。"""
    cfg = WaferConfig(nx=60, ny=20, nz=50, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="poly", thickness_um=0.4))
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    # 中央のみポリを残す（positive: 開口部のレジスト除去→開口部のポリをエッチ）
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=0.4))
    r.add(Strip(material="photoresist"))
    return r


def test_spacer_forms_on_sidewalls():
    """ゲート側壁にスペーサ材が残り、平坦フィールドには残らない。"""
    r = _gate_stack()
    r.add(Spacer(material="nitride", thickness_um=0.08))
    w = r.simulate()
    nid = materials.get("nitride").id
    grid = w.grid
    assert (grid == nid).any()  # スペーサが存在する
    # ゲート(中央)の左右脇にスペーサ、フィールド中央上には無いことを確認
    yc = w.config.ny // 2
    # フィールド左端カラムの最上面がnitrideでない（水平膜は除去済み）
    col = grid[:, yc, 2]
    solid = np.flatnonzero(col != materials.AIR)
    assert solid.size > 0
    assert col[solid[-1]] != nid


def test_spacer_adjacent_to_gate():
    """スペーサはゲート（ポリ）の隣接側壁列に存在する。"""
    r = _gate_stack()
    r.add(Spacer(material="nitride", thickness_um=0.08))
    w = r.simulate()
    nid = materials.get("nitride").id
    poly = materials.get("poly").id
    grid = w.grid
    yc = w.config.ny // 2
    # ポリが存在する列範囲を探す
    poly_cols = np.flatnonzero((grid[:, yc, :] == poly).any(axis=0))
    assert poly_cols.size > 0
    # ポリ列の左右どちらかの隣にnitrideがあること
    left = poly_cols.min() - 1
    right = poly_cols.max() + 1
    has_spacer = (grid[:, yc, left] == nid).any() or (grid[:, yc, right] == nid).any()
    assert has_spacer


def test_spacer_no_feature_clears():
    """段差が無い平坦面ではエッチバックで全て除去され材料が残らない。"""
    cfg = WaferConfig(nx=30, ny=20, nz=40, pitch_um=0.05, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(Spacer(material="nitride", thickness_um=0.08))
    w = r.simulate()
    nid = materials.get("nitride").id
    assert not (w.grid == nid).any()


def test_spacer_roundtrip():
    """params_dict / _from_params で往復できる。"""
    s = Spacer(material="oxide", thickness_um=0.06, overetch_um=0.02)
    s2 = Spacer._from_params(s.params_dict())
    assert s2.material == "oxide"
    assert abs(s2.thickness_um - 0.06) < 1e-9
    assert abs(s2.overetch_um - 0.02) < 1e-9


def test_spacer_thicker_more_material():
    """膜厚が大きいほどスペーサ材ボクセル数が増える。"""
    def count(th):
        r = _gate_stack()
        r.add(Spacer(material="nitride", thickness_um=th))
        w = r.simulate()
        return int((w.grid == materials.get("nitride").id).sum())

    assert count(0.12) > count(0.05)
