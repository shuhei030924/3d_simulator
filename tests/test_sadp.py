"""SADP（自己整合ダブルパターニング/ピッチダブリング）の統合テスト。

マンドレル→スペーサ→マンドレル除去→転写の一連で、線密度が 2 倍に
なる（1 マンドレルから 2 本の自立スペーサ／フィンが得られる）ことを検証する。
"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Spacer, Strip
from semisim.recipe import Recipe


def _mandrel_mask(centers, w):
    half = w / 2.0
    return Mask(shapes=[
        Shape("rect", {"x0": c - half, "y0": 0.0, "x1": c + half, "y1": 1.0})
        for c in centers
    ])


def _count_lines(grid, mat_id, z):
    """z 行で材料 mat_id の連結ライン本数を数える（中央 Y 断面）。"""
    yc = grid.shape[1] // 2
    row = (grid[z, yc, :] == mat_id).astype(int)
    rises = np.diff(np.concatenate(([0], row, [0]))) == 1
    return int(rises.sum())


def _run_sadp(n_mandrel):
    cfg = WaferConfig(nx=240, ny=8, nz=120, pitch_um=0.02, substrate_um=1.0)
    # 境界を避けてマンドレル中心を等間隔配置
    centers = [(i + 1) / (n_mandrel + 1) for i in range(n_mandrel)]
    r = Recipe(config=cfg)
    r.add(CVD(material="poly", thickness_um=0.30))
    r.add(Photo(mask=_mandrel_mask(centers, 0.10), thickness_um=0.30,
                polarity="positive"))
    r.add(Spacer(material="oxide", thickness_um=0.06, overetch_um=0.02))
    r.add(Strip(material="photoresist"))
    return cfg, r.simulate()


def test_sadp_doubles_line_density():
    """3 本のマンドレルから 6 本の自立スペーサが得られる（2×）。"""
    cfg, w = _run_sadp(3)
    # マンドレル高さ中央付近（poly上面 0.30µm + 0.15µm ＝ z=1.45µm）
    z = int(1.45 / cfg.pitch_um)
    n_spacer = _count_lines(w.grid, materials.get("oxide").id, z)
    assert n_spacer == 6


def test_sadp_pattern_transfer_to_substrate():
    """スペーサをマスクに下地ポリへ転写すると 6 本のフィンになる。"""
    cfg, w = _run_sadp(3)
    r = Recipe(config=cfg)
    # 同じレシピを再構築して転写まで実行
    centers = [(i + 1) / 4 for i in range(3)]
    r.add(CVD(material="poly", thickness_um=0.30))
    r.add(Photo(mask=_mandrel_mask(centers, 0.10), thickness_um=0.30,
                polarity="positive"))
    r.add(Spacer(material="oxide", thickness_um=0.06, overetch_um=0.02))
    r.add(Strip(material="photoresist"))
    r.add(DryEtch(targets=["poly"], depth_um=0.32))
    w2 = r.simulate()
    # ポリ下部（z=1.1µm, 基板上面 1.0µm 直上）でフィン本数を数える
    z = int(1.1 / cfg.pitch_um)
    n_fin = _count_lines(w2.grid, materials.get("poly").id, z)
    assert n_fin == 6


def test_sadp_spacers_are_freestanding():
    """マンドレル（レジスト）除去後はレジストが残っていない。"""
    _cfg, w = _run_sadp(2)
    assert int((w.grid == materials.get("photoresist").id).sum()) == 0


def test_sadp_spacer_width_is_sublithographic():
    """スペーサ幅は成膜厚で決まり、マンドレル幅より細い（サブリソ）。"""
    cfg, w = _run_sadp(2)
    z = int(1.45 / cfg.pitch_um)
    yc = w.grid.shape[1] // 2
    row = w.grid[z, yc, :] == materials.get("oxide").id
    xs = np.flatnonzero(row)
    assert xs.size > 0
    breaks = np.where(np.diff(xs) > 1)[0] + 1
    widths = [seg.size * cfg.pitch_um for seg in np.split(xs, breaks)]
    # スペーサ幅 ≈ 0.06µm（成膜厚）、マンドレル幅 0.10*4.8µm=0.48µm より十分細い
    for wv in widths:
        assert wv < 0.2
