"""IMPLANT ガウスプロファイルと OXIDE 消費比のパラメータ化テスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.masks import Mask, Shape
from semisim.processes import Implant, Oxidation, Photo
from semisim.recipe import Recipe


def _doped_depth_profile(wafer, dopant="doped_n"):
    """中央列におけるドープボクセルの z 集合を返す。"""
    dop = materials.get(dopant).id
    cx = wafer.config.nx // 2
    cy = wafer.config.ny // 2
    col = wafer.grid[:, cy, cx]
    return np.flatnonzero(col == dop)


def test_implant_gaussian_peaks_near_range(wafer):
    """ガウスプロファイルのドープ帯の中心が投影飛程付近にある。"""
    cfg = wafer.config
    Implant(dopant="doped_n", range_um=0.6, straggle_um=0.15).apply(wafer)
    zs = _doped_depth_profile(wafer)
    assert zs.size > 0
    z_top = wafer.top_surface_z()
    cx, cy = cfg.nx // 2, cfg.ny // 2
    surface = z_top[cy, cx]
    center_depth_vox = surface - float(zs.mean())
    expected = wafer.um_to_vox(0.6)
    # 帯の中心深さが Rp の ±2 ボクセル以内
    assert abs(center_depth_vox - expected) <= 2


def test_implant_lower_threshold_widens_band(wafer):
    cfg = wafer.config
    high = Recipe(config=cfg)
    high.add(Implant(dopant="doped_n", range_um=0.5, straggle_um=0.2, threshold=0.6))
    low = Recipe(config=cfg)
    low.add(Implant(dopant="doped_n", range_um=0.5, straggle_um=0.2, threshold=0.1))
    dop = materials.get("doped_n").id
    n_high = int((high.simulate().grid == dop).sum())
    n_low = int((low.simulate().grid == dop).sum())
    # しきい値を下げると注入帯が広くなる
    assert n_low > n_high


def test_implant_lateral_straggle_spreads_under_mask(wafer):
    """横方向ストラグルでマスク端の下にドープがにじむ。"""
    cfg = wafer.config
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    x0 = int(0.4 * cfg.nx)
    x1 = int(0.6 * cfg.nx)

    def stripe_dopant(lat):
        r = Recipe(config=cfg)
        r.add(Photo(mask=mask, thickness_um=0.6, polarity="negative"))
        r.add(
            Implant(
                dopant="doped_n",
                range_um=0.3,
                straggle_um=0.1,
                lateral_straggle_um=lat,
            )
        )
        w = r.simulate()
        dop = materials.get("doped_n").id
        # マスク直下（被覆ストライプ）の列にあるドープボクセル数
        return int((w.grid[:, :, x0:x1] == dop).sum())

    no_lat = stripe_dopant(0.0)
    with_lat = stripe_dopant(0.3)
    # 横散乱なし: レジスト直下は完全に保護され 0
    assert no_lat == 0
    # 横散乱あり: マスク端からにじんでドープが入り込む
    assert with_lat > 0


def test_implant_params_roundtrip():
    p = Implant(
        dopant="doped_p",
        range_um=0.5,
        straggle_um=0.2,
        lateral_straggle_um=0.1,
        threshold=0.25,
    )
    r = Implant._from_params(p.params_dict())
    assert r.dopant == "doped_p"
    assert r.range_um == 0.5
    assert r.lateral_straggle_um == 0.1
    assert r.threshold == 0.25


def test_oxidation_consume_fraction_affects_silicon(wafer):
    """消費比を上げるほどシリコン消費量が増える。"""
    cfg = wafer.config
    si = materials.get("silicon").id

    def si_count(frac):
        r = Recipe(config=cfg)
        r.add(Oxidation(thickness_um=0.5, consume_fraction=frac))
        return int((r.simulate().grid == si).sum())

    base = int((wafer.grid == si).sum())
    low = si_count(0.2)
    high = si_count(0.8)
    # どちらもシリコンを消費し、消費比が高いほど残存シリコンが少ない
    assert high < low < base


def test_oxidation_params_roundtrip():
    o = Oxidation(thickness_um=0.4, consume_fraction=0.6)
    r = Oxidation._from_params(o.params_dict())
    assert r.thickness_um == 0.4
    assert r.consume_fraction == 0.6
