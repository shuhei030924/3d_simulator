"""境界条件・統合フロー・計測精度のテスト。

通常の単体テストでは拾いきれないエッジケース（マスク全面、基板の完全除去、
極小パラメータ、各種グリッドサイズ）と、複数工程を連ねた現実的なフローを検証する。
"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import materials, metrology
from semisim.grid import Wafer, WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import (
    CMP,
    CVD,
    PVD,
    Anneal,
    Diffusion,
    DryEtch,
    Fill,
    Implant,
    Oxidation,
    Photo,
    WetEtch,
    _require_positive,
)

SI = materials.get("silicon").id
RESIST = materials.get("photoresist").id


# === マスク全面のエッジケース ===============================================
def test_photo_empty_mask_positive_removes_all(wafer):
    """空マスクは全面選択（全面開口）扱い。ポジでは全面現像されレジストが消える。"""
    Photo(mask=Mask(shapes=[]), thickness_um=0.5, polarity="positive").apply(wafer)
    assert not (wafer.grid == RESIST).any()


def test_photo_empty_mask_negative_keeps_full_coat(wafer):
    """空マスク（全面選択）＋ネガでは現像対象が無く、レジストが全面に残る。"""
    Photo(mask=Mask(shapes=[]), thickness_um=0.5, polarity="negative").apply(wafer)
    resist = wafer.grid == RESIST
    # すべての列にレジストが存在する
    assert (resist.any(axis=0)).all()


def test_photo_full_area_opening(wafer):
    """全面を覆う矩形開口（ポジ）ではレジストが全除去される。"""
    full = Mask(shapes=[Shape("rect", {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0})])
    Photo(mask=full, thickness_um=0.5, polarity="positive").apply(wafer)
    assert not (wafer.grid == RESIST).any()


# === 基板の完全除去 =========================================================
def test_dry_etch_through_entire_substrate(wafer):
    """非常に深いドライエッチで露出シリコンを掘り切ると、その列は空気になる。"""
    deep = wafer.config.nz * wafer.config.pitch_um + 5.0
    DryEtch(targets=["silicon"], depth_um=deep).apply(wafer)
    # 露出シリコンを全部削るので固体は残らない
    assert not (wafer.grid != materials.AIR).any()


def test_wet_etch_does_not_penetrate_barrier(wafer):
    """ウェットエッチは障壁材料（酸化膜）を貫通しない。"""
    # シリコン上に酸化膜のフタを作る → その下のシリコンはエッチされない
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    si_before = int((wafer.grid == SI).sum())
    WetEtch(targets=["silicon"], depth_um=1.0).apply(wafer)
    si_after = int((wafer.grid == SI).sum())
    # 酸化膜に覆われているのでシリコンは減らない
    assert si_after == si_before


# === 極小パラメータ =========================================================
def test_implant_tiny_params_no_crash(wafer):
    """極小の飛程・ばらつきでも例外なく動作する（1 ボクセルに丸まる）。"""
    Implant(dopant="doped_n", range_um=0.001, straggle_um=0.001).apply(wafer)
    # クラッシュしないことが主目的。ドープは入っても入らなくてもよい。
    assert wafer.grid.shape == (wafer.config.nz, wafer.config.ny, wafer.config.nx)


def test_cmp_on_flat_surface_no_error(wafer):
    """平坦面への CMP は例外を出さず、上面より上は空気のまま。"""
    before = wafer.grid.copy()
    CMP(remove_um=0.3).apply(wafer)
    # 平坦面では研磨しても基板の総量は減るが負の領域は作らない
    assert (wafer.grid != materials.AIR).sum() <= (before != materials.AIR).sum()


# === 各種グリッドサイズ =====================================================
@pytest.mark.parametrize(
    "nx,ny,nz",
    [(10, 10, 12), (8, 16, 20), (32, 24, 30)],
)
def test_processes_various_grid_sizes(nx, ny, nz):
    """非正方形・極小グリッドでも主要工程が破綻しない。"""
    cfg = WaferConfig(nx=nx, ny=ny, nz=nz, pitch_um=0.1, substrate_um=0.5)
    w = Wafer(cfg)
    CVD(material="oxide", thickness_um=0.2).apply(w)
    PVD(material="metal_al", thickness_um=0.2).apply(w)
    DryEtch(depth_um=0.2).apply(w)
    assert w.grid.shape == (nz, ny, nx)


# === 統合フロー =============================================================
def test_damascene_flow_fills_and_planarizes(wafer):
    """トレンチ → Cu 充填 → CMP のダマシンフローが平坦面を生む。"""
    # 中央にトレンチ開口を作って掘る
    opening = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    Photo(mask=opening, thickness_um=0.6, polarity="positive").apply(wafer)
    DryEtch(targets=["silicon"], depth_um=0.5).apply(wafer)
    Fill(material="metal_cu", overfill_um=0.2).apply(wafer)
    CMP(remove_um=0.2).apply(wafer)
    # Cu が埋め込まれている
    assert (wafer.grid == materials.get("metal_cu").id).any()
    # CMP 後は上面がほぼ平坦（段差が小さい）
    assert metrology.step_height_um(wafer) < 0.3


def test_implant_anneal_drives_dopant_wider(wafer):
    """注入 → アニールでドープ領域が広がる（横方向ドライブイン）。"""
    Implant(dopant="doped_n", range_um=0.3, straggle_um=0.1).apply(wafer)
    dop_id = materials.get("doped_n").id
    before = int((wafer.grid == dop_id).sum())
    Anneal(depth_um=0.3).apply(wafer)
    after = int((wafer.grid == dop_id).sum())
    assert after > before


def test_diffusion_then_oxide_sequence(wafer):
    """拡散 → 熱酸化を連続適用しても破綻しない。"""
    Diffusion(dopant="doped_n", depth_um=0.4).apply(wafer)
    Oxidation(thickness_um=0.3).apply(wafer)
    # 酸化膜が生成されている
    assert (wafer.grid == materials.get("oxide").id).any()


# === 計測精度 ===============================================================
def test_film_thickness_exact_on_flat(wafer):
    """平坦面への等方 CVD は厚みが格子分解能どおりに一定になる。"""
    t_um = 0.5
    CVD(material="oxide", thickness_um=t_um).apply(wafer)
    stats = metrology.film_thickness_stats(wafer, "oxide")
    expected = wafer.um_to_vox(t_um) * wafer.config.pitch_um
    assert abs(stats["max"] - expected) < 1e-9
    # 平坦面なので全列が同一厚（標準偏差ほぼ 0）
    assert stats["std"] < wafer.config.pitch_um
    assert stats["coverage"] == pytest.approx(1.0)


def test_feature_width_known_line():
    """既知幅の金属ラインを手動配置し、feature_width_um が一致する。"""
    cfg = WaferConfig(nx=40, ny=10, nz=20, pitch_um=0.1, substrate_um=1.0)
    w = Wafer(cfg)
    z = int(w.top_surface_z().max()) + 1
    metal = materials.get("metal_al").id
    # x=10..19（10 ボクセル）に金属を置く → 幅 1.0µm
    w.grid[z, :, 10:20] = metal
    width = metrology.feature_width_um(w, "metal_al", z, 5)
    assert width == pytest.approx(10 * cfg.pitch_um)


def test_step_height_after_trench():
    """トレンチを掘ると段差(深さ)が掘り込み量と一致する。"""
    cfg = WaferConfig(nx=40, ny=40, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Wafer(cfg)
    opening = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    Photo(mask=opening, thickness_um=0.6, polarity="positive").apply(w)
    DryEtch(targets=["silicon"], depth_um=0.5).apply(w)
    # レジストを剥がしてから測ると、シリコン段差が掘り込み量と一致
    w.grid[w.grid == RESIST] = materials.AIR
    depth = metrology.step_height_um(w)
    assert abs(depth - 0.5) < 0.15


# === バリデーション =========================================================
@pytest.mark.parametrize("bad", [0.0, -1.0, -0.0, float("nan"), float("inf")])
def test_require_positive_rejects(bad):
    with pytest.raises(ValueError):
        _require_positive(bad, "テスト値")


def test_require_positive_accepts():
    assert _require_positive(0.5, "テスト値") == 0.5


# === 等方拡散ヘルパの形状 ===================================================
def test_isotropic_dilate_is_round_not_diamond():
    """_isotropic_dilate は十字（ひし形）ではなく丸く広がる。"""
    from semisim.processes import _isotropic_dilate

    mask = np.zeros((21, 21, 21), dtype=bool)
    mask[10, 10, 10] = True
    grown = _isotropic_dilate(mask, 5)
    # 対角方向のボクセル(距離 ~7) は半径5外なので含まれない（ひし形なら含まれる）
    assert not grown[13, 13, 13]
    # 軸方向の半径5ちょうどは含まれる
    assert grown[10, 10, 15]
    assert grown[10, 10, 5]
