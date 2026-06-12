"""サブステップ補間（物理スケール）と輪郭スムージングのテスト。

中間フレームは工程の量パラメータを比例配分した工程を実エンジンで適用して
生成する（モーフィング禁止）。よって各フレームは物理的に到達可能な状態で
あり、成膜は単調増加・エッチは単調減少になるはず。
"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import animation, materials
from semisim.grid import Wafer, WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import ALD, CVD, DryEtch, Oxidation, Photo, Strip
from semisim.recipe import Recipe


def _cfg() -> WaferConfig:
    return WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=1.5)


# --- scaled_process -----------------------------------------------------------
def test_scaled_cvd_half_thickness():
    p = animation.scaled_process(CVD(material="oxide", thickness_um=0.8), 0.5)
    assert isinstance(p, CVD)
    assert p.thickness_um == pytest.approx(0.4)
    assert p.material == "oxide"


def test_scaled_ald_cycles_rounded_min1():
    p = animation.scaled_process(ALD(material="hafnia", cycles=12), 0.25)
    assert p.cycles == 3  # round(12*0.25)
    p2 = animation.scaled_process(ALD(material="hafnia", cycles=2), 0.1)
    assert p2.cycles == 1  # 最低 1 サイクル


def test_scaled_oxidation_time_mode_scales_time():
    """時間指定の熱酸化は時間を比例配分（Deal-Grove で厚さ∝√t を保つ）。"""
    p = animation.scaled_process(
        Oxidation(thickness_um=0.3, time_min=60.0, temperature_c=1000.0), 0.5
    )
    assert p.time_min == pytest.approx(30.0)
    assert p.thickness_um == pytest.approx(0.3)  # 厚さ指定値は触らない


def test_scaled_instant_process_returns_none():
    assert animation.scaled_process(Photo(), 0.5) is None
    assert animation.scaled_process(Strip(), 0.5) is None


# --- step_slices substeps -----------------------------------------------------
def _deposition_recipe() -> Recipe:
    r = Recipe(config=_cfg())
    r.add(CVD(material="nitride", thickness_um=0.8))
    return r


def test_substeps_monotone_deposition():
    """成膜のサブステップは膜量が単調増加し、最終フレームは正規結果と一致。"""
    r = _deposition_recipe()
    slices = animation.step_slices(r, substeps=4)
    assert len(slices) == 1 + 4  # 初期 + 25/50/75/100%
    nid = materials.get("nitride").id
    counts = [int(np.count_nonzero(p == nid)) for _t, p, _w, _h in slices]
    assert counts[0] == 0
    assert all(a < b for a, b in zip(counts[1:], counts[2:]))  # 単調増加
    # 最終フレーム = 通常シミュレーションの断面と完全一致
    ref = animation.step_slices(r, substeps=1)[-1][1]
    assert np.array_equal(slices[-1][1], ref)


def test_substeps_monotone_etch():
    """エッチのサブステップは除去量が単調増加（材料が単調減少）。"""
    r = Recipe(config=_cfg())
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="positive"))
    r.add(DryEtch(targets=["silicon"], depth_um=0.8))
    slices = animation.step_slices(r, substeps=4)
    si = materials.get("silicon").id
    # PHOTO は瞬時 1 フレーム、DRY は 4 フレーム → 初期+1+4
    assert len(slices) == 6
    si_counts = [int(np.count_nonzero(p == si)) for _t, p, _w, _h in slices[2:]]
    assert all(a > b for a, b in zip(si_counts, si_counts[1:]))  # 単調減少


def test_substep_titles_show_progress():
    slices = animation.step_slices(_deposition_recipe(), substeps=2)
    assert "（50%）" in slices[1][0]
    assert not slices[-1][0].endswith("%）")  # 確定フレームに % は付かない


def test_substeps_invalid():
    with pytest.raises(ValueError):
        animation.step_slices(_deposition_recipe(), substeps=0)


# --- smooth_plane -------------------------------------------------------------
def test_smooth_plane_preserves_ids_and_shape():
    """平滑化は形状を factor 倍し、存在しない材料 ID を発明しない。"""
    rng = np.random.default_rng(0)
    plane = rng.choice([0, 1, 2], size=(20, 24)).astype(np.uint8)
    out = animation.smooth_plane(plane, factor=3)
    assert out.shape == (60, 72)
    assert set(np.unique(out)) <= set(np.unique(plane))


def test_smooth_plane_keeps_bulk_regions():
    """大きな一様領域の内部は平滑化で変化しない（境界±半ボクセルのみ）。"""
    plane = np.zeros((20, 20), dtype=np.uint8)
    plane[:10] = 1  # 下半分が材料 1
    out = animation.smooth_plane(plane, factor=3)
    assert (out[:24] == 1).all()  # 境界から十分離れた内部
    assert (out[36:] == 0).all()


# --- 物理方向の検証（どこから増える/減るか） ---------------------------------
def test_fill_animates_bottom_up():
    """FILL の途中フレームは下から上へ充填レベルが上がる。"""
    from semisim.processes import Fill

    r = Recipe(config=_cfg())
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="positive"))
    r.add(DryEtch(targets=["silicon"], depth_um=1.0))
    r.add(Strip(material="photoresist"))
    r.add(Fill(material="metal_cu", overfill_um=0.2))
    slices = animation.step_slices(r, substeps=4)
    cu = materials.get("metal_cu").id
    frames = [p for t, p, _w, _h in slices if "FILL" in t]
    assert len(frames) == 4  # 25/50/75/100%
    tops, counts = [], []
    for p in frames:
        zs = np.nonzero((p == cu).any(axis=1))[0]
        assert zs.size > 0
        tops.append(int(zs.max()))
        counts.append(int(np.count_nonzero(p == cu)))
    # 充填上面が単調に上がり、量も単調増加（=下から埋まる）
    assert all(a <= b for a, b in zip(tops, tops[1:]))
    assert all(a < b for a, b in zip(counts, counts[1:]))
    # 25% 時点では完全充填レベルに達していない
    assert tops[0] < tops[-1]


def test_pvd_substeps_never_lose_material():
    """PVD の途中フレームは時間発展プレフィックス＝材料が消えない。"""
    from semisim.processes import PVD

    r = Recipe(config=_cfg())
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.4, polarity="positive"))
    r.add(DryEtch(targets=["silicon"], depth_um=0.8))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="metal_al", thickness_um=0.4, step_coverage=0.5))
    slices = animation.step_slices(r, substeps=4)
    pvd_frames = [p for t, p, _w, _h in slices if "PVD" in t]
    assert len(pvd_frames) == 4
    for a, b in zip(pvd_frames, pvd_frames[1:]):
        lost = (a != 0) & (b == 0)
        assert not lost.any()  # 成膜中に材料が消えない


def test_dry_lateral_undercut_develops_progressively():
    """DRY の横アンダーカットは進行とともに発達する（最初から全量出ない）。"""
    from semisim.processes import DryEtch as DE

    p25 = animation.scaled_process(
        DE(targets=["silicon"], depth_um=1.0, lateral_um=0.4, notch_um=0.2), 0.25
    )
    assert p25.depth_um == pytest.approx(0.25)
    assert p25.lateral_um == pytest.approx(0.1)
    assert p25.notch_um == pytest.approx(0.05)


def test_cmp_dishing_scales_with_progress():
    from semisim.processes import CMP as C

    p50 = animation.scaled_process(
        C(remove_um=0.6, soft_material="metal_cu", dishing_um=0.2, erosion_um=0.1), 0.5
    )
    assert p50.remove_um == pytest.approx(0.3)
    assert p50.dishing_um == pytest.approx(0.1)
    assert p50.erosion_um == pytest.approx(0.05)


def test_backgrind_substeps_do_not_corrupt_recipe_config():
    """BACKGRIND の途中フレーム生成がレシピの config を破壊しない。"""
    from semisim.processes import Backgrind

    r = Recipe(config=_cfg())
    r.add(CVD(material="oxide", thickness_um=0.3))
    r.add(Backgrind(thin_um=0.8))
    sub_before = r.config.substrate_um
    animation.step_slices(r, substeps=4)
    assert r.config.substrate_um == pytest.approx(sub_before)
    # 再シミュレーションも以前と同一結果（config 非破壊の外形的証拠）
    w1 = r.simulate()
    r.invalidate()
    w2 = r.simulate()
    assert np.array_equal(w1.grid, w2.grid)


def test_fill_pvd_default_unchanged():
    """level_fraction/growth_fraction の既定値 1.0 では従来結果と同一。"""
    from semisim.processes import PVD, Fill, Process

    f = Fill(material="metal_cu", overfill_um=0.2)
    f2 = Process.from_dict({"type": "FILL", "material": "metal_cu",
                            "overfill_um": 0.2})
    assert f.to_dict() == f2.to_dict()
    p = PVD(material="metal_al", thickness_um=0.3)
    p2 = Process.from_dict(p.to_dict())
    assert p2.growth_fraction == 1.0


# --- config 進化の一貫性（スナップショットに config も保存） -----------------
def test_cache_resume_preserves_evolved_config():
    """BACKGRIND 後のキャッシュ再開でも進化した config（基板厚）が復元される。

    スナップショットがグリッドのみだと、キャッシュ再開時に後続工程
    （CMP の研磨下限等）が初期 config を見てしまい、コールドリプレイと
    結果が食い違う。
    """
    from semisim.processes import Backgrind

    def build():
        r = Recipe(config=_cfg())
        r.add(CVD(material="oxide", thickness_um=0.4))
        r.add(Backgrind(thin_um=0.6))
        r.add(CVD(material="nitride", thickness_um=0.2))
        return r

    cold = build()
    w_cold = cold.simulate()  # 先頭から一括
    warm = build()
    warm.simulate(up_to=2)  # BACKGRIND までキャッシュを作る
    w_warm = warm.simulate()  # スナップショットから再開
    assert np.array_equal(w_cold.grid, w_warm.grid)
    # 再開ウェハの config も進化済み（基板が薄い）
    assert w_warm.config.substrate_um == pytest.approx(w_cold.config.substrate_um)
    assert w_warm.config.substrate_um < 1.5


def test_partial_frames_use_evolved_config():
    """BACKGRIND 後の CMP 途中フレームは進化後の基板厚を研磨下限に使う。"""
    from semisim.processes import CMP, Backgrind

    r = Recipe(config=_cfg())  # 基板 1.5µm
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(Backgrind(thin_um=0.5))  # 基板 1.0µm に薄化、上面は 1.5µm へ
    r.add(CMP(remove_um=5.0))  # 過大研磨 → 下限は「現在の」基板上面
    slices = animation.step_slices(r, substeps=4)
    cmp_frames = [p for t, p, _w, _h in slices if "CMP" in t]
    assert len(cmp_frames) == 4
    pre = [p for t, p, _w, _h in slices if "BACKGRIND" in t][-1]
    pre_top = int(np.nonzero((pre != 0).any(axis=1))[0].max())
    # 途中フレームでも研磨が実際に進む（古い config の下限だと何も削れない）
    mid_top = int(np.nonzero((cmp_frames[1] != 0).any(axis=1))[0].max())
    assert mid_top < pre_top


def test_fill_void_region_never_filled_in_partial_frames():
    """void_ar 付き FILL: キーホール予定領域は途中充填でも埋まらない（非単調防止）。"""
    from semisim.processes import Fill

    cfg = WaferConfig(nx=44, ny=44, nz=70, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=2.0))
    r.add(Photo(mask=Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0,
                                                 "x1": 0.55, "y1": 1.0})]),
                thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.8))
    r.add(Strip(material="photoresist"))
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=2.0))
    slices = animation.step_slices(r, substeps=4)
    frames = [p for t, p, _w, _h in slices if "FILL" in t]
    for a, b in zip(frames, frames[1:]):
        lost = (a != 0) & (b == 0)
        assert not lost.any()  # 充填中に材料が消えない（最終ボイド込み）


def test_cvd_roughness_never_removes_preexisting_material():
    """CVD ラフネスの凹みは新しく付いた膜のみを削る（下地は同材料でも不可侵）。"""
    w = Wafer(_cfg())
    CVD(material="oxide", thickness_um=0.5).apply(w)
    pre = w.grid.copy()
    # 薄い膜 + 大きなラフネス（凹みが膜厚を超えるワーストケース）
    CVD(material="oxide", thickness_um=0.1, roughness_um=0.3, seed=7).apply(w)
    lost = (pre != 0) & (w.grid == 0)
    assert not lost.any()
