"""重力デタッチ（支持を失った孤立片の脱落）と浮遊材料不変条件のテスト。

不変条件: どの工程の適用後も、基板底面（z=0）に 26 近傍で連結しない固体
（＝宙に浮いた材料）は存在しない。アンダーカットや欠陥の切断で完全に
切り離された材料は、物理的には脱落して系外へ除かれる（重力・リンス）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from semisim import materials, presets
from semisim.grid import Wafer, WaferConfig
from semisim.processes import CVD, Defect, DryEtch, WetEtch


def _floating_voxels(grid) -> int:
    """底面 z=0 に連結しない固体ボクセル数（26 近傍）。"""
    solid = grid != materials.AIR
    if not solid.any():
        return 0
    lbl, _ = ndimage.label(solid, structure=np.ones((3, 3, 3), dtype=bool))
    anchored = set(np.unique(lbl[0])) - {0}
    return int((solid & ~np.isin(lbl, list(anchored))).sum())


def _sealed_air_voxels(grid) -> int:
    """上面に連結しない空気（密閉ボイド）ボクセル数（6 近傍）。"""
    air = grid == materials.AIR
    if not air.any():
        return 0
    lbl, _ = ndimage.label(air, structure=ndimage.generate_binary_structure(3, 1))
    open_ids = set(np.unique(lbl[-1])) - {0}
    return int((air & ~np.isin(lbl, list(open_ids))).sum())


def _cfg() -> WaferConfig:
    return WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.1, substrate_um=2.0)


def test_undercut_particle_falls_off_dry():
    """横方向バイアスで直下を掘り抜かれたパーティクルは脱落する。"""
    w = Wafer(_cfg())
    Defect(kind="particle", size_um=0.5).apply(w)
    DryEtch(targets=["silicon"], depth_um=1.0, lateral_um=0.5).apply(w)
    assert _floating_voxels(w.grid) == 0
    # 粒子自体が除去されている（宙吊りで残らない）
    assert int(np.count_nonzero(w.grid == materials.get("particle").id)) == 0


def test_undercut_particle_falls_off_wet():
    """等方ウェットエッチでアンダーカットされたパーティクルは脱落する。"""
    w = Wafer(_cfg())
    Defect(kind="particle", size_um=0.5).apply(w)
    WetEtch(targets=["silicon"], depth_um=0.8).apply(w)
    assert _floating_voxels(w.grid) == 0


def test_void_severing_thin_pillar_drops_top():
    """細い柱より大きい埋込ボイドが柱を切断すると、上部は脱落する。"""
    w = Wafer(_cfg())
    w.grid[20:45, 18:22, 18:22] = materials.get("oxide").id
    Defect(kind="void", size_um=0.8, depth_um=1.0).apply(w)
    assert _floating_voxels(w.grid) == 0


def test_particle_agglomerate_no_sealed_air():
    """多数粒子の重なり散布でも意図しない密閉空気を作らない。"""
    w = Wafer(_cfg())
    Defect(kind="particle", size_um=0.6, count=30, seed=3).apply(w)
    assert _sealed_air_voxels(w.grid) == 0
    assert _floating_voxels(w.grid) == 0


def test_partial_undercut_keeps_anchored_overhang():
    """部分アンダーカットで繋がっている庇は脱落させない（過剰除去しない）。"""
    w = Wafer(_cfg())
    CVD(material="nitride", thickness_um=0.3).apply(w)
    # 窒化膜の下のシリコンを浅く等方エッチ → 庇はまだ周囲で支持されている
    w.grid[19, 20, 20] = materials.AIR  # 開口を 1 点開けて薬液を入れる
    WetEtch(targets=["silicon"], depth_um=0.3).apply(w)
    nid = materials.get("nitride").id
    assert int(np.count_nonzero(w.grid == nid)) > 0  # 膜は残る
    assert _floating_voxels(w.grid) == 0


def test_presets_no_floating_at_any_step():
    """代表プリセットの全中間ステップで浮遊材料が無い。"""
    for name in ("欠陥: マイクロマスキング", "欠陥: 埋込ボイド開口",
                 "多層 Cu 配線", "FinFET フィン形成"):
        r = presets.build(name)
        for k in range(1, len(r.steps) + 1):
            w = r.simulate(up_to=k)
            assert _floating_voxels(w.grid) == 0, f"{name} step {k}"
