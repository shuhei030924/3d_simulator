"""DEFECT 工程（欠陥注入）と後続工程の相互作用のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials, metrology, presets
from semisim.grid import Wafer, WaferConfig
from semisim.processes import CMP, CVD, Defect, DryEtch, Process, Strip
from semisim.recipe import Recipe


def _cfg(**kw) -> WaferConfig:
    base = dict(nx=50, ny=50, nz=60, pitch_um=0.1, substrate_um=2.0)
    base.update(kw)
    return WaferConfig(**base)


def _particle_id() -> int:
    return materials.get("particle").id


# --- 配置の基本 -------------------------------------------------------------
def test_particle_rests_on_surface():
    """パーティクルは表面に接して載り、空中に浮かない（連結性）。"""
    w = Wafer(_cfg())
    Defect(kind="particle", size_um=0.6, x_frac=0.5, y_frac=0.5).apply(w)
    pid = _particle_id()
    assert int(np.count_nonzero(w.grid == pid)) > 0
    # 最下点が基板上面の直上にある（浮遊しない）
    zs = np.nonzero((w.grid == pid).any(axis=(1, 2)))[0]
    sub_top = w.um_to_vox(2.0) - 1
    assert int(zs.min()) == sub_top + 1
    # 既存の基板は侵食されない
    assert int(np.count_nonzero(w.grid == materials.get("silicon").id)) == 50 * 50 * (sub_top + 1)


def test_particle_scatter_deterministic():
    """count>1 の乱数散布は seed 固定で決定的。"""
    w1, w2 = Wafer(_cfg()), Wafer(_cfg())
    d = Defect(kind="particle", size_um=0.3, count=8, seed=42)
    d.apply(w1)
    d.apply(w2)
    assert np.array_equal(w1.grid, w2.grid)
    # 別 seed では異なる配置
    w3 = Wafer(_cfg())
    Defect(kind="particle", size_um=0.3, count=8, seed=43).apply(w3)
    assert not np.array_equal(w1.grid, w3.grid)


def test_void_is_fully_buried():
    """埋込ボイドは表面を破らない密閉空洞になる。"""
    w = Wafer(_cfg())
    CVD(material="oxide", thickness_um=2.0).apply(w)
    Defect(kind="void", size_um=0.6, depth_um=0.8).apply(w)
    vm = metrology.void_metrics(w)
    assert vm["count"] == 1
    assert vm["total_um3"] > 0
    # 上面は平坦のまま（ボイドが表面に露出しない）
    z_top = w.top_surface_z()
    assert int(z_top.max()) == int(z_top.min())


def test_scratch_cuts_groove_across_x():
    """スクラッチは x 方向に横断する溝を掘る。"""
    w = Wafer(_cfg())
    Defect(kind="scratch", size_um=0.3, y_frac=0.5, depth_um=0.5).apply(w)
    z_top = w.top_surface_z()
    cy = 25
    sub_top = w.um_to_vox(2.0) - 1
    # 溝の中心行は全幅にわたり深さ 0.5µm 低い
    assert np.all(z_top[cy, :] == sub_top - w.um_to_vox(0.5))
    # 溝から離れた行は元の高さのまま
    assert np.all(z_top[0, :] == sub_top)


# --- 後続工程との相互作用 ----------------------------------------------------
def test_particle_blocks_etch_micromasking():
    """パーティクルが異方性エッチをブロックし、直下にピラーが残る。"""
    r = Recipe(config=_cfg())
    r.add(Defect(kind="particle", size_um=0.6, x_frac=0.5, y_frac=0.5))
    r.add(DryEtch(targets=["silicon"], depth_um=1.0))
    w = r.simulate()
    sub_top = w.um_to_vox(2.0) - 1
    si = materials.get("silicon").id
    # パーティクルから離れた列は 1.0µm 削れている
    far = w.grid[:, 5, 5]
    z_far = int(np.nonzero(far == si)[0].max())
    assert z_far == sub_top - w.um_to_vox(1.0)
    # パーティクル直下の列はシリコンが元の高さまで残る（ピラー）
    under = w.grid[:, 25, 25]
    z_under = int(np.nonzero(under == si)[0].max())
    assert z_under == sub_top
    # 洗浄（STRIP）でパーティクルを除去するとピラーだけが残る
    Strip(material="particle").apply(w)
    assert int(np.count_nonzero(w.grid == _particle_id())) == 0
    z_top = w.top_surface_z()
    assert int(z_top[25, 25]) == sub_top  # ピラー
    assert int(z_top[5, 5]) == sub_top - w.um_to_vox(1.0)


def test_particle_makes_bump_under_deposition():
    """パーティクル上に成膜すると表面バンプ（突起）になる。"""
    r = Recipe(config=_cfg())
    r.add(Defect(kind="particle", size_um=0.6, x_frac=0.5, y_frac=0.5))
    r.add(CVD(material="nitride", thickness_um=0.3))
    w = r.simulate()
    z_top = w.top_surface_z()
    # バンプ頂点は平坦部より パーティクル直径分 高い
    assert int(z_top[25, 25]) > int(z_top[5, 5])


def test_buried_void_opened_by_cmp():
    """埋込ボイドは CMP 研磨面が到達すると表面に開口する。"""
    r = Recipe(config=_cfg())
    r.add(CVD(material="oxide", thickness_um=2.0))
    r.add(Defect(kind="void", size_um=0.6, depth_um=0.5))
    r.add(CMP(remove_um=0.6))
    w = r.simulate()
    z_top = w.top_surface_z()
    # 開口: ボイド位置の上面が周囲より低い（ピット）
    assert int(z_top[25, 25]) < int(z_top[5, 5])
    # 密閉ボイドは消滅している（開口した）
    assert metrology.void_metrics(w)["count"] == 0


def test_default_etch_does_not_remove_particle():
    """ターゲット未指定のエッチでは硬質パーティクルは削れない（残渣）。"""
    w = Wafer(_cfg())
    CVD(material="oxide", thickness_um=1.0).apply(w)
    Defect(kind="particle", size_um=0.4).apply(w)
    n_before = int(np.count_nonzero(w.grid == _particle_id()))
    DryEtch(targets=[], depth_um=0.5).apply(w)
    assert int(np.count_nonzero(w.grid == _particle_id())) == n_before


# --- シリアライズ / 検証 ------------------------------------------------------
def test_defect_roundtrip_serialization():
    d = Defect(
        kind="void", size_um=0.45, x_frac=0.3, y_frac=0.7,
        count=3, seed=9, depth_um=0.8, material="particle",
    )
    d2 = Process.from_dict(d.to_dict())
    assert isinstance(d2, Defect)
    assert d2.to_dict() == d.to_dict()


def test_defect_invalid_params_raise():
    w = Wafer(_cfg())
    for bad in (
        Defect(kind="meteor"),
        Defect(size_um=0.0),
        Defect(x_frac=1.5),
        Defect(count=0),
        Defect(depth_um=-1.0),
    ):
        try:
            bad.apply(w)
        except ValueError:
            continue
        raise AssertionError(f"ValueError が送出されない: {bad}")


def test_defect_presets_build_and_run():
    """欠陥プリセットが構築・実行でき、欠陥の痕跡が現れる。"""
    r = presets.build("欠陥: マイクロマスキング")
    w = r.simulate()
    z_top = w.top_surface_z()
    assert int(z_top.max()) > int(z_top.min())  # ピラーが残る
    assert int(np.count_nonzero(w.grid == _particle_id())) == 0  # 洗浄済み

    r2 = presets.build("欠陥: 埋込ボイド開口")
    w2 = r2.simulate()
    z2 = w2.top_surface_z()
    assert int(z2.max()) > int(z2.min())  # 開口ピット
