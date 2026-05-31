"""WetEtch の横方向比（アンダーカット異方性）テスト。"""
from __future__ import annotations

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, Photo, WetEtch
from semisim.recipe import Recipe


def _undercut_run(lateral_ratio):
    """酸化膜をレジストマスクで保護しつつウェットエッチし、アンダーカット幅を測る。"""
    cfg = WaferConfig(nx=80, ny=20, nz=40, pitch_um=0.05, substrate_um=0.5)
    # 中央のみ開口（両側はレジストで保護）
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(WetEtch(targets=["oxide"], depth_um=0.8, lateral_ratio=lateral_ratio))
    w = r.simulate()
    # レジスト下(マスク外)で酸化膜が削れた最大横幅を測る
    pr_id = materials.get("photoresist").id
    g = w.grid
    y = cfg.ny // 2
    # レジストに覆われている列のうち、酸化膜が消えた(=空気)列数を数える
    under_pr = (g[:, y, :] == pr_id).any(axis=0)  # レジスト直下の列
    air_in_oxzone = (g[:, y, :] == materials.AIR).any(axis=0)
    undercut_cols = int((under_pr & air_in_oxzone).sum())
    return undercut_cols


def test_isotropic_has_most_undercut():
    """lateral_ratio=1.0 は等方で最大のアンダーカット。"""
    iso = _undercut_run(1.0)
    aniso = _undercut_run(0.2)
    assert iso > aniso


def test_zero_lateral_minimal_undercut():
    """lateral_ratio=0 ではアンダーカットがほぼ無い。"""
    none = _undercut_run(0.0)
    iso = _undercut_run(1.0)
    assert none < iso


def test_wet_still_etches_vertically():
    """横方向比 0 でも縦方向には削れる（深さは確保される）。"""
    cfg = WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(WetEtch(targets=["oxide"], depth_um=0.5, lateral_ratio=0.0))
    w = r.simulate()
    oxide_id = materials.get("oxide").id
    # 酸化膜の一部が削れている
    assert (w.grid == oxide_id).sum() < cfg.nx * cfg.ny * w.um_to_vox(1.0)


def test_wet_roundtrip_params():
    """params_dict / _from_params で lateral_ratio を保持。"""
    p = WetEtch(targets=["oxide"], depth_um=0.5, lateral_ratio=0.3)
    d = p.params_dict()
    assert d["lateral_ratio"] == 0.3
    p2 = WetEtch._from_params(d)
    assert p2.lateral_ratio == 0.3


def test_wet_rejects_bad_ratio():
    """範囲外の横方向比は弾く。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.1, substrate_um=0.5)
    w = Recipe(config=cfg).simulate()
    try:
        WetEtch(targets=["silicon"], depth_um=0.2, lateral_ratio=1.5).apply(w)
    except ValueError:
        return
    raise AssertionError("範囲外 lateral_ratio で ValueError が出るべき")
