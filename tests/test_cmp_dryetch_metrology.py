"""CMPディッシング・DryEtch横バイアス・WetEtch基板保護・追加メトロロジのテスト。"""
from __future__ import annotations

import json

from semisim import materials, metrology
from semisim.cli import main as cli_main
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CMP, CVD, DryEtch, Photo, WetEtch
from semisim.recipe import Recipe


# --- CMP ディッシング ------------------------------------------------------
def test_cmp_dishing_recesses_soft_material():
    cfg = WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.1, substrate_um=2.0)
    r2 = Recipe(config=cfg)
    r2.add(CVD(material="metal_cu", thickness_um=1.0))
    r2.add(CMP(remove_um=0.3, soft_material="metal_cu", dishing_um=0.2,
               dishing_width_um=0.3))
    w = r2.simulate()
    z_top = w.top_surface_z()
    center = int(z_top[20, 20])
    edge = int(z_top[1, 1])
    # ディッシングは中央ほど深く、縁ほど浅い凹面（皿状）になる
    assert center < edge


def test_cmp_dishing_roundtrip():
    c = CMP(remove_um=0.5, stop_material="nitride", soft_material="metal_cu", dishing_um=0.15)
    restored = CMP._from_params(c.params_dict())
    assert restored.soft_material == "metal_cu"
    assert restored.dishing_um == 0.15
    assert restored.stop_material == "nitride"


# --- DryEtch 横方向バイアス ------------------------------------------------
def test_dryetch_lateral_undercut():
    cfg = WaferConfig(nx=50, ny=50, nz=60, pitch_um=0.1, substrate_um=2.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])

    def remaining(lat):
        r = Recipe(config=cfg)
        r.add(CVD(material="oxide", thickness_um=0.6))
        r.add(Photo(mask=mask, thickness_um=0.6, polarity="positive"))
        r.add(DryEtch(targets=["oxide"], depth_um=0.6, lateral_um=lat))
        return int((r.simulate().grid == materials.get("oxide").id).sum())

    # 横バイアスありの方が酸化膜が多く削れる（アンダーカット）
    assert remaining(0.2) < remaining(0.0)


def test_dryetch_lateral_roundtrip():
    d = DryEtch(targets=["oxide"], depth_um=0.5, lateral_um=0.1)
    restored = DryEtch._from_params(d.params_dict())
    assert restored.lateral_um == 0.1


# --- WetEtch 基板保護 ------------------------------------------------------
def test_wetetch_keeps_substrate_bottom():
    cfg = WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    # シリコンを大量にウェットエッチしても最下層は残る
    WetEtch(targets=["silicon"], depth_um=50.0).apply(w)
    assert (w.grid[0] != materials.AIR).any()


# --- メトロロジ追加項目 ----------------------------------------------------
def test_void_volume_detects_buried_air(wafer):
    cfg = wafer.config
    cx, cy = cfg.nx // 2, cfg.ny // 2
    z = wafer.top_surface_z()[cy, cx]
    # 人工的に閉塞ボイドを作る
    wafer.grid[z + 1, cy, cx] = materials.AIR
    wafer.grid[z + 2, cy, cx] = materials.get("oxide").id
    assert metrology.void_volume_um3(wafer) > 0.0


def test_void_volume_zero_on_clean_surface(wafer):
    assert metrology.void_volume_um3(wafer) == 0.0


def test_cmp_uniformity_flat_is_low(wafer):
    # 平坦な基板は均一性指標が 0 に近い
    assert metrology.cmp_uniformity_pct(wafer) == 0.0


def test_cmp_uniformity_positive_after_step(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    Photo(mask=mask, thickness_um=1.0, polarity="negative").apply(wafer)
    assert metrology.cmp_uniformity_pct(wafer) > 0.0


def test_etch_depth_uniformity_keys(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    stats = metrology.etch_depth_uniformity(wafer, "oxide")
    assert set(stats) == {"mean_um", "std_um", "cv_pct", "min_um", "max_um"}
    assert stats["mean_um"] > 0.0


def test_etch_depth_uniformity_absent_material(wafer):
    stats = metrology.etch_depth_uniformity(wafer, "metal_cu")
    assert stats["mean_um"] == 0.0


def test_summary_has_new_keys(wafer):
    s = metrology.summary(wafer)
    assert "cmp_uniformity_pct" in s
    assert "void_volume_um3" in s


# --- CLI JSON レポート -----------------------------------------------------
def test_cli_json_report(tmp_path):
    out = tmp_path / "report.json"
    code = cli_main(["--preset", "MOSFET フロー", "--json-report", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "solid_fraction" in data
    assert "cmp_uniformity_pct" in data
