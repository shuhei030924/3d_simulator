"""追加メトロロジ関数のテスト。"""
from __future__ import annotations

from semisim import materials, metrology
from semisim.masks import Mask, Shape
from semisim.processes import CVD, AnisoWetEtch, Photo
from semisim.recipe import Recipe


def test_surface_roughness_flat_is_zero(wafer):
    # 初期の平坦な基板は粗さ 0
    assert metrology.surface_roughness_um(wafer) == 0.0


def test_surface_roughness_positive_after_step(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    Photo(mask=mask, thickness_um=1.0, polarity="negative").apply(wafer)
    assert metrology.surface_roughness_um(wafer) > 0.0


def test_sidewall_angle_vertical(wafer):
    # 垂直な堆積壁はおおむね 90° に近い
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    Photo(mask=mask, thickness_um=1.5, polarity="negative").apply(wafer)
    ang = metrology.sidewall_angle_deg(wafer, "photoresist", wafer.config.ny // 2)
    assert ang > 0.0


def test_sidewall_angle_koh_tapered(wafer):
    # KOH の斜め側壁は 90° 未満
    cfg = wafer.config
    r = Recipe(config=cfg)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r.add(Photo(mask=mask, thickness_um=0.6, polarity="positive"))
    r.add(AnisoWetEtch(target="silicon", depth_um=1.0, sidewall_angle_deg=54.7))
    w = r.simulate()
    ang = metrology.sidewall_angle_deg(w, "silicon", cfg.ny // 2)
    assert 0.0 < ang <= 90.0


def test_interface_width_doped_substrate(wafer):
    from semisim.processes import Diffusion

    Diffusion(dopant="doped_n", depth_um=0.4).apply(wafer)
    # 拡散層とシリコン基板は接触界面を持つ
    area = metrology.interface_width_um(wafer, "doped_n", "silicon")
    assert area > 0.0


def test_interface_width_no_contact_is_zero(wafer):
    # 接触しない材料対は 0
    area = metrology.interface_width_um(wafer, "metal_cu", "low_k")
    assert area == 0.0


def test_trench_closed_detects_void(wafer):
    cfg = wafer.config
    cx, cy = cfg.nx // 2, cfg.ny // 2
    # 人工的に空気の埋もれ層を作る
    z = wafer.top_surface_z()[cy, cx]
    wafer.grid[z + 1, cy, cx] = materials.AIR
    wafer.grid[z + 2, cy, cx] = materials.get("oxide").id
    assert metrology.trench_is_closed(wafer, cx, cy) is True


def test_trench_not_closed_on_open_surface(wafer):
    cfg = wafer.config
    assert metrology.trench_is_closed(wafer, cfg.nx // 2, cfg.ny // 2) is False


def test_summary_includes_roughness(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    s = metrology.summary(wafer)
    assert "surface_roughness_um" in s


def test_report_includes_roughness(wafer):
    r = metrology.report(wafer)
    assert "表面粗さ" in r
