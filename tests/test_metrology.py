"""メトロロジ（計測）関数のテスト。"""
from __future__ import annotations

from semisim import materials, metrology
from semisim.processes import CVD


def test_material_counts(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    counts = metrology.material_counts(wafer)
    assert "silicon" in counts
    assert "oxide" in counts
    assert "air" not in counts


def test_material_volume(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    vol = metrology.material_volume_um3(wafer, "oxide")
    expected = int((wafer.grid == materials.get("oxide").id).sum()) * (
        wafer.config.pitch_um ** 3
    )
    assert vol == expected


def test_film_thickness_map_and_stats(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    tmap = metrology.film_thickness_map(wafer, "oxide")
    assert tmap.shape == (wafer.config.ny, wafer.config.nx)
    stats = metrology.film_thickness_stats(wafer, "oxide")
    # コンフォーマル成膜なので平均厚は約 0.3µm
    assert abs(stats["mean"] - 0.3) < 0.05
    assert stats["coverage"] > 0.9


def test_film_thickness_stats_absent_material(wafer):
    stats = metrology.film_thickness_stats(wafer, "metal_cu")
    assert stats["mean"] == 0.0
    assert stats["coverage"] == 0.0


def test_surface_height_and_step(wafer):
    height = metrology.surface_height_map(wafer)
    assert height.shape == (wafer.config.ny, wafer.config.nx)
    # 平坦な基板なので段差はほぼ 0
    assert metrology.step_height_um(wafer) == 0.0


def test_solid_fraction(wafer):
    frac = metrology.solid_fraction(wafer)
    assert 0.0 < frac < 1.0


def test_feature_width(wafer):
    CVD(material="oxide", thickness_um=0.2).apply(wafer)
    z_top = int(wafer.top_surface_z().max())
    w = metrology.feature_width_um(wafer, "oxide", z_top, wafer.config.ny // 2)
    # 全面成膜なので幅はほぼウェハ幅
    assert w > 0
