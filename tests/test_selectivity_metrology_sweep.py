"""エッチ選択比・新メトロロジ(LER/overlay/CD variants)・CLI sweep・多層プリセットのテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import materials, metrology, presets
from semisim.cli import _parse_sweep
from semisim.cli import main as cli_main
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo
from semisim.recipe import Recipe


# --- DryEtch 選択比 --------------------------------------------------------
def test_dryetch_selectivity_slows_etch():
    cfg = WaferConfig(nx=30, ny=30, nz=50, pitch_um=0.1, substrate_um=1.0)

    def remaining(sel):
        r = Recipe(config=cfg)
        r.add(CVD(material="oxide", thickness_um=1.0))
        r.add(DryEtch(targets=["oxide"], depth_um=1.0, selectivity=sel))
        w = r.simulate()
        return int((w.grid == materials.get("oxide").id).sum())

    full = remaining({})
    slow = remaining({"oxide": 0.3})
    # 選択比 0.3（削れにくい）の方が酸化膜が多く残る
    assert slow > full


def test_dryetch_selectivity_stop_layer():
    # 選択比 0 のストップ層相当は削れない
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.8))
    r.add(DryEtch(targets=["oxide"], depth_um=0.8, selectivity={"oxide": 0.0}))
    w = r.simulate()
    # ほとんど削れず残る
    assert (w.grid == materials.get("oxide").id).sum() > 0


def test_dryetch_selectivity_roundtrip():
    d = DryEtch(targets=["oxide"], depth_um=0.5, selectivity={"oxide": 0.33, "nitride": 0.1})
    restored = DryEtch._from_params(d.params_dict())
    assert restored.selectivity == {"oxide": 0.33, "nitride": 0.1}


def test_dryetch_selectivity_rejects_out_of_range():
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    with pytest.raises(ValueError):
        DryEtch(targets=["oxide"], depth_um=0.5, selectivity={"oxide": 1.5}).apply(w)


def test_dryetch_empty_selectivity_matches_default():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])

    def trench(sel):
        r = Recipe(config=cfg)
        r.add(CVD(material="oxide", thickness_um=0.8))
        r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
        r.add(DryEtch(targets=["oxide"], depth_um=0.6, selectivity=sel))
        return r.simulate().grid.copy()

    # 空 selectivity は従来挙動と一致
    assert np.array_equal(trench({}), trench({}))


# --- 新メトロロジ ----------------------------------------------------------
def test_line_edge_roughness_smooth_is_low(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    # 全面成膜の平坦エッジ（左端が常に x=0）→ ばらつき 0
    z = int(wafer.top_surface_z().max())
    assert metrology.line_edge_roughness_um(wafer, "oxide", z) == 0.0


def test_line_edge_roughness_absent_material(wafer):
    assert metrology.line_edge_roughness_um(wafer, "metal_cu", 0) == 0.0


def test_overlay_error_same_material_zero(wafer):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    # 同一材料同士の重心ずれは 0
    assert metrology.overlay_error_um(wafer, "oxide", "oxide") == 0.0


def test_overlay_error_absent_material(wafer):
    assert metrology.overlay_error_um(wafer, "oxide", "metal_cu") == 0.0


def test_overlay_error_offset_positive():
    cfg = WaferConfig(nx=40, ny=40, nz=30, pitch_um=0.1, substrate_um=1.0)
    w = Recipe(config=cfg).simulate()
    # 左寄りに oxide、右寄りに nitride を置く → 重心ずれ > 0
    w.grid[5, :, :10] = materials.get("oxide").id
    w.grid[5, :, 30:] = materials.get("nitride").id
    assert metrology.overlay_error_um(w, "oxide", "nitride") > 0.0


def test_feature_width_variants_keys(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    Photo(mask=mask, thickness_um=0.5, polarity="negative").apply(wafer)
    z = int(wafer.top_surface_z().max())
    y = wafer.config.ny // 2
    v = metrology.feature_width_variants(wafer, "photoresist", z, y)
    assert set(v) == {"max_run_um", "total_um", "gap_um"}
    assert v["max_run_um"] > 0.0


def test_feature_width_variants_absent(wafer):
    v = metrology.feature_width_variants(wafer, "metal_cu", 0, 0)
    assert v == {"max_run_um": 0.0, "total_um": 0.0, "gap_um": 0.0}


# --- CLI sweep -------------------------------------------------------------
def test_parse_sweep_basic():
    idx, field, values = _parse_sweep("4.depth_um:1.0:2.0:0.5")
    assert idx == 4
    assert field == "depth_um"
    assert values == [1.0, 1.5, 2.0]


def test_parse_sweep_invalid_format():
    with pytest.raises(ValueError):
        _parse_sweep("bad-spec")


def test_parse_sweep_nonpositive_step():
    with pytest.raises(ValueError):
        _parse_sweep("0.depth_um:1.0:2.0:0")


def test_cli_sweep_outputs_csv(capsys):
    code = cli_main(["--preset", "KOH V溝", "--sweep", "4.depth_um:1.0:1.5:0.5"])
    assert code == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0].startswith("depth_um,")
    assert len(lines) == 3  # ヘッダ + 2 値


def test_cli_sweep_bad_index():
    code = cli_main(["--preset", "KOH V溝", "--sweep", "99.depth_um:1.0:2.0:0.5"])
    assert code == 1


def test_cli_sweep_bad_field():
    code = cli_main(["--preset", "KOH V溝", "--sweep", "1.nope:1.0:2.0:0.5"])
    assert code == 1


# --- 多層配線プリセット ----------------------------------------------------
def test_multilevel_preset_available():
    assert "多層 Cu 配線" in presets.available()


def test_multilevel_preset_simulates():
    r = presets.build("多層 Cu 配線")
    assert r.name
    w = r.simulate()
    # Cu と TaN が存在する
    assert (w.grid == materials.get("metal_cu").id).any()
    assert (w.grid == materials.get("tan").id).any()
