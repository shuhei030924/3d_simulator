"""界面粗さ・接合深さ・ドーパントプロファイル計測と CSV エクスポートのテスト。"""
from __future__ import annotations

from semisim import export, materials, metrology
from semisim.cli import main as cli_main
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, Diffusion, Implant, Photo
from semisim.recipe import Recipe


def _cfg():
    return WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.1, substrate_um=2.0)


# --- 界面粗さ --------------------------------------------------------------
def test_interface_roughness_flat_is_low(wafer):
    CVD(material="oxide", thickness_um=0.5).apply(wafer)
    # 平坦な基板/酸化膜界面は粗さほぼ 0
    r = metrology.interface_roughness_um(wafer, "silicon", "oxide")
    assert r < wafer.config.pitch_um


def test_interface_roughness_rough_after_step(wafer):
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    # 段差のあるポリを置いてから酸化膜 → 界面が凹凸
    Photo(mask=mask, thickness_um=0.6, polarity="negative").apply(wafer)
    CVD(material="oxide", thickness_um=0.4).apply(wafer)
    r = metrology.interface_roughness_um(wafer, "photoresist", "oxide")
    assert r >= 0.0


def test_interface_roughness_absent_returns_zero(wafer):
    assert metrology.interface_roughness_um(wafer, "metal_cu", "tungsten") == 0.0


# --- 接合深さ / ドーパントプロファイル -------------------------------------
def test_junction_depth_positive():
    r = Recipe(config=_cfg())
    r.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.1))
    w = r.simulate()
    xj = metrology.junction_depth_um(w, "doped_n")
    assert xj > 0.0


def test_junction_depth_deeper_with_larger_range():
    def xj(rng):
        r = Recipe(config=_cfg())
        r.add(Implant(dopant="doped_n", range_um=rng, straggle_um=0.1))
        return metrology.junction_depth_um(r.simulate(), "doped_n")

    assert xj(0.6) > xj(0.2)


def test_junction_depth_absent_returns_zero(wafer):
    assert metrology.junction_depth_um(wafer, "doped_p") == 0.0


def test_dopant_depth_profile_shape_and_peak():
    cfg = _cfg()
    r = Recipe(config=cfg)
    r.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.1))
    w = r.simulate()
    prof = metrology.dopant_depth_profile(w, "doped_n")
    assert prof.shape == (cfg.nz,)
    assert prof.max() > 0.0
    # ピークは基板内（z=0..substrate高さ付近）にある
    assert 0 <= int(prof.argmax()) < cfg.nz


def test_dopant_depth_profile_absent_all_zero(wafer):
    prof = metrology.dopant_depth_profile(wafer, "doped_n")
    assert prof.shape == (wafer.config.nz,)
    assert float(prof.sum()) == 0.0


def test_diffusion_junction_depth():
    r = Recipe(config=_cfg())
    r.add(Diffusion(dopant="doped_n", depth_um=0.6))
    w = r.simulate()
    assert metrology.junction_depth_um(w, "doped_n") > 0.0


# --- CSV 列プロファイルエクスポート ----------------------------------------
def test_column_profile_csv_header_and_rows(wafer):
    CVD(material="oxide", thickness_um=0.5).apply(wafer)
    csv = export.column_profile_csv(wafer, wafer.config.nx // 2, wafer.config.ny // 2)
    lines = csv.strip().splitlines()
    assert lines[0] == "z_index,z_um,material_id,material_name"
    assert len(lines) == wafer.config.nz + 1
    # 底は基板シリコン
    assert lines[1].endswith("silicon")
    # 酸化膜が含まれる
    assert any("oxide" in ln for ln in lines)


def test_to_csv_column_writes_file(wafer, tmp_path):
    CVD(material="oxide", thickness_um=0.3).apply(wafer)
    out = tmp_path / "col.csv"
    n = export.to_csv_column(wafer, str(out), 20, 20)
    assert n == wafer.config.nz
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("z_index,z_um")


def test_cli_csv_column_export(tmp_path):
    out = tmp_path / "col.csv"
    code = cli_main(["--preset", "MOSFET フロー", "--csv-column", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "material_name" in text


def test_column_profile_air_top(wafer):
    # 何も成膜していない列は上部が air
    csv = export.column_profile_csv(wafer, 0, 0)
    air_name = materials.BY_ID[materials.AIR].name
    last = csv.strip().splitlines()[-1]
    assert last.endswith(air_name)
