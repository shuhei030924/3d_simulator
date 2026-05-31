"""界面粗さ・接合深さ・ドーパントプロファイル計測と CSV エクスポートのテスト。"""
from __future__ import annotations

import numpy as np

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


# --- CLI 2D スイープ -------------------------------------------------------
def _sweep_recipe():
    r = Recipe(config=_cfg())
    r.add(CVD(material="oxide", thickness_um=0.5))  # index 0
    r.add(CVD(material="poly", thickness_um=0.3))  # index 1
    return r


def test_cli_sweep2_outputs_grid(capsys, tmp_path):
    path = tmp_path / "r.json"
    _sweep_recipe().save(str(path))
    code = cli_main([
        str(path),
        "--sweep2",
        "0.thickness_um:0.3:0.5:0.2,1.thickness_um:0.2:0.4:0.2",
    ])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    # ヘッダ + 2x2 = 4 行
    assert lines[0].startswith("thickness_um,thickness_um,")
    assert len(lines) == 1 + 4


def test_cli_sweep2_bad_count(tmp_path):
    path = tmp_path / "r.json"
    _sweep_recipe().save(str(path))
    code = cli_main([str(path), "--sweep2", "0.thickness_um:0.3:0.5:0.2"])
    assert code == 1


def test_cli_sweep2_bad_field(tmp_path):
    path = tmp_path / "r.json"
    _sweep_recipe().save(str(path))
    code = cli_main([
        str(path),
        "--sweep2",
        "0.nope:0.3:0.5:0.2,1.thickness_um:0.2:0.4:0.2",
    ])
    assert code == 1


# --- 表面粗さスペクトル（支配波長） ---------------------------------------
def test_dominant_wavelength_flat_is_zero(wafer):
    CVD(material="oxide", thickness_um=0.5).apply(wafer)
    # 平坦面は支配波長 0
    assert metrology.dominant_wavelength_um(wafer) == 0.0


def test_dominant_wavelength_matches_period(wafer):

    # x 方向に周期 8 ピクセルの矩形凹凸表面を人工的に作る
    grid = wafer.grid
    nz, ny, nx = grid.shape
    grid[:] = materials.AIR
    base = 10
    period = 8
    for x in range(nx):
        top = base + (4 if (x % period) < period // 2 else 0)
        grid[:top, :, x] = materials.BY_NAME["silicon"].id
    wl = metrology.dominant_wavelength_um(wafer)
    # 支配波長は周期 8px × pitch に近いはず
    assert abs(wl - period * wafer.config.pitch_um) < wafer.config.pitch_um


# --- ボイド連結成分統計 ---------------------------------------------------
def test_void_metrics_none(wafer):
    CVD(material="oxide", thickness_um=0.5).apply(wafer)
    m = metrology.void_metrics(wafer)
    assert m["count"] == 0
    assert m["largest_um3"] == 0.0


def test_void_metrics_detects_buried_void(wafer):
    import numpy as np

    grid = wafer.grid
    nz, ny, nx = grid.shape
    grid[:] = materials.AIR
    sid = materials.BY_NAME["silicon"].id
    # 厚い固体ブロックの内部に閉塞空気のボイドを 1 つ作る
    grid[:20, :, :] = sid
    grid[8:12, 18:22, 18:22] = materials.AIR
    m = metrology.void_metrics(wafer)
    assert m["count"] == 1
    assert m["largest_um3"] > 0.0
    assert np.isclose(m["max_height_um"], 4 * wafer.config.pitch_um)


# --- Deal-Grove 熱酸化モデル ----------------------------------------------
def test_deal_grove_monotonic_in_time():
    from semisim.processes import deal_grove_thickness_um

    x1 = deal_grove_thickness_um(30, 1000, "dry")
    x2 = deal_grove_thickness_um(120, 1000, "dry")
    assert 0 < x1 < x2  # 時間が長いほど厚い


def test_deal_grove_wet_thicker_than_dry():
    from semisim.processes import deal_grove_thickness_um

    dry = deal_grove_thickness_um(60, 1000, "dry")
    wet = deal_grove_thickness_um(60, 1000, "wet")
    assert wet > dry  # 水蒸気酸化の方が速い


def test_deal_grove_zero_time_is_zero():
    from semisim.processes import deal_grove_thickness_um

    assert deal_grove_thickness_um(0, 1000, "dry") == 0.0


def test_oxidation_time_mode_grows_oxide(wafer):
    from semisim.processes import Oxidation, deal_grove_thickness_um

    ox = Oxidation(time_min=120, temperature_c=1100, ambient="wet")
    ox.apply(wafer)
    # Deal-Grove 由来の厚みで酸化膜が生成される
    assert (wafer.grid == materials.BY_NAME["oxide"].id).any()
    assert ox._effective_thickness_um() == deal_grove_thickness_um(120, 1100, "wet")


def test_oxidation_time_mode_roundtrip():
    from semisim.processes import Oxidation, Process

    ox = Oxidation(time_min=90, temperature_c=950, ambient="wet")
    d = ox.to_dict()
    ox2 = Process.from_dict(d)
    assert ox2.time_min == 90
    assert ox2.ambient == "wet"
    assert ox2._effective_thickness_um() == ox._effective_thickness_um()


# --- Line Width Roughness (LWR) -------------------------------------------
def test_lwr_straight_line_is_zero(wafer):
    # まっすぐな一定幅ラインは LWR ≈ 0
    grid = wafer.grid
    poly = materials.BY_NAME["poly"].id
    z = 25
    grid[z, :, 18:23] = poly  # 全 y で幅 5 の直線
    lwr = metrology.line_width_roughness_um(wafer, "poly", z)
    assert lwr == 0.0


def test_lwr_wiggly_line_positive(wafer):
    grid = wafer.grid
    poly = materials.BY_NAME["poly"].id
    z = 25
    ny = grid.shape[1]
    # y ごとに幅を変えてギザギザにする
    for y in range(ny):
        w = 4 + (y % 3)
        grid[z, y, 18 : 18 + w] = poly
    lwr = metrology.line_width_roughness_um(wafer, "poly", z)
    assert lwr > 0.0


def test_lwr_absent_zero(wafer):
    assert metrology.line_width_roughness_um(wafer, "tungsten", 25) == 0.0


# --- ARDE / RIE ラグ -------------------------------------------------------
def _arde_wafer():
    from semisim.grid import Wafer

    cfg = WaferConfig(nx=60, ny=20, nz=60, pitch_um=0.1, substrate_um=4.0)
    w = Wafer(cfg)
    grid = w.grid
    sid = materials.BY_NAME["silicon"].id
    rid = next(m.id for m in materials.all_materials() if m.is_resist)
    # シリコン上面にレジストを被せ、広い開口(幅16)と狭い開口(幅3)を空ける
    top = (grid[:, 0, 0] == sid).nonzero()[0].max()
    grid[top + 1 : top + 4, :, :] = rid  # レジスト層
    grid[top + 1 : top + 4, :, 5:21] = materials.AIR  # 広い開口
    grid[top + 1 : top + 4, :, 35:38] = materials.AIR  # 狭い開口
    return w, top


def _depth_in(wafer, x0, x1, top):
    # 開口内のシリコン上面の平均エッチ深さ(ボクセル) = top - 現在の最上シリコン
    from semisim import materials as M

    grid = wafer.grid
    sid = M.BY_NAME["silicon"].id
    col = grid[:, :, x0:x1]
    sil = col == sid
    z_top = np.where(sil.any(axis=0), sil.shape[0] - 1 - np.argmax(sil[::-1], axis=0), -1)
    return top - float(np.mean(z_top))


def test_arde_narrow_etches_shallower():
    w, top = _arde_wafer()
    from semisim.processes import DryEtch

    DryEtch(targets=["silicon"], depth_um=2.0, arde_lag_um=0.6).apply(w)
    wide = _depth_in(w, 5, 21, top)
    narrow = _depth_in(w, 35, 38, top)
    assert wide > narrow > 0  # 広い開口の方が深く、狭い開口も多少は削れる


def test_arde_off_equal_depth():
    w, top = _arde_wafer()
    from semisim.processes import DryEtch

    DryEtch(targets=["silicon"], depth_um=1.0, arde_lag_um=0.0).apply(w)
    wide = _depth_in(w, 5, 21, top)
    narrow = _depth_in(w, 35, 38, top)
    assert abs(wide - narrow) < 1.0  # ラグ無しなら深さは等しい




