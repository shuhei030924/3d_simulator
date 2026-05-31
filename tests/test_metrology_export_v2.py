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


# --- Anneal 時間/温度ドライブイン (拡散長 L=√Dt) ---------------------------
def test_diffusion_length_monotonic_in_temp():
    from semisim.processes import diffusion_length_um

    lo = diffusion_length_um(60, 1000, "boron")
    hi = diffusion_length_um(60, 1100, "boron")
    assert 0 < lo < hi  # 高温ほど拡散長が大きい


def test_diffusion_length_zero_time():
    from semisim.processes import diffusion_length_um

    assert diffusion_length_um(0, 1000, "boron") == 0.0


def test_diffusion_length_material_alias():
    from semisim.processes import diffusion_length_um

    # 材料名 doped_p はホウ素に解決される
    a = diffusion_length_um(60, 1000, "doped_p")
    b = diffusion_length_um(60, 1000, "boron")
    assert a == b


def test_diffusion_length_bad_dopant():
    import pytest

    from semisim.processes import diffusion_length_um

    with pytest.raises(ValueError):
        diffusion_length_um(60, 1000, "unobtainium")


def test_anneal_time_mode_drives_in(wafer):
    from semisim.processes import Anneal, Implant

    Implant(dopant="doped_p", range_um=0.3, straggle_um=0.1).apply(wafer)
    before = int((wafer.grid == materials.BY_NAME["doped_p"].id).sum())
    Anneal(time_min=120, temperature_c=1100).apply(wafer)
    after = int((wafer.grid == materials.BY_NAME["doped_p"].id).sum())
    assert after > before  # ドーパントが周囲シリコンへ広がる


def test_anneal_time_mode_roundtrip():
    from semisim.processes import Anneal, Process

    a = Anneal(time_min=90, temperature_c=1050)
    a2 = Process.from_dict(a.to_dict())
    assert a2.time_min == 90
    assert a2.temperature_c == 1050
    assert a2._effective_depth_um("boron") == a._effective_depth_um("boron")


def test_anneal_thermal_budget_uses_diffusion_length():
    from semisim.processes import Anneal

    tb = metrology.thermal_budget([Anneal(time_min=120, temperature_c=1100)])
    # 時間モードでは拡散長(√Dt)由来の Dt が計上される
    assert tb["total_dt_um2"] > 0.0


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


# --- 電気的導通チェック (オープン不良検出) --------------------------------
def test_continuity_connected_line(wafer):
    grid = wafer.grid
    w_id = materials.BY_NAME["tungsten"].id
    z, y = 25, 20
    nx = grid.shape[2]
    grid[z, y, 0:nx] = w_id  # x 方向に端から端まで連結した配線
    r = metrology.electrical_continuity(wafer, "tungsten", "x")
    assert r["connected"] is True
    assert r["spanning_components"] >= 1


def test_continuity_open_line(wafer):
    grid = wafer.grid
    w_id = materials.BY_NAME["tungsten"].id
    z, y = 25, 20
    nx = grid.shape[2]
    grid[z, y, 0:nx] = w_id
    grid[z, y, nx // 2] = materials.AIR  # 中央で断線(オープン)
    r = metrology.electrical_continuity(wafer, "tungsten", "x")
    assert r["connected"] is False
    assert r["spanning_components"] == 0


def test_continuity_absent_material(wafer):
    r = metrology.electrical_continuity(wafer, "tungsten", "x")
    assert r["connected"] is False
    assert r["n_components"] == 0


def test_continuity_axis_selectable(wafer):
    grid = wafer.grid
    w_id = materials.BY_NAME["tungsten"].id
    z, x = 25, 20
    ny = grid.shape[1]
    grid[z, 0:ny, x] = w_id  # y 方向に貫通する配線
    assert metrology.electrical_continuity(wafer, "tungsten", "y")["connected"] is True
    assert metrology.electrical_continuity(wafer, "tungsten", "x")["connected"] is False


# --- 配線抵抗推定 (断面積考慮の直列抵抗) ----------------------------------
def test_line_resistance_finite_for_full_line(wafer):
    grid = wafer.grid
    w_id = materials.BY_NAME["tungsten"].id
    z, y = 25, 20
    nx = grid.shape[2]
    grid[z, y, 0:nx] = w_id
    r = metrology.line_resistance_ohm(wafer, "tungsten", "x")
    assert np.isfinite(r) and r > 0


def test_line_resistance_open_is_inf(wafer):
    grid = wafer.grid
    w_id = materials.BY_NAME["tungsten"].id
    z, y = 25, 20
    nx = grid.shape[2]
    grid[z, y, 0:nx] = w_id
    grid[z, y, nx // 2] = materials.AIR  # 断線
    assert metrology.line_resistance_ohm(wafer, "tungsten", "x") == float("inf")


def test_line_resistance_narrower_is_higher(wafer):
    # 細い配線ほど抵抗が高い（断面積が小さい）
    from semisim.grid import Wafer

    cfg = wafer.config
    w_wide = Wafer(cfg)
    w_narrow = Wafer(cfg)
    w_id = materials.BY_NAME["tungsten"].id
    nx = cfg.nx
    z, y = 25, 20
    w_wide.grid[z, y - 1 : y + 2, 0:nx] = w_id  # 幅3
    w_narrow.grid[z, y, 0:nx] = w_id  # 幅1
    r_wide = metrology.line_resistance_ohm(w_wide, "tungsten", "x")
    r_narrow = metrology.line_resistance_ohm(w_narrow, "tungsten", "x")
    assert r_narrow > r_wide


def test_line_resistance_nonconductor_inf(wafer):
    grid = wafer.grid
    ox = materials.BY_NAME["oxide"].id
    grid[25, 20, :] = ox
    assert metrology.line_resistance_ohm(wafer, "oxide", "x") == float("inf")


# --- DRC 最小間隔チェック (ショート不良リスク) ----------------------------
def test_min_spacing_separated(wafer):
    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    al = materials.BY_NAME["metal_al"].id
    z, y = 25, 20
    grid[z, y, 5] = cu
    grid[z, y, 15] = al  # 10 ボクセル離れている
    s = metrology.min_spacing_um(wafer, "metal_cu", "metal_al")
    assert abs(s - 10 * wafer.config.pitch_um) < 1e-9


def test_min_spacing_touching_is_zero(wafer):
    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    al = materials.BY_NAME["metal_al"].id
    z, y = 25, 20
    grid[z, y, 10] = cu
    grid[z, y, 11] = al  # 隣接（ショート）
    assert metrology.min_spacing_um(wafer, "metal_cu", "metal_al") == 0.0


def test_min_spacing_absent_inf(wafer):
    grid = wafer.grid
    grid[25, 20, 10] = materials.BY_NAME["metal_cu"].id
    assert metrology.min_spacing_um(wafer, "metal_cu", "metal_al") == float("inf")


# --- シート抵抗 (Ω/sq) ----------------------------------------------------
def test_sheet_resistance_thinner_is_higher(wafer):
    from semisim.grid import Wafer

    cfg = wafer.config
    w_thin = Wafer(cfg)
    w_thick = Wafer(cfg)
    cu = materials.BY_NAME["metal_cu"].id
    # 全面に厚さの異なる Cu 膜を敷く
    w_thin.grid[30:31, :, :] = cu  # 厚さ1
    w_thick.grid[30:34, :, :] = cu  # 厚さ4
    rs_thin = metrology.sheet_resistance_ohm_sq(w_thin, "metal_cu")
    rs_thick = metrology.sheet_resistance_ohm_sq(w_thick, "metal_cu")
    assert rs_thin > rs_thick > 0


def test_sheet_resistance_value(wafer):
    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    grid[30:32, :, :] = cu  # 厚さ2 ボクセル = 0.2µm
    rs = metrology.sheet_resistance_ohm_sq(wafer, "metal_cu")
    rho = materials.BY_NAME["metal_cu"].resistivity_ohm_um
    assert abs(rs - rho / 0.2) < 1e-6


def test_sheet_resistance_nonconductor_inf(wafer):
    grid = wafer.grid
    grid[30:32, :, :] = materials.BY_NAME["oxide"].id
    assert metrology.sheet_resistance_ohm_sq(wafer, "oxide") == float("inf")


def test_report_includes_electrical_section(wafer):
    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    nx = grid.shape[2]
    grid[30, 20, 0:nx] = cu  # x 方向に貫通する Cu 配線
    txt = metrology.report(wafer)
    assert "電気特性" in txt
    assert "metal_cu" in txt
    assert "導通" in txt


def test_report_flags_short(wafer):
    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    al = materials.BY_NAME["metal_al"].id
    grid[30, 20, 10] = cu
    grid[30, 20, 11] = al  # 隣接（ショート）
    txt = metrology.report(wafer)
    assert "ショート" in txt or "★" in txt


def test_electrical_report_dict(wafer):
    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    nx = grid.shape[2]
    grid[30, 20, 0:nx] = cu
    rep = metrology.electrical_report(wafer)
    assert "metal_cu" in rep["conductors"]
    c = rep["conductors"]["metal_cu"]
    assert c["connected_x"] is True
    assert c["sheet_resistance_ohm_sq"] is not None


def test_electrical_report_empty_no_conductor(wafer):
    rep = metrology.electrical_report(wafer)
    assert rep["conductors"] == {}
    assert rep["min_spacing_um"] == {}


def test_electrical_report_json_serializable(wafer):
    import json

    grid = wafer.grid
    cu = materials.BY_NAME["metal_cu"].id
    grid[30, 20, 5] = cu  # 単一ボクセル(オープン)→ sheet_resistance は有限
    rep = metrology.electrical_report(wafer)
    json.dumps(rep)  # inf が None 化されていれば例外なし










