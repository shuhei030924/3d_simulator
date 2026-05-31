"""STL エクスポータ（semisim.export）のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import export, materials
from semisim.grid import WaferConfig
from semisim.processes import CVD
from semisim.recipe import Recipe


def _flat_wafer():
    cfg = WaferConfig(nx=4, ny=4, nz=6, pitch_um=0.1, substrate_um=0.2)
    return Recipe(config=cfg).simulate()


def test_stl_header_and_footer():
    """ASCII STL の solid/endsolid で囲まれる。"""
    s = export.stl_string(_flat_wafer(), solid_name="test")
    assert s.startswith("solid test")
    assert s.rstrip().endswith("endsolid test")


def test_facet_count_matches_loops():
    """facet normal 行数と outer loop 行数が一致する。"""
    s = export.stl_string(_flat_wafer())
    assert s.count("facet normal") == s.count("outer loop")
    # 各 facet は頂点 3 つ
    assert s.count("vertex") == 3 * s.count("facet normal")


def test_solid_slab_facet_count():
    """nx×ny×(基板2層)の直方体の露出面数を解析的に検証。

    4×4×2 の塊。外側表面の単位正方形数 = 2*(4*4) + 4*(4*2) = 32 + 32 = 64。
    各正方形は三角形 2 枚なので facet 数 = 128。
    """
    cfg = WaferConfig(nx=4, ny=4, nz=4, pitch_um=0.1, substrate_um=0.2)
    w = Recipe(config=cfg).simulate()  # 底 2 層がシリコン
    n = export.to_stl(w, _tmp_path(), name_or_id="silicon")
    assert n == 128


def test_to_stl_writes_file_and_returns_count(tmp_path):
    """ファイルが書き出され、戻り値が facet 数と一致。"""
    w = _flat_wafer()
    path = str(tmp_path / "out.stl")
    n = export.to_stl(w, path)
    with open(path, encoding="ascii") as f:
        text = f.read()
    assert n == text.count("facet normal")
    assert n > 0


def test_material_filter(tmp_path):
    """材料指定で別材料の面は出力されない。"""
    cfg = WaferConfig(nx=4, ny=4, nz=8, pitch_um=0.1, substrate_um=0.2)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.2))
    w = r.simulate()
    si = export.to_stl(w, str(tmp_path / "si.stl"), name_or_id="silicon")
    ox = export.to_stl(w, str(tmp_path / "ox.stl"), name_or_id="oxide")
    full = export.to_stl(w, str(tmp_path / "all.stl"))
    assert si > 0 and ox > 0
    # 全体は各材料単独より facet が少ない（内部界面は出ない）
    assert full < si + ox


def test_vertices_are_finite():
    """頂点座標がすべて有限値。"""
    s = export.stl_string(_flat_wafer())
    vals = [float(tok) for line in s.splitlines() if "vertex" in line
            for tok in line.split()[1:]]
    assert np.all(np.isfinite(vals))


# tmp ファイル用ヘルパ（pytest tmp_path を使わないテスト向け）
_TMP_HOLDER: list = []


def _tmp_path():
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    f.close()
    _TMP_HOLDER.append(f.name)
    return f.name


def test_air_only_grid_minimal():
    """固体が無いグリッドでは facet が 0。"""
    cfg = WaferConfig(nx=3, ny=3, nz=3, pitch_um=0.1, substrate_um=0.2)
    w = Recipe(config=cfg).simulate()
    # シリコンを全部空気に置換
    w.grid[:] = materials.AIR
    s = export.stl_string(w)
    assert s.count("facet normal") == 0
