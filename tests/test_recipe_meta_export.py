"""レシピメタデータと CLI/エクスポートの検証テスト。"""
from __future__ import annotations

import pytest

from semisim.cli import main as cli_main
from semisim.grid import WaferConfig
from semisim.processes import CVD
from semisim.recipe import Recipe


def test_recipe_metadata_roundtrip(tmp_path):
    r = Recipe(config=WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.1, substrate_um=1.0))
    r.name = "テストフロー"
    r.description = "メタデータ確認用"
    r.author = "tester"
    r.add(CVD(material="oxide", thickness_um=0.2))
    path = tmp_path / "recipe.json"
    r.save(str(path))
    loaded = Recipe.load(str(path))
    assert loaded.name == "テストフロー"
    assert loaded.description == "メタデータ確認用"
    assert loaded.author == "tester"
    assert len(loaded.steps) == 1


def test_recipe_metadata_defaults_empty():
    r = Recipe()
    d = r.to_dict()
    assert d["name"] == ""
    assert d["description"] == ""
    assert d["author"] == ""


def test_cli_png_export(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "slice.png"
    code = cli_main(["--preset", "MOSFET フロー", "--png", str(out)])
    assert code == 0
    assert out.exists() and out.stat().st_size > 0


def test_cli_stl_export(tmp_path):
    pytest.importorskip("pyvista")
    out = tmp_path / "shape.stl"
    code = cli_main(["--preset", "KOH V溝", "--stl", str(out)])
    assert code == 0
    assert out.exists() and out.stat().st_size > 0
