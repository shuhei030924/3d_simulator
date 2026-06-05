"""CLI（ヘッドレス実行）のテスト。"""
from __future__ import annotations

import json

import pytest

from semisim import presets
from semisim.cli import main


def test_list_presets(capsys):
    rc = main(["--list-presets"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert sorted(out) == sorted(presets.available())


def test_run_preset_prints_report(capsys):
    rc = main(["--preset", "MOSFET フロー"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "計測レポート" in out
    assert "固体率" in out


def test_run_recipe_file(tmp_path, capsys):
    recipe = presets.build("Cu ダマシン配線")
    path = tmp_path / "r.json"
    recipe.save(str(path))
    rc = main([str(path)])
    assert rc == 0
    assert "計測レポート" in capsys.readouterr().out


def test_report_saved_to_file(tmp_path, capsys):
    out = tmp_path / "report.txt"
    rc = main(["--preset", "KOH V溝", "--report", str(out)])
    assert rc == 0
    assert out.exists()
    assert "計測レポート" in out.read_text(encoding="utf-8")


def test_missing_recipe_file_returns_error(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.json")])
    assert rc == 1
    assert "エラー" in capsys.readouterr().err


def test_unknown_preset_returns_error(capsys):
    rc = main(["--preset", "存在しないプリセット"])
    assert rc == 1
    assert "エラー" in capsys.readouterr().err


def test_corrupt_recipe_returns_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    rc = main([str(bad)])
    assert rc == 1
    assert "エラー" in capsys.readouterr().err


def test_no_args_errors():
    with pytest.raises(SystemExit):
        main([])


def _write_recipe(path, steps):
    cfg = {"nx": 20, "ny": 20, "nz": 30, "pitch_um": 0.1, "substrate_um": 2.0}
    path.write_text(json.dumps({"config": cfg, "steps": steps}), encoding="utf-8")


def test_invalid_param_returns_clean_error(tmp_path, capsys):
    """シミュレーション中の検証エラー（不正な膜厚）はトレースバックでなく
    「エラー: ...」で返す（rc=1）。"""
    p = tmp_path / "neg.json"
    _write_recipe(p, [{"type": "CVD", "material": "oxide", "thickness_um": -1.0}])
    rc = main([str(p)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "エラー" in err
    assert "Traceback" not in err


def test_unknown_material_returns_clean_error(tmp_path, capsys):
    """未知の材料名もトレースバックでなくクリーンなエラーで返す。"""
    p = tmp_path / "mat.json"
    _write_recipe(p, [{"type": "CVD", "material": "unobtainium", "thickness_um": 0.3}])
    rc = main([str(p)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "エラー" in err
    assert "Traceback" not in err


def test_png_export(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "slice.png"
    rc = main(["--preset", "イオン注入 埋込層", "--png", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


def test_recipe_roundtrip_via_json(tmp_path):
    recipe = presets.build("MOSFET フロー")
    d = json.loads(json.dumps(recipe.to_dict()))
    assert d["config"]["nx"] > 0
    assert len(d["steps"]) > 0
