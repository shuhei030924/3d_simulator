"""プリセットレシピ (presets.py) と計測レポート (metrology.report) のテスト。"""
from __future__ import annotations

import pytest

from semisim import metrology, presets
from semisim.recipe import Recipe


def test_available_lists_all():
    names = presets.available()
    assert len(names) == len(presets.PRESETS)
    assert "MOSFET フロー" in names


@pytest.mark.parametrize("name", list(presets.PRESETS.keys()))
def test_each_preset_builds_and_simulates(name):
    r = presets.build(name)
    assert isinstance(r, Recipe)
    assert len(r.steps) > 0
    # 例外なくシミュレートでき、固体が残る
    wafer = r.simulate()
    assert (wafer.grid != 0).any()


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        presets.build("存在しないプリセット")


def test_preset_roundtrip_via_recipe(tmp_path):
    """プリセットを保存→読込しても工程数が保たれる。"""
    r = presets.build("Cu ダマシン配線")
    path = tmp_path / "preset.json"
    r.save(str(path))
    loaded = Recipe.load(str(path))
    assert len(loaded.steps) == len(r.steps)


def test_report_contains_key_metrics():
    r = presets.build("選択エピ成長")
    wafer = r.simulate()
    text = metrology.report(wafer)
    assert "計測レポート" in text
    assert "固体率" in text
    assert "表面段差" in text
    # エピ材料が体積行に現れる
    assert "epi_si" in text


def test_report_is_multiline_string():
    r = presets.build("MOSFET フロー")
    text = metrology.report(r.simulate())
    assert isinstance(text, str)
    assert text.count("\n") > 3
