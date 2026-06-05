"""レシピ（保存/読込・スナップショットキャッシュ・全工程往復）のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import processes
from semisim.masks import Mask, Shape
from semisim.processes import Process
from semisim.recipe import Recipe


def _sample_recipe(cfg) -> Recipe:
    r = Recipe(config=cfg)
    r.add(processes.CVD(material="oxide", thickness_um=0.3))
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.35, "x1": 0.65, "y1": 0.65})])
    r.add(processes.Photo(mask=mask, thickness_um=0.8, polarity="positive"))
    r.add(processes.DryEtch(targets=["oxide"], depth_um=0.3))
    r.add(processes.Strip(material="photoresist"))
    r.add(processes.Diffusion(dopant="doped_n", depth_um=0.4))
    return r


def test_recipe_save_load_roundtrip(tmp_path, small_cfg):
    r = _sample_recipe(small_cfg)
    w = r.simulate()
    path = tmp_path / "recipe.json"
    r.save(str(path))
    r2 = Recipe.load(str(path))
    w2 = r2.simulate()
    assert np.array_equal(w.grid, w2.grid)


def test_recipe_load_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        Recipe.load(str(tmp_path / "no_such.json"))


def test_recipe_load_corrupt_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        Recipe.load(str(p))


def test_all_processes_roundtrip():
    """全工程タイプが to_dict/from_dict で往復できる。"""
    for t, _ in processes.available_types():
        cls = processes._REGISTRY[t]
        try:
            inst = cls()
        except TypeError:
            # 必須引数がある場合はスキップ（基本的に既定値あり）
            continue
        d = inst.to_dict()
        assert d["type"] == t
        back = Process.from_dict(d)
        assert back.type == t


def test_snapshot_cache_limit(small_cfg):
    """スナップショット上限を超えても正しくシミュレートできる。"""
    r = Recipe(config=small_cfg, max_snapshots=4)
    for _ in range(10):
        r.add(processes.CVD(material="oxide", thickness_um=0.1))
    w_full = r.simulate()
    # 途中段階も正しく取得できる
    w_mid = r.simulate(up_to=5)
    assert w_mid.grid.shape == w_full.grid.shape
    # 上限を超えて保持されていない
    live = [s for s in r._snapshots if s is not None]
    assert len(live) <= r.max_snapshots


def test_invalidate_on_edit(small_cfg):
    r = _sample_recipe(small_cfg)
    w1 = r.simulate()
    r.replace(0, processes.CVD(material="oxide", thickness_um=0.6))
    w2 = r.simulate()
    assert not np.array_equal(w1.grid, w2.grid)


def test_from_dict_version_check(small_cfg):
    r = _sample_recipe(small_cfg)
    d = r.to_dict()
    d["format_version"] = 999
    with pytest.raises(ValueError):
        Recipe.from_dict(d)


_CFG = {"nx": 20, "ny": 20, "nz": 30, "pitch_um": 0.1, "substrate_um": 2.0}


@pytest.mark.parametrize("bad", [
    {"config": [1, 2], "steps": []},            # config が非 dict
    {"config": "x", "steps": []},               # config が文字列
    {"config": _CFG, "steps": "notalist"},      # steps が非 list
    {"config": _CFG, "steps": {"type": "CVD"}},  # steps が dict
    {"config": _CFG, "steps": [123]},           # step 要素が非 dict
    {"config": _CFG, "steps": ["CVD"]},         # step 要素が文字列
])
def test_from_dict_malformed_raises_valueerror(bad):
    """壊れた構造は AttributeError でなく ValueError を投げる（CLI/GUI が捕捉可能）。"""
    with pytest.raises(ValueError):
        Recipe.from_dict(bad)


def test_from_dict_valid_still_loads():
    """正常なレシピは引き続き読み込める。"""
    r = Recipe.from_dict({
        "format_version": 1, "config": _CFG,
        "steps": [{"type": "CVD", "material": "oxide", "thickness_um": 0.3}],
    })
    assert len(r.steps) == 1
