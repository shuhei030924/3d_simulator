"""アプリ設定の永続化 (settings.py) のテスト。"""
from __future__ import annotations

from semisim.settings import AppSettings


def test_default_settings_roundtrip(tmp_path):
    s = AppSettings()
    path = tmp_path / "settings.json"
    s.save(path)
    loaded = AppSettings.load(path)
    assert loaded.last_dir == ""
    assert loaded.recent_recipes == []
    assert loaded.show_resist is True


def test_add_recent_dedup_and_order(tmp_path):
    s = AppSettings(max_recent=3)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for p in (a, b, a):  # a を再追加 → 先頭に来て重複しない
        p.write_text("{}", encoding="utf-8")
        s.add_recent(str(p))
    assert len(s.recent_recipes) == 2
    # 最後に追加した a が先頭
    assert s.recent_recipes[0].endswith("a.json")
    # last_dir は親フォルダ
    assert s.last_dir == str(tmp_path)


def test_recent_respects_max(tmp_path):
    s = AppSettings(max_recent=2)
    for name in ("a", "b", "c"):
        s.add_recent(str(tmp_path / f"{name}.json"))
    assert len(s.recent_recipes) == 2
    # 新しい順に c, b が残る
    assert s.recent_recipes[0].endswith("c.json")
    assert s.recent_recipes[1].endswith("b.json")


def test_prune_missing(tmp_path):
    s = AppSettings()
    exist = tmp_path / "exist.json"
    exist.write_text("{}", encoding="utf-8")
    s.add_recent(str(exist))
    s.add_recent(str(tmp_path / "ghost.json"))  # 実在しない
    s.prune_missing()
    assert len(s.recent_recipes) == 1
    assert s.recent_recipes[0].endswith("exist.json")


def test_load_missing_file_returns_defaults(tmp_path):
    loaded = AppSettings.load(tmp_path / "nope.json")
    assert isinstance(loaded, AppSettings)
    assert loaded.recent_recipes == []


def test_load_corrupt_file_returns_defaults(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    loaded = AppSettings.load(bad)
    assert isinstance(loaded, AppSettings)


def test_load_bad_typed_fields_falls_back_per_field(tmp_path):
    """型が壊れたフィールドは既定値へ、妥当なフィールドは保持する。"""
    import json
    p = tmp_path / "bt.json"
    p.write_text(json.dumps({
        "last_dir": "/keep", "ui_theme": "dark",
        "max_recent": "abc", "recent_recipes": "notalist",
        "default_config": [1, 2],
    }), encoding="utf-8")
    s = AppSettings.load(p)
    assert s.last_dir == "/keep"          # 妥当な値は保持
    assert s.ui_theme == "dark"           # 妥当な値は保持
    assert s.max_recent == 10             # 不正 int -> 既定
    assert s.recent_recipes == []         # 非 list -> []
    assert s.default_config == {}         # 非 dict -> {}


def test_load_nondict_json_returns_defaults(tmp_path):
    """JSON が妥当でも dict でない（list/数値）場合は既定値。"""
    for content in ("[1,2,3]", "42", '"a string"'):
        p = tmp_path / "nd.json"
        p.write_text(content, encoding="utf-8")
        s = AppSettings.load(p)
        assert isinstance(s, AppSettings)
        assert s.ui_theme == "light"


def test_from_dict_invalid_recent_items_coerced_to_str(tmp_path):
    """recent_recipes の要素が文字列でなくても str 化される。"""
    s = AppSettings.from_dict({"recent_recipes": [1, 2.5, "x"]})
    assert s.recent_recipes == ["1", "2.5", "x"]


def test_config_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMISIM_CONFIG_DIR", str(tmp_path))
    from semisim import settings as st

    assert st.config_dir() == tmp_path
    assert st.default_path() == tmp_path / "settings.json"


def test_default_config_persisted(tmp_path):
    s = AppSettings(default_config={"nx": 80, "ny": 80, "nz": 100})
    path = tmp_path / "s.json"
    s.save(path)
    loaded = AppSettings.load(path)
    assert loaded.default_config["nx"] == 80
