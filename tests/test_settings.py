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
