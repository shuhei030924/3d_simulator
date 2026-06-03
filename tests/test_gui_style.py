"""GUI スタイルシート（QSS）と設定テーマの単体テスト。

QtInteractor（OpenGL）に依存しない部分のみを検証する。実ウィジェットへの
適用テストは PyQt5 が利用できる環境でのみ実行（offscreen）。
"""
import os

import pytest

from semisim import gui_style
from semisim.settings import AppSettings


def test_themes_available():
    """ライト/ダークの 2 テーマが定義されている。"""
    assert "light" in gui_style.THEMES
    assert "dark" in gui_style.THEMES


def test_stylesheet_nonempty_for_each_theme():
    """各テーマの QSS が非空で主要セレクタを含む。"""
    for theme in gui_style.THEMES:
        qss = gui_style.stylesheet(theme)
        assert len(qss) > 500
        for sel in ("QPushButton", "QListWidget", "QLineEdit", "QGroupBox",
                    "QTabBar::tab", "QToolButton", "QMenu", "QToolTip"):
            assert sel in qss


def test_unknown_theme_falls_back_to_light():
    """未知のテーマ名はライトにフォールバックする。"""
    assert gui_style.stylesheet("nonsense") == gui_style.stylesheet("light")


def test_themes_have_distinct_palettes():
    """ライトとダークで配色が異なる。"""
    assert gui_style.stylesheet("light") != gui_style.stylesheet("dark")


def test_accent_color_present():
    """アクセント色（マニュアルと統一）が含まれる。"""
    assert "#2d6cdf" in gui_style.stylesheet("light")


def test_settings_ui_theme_roundtrip(tmp_path):
    """ui_theme が保存・復元される。"""
    p = tmp_path / "s.json"
    s = AppSettings(ui_theme="dark")
    s.save(p)
    assert AppSettings.load(p).ui_theme == "dark"


def test_settings_ui_theme_default_light():
    """既定テーマはライト。"""
    assert AppSettings().ui_theme == "light"


def test_settings_ui_theme_invalid_falls_back(tmp_path):
    """不正なテーマ値はライトにフォールバックする。"""
    p = tmp_path / "s.json"
    AppSettings(ui_theme="rainbow").save(p)
    assert AppSettings.load(p).ui_theme == "light"


@pytest.mark.skipif(
    os.environ.get("SEMISIM_SKIP_QT") == "1", reason="Qt 無効")
def test_stylesheet_applies_to_widget():
    """QSS を実ウィジェットに適用してもエラーにならない（offscreen）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(gui_style.stylesheet("dark"))
    w = QtWidgets.QPushButton("テスト")
    pm = w.grab()
    assert pm.width() > 0 and pm.height() > 0
