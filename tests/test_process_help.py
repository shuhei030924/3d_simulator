"""工程の説明テキスト（GUI ダイアログのヒントバナー用, GUI 非依存）のテスト。"""
import os

import pytest

from semisim import processes


def test_all_types_have_help():
    """すべての工程タイプに 1 行説明が登録されている。"""
    for t, _ in processes.available_types():
        h = processes.process_help(t)
        assert h and len(h) > 8, f"{t} に説明が無い"


def test_unknown_type_empty():
    """未登録タイプは空文字を返す。"""
    assert processes.process_help("NOPE") == ""


def test_help_is_concise_single_line():
    """説明は 1 行（改行を含まない）で簡潔。"""
    for t, _ in processes.available_types():
        h = processes.process_help(t)
        assert "\n" not in h
        assert len(h) < 120


@pytest.mark.skipif(os.environ.get("SEMISIM_SKIP_QT") == "1", reason="Qt 無効")
def test_dialog_shows_help_banner():
    """各工程ダイアログ上部に説明バナー（ヒント）が表示される（offscreen）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")
    from semisim.gui import ProcessDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    for t, _ in processes.available_types():
        d = ProcessDialog(t)
        texts = [w.text() for w in d.findChildren(QtWidgets.QLabel)]
        help_text = processes.process_help(t)
        assert any(help_text[:12] in tx for tx in texts), f"{t} のバナーが無い"
        d.deleteLater()
