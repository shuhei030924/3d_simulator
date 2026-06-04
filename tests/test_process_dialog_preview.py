"""ProcessDialog のライブサマリプレビューのテスト（offscreen Qt）。"""
import os

import pytest

_APP = None


def _qt_or_skip():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")


def _app():
    global _APP
    from PyQt5 import QtWidgets
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def test_preview_label_present_and_nonempty():
    """CVD ダイアログにプレビューラベルがあり、工程名を含む。"""
    _qt_or_skip()
    _app()
    from semisim.gui import ProcessDialog
    d = ProcessDialog("CVD")
    assert hasattr(d, "preview_lbl")
    assert "CVD" in d.preview_lbl.text()


def test_preview_updates_on_value_change():
    """厚みを変えるとプレビューのサマリが更新される。"""
    _qt_or_skip()
    _app()
    from semisim.gui import ProcessDialog
    d = ProcessDialog("CVD")
    before = d.preview_lbl.text()
    d.thick.setValue(d.thick.value() + 0.5)
    after = d.preview_lbl.text()
    assert before != after
    # build_process().summary() と一致する
    assert after == d.build_process().summary()


def test_preview_never_raises_for_all_types():
    """全工程タイプでプレビュー生成が例外を出さない。"""
    _qt_or_skip()
    _app()
    from semisim import processes
    from semisim.gui import ProcessDialog
    for t, _label in processes.available_types():
        d = ProcessDialog(t)
        d._update_preview()
        assert d.preview_lbl.text()  # 非空（"—" でも可）
        d.deleteLater()
