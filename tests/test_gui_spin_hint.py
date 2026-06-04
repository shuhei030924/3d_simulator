"""数値入力（spinbox）の範囲ヒントツールチップのテスト（offscreen）。"""
import os

import pytest


def _qt_or_skip():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")


def test_range_hint_format():
    """範囲ヒントは min〜max を指定桁で整形する。"""
    _qt_or_skip()
    from semisim.gui import _range_hint
    assert _range_hint(0.05, 20.0, 2) == "入力範囲: 0.05 〜 20.00"
    assert _range_hint(0.0, 100.0, 0) == "入力範囲: 0 〜 100"


def test_spin_has_range_tooltip():
    """_spin で作る spinbox に範囲ツールチップが設定される。"""
    _qt_or_skip()
    from PyQt5 import QtWidgets

    from semisim.gui import _spin
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = _spin(1.0, 0.05, 20.0, 0.1, 2)
    assert "入力範囲" in s.toolTip()
    assert s.minimum() == pytest.approx(0.05)
    assert s.maximum() == pytest.approx(20.0)


def test_dialog_spinboxes_all_have_tooltips():
    """各工程ダイアログの全 spinbox に範囲ツールチップが付く。"""
    _qt_or_skip()
    from PyQt5 import QtWidgets

    from semisim import processes
    from semisim.gui import ProcessDialog
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for t, _label in processes.available_types():
        d = ProcessDialog(t)
        for sb in d.findChildren(QtWidgets.QDoubleSpinBox):
            assert "入力範囲" in sb.toolTip(), f"{t} の spinbox にヒント無し"
        d.deleteLater()
