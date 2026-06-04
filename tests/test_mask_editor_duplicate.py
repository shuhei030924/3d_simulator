"""MaskEditor の図形複製機能のテスト（offscreen Qt）。"""
import os

import pytest


def _qt_or_skip():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")


_APP = None


def _app():
    """QApplication をモジュール全体で保持し GC による破棄を防ぐ。"""
    global _APP
    from PyQt5 import QtWidgets
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _editor():
    _qt_or_skip()
    _app()
    from semisim.gui import MaskEditor
    from semisim.masks import Mask, Shape
    m = Mask(shapes=[Shape("rect", {"x0": 0.2, "y0": 0.2, "x1": 0.6, "y1": 0.6})])
    return MaskEditor(m)


def test_duplicate_adds_identical_shape():
    """複製で図形数が増え、内容が一致した独立コピーが直後に入る。"""
    ed = _editor()
    ed.list.setCurrentRow(0)
    ed._duplicate()
    shapes = ed.mask.shapes
    assert len(shapes) == 2
    assert shapes[0].kind == shapes[1].kind
    assert shapes[0].params == shapes[1].params
    # 独立コピー（params が共有されていない）
    shapes[1].params["x0"] = 0.9
    assert shapes[0].params["x0"] != shapes[1].params["x0"]


def test_duplicate_selects_new_row():
    """複製後は新しい行が選択される。"""
    ed = _editor()
    ed.list.setCurrentRow(0)
    ed._duplicate()
    assert ed.list.currentRow() == 1


def test_duplicate_noop_when_empty():
    """図形が無ければ複製しても何も起きない。"""
    _qt_or_skip()
    _app()
    from semisim.gui import MaskEditor
    from semisim.masks import Mask
    ed = MaskEditor(Mask(shapes=[]))
    ed._duplicate()
    assert ed.mask.shapes == []


def test_duplicate_button_exists():
    _qt_or_skip()
    ed = _editor()
    from PyQt5 import QtWidgets
    labels = [b.text() for b in ed.findChildren(QtWidgets.QPushButton)]
    assert "複製" in labels
