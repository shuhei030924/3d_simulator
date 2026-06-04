"""2D 断面ビューの画像保存機能のテスト（offscreen Qt）。"""
import os

import pytest


def _qt_or_skip():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")


def _make_wafer():
    from semisim import processes
    from semisim.grid import Wafer, WaferConfig
    w = Wafer(WaferConfig(nx=40, ny=40, nz=50, pitch_um=0.1, substrate_um=2.0))
    # 何か材料を載せて断面に内容を作る
    processes.CVD(material="oxide", thickness_um=0.3).apply(w)
    return w


def test_save_image_writes_png(tmp_path):
    """save_image が PNG ファイルを書き出す。"""
    _qt_or_skip()
    from PyQt5 import QtWidgets

    from semisim.gui import CrossSection2D
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = CrossSection2D()
    view.set_wafer(_make_wafer(), include_resist=True)
    out = tmp_path / "slice.png"
    ret = view.save_image(str(out))
    assert ret == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_save_image_no_wafer_returns_none(tmp_path):
    """ウェハ未設定なら None を返し例外を出さない。"""
    _qt_or_skip()
    from PyQt5 import QtWidgets

    from semisim.gui import CrossSection2D
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = CrossSection2D()
    assert view.save_image(str(tmp_path / "x.png")) is None


def test_save_button_exists():
    """2D ビューに画像保存ボタンがある。"""
    _qt_or_skip()
    from PyQt5 import QtWidgets

    from semisim.gui import CrossSection2D
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = CrossSection2D()
    assert hasattr(view, "save_btn")
    assert view.save_btn.text() == "画像保存"
