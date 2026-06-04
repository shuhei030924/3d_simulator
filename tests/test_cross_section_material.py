"""2D 断面ビューのカーソル下材料リードアウトのテスト（offscreen Qt）。"""
import os

import pytest


def _qt_or_skip():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")


def _make_view():
    from PyQt5 import QtWidgets

    from semisim import processes
    from semisim.grid import Wafer, WaferConfig
    from semisim.gui import CrossSection2D
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Wafer(WaferConfig(nx=40, ny=40, nz=50, pitch_um=0.1, substrate_um=2.0))
    processes.CVD(material="oxide", thickness_um=0.3).apply(w)
    view = CrossSection2D()
    view.set_wafer(w, include_resist=True)  # 軸 Y（XZ 面）, 中央
    return view


def test_material_at_substrate_and_oxide():
    """下部=シリコン基板、その上=酸化膜、上空=空気を識別する。"""
    _qt_or_skip()
    view = _make_view()
    xc = 2.0  # 中央列 (40*0.1/2)
    assert "シリコン" in view._material_at(xc, 0.5)
    assert "酸化膜" in view._material_at(xc, 2.15)
    assert "空気" in view._material_at(xc, 4.5)


def test_material_at_out_of_bounds_none():
    """断面の外側は None を返す。"""
    _qt_or_skip()
    view = _make_view()
    assert view._material_at(-1.0, 1.0) is None
    assert view._material_at(1.0, 99.0) is None


def test_material_at_no_plane_none():
    """未描画（ウェハ未設定）なら None。"""
    _qt_or_skip()
    from PyQt5 import QtWidgets

    from semisim.gui import CrossSection2D
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = CrossSection2D()
    assert view._material_at(1.0, 1.0) is None


def test_mat_lbl_exists():
    """材料リードアウト用ラベルがある。"""
    _qt_or_skip()
    view = _make_view()
    assert hasattr(view, "mat_lbl")
