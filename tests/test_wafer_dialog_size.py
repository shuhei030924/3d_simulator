"""WaferDialog の規模（格子点数・メモリ）ライブリードアウトのテスト。"""
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


def test_grid_size_hint_format():
    """格子点数（カンマ区切り）と MB 概算を含む。"""
    _qt_or_skip()
    from semisim.gui import _grid_size_hint
    s = _grid_size_hint(100, 100, 100)
    assert "1,000,000" in s
    assert "MB" in s
    # uint8: 1,000,000 cells ≈ 0.95 MB → 1 桁丸めで 1.0 MB
    assert "1.0 MB" in s


def test_dialog_shows_and_updates_size():
    """ダイアログに規模ラベルがあり、ボクセル数変更で更新される。"""
    _qt_or_skip()
    _app()
    from semisim.grid import WaferConfig
    from semisim.gui import WaferDialog
    d = WaferDialog(WaferConfig(nx=40, ny=40, nz=50, pitch_um=0.1, substrate_um=2.0))
    assert hasattr(d, "size_lbl")
    before = d.size_lbl.text()
    d.nx.setValue(d.nx.value() + 50)
    assert d.size_lbl.text() != before
    assert "格子点数" in d.size_lbl.text()
