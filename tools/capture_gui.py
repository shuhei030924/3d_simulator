"""GUI スクリーンショット生成（ヘッドレス・グラブ方式）。

PyQt5 のウィジェットを画面に表示せずにオフスクリーン・グラブで PNG 化し、
docs/manual/img/gui_*.png に保存する。設定ダイアログ（工程パラメータ入力）と
メインウィンドウ（操作画面）のスクリーンショットを使い方説明書に載せる用途。

ポイント:
  - QT_QPA_PLATFORM=offscreen は QFormLayout のラベルが空白になるため使わない。
    実プラットフォーム上で WA_DontShowOnScreen を立ててグラブする。
  - MainWindow の 3D ビュー(VTK)はグラブで黒くなるため、2D 断面タブへ切替えて撮る。

使い方:
    py tools/capture_gui.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from semisim.grid import WaferConfig  # noqa: E402
from semisim.gui import MainWindow, ProcessDialog, WaferDialog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "docs", "manual", "img")


def _grab(widget: QtWidgets.QWidget, app: QtWidgets.QApplication, path: str) -> None:
    """ウィジェットを画面に出さずにグラブして PNG 保存する。"""
    widget.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    widget.show()
    app.processEvents()
    widget.adjustSize()
    app.processEvents()
    pm = widget.grab()
    os.makedirs(IMG_DIR, exist_ok=True)
    pm.save(path)
    widget.close()


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # --- 工程設定ダイアログ（タイプごとに入力欄が変わる） ---
    dialogs = [
        ("PHOTO", "gui_dialog_photo.png"),
        ("CVD", "gui_dialog_cvd.png"),
        ("PVD", "gui_dialog_pvd.png"),
        ("KOH", "gui_dialog_koh.png"),
        ("IMPLANT", "gui_dialog_implant.png"),
        ("CMP", "gui_dialog_cmp.png"),
    ]
    for proc_type, fname in dialogs:
        dlg = ProcessDialog(proc_type)
        _grab(dlg, app, os.path.join(IMG_DIR, fname))

    # --- ウェハ設定ダイアログ ---
    wdlg = WaferDialog(WaferConfig())
    _grab(wdlg, app, os.path.join(IMG_DIR, "gui_dialog_wafer.png"))

    # --- メインウィンドウ（操作画面）。2D 断面タブを表示してグラブ ---
    win = MainWindow()
    win.resize(1180, 760)
    # 2D 断面タブへ切替（3D の VTK はグラブで黒くなるため）
    for i in range(win.tabs.count()):
        if win.tabs.tabText(i).startswith("2D"):
            win.tabs.setCurrentIndex(i)
            break
    win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()
    app.processEvents()
    pm = win.grab()
    pm.save(os.path.join(IMG_DIR, "gui_main.png"))
    win.close()

    print("done")


if __name__ == "__main__":
    main()
