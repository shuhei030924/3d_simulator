"""GUI 構築のスモークテスト（オフスクリーン、ウィンドウを開かない）。"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvista as pv

pv.OFF_SCREEN = True

from PyQt5 import QtWidgets  # noqa: E402

from semisim.gui import MainWindow  # noqa: E402


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    app.processEvents()
    # 断面モードを一通り切り替えてレンダリングパスを検証
    for idx in range(5):
        win.clip_combo.setCurrentIndex(idx)
        app.processEvents()
    win.slider.setValue(20)
    app.processEvents()
    win.resist_cb.setChecked(False)
    app.processEvents()
    win.smooth_cb.setChecked(True)
    app.processEvents()
    print("GUI スモークテスト OK / 工程数:", len(win.recipe.steps))
    win.close()


if __name__ == "__main__":
    main()
