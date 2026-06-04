"""マスク図形ラベル（GUI 一覧用, GUI 非依存）と MaskEditor 空状態のテスト。"""
import os

import pytest

from semisim.masks import Mask, Shape


def test_rect_label():
    s = Shape("rect", {"x0": 0.1, "y0": 0.2, "x1": 0.8, "y1": 0.9})
    lbl = s.label()
    assert lbl.startswith("▭ 矩形")
    assert "0.10" in lbl and "0.80" in lbl


def test_rect_label_with_angle():
    s = Shape("rect", {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "angle": 30})
    assert "∠30°" in s.label()


def test_circle_label():
    s = Shape("circle", {"cx": 0.5, "cy": 0.5, "r": 0.25})
    assert s.label().startswith("● 円")
    assert "r=0.25" in s.label()


def test_stripe_label():
    s = Shape("stripe", {"cx": 0.5, "cy": 0.5, "angle": 90, "width": 0.2})
    assert s.label().startswith("▬ 帯")


def test_grating_label():
    s = Shape("grating", {"angle": 0, "period": 0.3, "width": 0.15})
    lbl = s.label()
    assert lbl.startswith("☰ 周期ライン")
    assert "周期0.30" in lbl


def test_each_kind_has_distinct_icon():
    """種別ごとに先頭アイコンが異なる。"""
    icons = {Shape(k, {}).label()[0] for k in ("rect", "circle", "stripe", "grating")}
    assert len(icons) == 4


def test_mask_editor_empty_state():
    """図形が無いとき MaskEditor は空状態プレースホルダを表示する（offscreen）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")
    from semisim.gui import MaskEditor
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ed = MaskEditor(Mask(shapes=[]))
    assert ed.list.count() == 1
    assert "全面が対象" in ed.list.item(0).text()
    # 図形ありでは各図形のラベルが並ぶ（プレースホルダなし）
    ed2 = MaskEditor(Mask(shapes=[Shape("circle", {"cx": 0.5, "cy": 0.5, "r": 0.2})]))
    assert ed2.list.count() == 1
    assert ed2.list.item(0).text().startswith("● 円")
