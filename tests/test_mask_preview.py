"""マスクプレビュー画像（GUI 非依存）と MaskEditor のプレビュー表示テスト。"""
import os

import numpy as np
import pytest

from semisim.masks import Mask, Shape


def test_preview_rgb_shape_and_dtype():
    """preview_rgb は size×size×3 の uint8 を返す。"""
    rgb = Mask(shapes=[]).preview_rgb(40)
    assert rgb.shape == (40, 40, 3)
    assert rgb.dtype == np.uint8


def test_empty_mask_all_selected():
    """図形なしは全面が選択色（アクセント青）。"""
    rgb = Mask(shapes=[]).preview_rgb(32)
    assert np.all(rgb == (45, 108, 223))


def test_inverted_empty_mask_all_unselected():
    """反転＋図形なしは全面が非選択色。"""
    rgb = Mask(shapes=[], invert=True).preview_rgb(32)
    assert np.all(rgb == (238, 242, 247))


def test_partial_shape_has_both_colors():
    """中央の小矩形は選択・非選択の両色を含む。"""
    m = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    rgb = m.preview_rgb(48)
    sel = np.all(rgb == (45, 108, 223), axis=-1)
    assert sel.any() and (~sel).any()
    # 中央付近は選択
    assert sel[24, 24]


def test_preview_reflects_invert():
    """invert で選択/非選択が反転する。"""
    m = Mask(shapes=[Shape("circle", {"cx": 0.5, "cy": 0.5, "r": 0.3})])
    a = m.preview_rgb(40)
    m.invert = True
    b = m.preview_rgb(40)
    assert not np.array_equal(a, b)


def test_mask_editor_preview_pixmap():
    """MaskEditor がプレビュー QPixmap を生成する（offscreen）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")
    from semisim.gui import MaskEditor
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ed = MaskEditor(Mask(shapes=[Shape("rect", {"x0": 0.2, "y0": 0.2,
                                                "x1": 0.8, "y1": 0.8})]))
    pm = ed.preview.pixmap()
    assert pm is not None and not pm.isNull()
    assert pm.width() == 72 and pm.height() == 72
