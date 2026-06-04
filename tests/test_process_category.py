"""工程カテゴリ分け（GUI の追加メニューのサブメニュー化, GUI 非依存）のテスト。"""
import os

import pytest

from semisim import processes


def test_every_type_has_category():
    """すべての工程タイプがカテゴリを持つ（'その他' 含む）。"""
    for t, _ in processes.available_types():
        assert processes.process_category(t)


def test_known_categories_assigned():
    """既知の分類が割り当てられている。"""
    assert processes.process_category("CVD") == "成膜"
    assert processes.process_category("DRY") == "エッチング"
    assert processes.process_category("IMPLANT") == "ドーピング・熱処理"
    assert processes.process_category("CMP") == "平坦化・仕上げ"
    assert processes.process_category("PHOTO") == "リソグラフィ"


def test_unknown_category_is_other():
    assert processes.process_category("NOPE") == "その他"


def test_categorized_covers_all_types_once():
    """categorized_types は全タイプを重複なく網羅する。"""
    all_types = {t for t, _ in processes.available_types()}
    seen = []
    for _cat, items in processes.categorized_types():
        for t, _label in items:
            seen.append(t)
    assert set(seen) == all_types
    assert len(seen) == len(all_types)  # 重複なし


def test_categories_in_defined_order():
    """カテゴリは定義順（リソグラフィ→成膜→…）で並ぶ。"""
    cats = [c for c, _ in processes.categorized_types()]
    assert cats[0] == "リソグラフィ"
    assert cats.index("成膜") < cats.index("エッチング")


@pytest.mark.skipif(os.environ.get("SEMISIM_SKIP_QT") == "1", reason="Qt 無効")
def test_add_menu_has_category_submenus():
    """「工程を追加」メニューがカテゴリのサブメニューを持つ（offscreen は不可なので
    メニュー構築ロジックを直接検証）。"""
    try:
        from PyQt5 import QtWidgets
    except Exception:  # noqa: BLE001
        pytest.skip("PyQt5 が無い")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    menu = QtWidgets.QMenu()
    for category, types in processes.categorized_types():
        sub = menu.addMenu(category)
        for t, label in types:
            sub.addAction(f"{t} — {label}")
    submenus = [a.menu() for a in menu.actions() if a.menu() is not None]
    assert len(submenus) == len(processes.categorized_types())
    assert len(submenus) >= 5


def test_category_colors_valid_hex():
    """全カテゴリに有効な #rrggbb 識別色がある。"""
    import re
    for cat, _ in processes.categorized_types():
        c = processes.category_color(cat)
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", c), f"{cat}: {c}"


def test_category_colors_distinct():
    """主要カテゴリの色は互いに異なる。"""
    cats = [c for c, _ in processes.categorized_types()]
    colors = [processes.category_color(c) for c in cats]
    assert len(set(colors)) == len(colors)


def test_process_color_matches_category():
    """process_color は工程のカテゴリ色と一致する。"""
    for t, _ in processes.available_types():
        assert processes.process_color(t) == \
            processes.category_color(processes.process_category(t))


def test_pixmap_icon_from_color():
    """カテゴリ色から実際にアイコン（QPixmap）が作れる（offscreen）。"""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtGui, QtWidgets
    except Exception:  # noqa: BLE001
        import pytest
        pytest.skip("PyQt5 が無い")
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pm = QtGui.QPixmap(11, 11)
    pm.fill(QtGui.QColor(processes.process_color("CVD")))
    assert pm.width() == 11 and not pm.isNull()
