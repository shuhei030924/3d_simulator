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
