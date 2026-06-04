"""HTML マニュアルの UX 要素（テンプレート構造）のテスト。

build_manual.PAGE_TMPL を実際の format 引数と同じキーで描画し、ダークモード・
検索・スクロールスパイ等のインタラクティブ要素が含まれることを検証する
（画像生成を伴わない高速な構造テスト）。
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "build_manual", os.path.join(ROOT, "tools", "build_manual.py"))
build_manual = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_manual)


@pytest.fixture(scope="module")
def page() -> str:
    return build_manual.PAGE_TMPL.format(
        nav="<a href='#x'>x</a>", sections="<section id='x'></section>",
        gui="", device_curves_body="", defect_title="d", defect_body="",
        verification_body="", stats=build_manual._stats_html(19, 13),
        extra_style=build_manual.MANUAL_CSS, scripts=build_manual.MANUAL_JS,
    )


def test_template_formats_without_brace_error(page):
    """テンプレートがブレース衝突なく描画できる。"""
    assert len(page) > 1000


def test_has_progress_bar(page):
    assert 'id="progress"' in page


def test_has_theme_toggle_and_dark_css(page):
    """ダークモード切替ボタンと data-theme CSS を含む。"""
    assert 'id="themeBtn"' in page
    assert '[data-theme="dark"]' in page
    assert "semisim-theme" in page  # localStorage 永続化


def test_has_search_filter(page):
    assert 'id="search"' in page
    assert 'id="noResult"' in page


def test_has_scrollspy(page):
    assert "IntersectionObserver" in page


def test_has_back_to_top(page):
    assert 'id="toTop"' in page


def test_has_lightbox(page):
    assert 'id="lightbox"' in page
    assert "zoomable" in page


def test_print_button(page):
    """印刷 / PDF ボタンがあり、印刷時は折りたたみ章も展開される。"""
    assert 'id="printBtn"' in page
    assert "window.print()" in page
    # 印刷スタイルで collapsed の本文を復活させる
    assert "section.collapsed > :not(h2) { display:revert" in page


def test_search_hit_count(page):
    """検索ヒット件数のリードアウト（searchCount）がある。"""
    assert 'id="searchCount"' in page
    assert "hits++" in page
    assert "' 件'" in page


def test_lightbox_navigation(page):
    """ライトボックスに前後ナビ・キャプション・カウンタがある。"""
    assert 'id="lbPrev"' in page
    assert 'id="lbNext"' in page
    assert 'id="lbCount"' in page
    assert 'id="lbCap"' in page
    assert "ArrowRight" in page and "ArrowLeft" in page


def test_keyboard_shortcuts(page):
    """'/' で検索フォーカス・Esc 対応のキーボードショートカットがある。"""
    assert "keydown" in page
    assert "Escape" in page


def test_has_copy_buttons(page):
    assert "navigator.clipboard" in page
    assert "copy-btn" in page


def test_responsive_media_query(page):
    assert "@media" in page


def test_smooth_scroll(page):
    assert "scroll-behavior:smooth" in page


def test_has_stats_dashboard(page):
    """概要ダッシュボード（統計カード）が含まれる。"""
    assert "class='stats'" in page or 'class="stats"' in page
    assert "検証関数" in page


def test_print_stylesheet(page):
    """印刷用スタイル（@media print）でナビ等を隠す。"""
    assert "@media print" in page


def test_anchor_link_copy(page):
    """見出しアンカーリンクのコピー機能とターゲットハイライトがある。"""
    assert "anchor-link" in page
    assert "section:target" in page
    assert "replaceState" in page


def test_collapsible_nav_categories(page):
    """目次カテゴリの折りたたみ機能がある（検索時は自動展開）。"""
    assert "cat-toggle" in page
    assert "expandAllCats" in page
    assert "collapsed" in page


def test_help_overlay(page):
    """? キーで開くショートカットヘルプオーバーレイがある。"""
    assert 'id="help-overlay"' in page
    assert 'id="helpBtn"' in page
    assert "toggleHelp" in page
    assert "キーボードショートカット" in page


def test_section_collapse(page):
    """セクション本文の折りたたみ（見出しクリック）と全展開/折りたたみがある。"""
    assert "sec-caret" in page
    assert "section.collapsed" in page
    assert 'id="collapseAllBtn"' in page


def test_collapse_persistence(page):
    """折りたたみ状態が localStorage に永続化される。"""
    assert "semisim-collapsed" in page
    assert "persistCollapsed" in page


def test_font_size_adjuster(page):
    """文字サイズ調整（A− / A＋）ボタンと localStorage 永続化がある。"""
    assert 'id="fontDecBtn"' in page
    assert 'id="fontIncBtn"' in page
    assert "semisim-fontscale" in page


def test_card_var_defined_in_light(page):
    """--card 変数がライト（:root）でも定義されている（不透明背景の保証）。

    var(--card) を使う help-card 等がライトで透明背景にならないことを保証する。
    """
    i = page.index(":root")
    root_block = page[i:i + 220]
    assert "--card" in root_block


def test_stats_html_counts():
    """_stats_html は指定した工程数・カーブ数と metrology 関数数を埋め込む。"""
    s = build_manual._stats_html(19, 13)
    assert ">19<" in s and ">13<" in s
    # metrology の検証関数が多数カウントされる
    import re
    nums = [int(x) for x in re.findall(r">(\d+)<", s)]
    assert max(nums) > 50
