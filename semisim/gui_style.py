"""GUI の外観（QSS スタイルシート）。

フラットでモダンな配色を提供する。HTML マニュアルと同じアクセント色
（#2d6cdf）で統一し、ライト/ダークの 2 テーマを返す。アプリ起動時に
QApplication.setStyleSheet(stylesheet()) で適用する。
"""
from __future__ import annotations

# テーマ別パレット（HTML マニュアルと統一）
_PALETTES = {
    "light": {
        "bg": "#f4f6fa", "card": "#ffffff", "fg": "#1c2530", "muted": "#5a6b7b",
        "line": "#d4dce6", "accent": "#2d6cdf", "accent_dark": "#1f54b5",
        "sel": "#e7eef9", "field": "#ffffff", "header": "#eef2f8",
        "disabled": "#9aa7b4",
    },
    "dark": {
        "bg": "#0d1117", "card": "#161b22", "fg": "#e6edf3", "muted": "#9aa7b4",
        "line": "#2a3340", "accent": "#5b9dff", "accent_dark": "#3f7fe0",
        "sel": "#1f2a3a", "field": "#0d1117", "header": "#1c2530",
        "disabled": "#5a6b7b",
    },
}

THEMES = tuple(_PALETTES.keys())


def stylesheet(theme: str = "light") -> str:
    """指定テーマの QSS 文字列を返す（既定はライト）。

    QPushButton / QToolButton / QListWidget / QLineEdit / QSpinBox /
    QComboBox / QGroupBox / QTabWidget / QHeaderView / QMenu / QToolTip 等を
    フラットでモダンに整える。未知のテーマ名はライトにフォールバックする。
    """
    c = _PALETTES.get(theme, _PALETTES["light"])
    return f"""
QWidget {{
  background: {c['bg']};
  color: {c['fg']};
  font-size: 13px;
}}
QMainWindow, QDialog {{ background: {c['bg']}; }}
QLabel {{ background: transparent; }}

/* 一般ボタン */
QPushButton {{
  background: {c['card']};
  border: 1px solid {c['line']};
  border-radius: 7px;
  padding: 6px 13px;
  color: {c['fg']};
}}
QPushButton:hover {{ border-color: {c['accent']}; color: {c['accent']}; }}
QPushButton:pressed {{ background: {c['sel']}; }}
QPushButton:disabled {{ color: {c['disabled']}; border-color: {c['line']}; }}
QPushButton:default {{
  background: {c['accent']}; border-color: {c['accent']}; color: #ffffff;
}}
QPushButton:default:hover {{ background: {c['accent_dark']}; color: #ffffff; }}

/* ツールボタン（工程を追加など） */
QToolButton {{
  background: {c['accent']};
  border: 1px solid {c['accent']};
  border-radius: 7px;
  padding: 6px 12px;
  color: #ffffff;
  font-weight: 600;
}}
QToolButton:hover {{ background: {c['accent_dark']}; }}
QToolButton::menu-indicator {{ image: none; }}

/* リスト（レシピ） */
QListWidget {{
  background: {c['card']};
  border: 1px solid {c['line']};
  border-radius: 9px;
  padding: 4px;
  outline: none;
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
QListWidget::item:hover {{ background: {c['sel']}; }}
QListWidget::item:selected {{ background: {c['accent']}; color: #ffffff; }}

/* テキスト入力・数値・コンボ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
  background: {c['field']};
  border: 1px solid {c['line']};
  border-radius: 6px;
  padding: 4px 7px;
  selection-background-color: {c['accent']};
  selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
  background: {c['card']}; border: 1px solid {c['line']};
  selection-background-color: {c['accent']}; selection-color: #ffffff;
}}

/* グループ枠 */
QGroupBox {{
  background: {c['card']};
  border: 1px solid {c['line']};
  border-radius: 9px;
  margin-top: 14px;
  padding: 10px 12px 12px;
}}
QGroupBox::title {{
  subcontrol-origin: margin; left: 12px; padding: 0 5px;
  color: {c['muted']}; font-weight: 600;
}}

/* タブ（3D / 2D 断面） */
QTabWidget::pane {{
  border: 1px solid {c['line']}; border-radius: 9px; top: -1px; background: {c['card']};
}}
QTabBar::tab {{
  background: transparent; color: {c['muted']};
  padding: 7px 16px; margin-right: 2px;
  border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
  background: {c['card']}; color: {c['accent']};
  border-color: {c['line']}; border-bottom-color: {c['card']}; font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ color: {c['accent']}; }}

/* テーブルヘッダ */
QHeaderView::section {{
  background: {c['header']}; color: {c['fg']};
  border: none; border-right: 1px solid {c['line']}; border-bottom: 1px solid {c['line']};
  padding: 5px 8px;
}}
QTableWidget, QTableView {{
  background: {c['card']}; border: 1px solid {c['line']}; border-radius: 8px;
  gridline-color: {c['line']};
}}

/* メニュー */
QMenu {{
  background: {c['card']}; border: 1px solid {c['line']}; border-radius: 8px; padding: 4px;
}}
QMenu::item {{ padding: 6px 22px; border-radius: 5px; }}
QMenu::item:selected {{ background: {c['accent']}; color: #ffffff; }}
QMenuBar {{ background: {c['bg']}; }}
QMenuBar::item:selected {{ background: {c['sel']}; border-radius: 5px; }}

/* ツールチップ・ステータスバー */
QToolTip {{
  background: {c['fg']}; color: {c['bg']};
  border: none; border-radius: 5px; padding: 5px 8px;
}}
QStatusBar {{ background: {c['bg']}; color: {c['muted']}; }}

/* スクロールバー */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['line']}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {c['muted']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['line']}; border-radius: 5px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

/* チェック・スライダ */
QCheckBox {{ spacing: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['line']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
  background: {c['accent']}; width: 14px; margin: -6px 0; border-radius: 7px;
}}
"""
