"""Light/dark theme from shared tokens: Fusion style, palette, one stylesheet, fonts.

Spacing follows an 8 px grid (4/8/16/24); one accent colour drives selection, focus,
progress and primary buttons.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from sessionscribe.gui.models.colors import DARK, LIGHT, Tokens

SPACE = (4, 8, 16, 24)
RADIUS = 6


def system_is_dark() -> bool:
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is None:
        return False
    from PySide6.QtCore import Qt

    return bool(scheme() == Qt.ColorScheme.Dark)


def tokens_for(name: str) -> Tokens:
    if name == "dark" or (name == "system" and system_is_dark()):
        return DARK
    return LIGHT


def _palette(t: Tokens) -> QPalette:
    p = QPalette()
    c = QColor
    roles = {
        QPalette.ColorRole.Window: t.background,
        QPalette.ColorRole.WindowText: t.text,
        QPalette.ColorRole.Base: t.surface,
        QPalette.ColorRole.AlternateBase: t.surface_alt,
        QPalette.ColorRole.Text: t.text,
        QPalette.ColorRole.Button: t.surface,
        QPalette.ColorRole.ButtonText: t.text,
        QPalette.ColorRole.Highlight: t.accent,
        QPalette.ColorRole.HighlightedText: t.accent_text,
        QPalette.ColorRole.ToolTipBase: t.surface,
        QPalette.ColorRole.ToolTipText: t.text,
        QPalette.ColorRole.PlaceholderText: t.text_muted,
        QPalette.ColorRole.Link: t.accent,
    }
    for role, color in roles.items():
        p.setColor(role, c(color))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, c(t.text_muted))
    return p


def stylesheet(t: Tokens) -> str:
    s1, s2, s3, _ = SPACE
    return f"""
QWidget {{ font-size: 10pt; }}
QMainWindow, QDialog {{ background: {t.background}; }}
QTabWidget::pane {{ border: none; background: {t.background}; top: -1px; }}
QTabBar::tab {{
    padding: {s2}px {s3}px; margin-right: {s1}px; border: none;
    color: {t.text_muted}; background: transparent; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:hover {{ color: {t.text}; }}
QGroupBox {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: {RADIUS}px;
    margin-top: {s3 + s1}px; padding: {s3}px {s2 + s1}px {s2}px {s2 + s1}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: {s2}px; padding: 0 {s1}px; color: {t.text_muted};
    font-weight: 600;
}}
QPushButton {{
    padding: {s1 + 2}px {s3 - 2}px; border: 1px solid {t.border}; border-radius: {RADIUS}px;
    background: {t.surface}; color: {t.text};
}}
QPushButton:hover {{ border-color: {t.accent}; }}
QPushButton:disabled {{ color: {t.text_muted}; background: {t.surface_alt}; }}
QPushButton[primary="true"] {{
    background: {t.accent}; color: {t.accent_text}; border-color: {t.accent}; font-weight: 600;
}}
QPushButton[primary="true"]:disabled {{ background: {t.border}; border-color: {t.border}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    padding: {s1}px {s2}px; border: 1px solid {t.border}; border-radius: {RADIUS - 2}px;
    background: {t.surface}; color: {t.text}; selection-background-color: {t.accent};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t.accent};
}}
QTableView, QTableWidget, QListWidget {{
    background: {t.surface}; alternate-background-color: {t.surface_alt};
    border: 1px solid {t.border}; border-radius: {RADIUS}px; gridline-color: transparent;
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QHeaderView::section {{
    background: {t.surface}; color: {t.text_muted}; border: none;
    border-bottom: 1px solid {t.border}; padding: {s1}px {s2}px; font-weight: 600;
}}
QProgressBar {{
    border: none; border-radius: {RADIUS - 2}px; background: {t.surface_alt};
    height: {s2}px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {t.accent}; border-radius: {RADIUS - 2}px; }}
QStatusBar {{ color: {t.text_muted}; }}
QLabel[muted="true"] {{ color: {t.text_muted}; }}
QLabel[heading="true"] {{ font-size: 13pt; font-weight: 600; }}
QLabel[error="true"] {{ color: {t.error}; }}
QLabel[warning="true"] {{ color: {t.warning}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; }}
QCheckBox {{ spacing: {s1 + 2}px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {t.text_muted}; border-radius: 3px;
    background: {t.surface};
}}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QCheckBox::indicator:disabled {{ border-color: {t.border}; }}
"""


UI_FAMILIES = ["Segoe UI", "Noto Sans", "Cantarell", "DejaVu Sans", "Helvetica Neue", "Arial"]


def ui_font(points: float = 10.0) -> QFont:
    """The platform UI sans (never left to the platform plugin's first match)."""
    font = QFont()
    font.setFamilies(UI_FAMILIES)
    font.setPointSizeF(points)
    return font


def transcript_font() -> QFont:
    """Readable transcript text: the UI sans at 11 pt."""
    return ui_font(11.0)


def apply(app: QApplication, name: str) -> Tokens:
    tokens = tokens_for(name)
    app.setStyle("Fusion")
    app.setFont(ui_font())
    app.setPalette(_palette(tokens))
    app.setStyleSheet(stylesheet(tokens))
    return tokens
