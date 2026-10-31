"""Small shared widgets: headings, muted labels, primary buttons, empty states."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ttrpg_recorder.gui import icons
from ttrpg_recorder.gui.models.colors import Tokens


def label(text: str = "", *, muted: bool = False, heading: bool = False) -> QLabel:
    w = QLabel(text)
    w.setWordWrap(True)
    if muted:
        w.setProperty("muted", True)
    if heading:
        w.setProperty("heading", True)
    return w


def button(
    text: str, tokens: Tokens, icon: str | None = None, primary: bool = False
) -> QPushButton:
    b = QPushButton(text)
    if icon:
        b.setIcon(icons.icon(icon, tokens.accent_text if primary else tokens.text_muted))
    if primary:
        b.setProperty("primary", True)
    return b


class EmptyState(QWidget):
    """A friendly placeholder instead of an empty table."""

    def __init__(self, tokens: Tokens, icon_name: str, title: str, hint: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)
        art = QLabel()
        art.setPixmap(icons.icon(icon_name, tokens.text_muted, 48).pixmap(48, 48))
        art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(art)
        t = label(title, heading=True)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(t)
        h = label(hint, muted=True)
        h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(h)
