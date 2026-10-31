"""Own single-colour line icons (24 x 24 SVG, MIT like the project), tinted per theme."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    "{body}</svg>"
)

PATHS = {
    # tabs
    "session": '<path d="M3 7h6l2 2h10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "settings": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
    '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "run": '<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5z"/>',
    "result": '<path d="M5 6h14M5 10h14M5 14h9M5 18h6"/>',
    "record": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4" fill="{color}"/>',
    "check": '<path d="M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7z"/>'
    '<path d="M8.5 12l2.5 2.5 4.5-5"/>',
    # flags / row kinds
    "review": '<path d="M5 21V4h11l-2 4 2 4H5"/>',
    "overlap": '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
    "bleed": '<path d="M3 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/>',
    "duplicate": '<rect x="8" y="8" width="12" height="12" rx="2"/>'
    '<path d="M4 16V6a2 2 0 0 1 2-2h10"/>',
    "missed": '<circle cx="12" cy="12" r="9"/>'
    '<path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7"/><path d="M12 17h0"/>',
    "reassigned": '<path d="M4 8h13l-3-3M20 16H7l3 3"/>',
    "gap": '<path d="M3 12h5M16 12h5M10 7l-2 10M16 7l-2 10"/>',
    "alarm": '<path d="M6 16V11a6 6 0 0 1 12 0v5l2 2H4z"/><path d="M10 21h4"/>',
    "suppressed": '<circle cx="12" cy="12" r="9"/><path d="M6 6l12 12"/>',
    # actions
    "play": '<path d="M8 5v14l11-7z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "open": '<path d="M3 7h6l2 2h10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
    '<path d="M12 12v5M9.5 14.5 12 12l2.5 2.5"/>',
    "save": '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    "add": '<path d="M12 5v14M5 12h14"/>',
    "remove": '<path d="M5 12h14"/>',
    "export": '<path d="M12 4v11M8 8l4-4 4 4"/><path d="M5 14v5h14v-5"/>',
    "undo": '<path d="M9 7 4 12l5 5"/><path d="M4 12h10a6 6 0 0 1 0 12h-1"/>',
}


def svg(name: str, color: str) -> bytes:
    body = PATHS[name].replace("{color}", color)
    return _SVG.format(color=color, body=body).encode()


def icon(name: str, color: str, size: int = 32) -> QIcon:
    renderer = QSvgRenderer(QByteArray(svg(name, color)))
    result = QIcon()
    for px in (size, size * 2):
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, px, px))
        painter.end()
        result.addPixmap(pixmap)
    return result
