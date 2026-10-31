"""Translations: English source strings, Hungarian first (``ttrpg_recorder_hu.qm``)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLocale, QTranslator
from PySide6.QtWidgets import QApplication

FOLDER = Path(__file__).resolve().parent
LANGUAGES = ("en", "hu")
_installed: list[QTranslator] = []


def resolve(choice: str) -> str:
    if choice in LANGUAGES:
        return choice
    return "hu" if QLocale.system().name().lower().startswith("hu") else "en"


def install(app: QApplication, choice: str) -> str:
    """Install the translator for ``choice`` (system/en/hu); returns the language used."""
    for old in _installed:
        app.removeTranslator(old)
    _installed.clear()
    lang = resolve(choice)
    if lang != "en":
        translator = QTranslator(app)
        if translator.load(str(FOLDER / f"ttrpg_recorder_{lang}.qm")):
            app.installTranslator(translator)
            _installed.append(translator)
    return lang
