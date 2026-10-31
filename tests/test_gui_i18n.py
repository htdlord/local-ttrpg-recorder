"""Translation completeness and the window starting in Hungarian."""

import ast
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[1] / "src" / "sessionscribe" / "gui"
TS = GUI / "i18n" / "sessionscribe_hu.ts"


def tr_literals() -> set[str]:
    found = set()
    for path in GUI.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tr"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
    return found


def test_every_message_has_a_finished_hungarian_translation():
    messages = {}
    for m in ET.parse(TS).getroot().iter("message"):
        translation = m.find("translation")
        assert translation is not None
        messages[m.findtext("source")] = translation
    missing = tr_literals() - set(messages)
    assert not missing, f"run pyside6-lupdate: {sorted(missing)}"
    for source, translation in messages.items():
        assert translation.get("type") not in ("unfinished", "obsolete", "vanished"), source
        assert translation.text, source
        for i in range(10):  # placeholders survive the translation
            assert (f"{{{i}" in source) == (f"{{{i}" in translation.text), source


def test_compiled_translation_exists():
    qm = TS.with_suffix(".qm")
    assert qm.is_file() and qm.stat().st_size > 0


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="gui extra")
def test_window_starts_in_hungarian(qtbot, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from sessionscribe.gui import i18n
    from sessionscribe.gui.main_window import MainWindow

    app = QApplication.instance()
    assert i18n.install(app, "hu") == "hu"
    try:
        prefs = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
        w = MainWindow(settings=prefs)
        qtbot.addWidget(w)
        titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
        assert titles == ["Session", "Felvétel", "Beállítások", "Futtatás", "Eredmény"]
        assert w.statusBar().currentMessage() == "Nincs megnyitott session"
    finally:
        i18n.install(app, "en")
