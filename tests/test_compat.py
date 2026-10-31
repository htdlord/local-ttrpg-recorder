"""Deprecated sessionscribe / sessionscribe-gui commands from before the rename."""

import sys

import pytest

from ttrpg_recorder import __version__, compat


def test_old_cli_warns_and_runs_the_new_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sessionscribe", "--version"])
    with pytest.raises(SystemExit) as done:
        compat.old_cli()
    assert done.value.code == 0
    out = capsys.readouterr()
    assert "'sessionscribe' is deprecated" in out.err and "use 'ttrpg'" in out.err
    assert __version__ in out.out


def test_old_gui_warns_and_starts_the_new_gui(monkeypatch, capsys):
    from ttrpg_recorder.gui import app

    calls = []
    monkeypatch.setattr(app, "main", lambda **kw: calls.append(kw) or 0)
    assert compat.old_gui() == 0
    assert calls == [{"deprecated_alias": "sessionscribe-gui"}]
    assert "use 'ttrpg-gui'" in capsys.readouterr().err


def test_old_gui_message_in_the_status_bar(qtbot, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings

    from ttrpg_recorder.gui.main_window import MainWindow

    w = MainWindow(settings=QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show_deprecated_alias("sessionscribe-gui")
    assert "ttrpg-gui" in w.statusBar().currentMessage()
