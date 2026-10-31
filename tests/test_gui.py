"""GUI end to end with pytest-qt (offscreen in CI) and fake ASR engines."""

from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from tests.fakes import FakeFactory, SlowFactory
from ttrpg_recorder.core.session import Session
from ttrpg_recorder.gui.audio_preview import PreviewPlayer
from ttrpg_recorder.gui.main_window import MainWindow
from ttrpg_recorder.gui.state import AppState

R = 48_000
IDENTITY = {"id": "fake", "revision": "r1", "device": "cpu", "compute_type": "int8"}


class FakeSink:
    started: ClassVar[list] = []

    def __init__(self, rate):
        self.rate = rate

    def start(self, device):
        FakeSink.started.append((self.rate, device.size()))

    def stop(self):
        pass


def recordings(folder, n=3, seconds=8.0):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(0.8 * R)) / R
    for i in range(n):
        x = np.zeros(int(seconds * R), np.float32)
        for start in (1.0 + i, 5.0 + 0.2 * i):
            a = int(start * R)
            x[a : a + len(t)] = 0.4 * np.sin(2 * np.pi * (400 + 50 * i) * t)
        sf.write(str(folder / f"ch{i + 1:02d}_p{i}.wav"), x, R, subtype="FLOAT")
    return folder


def window(qtbot, tmp_path, factory=None, log=None):
    state = AppState(
        {
            "isolation": "inline",
            "factory": factory or FakeFactory(calls=log),
            "model_identity": lambda s: IDENTITY,
        }
    )
    prefs = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    w = MainWindow(state, prefs, PreviewPlayer(sink_factory=FakeSink))
    qtbot.addWidget(w)
    w.show()
    return w, state


def test_window_starts_with_empty_states(qtbot, tmp_path):
    w, _ = window(qtbot, tmp_path)
    assert w.tabs.count() == 5
    assert w.session_tab.stack.currentIndex() == 0  # empty state
    assert w.result_tab.stack.currentIndex() == 0
    assert not w.run_tab.run_btn.isEnabled()


def test_workflow_a_with_review_and_reassignment(qtbot, tmp_path):
    log = tmp_path / "calls.txt"
    w, state = window(qtbot, tmp_path, log=log)
    folder = recordings(tmp_path / "rec")
    w.session_tab.new_from_folder(folder)
    assert w.session_tab.table.rowCount() == 3
    w.session_tab.name.setText("Próba")
    assert w.session_tab.save()
    assert Session.load(folder).config.name == "Próba"

    # Basic setting: language, saved into the session's [settings]
    lang = w.settings_tab.editors["basic:decoding.language"]
    w.settings_tab._changed(lang.spec, "hu")
    w.settings_tab.save_to_session()
    assert Session.load(folder).config.settings == {"decoding": {"language": "hu"}}

    w.run_tab.run_btn.click()
    qtbot.waitUntil(lambda: state.result is not None and not w.run_tab.running, timeout=20000)
    assert w.tabs.currentWidget() is w.result_tab
    calls = log.read_text(encoding="utf-8")
    rows = w.result_tab.model.rows
    assert rows and all(r.kind in ("segment", "missed", "gap") for r in rows)

    # filters
    w.result_tab.search.setText("ő0")
    assert all("ő0" in r.text for r in w.result_tab.model.rows)
    w.result_tab.search.setText("")

    # playback through the injected sink
    assert w.result_tab.select_id(rows[0].id)
    FakeSink.started.clear()
    w.result_tab.play_btn.click()
    assert FakeSink.started and FakeSink.started[0][0] == R

    # reassign the first segment to another participant; re-export without ASR
    target = rows[0]
    other = next(p.id for p in state.session.participants if p.id != target.speaker)
    w.result_tab.target.setCurrentIndex(w.result_tab.target.findData(other))
    w.result_tab.reassign_btn.click()
    qtbot.waitUntil(
        lambda: (
            not w.run_tab.running
            and any(r.id == target.id and r.speaker == other for r in w.result_tab.model.rows)
        ),
        timeout=20000,
    )
    moved = next(r for r in w.result_tab.model.rows if r.id == target.id)
    assert moved.reassigned_from == target.speaker and "reassigned" in moved.flags
    assert Session.load(folder).config.reassignments[0].to == other
    assert log.read_text(encoding="utf-8") == calls  # no ASR for the re-export
    w.result_tab.select_id(target.id)
    w.result_tab.undo_btn.click()
    qtbot.waitUntil(
        lambda: (
            not w.run_tab.running
            and any(
                r.id == target.id and r.speaker == target.speaker for r in w.result_tab.model.rows
            )
        ),
        timeout=20000,
    )
    assert Session.load(folder).config.reassignments == []


def test_cancel_and_resume(qtbot, tmp_path):
    w, state = window(qtbot, tmp_path, factory=SlowFactory(0.4))
    folder = recordings(tmp_path / "rec", n=3, seconds=12.0)
    w.session_tab.new_from_folder(folder)
    w.session_tab.save()
    state.set_settings(
        state.settings.model_copy(
            update={"asr": state.settings.asr.model_copy(update={"chunk_owned_s": 4.0})}
        )
    )
    w.run_tab.run_btn.click()
    qtbot.waitUntil(lambda: w.run_tab.progress.value() > 0, timeout=20000)
    w.run_tab.cancel_btn.click()
    qtbot.waitUntil(lambda: not w.run_tab.running, timeout=20000)
    texts = [w.run_tab.log.item(i).text() for i in range(w.run_tab.log.count())]
    assert any(t.startswith("Cancelled") for t in texts)
    assert state.result is None
    w.run_tab.run_btn.click()
    qtbot.waitUntil(lambda: state.result is not None and not w.run_tab.running, timeout=30000)
    done = w.run_tab.log.item(w.run_tab.log.count() - 1).text()
    assert "from cache" in done and not done.startswith("Done: 9 chunk(s), 0 from cache")


def test_settings_are_mode_aware(qtbot, tmp_path):
    w, state = window(qtbot, tmp_path)
    best_of = w.settings_tab.editors["decoding.best_of"]
    assert best_of.control.isEnabled()
    mode = w.settings_tab.editors["basic:asr.mode"]
    w.settings_tab._changed(mode.spec, "fast")
    assert not best_of.control.isEnabled()
    assert best_of.note.text() == "no effect in this mode"
    temp = w.settings_tab.editors["decoding.temperature"]
    assert temp.note.text() == "only the first value is used in this mode"
    beam = w.settings_tab.editors["decoding.beam_size"]
    beam.control.setText("0")
    beam._text_changed(beam.control)
    assert state.settings.decoding.beam_size == 5  # rejected
    assert beam.note.text()


def test_theme_switch_rebuilds_the_window(qtbot, tmp_path):
    w, _ = window(qtbot, tmp_path)
    w.set_preference("theme", "dark")
    assert w.tokens.background == "#14181d" and w.tabs.count() == 5
    w.set_preference("theme", "light")
    assert w.tokens.background == "#f4f6f8"


def test_superseded_export_and_close_leave_no_running_thread(qtbot, tmp_path):
    from ttrpg_recorder.gui import workers

    w, state = window(qtbot, tmp_path)
    first = recordings(tmp_path / "a", n=2)
    second = recordings(tmp_path / "b", n=3)
    for folder in (first, second):
        w.session_tab.new_from_folder(folder)
        assert w.session_tab.save()
        w.run_tab.run_btn.click()
        qtbot.waitUntil(lambda: not w.run_tab.running and workers.running() == 0, timeout=20000)
    w.open_path(first / "session.toml")
    w.open_path(second / "session.toml")  # replaces the running export of the first
    qtbot.waitUntil(lambda: workers.running() == 0, timeout=20000)
    qtbot.wait(50)
    assert state.result is not None
    assert {p.id for p in state.result.participants} == {
        p.id for p in Session.load(second).participants
    }
    w.close()
    assert workers.running() == 0


def test_file_buttons_open_their_dialogs(qtbot, tmp_path, monkeypatch):
    """Clicked through the real buttons: clicked(bool) must not reach the path argument."""
    from PySide6.QtWidgets import QFileDialog

    from ttrpg_recorder.gui.tabs import settings_tab

    w, state = window(qtbot, tmp_path)
    folder = recordings(tmp_path / "rec", n=2)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    w.session_tab.new_btn.click()
    assert state.session is not None and state.session.folder == folder
    assert w.session_tab.save()

    state.set_session(None)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (str(folder / "session.toml"), "")
    )
    w.session_tab.open_btn.click()
    assert state.session is not None and state.session.folder == folder

    preset = tmp_path / "presets" / "mine.toml"
    monkeypatch.setattr(settings_tab, "user_presets_dir", lambda: tmp_path / "presets")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(preset), ""))
    w.settings_tab.save_preset_btn.click()
    assert preset.is_file()

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    w.session_tab.new_btn.click()  # dialog cancelled: nothing changes
    assert state.session.folder == folder


def test_closing_during_a_run_does_not_hang(qtbot, tmp_path):
    import time

    from ttrpg_recorder.gui import workers

    w, _ = window(qtbot, tmp_path, factory=SlowFactory(0.4))
    w.session_tab.new_from_folder(recordings(tmp_path / "rec", n=3, seconds=12.0))
    w.session_tab.save()
    w.run_tab.run_btn.click()
    qtbot.waitUntil(lambda: w.run_tab.progress.value() > 0, timeout=20000)
    t0 = time.monotonic()
    w.close()  # cancels; completed chunks stay cached
    assert time.monotonic() - t0 < 10 and workers.running() == 0


def test_every_advanced_setting_has_a_title_and_its_toml_key(qtbot, tmp_path):
    from PySide6.QtWidgets import QLabel

    from ttrpg_recorder.gui.models.settings_form import field_specs

    w, _ = window(qtbot, tmp_path)
    titles = w.settings_tab.field_titles()
    assert set(titles) == {s.path for s in field_specs()}
    captions = {
        lbl.text() for lbl in w.settings_tab.findChildren(QLabel) if lbl.property("caption") is True
    }
    assert "model.compute_type" in captions and len(captions) == len(titles)


def test_tooltips_are_the_core_descriptions_and_go_through_tr(qtbot, tmp_path):
    from ttrpg_recorder.gui.models.settings_form import field_specs

    w, _ = window(qtbot, tmp_path)
    tips = w.settings_tab.field_tips()
    described = {s.path: s.description for s in field_specs() if s.description}
    assert tips == described  # English source = core.config description, nothing missing
    editor = w.settings_tab.editors["model.compute_type"]
    assert editor.control.toolTip() == tips["model.compute_type"]


def test_about_shows_versions_and_opens_the_local_notices(qtbot, tmp_path):
    from ttrpg_recorder import __version__

    w, _ = window(qtbot, tmp_path)
    dialog = w.about()
    qtbot.addWidget(dialog)
    assert "ctranslate2" in dialog.versions.text() and "PySide6" in dialog.versions.text()
    assert __version__ in dialog.windowTitle() or any(
        __version__ in lbl.text() for lbl in dialog.findChildren(type(dialog.versions))
    )
    opened = []
    dialog.open_url = lambda url: opened.append(url) or True
    dialog.notices_btn.click()
    assert opened and opened[0].isLocalFile()
    assert opened[0].toLocalFile().endswith("THIRD_PARTY_NOTICES.md")
    dialog.accept()
