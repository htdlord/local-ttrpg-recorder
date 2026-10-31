"""Record tab end to end: a real detached recorder process with the fake device."""

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from tests.test_gui import FakeSink
from tests.test_record_child import ROOT, kill, rig, status
from ttrpg_recorder.core.audio import control
from ttrpg_recorder.core.audio.recovery import verify
from ttrpg_recorder.gui.audio_preview import PreviewPlayer
from ttrpg_recorder.gui.main_window import MainWindow
from ttrpg_recorder.gui.models import recording as rm
from ttrpg_recorder.gui.state import AppState

DEVICES = [{"name": "Fake UMC", "host_api": "ASIO", "channels": 8, "samplerate": 48000}]


@pytest.fixture
def fake_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("TTRPG_RECORDER_TEST_OPENER", "tests.fake_stream:child_opener")
    monkeypatch.setenv("TTRPG_RECORDER_TEST_DEVICES", json.dumps(DEVICES))
    # the fake voices follow the recorder's default mic-check timing (5 s quiet, 5 s turns)
    monkeypatch.setenv("TTRPG_RECORDER_TEST_TURN_S", "5")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.chdir(ROOT)
    yield
    # a failed test must not leave its recorder running into the next tests
    pid = status(tmp_path / "session").get("pid")
    if pid and control.pid_alive(int(pid)):
        kill(int(pid))


def window(qtbot, tmp_path, scenario="steady", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("TTRPG_RECORDER_TEST_SCENARIO", scenario)
    prefs = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    w = MainWindow(AppState(), prefs, PreviewPlayer(sink_factory=FakeSink))
    w.confirm_close = lambda: True
    qtbot.addWidget(w)
    w.show()
    return w


def setup_tab(w, tmp_path, *, check=False):
    tab = w.record_tab
    w.tabs.setCurrentWidget(tab)
    assert tab.load_rig(rig(tmp_path))
    tab.folder.setText(str(tmp_path / "session"))
    tab.check_box.setChecked(check)
    tab.hours.setValue(0.5)
    return tab


def test_rig_rows_and_arguments(tmp_path):
    rows = [rm.RigRow("1", "gm", "Mesélő"), rm.RigRow("", "", ""), rm.RigRow("x", "anna", "")]
    bad = rm.build_rig("UMC", "asio", rows)
    assert not bad.ok and "row 3" in bad.errors[0]
    good = rm.build_rig("UMC", "asio", rows[:2])
    assert good.ok and [c.id for c in good.rig.channels] == ["gm"]
    assert rm.child_args(tmp_path / "r.toml", hours=8, check=False, force=True)[2:] == [
        "--hours",
        "8",
        "--no-check",
        "--force",
    ]
    m = rm.meters({"participants": ["gm"], "levels": {"peak": [-6.0], "rms": [-90.0]}})[0]
    assert m.fraction(m.peak_db) == pytest.approx(0.9) and m.fraction(m.rms_db) == 0.0
    assert rm.alarm_severity({"kind": "device_lost"}) == "error"
    assert rm.alarm_severity({"kind": "reconnected"}) == "info"
    assert rm.format_clock(3725.4) == "1:02:05"


def test_devices_come_from_a_short_lived_process(qtbot, tmp_path, fake_audio, monkeypatch):
    w = window(qtbot, tmp_path, monkeypatch=monkeypatch)
    tab = w.record_tab
    tab.refresh_btn.click()
    qtbot.waitUntil(lambda: tab.device.count() == 1, timeout=30000)
    tab.device.setCurrentIndex(0)
    tab._device_chosen()
    assert tab.device.currentText() == "Fake UMC" and tab.host_api.text() == "asio"


def test_record_meters_stop_and_open_the_session(qtbot, tmp_path, fake_audio, monkeypatch):
    w = window(qtbot, tmp_path, monkeypatch=monkeypatch)
    tab = setup_tab(w, tmp_path)
    assert tab.start_recording()
    qtbot.waitUntil(lambda: len(tab.meter_widgets) == 3, timeout=30000)
    qtbot.waitUntil(lambda: tab.meter_widgets[0][1].rms > 0.5, timeout=10000)
    assert not tab.start_btn.isEnabled() and tab.stop_btn.isEnabled()
    tab.stop_btn.click()
    qtbot.waitUntil(lambda: w.state.session is not None, timeout=30000)
    assert w.tabs.currentWidget() is w.session_tab
    assert [p.id for p in w.state.session.participants] == ["gm", "anna", "bela"]
    assert verify(tmp_path / "session").ok


def test_mic_check_prompt_and_results(qtbot, tmp_path, fake_audio, monkeypatch):
    w = window(qtbot, tmp_path, scenario="mic_dead2", monkeypatch=monkeypatch)
    tab = setup_tab(w, tmp_path, check=True)
    tab.start_recording()
    qtbot.waitUntil(lambda: "Mesélő" in tab.prompt.text() and tab.prompt.isVisible(), timeout=30000)
    qtbot.waitUntil(lambda: tab.check_table.rowCount() == 3, timeout=60000)
    assert tab.check_table.item(1, 0).text() == "Anna"  # names, not ids
    verdicts = [tab.check_table.item(r, 1).text() for r in range(3)]
    assert verdicts == ["OK", "Failed", "OK"]  # only the dead mic
    # a failed check stops the recording by itself (no force)
    qtbot.waitUntil(lambda: tab.link_state == "ended", timeout=30000)
    texts = [tab.alarm_list.item(i).text() for i in range(tab.alarm_list.count())]
    assert any("mic check failed" in t for t in texts)


def test_closing_the_window_keeps_recording_and_a_new_window_reconnects(
    qtbot, tmp_path, fake_audio, monkeypatch
):
    w = window(qtbot, tmp_path, monkeypatch=monkeypatch)
    tab = setup_tab(w, tmp_path)
    tab.start_recording()
    session = tmp_path / "session"
    qtbot.waitUntil(lambda: status(session).get("time_s", 0) > 0.5, timeout=30000)
    asked = []
    w.confirm_close = lambda: asked.append(1) or True
    w.close()
    assert asked  # recording: the window asked first
    child = status(session)["pid"]
    assert control.pid_alive(child)

    again = window(qtbot, tmp_path, monkeypatch=monkeypatch)  # same preferences
    assert again.tabs.currentWidget() is again.record_tab
    assert again.record_tab.recording and again.record_tab.session_dir == session.resolve()
    again.record_tab.stop_btn.click()
    qtbot.waitUntil(lambda: not control.pid_alive(child), timeout=30000)
    manifest_ok = verify(session)
    assert manifest_ok.ok and manifest_ok.gaps == 0


def test_a_crashed_recorder_is_offered_recovery(qtbot, tmp_path, fake_audio, monkeypatch):
    w = window(qtbot, tmp_path, monkeypatch=monkeypatch)
    tab = setup_tab(w, tmp_path)
    tab.start_recording()
    session = tmp_path / "session"
    qtbot.waitUntil(lambda: status(session).get("time_s", 0) > 0.5, timeout=30000)
    kill(status(session)["pid"])
    qtbot.waitUntil(lambda: tab.link_state == "crashed", timeout=30000)
    assert tab.recover_btn.isVisible() and tab.banner.isVisible()
    tab.recover_btn.click()
    assert tab.link_state == "ended" and "Recovered" in tab.banner.text()
    assert control.probe(session).state == "ended"
