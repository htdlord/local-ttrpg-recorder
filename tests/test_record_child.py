"""The detached recorder process: status/event files, stop command, mic check, and a
GUI that dies while the recording goes on (fake device, about real time)."""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from sessionscribe.core.audio import control
from sessionscribe.core.audio.manifest import RigChannel, RigConfig, write_rig
from sessionscribe.core.audio.recovery import recover, verify
from sessionscribe.core.session import Session

ROOT = Path(__file__).resolve().parents[1]


def rig(tmp_path) -> Path:
    path = tmp_path / "rig.toml"
    write_rig(
        path,
        RigConfig(
            device="Fake UMC",
            host_api="asio",
            channels=[
                RigChannel(channel=1, id="gm", name="Mesélő"),
                RigChannel(channel=4, id="anna", name="Anna"),
                RigChannel(channel=5, id="bela", name="Béla"),
            ],
            segment_min=0.05,
        ),
    )
    return path


def env(scenario="steady"):
    out = dict(os.environ)
    out["SESSIONSCRIBE_TEST_OPENER"] = "tests.fake_stream:child_opener"
    out["SESSIONSCRIBE_TEST_SCENARIO"] = scenario
    out["PYTHONPATH"] = str(ROOT)
    return out


def spawn(tmp_path, *args, scenario="steady"):
    session = tmp_path / "session"
    proc = control.spawn_recorder(
        session,
        ["--rig", str(rig(tmp_path)), "--hours", "0.1", "--noise-s", "1", "--turn-s", "1", *args],
        env=env(scenario),
        cwd=ROOT,
    )
    return session, proc


def wait_for(predicate, timeout=30.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def status(session):
    return control.read_status(control.control_dir(session)) or {}


def log_text(session):
    path = control.control_dir(session) / control.LOG_FILE
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def recording_for(session, seconds):
    wait_for(
        lambda: (
            status(session).get("state") == "recording"
            and status(session).get("time_s", 0) >= seconds
        ),
        what=f"{seconds} s recorded\n{log_text(session)}",
    )


def kill(pid):
    """Hard kill (no cleanup), like a crash."""
    import signal

    os.kill(
        pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
    )  # Windows: Terminate


def stop_and_wait(session, pid):
    control.send_command(control.control_dir(session), "stop")
    wait_for(lambda: not control.pid_alive(pid), what="recorder exit")
    return status(session)


def test_status_events_and_stop_command(tmp_path):
    session, proc = spawn(tmp_path, "--no-check")
    recording_for(session, 1.0)
    s = status(session)
    # on Windows a venv python is a launcher: the recorder's own pid is in status and lock
    assert s["pid"] == control.read_lock(control.control_dir(session))["pid"]
    assert s["participants"] == ["gm", "anna", "bela"]
    assert len(s["levels"]["peak"]) == 3 and s["buffer_fill"] < 0.5
    assert control.probe(session).state == "running"
    final = stop_and_wait(session, s["pid"])
    assert proc.wait(timeout=10) == 0
    assert final["state"] == "stopped" and final["summary"]["gaps"] == 0
    types = [e["type"] for e in control.EventReader(control.control_dir(session)).poll()]
    assert types[0] == "started" and "command" in types and types[-1] == "finished"
    assert verify(session).ok
    assert control.read_lock(control.control_dir(session)) is None
    assert control.probe(session).state == "ended"


def test_recording_goes_on_when_the_gui_dies_and_a_new_gui_reconnects(tmp_path):
    session = tmp_path / "session"
    gui = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from pathlib import Path
                from sessionscribe.core.audio import control
                control.spawn_recorder(
                    Path({str(session)!r}),
                    ["--rig", {str(rig(tmp_path))!r}, "--hours", "0.1", "--no-check"],
                    cwd=Path({str(ROOT)!r}),
                )
                time.sleep(120)
                """
            ),
        ],
        env=env(),
        cwd=ROOT,
    )
    recording_for(session, 1.0)
    child = status(session)["pid"]
    gui.kill()  # the GUI crashes (no clean shutdown)
    gui.wait(timeout=10)
    before = status(session)["time_s"]
    time.sleep(1.5)  # nobody reads the files meanwhile
    after = status(session)
    assert control.pid_alive(child) and after["time_s"] > before + 0.5
    # a new GUI finds the running recorder and stops it
    link = control.probe(session)
    assert link.state == "running" and link.pid == child
    final = stop_and_wait(session, child)
    assert final["state"] == "stopped"
    manifest = Session.load(session).config.recording
    assert manifest is not None and manifest.gaps == [] and verify(session).ok


def test_mic_check_passes_and_recording_continues(tmp_path):
    session, proc = spawn(tmp_path, scenario="mic_ok")
    seen = wait_for(
        lambda: [
            e
            for e in control.EventReader(control.control_dir(session)).poll()
            if e["type"] == "mic_check"
        ],
        what=f"mic check\n{log_text(session)}",
    )
    result = seen[0]
    assert result["verdict"] in ("PASS", "WARN") and not result["stopping"]
    prompts = [
        e for e in control.EventReader(control.control_dir(session)).poll() if e["type"] == "prompt"
    ]
    assert [p["kind"] for p in prompts] == ["noise", "turn", "turn", "turn"]
    assert [p["label"] for p in prompts[1:]] == ["Mesélő", "Anna", "Béla"]
    wait_for(lambda: status(session).get("state") == "recording", what="recording after check")
    stop_and_wait(session, status(session)["pid"])
    assert proc.wait(timeout=10) == 0
    assert set(Session.load(session).config.mic_check) == {"gm", "anna", "bela"}


def test_failed_mic_check_stops_by_itself(tmp_path):
    session, proc = spawn(tmp_path, scenario="mic_dead2")
    assert proc.wait(timeout=60) == 0, log_text(session)
    events = control.EventReader(control.control_dir(session)).poll()
    check = next(e for e in events if e["type"] == "mic_check")
    assert check["verdict"] == "FAIL" and check["stopping"]
    anna = next(c for c in check["checks"] if c["participant"] == "anna")
    assert anna["verdict"] == "FAIL" and anna["alarms"][0]["kind"] == "no_signal"
    assert status(session)["state"] == "stopped"


def test_a_second_recorder_is_refused_and_a_killed_one_is_recoverable(tmp_path):
    session, proc = spawn(tmp_path, "--no-check")
    recording_for(session, 0.5)
    _, second = spawn(tmp_path, "--no-check")
    assert second.wait(timeout=30) == 3
    pid = status(session)["pid"]
    assert control.read_lock(control.control_dir(session))["pid"] == pid  # untouched
    kill(pid)
    proc.wait(timeout=10)
    wait_for(lambda: not control.pid_alive(pid), what="killed recorder")
    assert control.probe(session).state == "crashed"
    recover(session)
    report = verify(session)
    # a segment opened just before the kill may hold no readable audio: reported, not hidden
    assert report.state == "interrupted" and min(report.frames_per_channel.values()) > 0
    assert all(s.ok or s.problem.startswith("no readable audio") for s in report.segments)
