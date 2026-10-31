"""Control files between the GUI and the recorder process (no audio, no processes)."""

import json
import os
import subprocess
import sys
import time

import pytest

from ttrpg_recorder.core.audio import control


def test_events_rotate_and_a_late_reader_misses_nothing_recent(tmp_path):
    log = control.EventLog(tmp_path, max_bytes=400)
    reader = control.EventReader(tmp_path)
    for i in range(5):
        log.append("alarm", kind="idle_channel", n=i)
    assert [e["n"] for e in reader.poll()] == list(range(5))
    for i in range(5, 30):  # nobody reads: several rotations
        log.append("alarm", kind="idle_channel", n=i)
    assert (tmp_path / control.EVENTS_OLD).is_file()
    assert (tmp_path / control.EVENTS_FILE).stat().st_size <= 400
    late = reader.poll()
    seqs = [e["seq"] for e in late]
    assert seqs == sorted(seqs) and late[-1]["n"] == 29 and late[-1]["type"] == "alarm"
    assert reader.poll() == []
    # a restarted writer continues the numbering
    assert control.EventLog(tmp_path, max_bytes=400).append("x") == 31


def test_a_half_written_line_waits_for_the_next_poll(tmp_path):
    log = control.EventLog(tmp_path)
    log.append("started")
    with (tmp_path / control.EVENTS_FILE).open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "type": "ala')
    reader = control.EventReader(tmp_path)
    assert [e["seq"] for e in reader.poll()] == [1]


def test_commands_are_numbered_and_acknowledged(tmp_path):
    assert control.read_command(tmp_path, 0) is None
    cid = control.send_command(tmp_path, "stop")
    assert control.read_command(tmp_path, 0) == (cid, "stop")
    control.write_status(tmp_path, {"ack": cid})
    assert control.read_command(tmp_path, cid) is None
    assert control.send_command(tmp_path, "stop") == cid + 1


def finished_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_pid_alive():
    assert control.pid_alive(os.getpid())
    assert not control.pid_alive(finished_pid())
    assert not control.pid_alive(0)


def test_lock_is_exclusive_but_a_dead_owner_is_replaced(tmp_path):
    folder = tmp_path / "c"
    control.acquire_lock(folder, tmp_path)
    with pytest.raises(control.LockedError):
        control.acquire_lock(folder, tmp_path)  # this process is alive
    (folder / control.LOCK_FILE).write_text(json.dumps({"pid": finished_pid()}))
    control.acquire_lock(folder, tmp_path)
    assert control.read_lock(folder)["pid"] == os.getpid()
    control.release_lock(folder)
    assert control.read_lock(folder) is None


def test_probe_states(tmp_path):
    folder = control.control_dir(tmp_path)
    assert control.probe(tmp_path).state == "none"
    folder.mkdir(parents=True)
    now = time.time()
    me = os.getpid()
    (folder / control.LOCK_FILE).write_text(json.dumps({"pid": me}))
    control.write_status(folder, {"pid": me, "state": "recording", "heartbeat": now})
    assert control.probe(tmp_path).state == "running"
    assert control.probe(tmp_path, now=now + 10).state == "unresponsive"
    dead = finished_pid()
    (folder / control.LOCK_FILE).write_text(json.dumps({"pid": dead}))
    control.write_status(folder, {"pid": dead, "state": "recording", "heartbeat": now})
    assert control.probe(tmp_path).state == "crashed"
    (folder / control.LOCK_FILE).unlink()
    assert control.probe(tmp_path).state == "crashed"  # status never reached a final state
    control.write_status(folder, {"pid": dead, "state": "stopped", "heartbeat": now})
    assert control.probe(tmp_path).state == "ended"


def test_appends_from_several_threads_keep_every_line(tmp_path):
    import threading

    log = control.EventLog(tmp_path)
    threads = [
        threading.Thread(target=lambda k=k: [log.append("t", k=k, i=i) for i in range(200)])
        for k in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events = control.EventReader(tmp_path).poll()
    assert len(events) == 800 and [e["seq"] for e in events] == list(range(1, 801))
