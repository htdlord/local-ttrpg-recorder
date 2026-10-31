"""Files through which the GUI follows and steers a detached recorder process.

Everything lives in ``<session>/rec/control/``; there is no pipe, so a GUI that crashes
or is closed cannot stop a recording, and a new GUI can reconnect:

- ``status.json``: written atomically about 4 times a second by the recorder (state,
  heartbeat, levels, buffer, disk, gaps, the current mic-check prompt, command ack);
- ``events.jsonl``: append-only alarms, prompts, the mic-check result and the end, each
  with a sequence number; rotated to ``events.1.jsonl`` above a size limit, so nobody
  has to read it;
- ``command.json``: written by the GUI (``{"id": n, "command": "stop"}``); the recorder
  polls it and acknowledges ``id`` in the status;
- ``recorder.lock``: pid, start time and session of the running recorder;
- ``child.log``: the recorder process's own output.

Readers open, read and close at once (no handle is kept), so the writer's renames
never collide with them for long; ``fileio`` retries a blocked rename on Windows.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ttrpg_recorder.core.fileio import _replace, atomic_write_json

CONTROL_DIR = Path("rec") / "control"
STATUS_FILE = "status.json"
EVENTS_FILE = "events.jsonl"
EVENTS_OLD = "events.1.jsonl"
COMMAND_FILE = "command.json"
LOCK_FILE = "recorder.lock"
LOG_FILE = "child.log"
EVENTS_MAX_BYTES = 1_000_000
HEARTBEAT_STALE_S = 5.0
FINAL_STATES = frozenset({"stopped", "failed"})


def control_dir(session_dir: Path) -> Path:
    return Path(session_dir) / CONTROL_DIR


# --- processes --------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """True while ``pid`` runs (a zombie on Linux counts as ended)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


# --- lock --------------------------------------------------------------------------------


class LockedError(RuntimeError):
    """Another recorder process is running for this session."""


def acquire_lock(folder: Path, session_dir: Path) -> Path:
    """Create ``recorder.lock`` for this process; a lock of a dead process is replaced."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOCK_FILE
    data = {"pid": os.getpid(), "started": time.time(), "session": str(session_dir)}
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            old = read_lock(folder)
            if old is not None and pid_alive(int(old.get("pid", 0))):
                raise LockedError(f"a recorder (pid {old['pid']}) is already running") from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path
    raise LockedError(f"cannot create {path}")


def read_lock(folder: Path) -> dict[str, Any] | None:
    return _read_json(folder / LOCK_FILE)


def release_lock(folder: Path) -> None:
    lock = read_lock(folder)
    if lock is not None and int(lock.get("pid", 0)) == os.getpid():
        (folder / LOCK_FILE).unlink(missing_ok=True)


# --- status, commands --------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_status(folder: Path, status: dict[str, Any]) -> None:
    atomic_write_json(folder / STATUS_FILE, status)


def read_status(folder: Path) -> dict[str, Any] | None:
    return _read_json(folder / STATUS_FILE)


def send_command(folder: Path, command: str) -> int:
    """Write a command for the recorder; returns its id (acknowledged in the status)."""
    previous = _read_json(folder / COMMAND_FILE) or {}
    status = read_status(folder) or {}
    cid = max(int(previous.get("id", 0)), int(status.get("ack", 0))) + 1
    atomic_write_json(folder / COMMAND_FILE, {"id": cid, "command": command, "sent": time.time()})
    return cid


def read_command(folder: Path, after: int) -> tuple[int, str] | None:
    data = _read_json(folder / COMMAND_FILE)
    if data is None or int(data.get("id", 0)) <= after:
        return None
    return int(data["id"]), str(data.get("command", ""))


# --- events ------------------------------------------------------------------------------


class EventLog:
    """Append-only JSON lines with sequence numbers, rotated above ``max_bytes``."""

    def __init__(self, folder: Path, max_bytes: int = EVENTS_MAX_BYTES) -> None:
        self.folder = folder
        self.max_bytes = max_bytes
        folder.mkdir(parents=True, exist_ok=True)
        self.seq = self._last_seq()
        # alarms (writer thread), prompts (mic-check thread) and the main thread append;
        # appends from two handles are not atomic on Windows
        self._lock = threading.Lock()

    def _last_seq(self) -> int:
        last = 0
        for name in (EVENTS_OLD, EVENTS_FILE):
            for event in _read_lines(self.folder / name):
                last = max(last, int(event.get("seq", 0)))
        return last

    def append(self, event_type: str, /, **fields: Any) -> int:
        with self._lock:
            self.seq += 1
            line = json.dumps({"seq": self.seq, "type": event_type, "wall": time.time(), **fields})
            path = self.folder / EVENTS_FILE
            try:
                if path.stat().st_size + len(line) > self.max_bytes:
                    _replace(str(path), self.folder / EVENTS_OLD)
            except FileNotFoundError:
                pass
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return self.seq


def _read_lines(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.split("\n")[:-1]:  # the last piece may be a line still being written
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


@dataclass
class EventReader:
    """Events newer than the last one seen, across a rotation."""

    folder: Path
    last_seq: int = 0

    def poll(self) -> list[dict[str, Any]]:
        events = _read_lines(self.folder / EVENTS_FILE)
        if not events or int(events[0].get("seq", 0)) > self.last_seq + 1:
            events = _read_lines(self.folder / EVENTS_OLD) + events  # rotated meanwhile
        new = [e for e in events if int(e.get("seq", 0)) > self.last_seq]
        new.sort(key=lambda e: int(e["seq"]))
        if new:
            self.last_seq = int(new[-1]["seq"])
        return new


# --- what a (new) GUI finds --------------------------------------------------------------

LinkState = Literal["none", "running", "unresponsive", "crashed", "ended"]


@dataclass
class Link:
    state: LinkState
    status: dict[str, Any] = field(default_factory=dict)
    pid: int | None = None

    @property
    def heartbeat_age(self) -> float | None:
        beat = self.status.get("heartbeat")
        return None if beat is None else max(0.0, time.time() - float(beat))


def probe(session_dir: Path, now: float | None = None) -> Link:
    """The recorder of a session as seen from outside.

    - ``running``: the lock's process lives and the heartbeat is fresh;
    - ``unresponsive``: it lives but the heartbeat is older than 5 s;
    - ``crashed``: a lock (or a non-final status) whose process is gone, so the
      recording ended without a clean stop: offer ``record recover``;
    - ``ended``: a final status and no lock; ``none``: never recorded from the GUI.
    """
    folder = control_dir(session_dir)
    status = read_status(folder) or {}
    lock = read_lock(folder)
    now = time.time() if now is None else now
    pid = int(lock["pid"]) if lock and "pid" in lock else status.get("pid")
    alive = pid is not None and pid_alive(int(pid))
    if lock is not None or (status and status.get("state") not in FINAL_STATES):
        if not alive:
            return Link("crashed" if status or lock else "none", status, pid)
        beat = status.get("heartbeat")
        if beat is not None and now - float(beat) > HEARTBEAT_STALE_S:
            return Link("unresponsive", status, pid)
        return Link("running", status, pid)
    if status:
        return Link("ended", status, pid)
    return Link("none", status, None)


def mark_recovered(session_dir: Path) -> None:
    """After ``record recover``: drop a dead recorder's lock and close its status, so the
    session reads as ended instead of crashed."""
    folder = control_dir(session_dir)
    lock = read_lock(folder)
    if lock is not None and not pid_alive(int(lock.get("pid", 0))):
        (folder / LOCK_FILE).unlink(missing_ok=True)
    status = read_status(folder) or {}
    if status.get("state") not in FINAL_STATES:
        write_status(folder, {**status, "state": "failed", "error": "recovered after a crash"})


# --- starting the recorder ---------------------------------------------------------------


def query_devices(timeout_s: float = 30.0) -> list[dict[str, Any]]:
    """Input devices as listed by a short-lived recorder process (name, host_api,
    channels, samplerate); raises RuntimeError with its output when it fails."""
    import subprocess

    done = subprocess.run(
        [sys.executable, "-m", "ttrpg_recorder.core.audio.record_child", "--list-devices"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip()[-500:] or "device query failed")
    return list(json.loads(done.stdout.strip().splitlines()[-1]))


def spawn_recorder(
    session_dir: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> Any:
    """Start ``python -m ttrpg_recorder.core.audio.record_child --session S <args>``,
    detached from the caller (own process group / session, no console, no pipes)."""
    import subprocess

    folder = control_dir(session_dir)
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / LOG_FILE).open("ab")
    command = [
        sys.executable,
        "-m",
        "ttrpg_recorder.core.audio.record_child",
        "--session",
        str(session_dir),
        *args,
    ]
    common: dict[str, Any] = dict(
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        close_fds=True,
    )
    try:
        if sys.platform != "win32":
            return subprocess.Popen(command, start_new_session=True, **common)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:
            # Leave a job object the GUI may run in (launchers kill their job on exit).
            return subprocess.Popen(
                command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **common
            )
        except OSError:  # the job does not allow breaking away
            return subprocess.Popen(command, creationflags=flags, **common)
    finally:
        log.close()
