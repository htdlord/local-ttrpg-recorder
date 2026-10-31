"""The recorder as a detached process for the GUI: ``python -m
ttrpg_recorder.core.audio.record_child --session S --rig R [options]``.

It records exactly like ``record start`` (stream on the main thread for ASIO, guided
mic check inside the recording) and talks only through the files of
``core.audio.control``. It never waits for a reader: status is overwritten, events are
appended and rotated, so a closed or crashed GUI does not affect the recording.

Exit codes: 0 stopped, 1 failed, 3 another recorder holds the session.

Tests replace the audio device with ``TTRPG_RECORDER_TEST_OPENER=module:function``
(``function(setup) -> opener``); it is ignored unless set.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ttrpg_recorder.core.audio import control

log = logging.getLogger(__name__)
STATUS_EVERY_S = 0.25
TEST_OPENER_ENV = "TTRPG_RECORDER_TEST_OPENER"
TEST_DEVICES_ENV = "TTRPG_RECORDER_TEST_DEVICES"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="record_child")
    p.add_argument("--list-devices", action="store_true", help="Print input devices as JSON.")
    p.add_argument("--session", type=Path)
    p.add_argument("--rig", type=Path)
    p.add_argument("--device")
    p.add_argument("--host-api")
    p.add_argument("--channels")
    p.add_argument("--names")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--noise-s", type=float, default=5.0)
    p.add_argument("--turn-s", type=float, default=5.0)
    p.add_argument("--seconds", type=float)
    p.add_argument("--events-max-bytes", type=int, default=control.EVENTS_MAX_BYTES)
    return p.parse_args(argv)


def _opener(setup: Any) -> Any:
    spec = os.environ.get(TEST_OPENER_ENV)
    if spec:
        module, _, name = spec.partition(":")
        return getattr(importlib.import_module(module), name)(setup)
    from ttrpg_recorder.core.audio.guided import open_device

    return open_device(setup)


class Reporter:
    """Status (heartbeat thread) and events for one recording; polls commands."""

    def __init__(self, folder: Path, events: control.EventLog, stop: threading.Event) -> None:
        self.folder = folder
        self.events = events
        self.stop = stop
        self.lock = threading.Lock()
        self.status: dict[str, Any] = {
            "pid": os.getpid(),
            "state": "starting",
            "started": time.time(),
            "ack": 0,
        }
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="status", daemon=True)

    def start(self) -> None:
        self.write()
        self._thread.start()

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.status.update(fields)

    def write(self) -> None:
        with self.lock:
            self.status["heartbeat"] = time.time()
            snapshot = dict(self.status)
        try:
            control.write_status(self.folder, snapshot)
        except OSError:  # a reader held the file too long; the next tick writes again
            log.warning("status write failed", exc_info=True)

    def _loop(self) -> None:
        while not self._done.wait(STATUS_EVERY_S):
            command = control.read_command(self.folder, int(self.status["ack"]))
            if command is not None:
                cid, name = command
                if name == "stop":
                    self.event("command", command=name)
                    self.stop.set()
                    self.update(state="stopping")
                self.update(ack=cid)
            self.write()

    def finish(self, **fields: Any) -> None:
        self._done.set()
        self._thread.join(timeout=2)
        self.update(**fields)
        self.write()

    def event(self, event_type: str, /, **fields: Any) -> None:
        try:
            self.events.append(event_type, **fields)
        except OSError:
            log.warning("event write failed", exc_info=True)


def list_devices() -> list[dict[str, Any]]:
    """Input devices for the GUI, which does not load PortAudio itself (an ASIO driver
    serves one client at a time)."""
    fake = os.environ.get(TEST_DEVICES_ENV)
    if fake:
        return list(json.loads(fake))
    import sys

    from ttrpg_recorder.core.audio.backend import list_input_devices, load_sounddevice

    sd = load_sounddevice(enable_asio=sys.platform == "win32")
    return [
        {
            "name": d.name,
            "host_api": d.host_api,
            "channels": d.max_input_channels,
            "samplerate": d.default_samplerate,
        }
        for d in list_input_devices(sd)
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.list_devices:
        print(json.dumps(list_devices()))
        return 0
    if args.session is None:
        log.error("--session is required")
        return 2
    session = args.session.resolve()
    folder = control.control_dir(session)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        control.acquire_lock(folder, session)
    except control.LockedError as exc:
        log.error("%s", exc)  # the running recorder's status is left alone
        return 3
    stop = threading.Event()
    reporter = Reporter(folder, control.EventLog(folder, args.events_max_bytes), stop)
    reporter.start()
    try:
        return _record(args, session, reporter, stop)
    finally:
        control.release_lock(folder)


def _record(
    args: argparse.Namespace, session: Path, reporter: Reporter, stop: threading.Event
) -> int:
    from ttrpg_recorder.core.audio.backend import AudioSetupError
    from ttrpg_recorder.core.audio.guided import (
        MicCheckResult,
        Prompt,
        SetupError,
        check_in_recording,
        recorder_config,
        resolve_setup,
    )
    from ttrpg_recorder.core.audio.power import inhibit_sleep
    from ttrpg_recorder.core.audio.recorder import (
        LevelEvent,
        Recorder,
        RecorderAlarm,
        RecorderError,
        RecorderStatus,
    )

    # Load now what the recording may import later: an update of the installed package
    # during a recording must not break the stop (INSTALL.md still says not to update).
    importlib.import_module("ttrpg_recorder.core.session")
    importlib.import_module("ttrpg_recorder.core.audio.recovery")

    def failed(message: str) -> int:
        reporter.event("failed", message=message)
        reporter.finish(state="failed", error=message, prompt=None)
        return 1

    try:
        setup = resolve_setup(args.rig, args.device, args.host_api, args.channels, args.names)
        opener = _opener(setup)
    except (SetupError, AudioSetupError, OSError, ValueError) as exc:
        return failed(str(exc))
    reporter.update(
        device=opener.device_name,
        host_api=opener.host_api,
        channels=list(setup.channels),
        participants=list(setup.ids),
        names=list(setup.names),
    )
    check_events: list[LevelEvent] = []
    checking = threading.Event()
    if not args.no_check:
        checking.set()

    def sink(event: Any) -> None:
        if isinstance(event, LevelEvent):
            if checking.is_set():
                check_events.append(event)
            reporter.update(
                time_s=round(event.time_s, 2),  # the 4 Hz clock (status events are rarer)
                levels={
                    "time_s": event.time_s,
                    "peak": [round(v, 1) for v in event.peak_dbfs],
                    "rms": [round(v, 1) for v in event.rms_dbfs],
                    "clipped": list(event.clipped),
                },
            )
        elif isinstance(event, RecorderAlarm):
            reporter.event("alarm", **asdict(event))
        elif isinstance(event, RecorderStatus):
            reporter.update(
                state="stopping"
                if stop.is_set()
                else "mic_check"
                if checking.is_set()
                else "recording",
                time_s=round(event.time_s, 2),
                frames=event.frames,
                buffer_fill=round(event.buffer_fill, 4),
                disk_left_s=event.disk_left_s,
                gaps=event.gaps,
            )

    def inhibit() -> Any:
        inh = inhibit_sleep()
        if not inh.active:
            reporter.event("alarm", kind="no_sleep_inhibit", message=inh.how, fatal=False)
        return inh

    try:
        recorder = Recorder(
            recorder_config(setup, session, args.hours), opener, sink=sink, inhibit=inhibit
        )
        recorder.check_disk()
    except RecorderError as exc:
        return failed(str(exc))

    def on_prompt(prompt: Prompt) -> None:
        fields = asdict(prompt)
        reporter.event("prompt", **fields)
        reporter.update(prompt=None if prompt.kind == "stalled" else fields)

    def on_result(result: MicCheckResult) -> None:
        checking.clear()
        checks = [
            {
                "channel": c.channel,
                "participant": c.participant,
                "verdict": c.verdict,
                "rise_db": round(c.rise_db, 1),
                "speech_dbfs": round(c.speech_dbfs, 1),
                "alarms": [{"kind": k, "message": m} for k, m in c.alarms],
            }
            for c in result.checks
        ]
        reporter.event(
            "mic_check",
            verdict=result.verdict,
            checks=checks,
            error=result.error,
            stopping=result.stopping,
        )
        reporter.update(prompt=None, mic_check=result.verdict)

    if checking.is_set():
        reporter.update(state="mic_check")
        threading.Thread(
            target=check_in_recording,
            args=(recorder, lambda: list(check_events), setup, args.noise_s, args.turn_s, stop),
            kwargs={"force": args.force, "on_prompt": on_prompt, "on_result": on_result},
            name="mic-check",
            daemon=True,
        ).start()
    reporter.event("started", device=opener.device_name, channels=list(setup.channels))
    try:
        manifest = recorder.run(stop, max_seconds=args.seconds)  # main thread (ASIO)
    except (AudioSetupError, RecorderError) as exc:
        stop.set()
        hint = ""
        if "ASIO" in str(exc):
            hint = " Is another program using the ASIO driver? Close it and try again."
        return failed(f"{exc}{hint}")
    m = manifest.metrics
    summary = {
        "state": manifest.state,
        "seconds": round(m.frames / manifest.samplerate, 2),
        "segments": len(manifest.segments),
        "gaps": len(manifest.gaps),
        "dropped_frames": m.dropped_frames,
        "input_overflows": m.input_overflows,
        "max_buffer_fill": m.max_buffer_fill,
    }
    reporter.event("finished", **summary)
    reporter.finish(state="stopped", summary=summary, prompt=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
