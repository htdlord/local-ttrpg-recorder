"""``sessionscribe record``: devices, rig profiles, meters, mic check, recording, recovery.

The device is always named explicitly (``--device`` + ``--host-api`` or a ``rig.toml``);
nothing is auto-selected (SPEC §8.1). Explicit flags override the rig.
"""

from __future__ import annotations

import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

record_app = typer.Typer(help="Record a session (multi-channel interface).", no_args_is_help=True)
rig_app = typer.Typer(help="Saved recording setups (rig.toml).", no_args_is_help=True)
record_app.add_typer(rig_app, name="rig")

RigOpt = Annotated[
    Path | None, typer.Option("--rig", help="Recording profile (rig.toml).", exists=True)
]
DeviceOpt = Annotated[str | None, typer.Option(help="Device name (or a unique part of it).")]
HostApiOpt = Annotated[str | None, typer.Option("--host-api", help="asio, wasapi, alsa, ...")]
ChannelsOpt = Annotated[str | None, typer.Option(help="Interface channels, e.g. 1-8 or 1,3,5.")]
NamesOpt = Annotated[
    str | None, typer.Option(help="Participant ids per channel, comma-separated (gm,anna,...).")
]


def _badge(status: str) -> str:
    colours = {
        "OK": typer.colors.GREEN,
        "PASS": typer.colors.GREEN,
        "WARN": typer.colors.YELLOW,
        "FAIL": typer.colors.RED,
        "INFO": typer.colors.CYAN,
    }
    return typer.style(f"[{status}]", fg=colours.get(status), bold=True)


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(f"{_badge('FAIL')} {message}", err=True)
    return typer.Exit(code)


@dataclass
class Setup:
    device: str
    host_api: str
    channels: tuple[int, ...]
    ids: tuple[str, ...]
    names: tuple[str, ...]
    samplerate: int
    blocksize: int
    segment_min: float
    format: str
    idle_warn_min: float
    wasapi_exclusive: bool


def resolve_setup(
    rig: Path | None,
    device: str | None,
    host_api: str | None,
    channels: str | None,
    names: str | None,
    *,
    segment_min: float | None = None,
    fmt: str | None = None,
    idle_warn_min: float | None = None,
) -> Setup:
    """Rig values, overridden by explicit flags; device and host API are required."""
    from sessionscribe.core.audio.backend import parse_channel_spec
    from sessionscribe.core.audio.manifest import RigConfig, read_rig

    profile: RigConfig | None = read_rig(rig) if rig else None
    dev = device or (profile.device if profile else None)
    api = host_api or (profile.host_api if profile else None)
    if not dev or not api:
        raise _fail("name the device and its host API (--device, --host-api) or give --rig", 2)
    if channels:
        chans = parse_channel_spec(channels)
    elif profile:
        chans = tuple(c.channel for c in profile.channels)
    else:
        raise _fail("give --channels (or --rig)", 2)
    if names:
        ids = tuple(n.strip() for n in names.split(","))
        labels = ids
    elif profile and not channels:
        ids = tuple(c.id for c in profile.channels)
        labels = tuple(c.name or c.id for c in profile.channels)
    elif profile:
        by_ch = {c.channel: c for c in profile.channels}
        ids = tuple(by_ch[c].id if c in by_ch else f"ch{c:02d}" for c in chans)
        labels = tuple(by_ch[c].name or by_ch[c].id if c in by_ch else f"ch{c:02d}" for c in chans)
    else:
        ids = tuple(f"ch{c:02d}" for c in chans)
        labels = ids
    if len(ids) != len(chans):
        raise _fail(f"{len(chans)} channel(s) but {len(ids)} name(s)", 2)
    return Setup(
        device=dev,
        host_api=api,
        channels=chans,
        ids=ids,
        names=labels,
        samplerate=profile.samplerate if profile else 48_000,
        blocksize=profile.blocksize if profile else 480,
        segment_min=segment_min or (profile.segment_min if profile else 5.0),
        format=fmt or (profile.format if profile else "flac"),
        idle_warn_min=idle_warn_min or (profile.idle_warn_min if profile else 10.0),
        wasapi_exclusive=profile.wasapi_exclusive if profile else False,
    )


def _opener(setup: Setup) -> Any:
    from sessionscribe.core.audio.backend import AudioSetupError
    from sessionscribe.core.audio.live import SoundDeviceOpener

    try:
        return SoundDeviceOpener(
            setup.device,
            setup.host_api,
            setup.channels,
            setup.samplerate,
            setup.blocksize,
            wasapi_exclusive=setup.wasapi_exclusive,
        )
    except AudioSetupError as exc:
        raise _fail(str(exc)) from exc


def _level_line(event: Any, setup: Setup, reserve: int = 0) -> str:
    """One meter line that fits the terminal, so a carriage return can overwrite it."""
    from sessionscribe.core.audio.live import bar

    columns = shutil.get_terminal_size((100, 24)).columns - 1 - reserve
    n = max(1, len(setup.ids))
    per = max(6, columns // n)
    name_w = min(4, max(2, per - 6))
    bar_w = max(2, per - name_w - 3)
    parts = [
        f"{name[:name_w]:>{name_w}} {bar(rms, bar_w)}{'!' if clip else ' '}"
        for name, rms, clip in zip(setup.ids, event.rms_dbfs, event.clipped, strict=False)
    ]
    return " ".join(parts)[: max(0, columns)]


def _overwrite(line: str) -> None:
    columns = shutil.get_terminal_size((100, 24)).columns - 1
    typer.echo("\r" + line[:columns].ljust(columns), nl=False)


@record_app.command("devices")
def record_devices() -> None:
    """List input devices with host API and channel count (index is informational)."""
    from sessionscribe.core.audio.backend import (
        host_api_names,
        list_input_devices,
        load_sounddevice,
    )

    sd = load_sounddevice(enable_asio=sys.platform == "win32")
    typer.echo("Host APIs: " + ", ".join(host_api_names(sd)))
    for d in list_input_devices(sd):
        typer.echo(
            f"  {d.name[:44]!r:46} {d.host_api:20} {d.max_input_channels:3d} ch  "
            f"{d.default_samplerate:.0f} Hz  (index {d.index})"
        )


@rig_app.command("save")
def rig_save(
    path: Annotated[Path, typer.Argument(help="rig.toml to write.")],
    device: Annotated[str, typer.Option(help="Device name.")],
    host_api: Annotated[str, typer.Option("--host-api", help="Host API.")],
    channels: Annotated[str, typer.Option(help="Interface channels, e.g. 1-8.")],
    names: Annotated[str, typer.Option(help="Participant ids per channel.")],
    labels: Annotated[str | None, typer.Option(help="Display names per channel.")] = None,
    segment_min: Annotated[float, typer.Option(help="Segment length in minutes.")] = 5.0,
    fmt: Annotated[str, typer.Option("--format", help="flac or rf64.")] = "flac",
    idle_warn_min: Annotated[
        float, typer.Option(help="Idle-channel warning after minutes.")
    ] = 10.0,
    force: Annotated[bool, typer.Option(help="Overwrite.")] = False,
) -> None:
    """Save a device + channel -> participant assignment for --rig."""
    from pydantic import ValidationError

    from sessionscribe.core.audio.backend import parse_channel_spec
    from sessionscribe.core.audio.manifest import RigChannel, RigConfig, write_rig

    if path.exists() and not force:
        raise _fail(f"{path} exists; use --force")
    chans = parse_channel_spec(channels)
    ids = [n.strip() for n in names.split(",")]
    shown = [n.strip() for n in labels.split(",")] if labels else ids
    if not len(chans) == len(ids) == len(shown):
        raise _fail("channels, names and labels must have the same count", 2)
    try:
        rig = RigConfig(
            device=device,
            host_api=host_api,
            segment_min=segment_min,
            format=fmt,
            idle_warn_min=idle_warn_min,
            channels=[
                RigChannel(channel=c, id=i, name=n)
                for c, i, n in zip(chans, ids, shown, strict=True)
            ],
        )
    except ValidationError as exc:
        raise _fail(str(exc), 2) from exc
    write_rig(path, rig)
    typer.echo(f"Wrote {path}")


@record_app.command("meters")
def record_meters(
    rig: RigOpt = None,
    device: DeviceOpt = None,
    host_api: HostApiOpt = None,
    channels: ChannelsOpt = None,
    names: NamesOpt = None,
    seconds: Annotated[float | None, typer.Option(help="Stop after this many seconds.")] = None,
) -> None:
    """Live level meters (nothing is written); Ctrl+C stops."""
    from sessionscribe.core.audio.live import Meter

    setup = resolve_setup(rig, device, host_api, channels, names)
    opener = _opener(setup)
    typer.echo(f"{opener.device_name} ({opener.host_api}), channels {list(setup.channels)}")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    Meter(opener, len(setup.channels), setup.samplerate).run(
        seconds, lambda e: _overwrite(_level_line(e, setup)), stop
    )
    typer.echo("")


def _print_checks(checks: list[Any]) -> str:
    from sessionscribe.core.audio.mic_check import overall

    for c in checks:
        typer.echo(
            f"{_badge(c.verdict)} ch {c.channel:2d} {c.participant:10} noise {c.noise_dbfs:6.1f} "
            f"speech {c.speech_dbfs:6.1f} dBFS  rise {c.rise_db:5.1f} dB  peak {c.peak_dbfs:6.1f}"
        )
        for _, message in c.alarms:
            typer.echo(f"        {message}")
    verdict = overall(checks)
    typer.echo(f"Mic check: {_badge(verdict)}")
    return verdict


STALL_S = 3.0  # no audio time progress for this long: the stream failed


def _prompt_turns(
    setup: Setup, noise_s: float, turn_s: float, now: Any, stop: threading.Event
) -> tuple[Any, Any] | None:
    """Print the prompts on time; ``now()`` is the audio time in seconds (frames read).

    Returns None when stopped (Ctrl+C) or when the audio time stops advancing.
    """
    from sessionscribe.core.audio.mic_check import plan_turns

    noise, turns = plan_turns(setup.ids, noise_s, turn_s)
    last = [time.monotonic(), now()]

    def wait_until(t: float) -> bool:
        while now() < t:
            if stop.is_set():
                return False
            current = now()
            if current > last[1]:
                last[0], last[1] = time.monotonic(), current
            elif time.monotonic() - last[0] > STALL_S:
                typer.echo(f"\n{_badge('FAIL')} no audio is arriving; mic check aborted")
                return False
            time.sleep(0.05)
        return True

    if noise:
        typer.echo(f"Everyone quiet for {noise_s:.0f} s (noise level) ...")
        if not wait_until(noise[1]):
            return None
    for t, label in zip(turns, setup.names, strict=True):
        if not wait_until(t.start_s - 0.5):
            return None
        typer.echo(f"  {label}: please speak now ({turn_s:.0f} s) ...")
        if not wait_until(t.end_s):
            return None
    return noise, turns


def _open_failed(exc: Exception) -> typer.Exit:
    hint = ""
    if "ASIO" in str(exc):
        hint = (
            " Is another program using the ASIO driver (a DAW, or an earlier sessionscribe"
            " that is still running)? Close it and try again."
        )
    return _fail(f"{exc}{hint}")


@record_app.command("check")
def record_check(
    rig: RigOpt = None,
    device: DeviceOpt = None,
    host_api: HostApiOpt = None,
    channels: ChannelsOpt = None,
    names: NamesOpt = None,
    noise_s: Annotated[float, typer.Option(help="Seconds of silence first (0 = skip).")] = 5.0,
    turn_s: Annotated[float, typer.Option(help="Seconds each participant speaks.")] = 5.0,
) -> None:
    """Guided mic check: silence, then each participant in turn; alarms per channel."""
    from sessionscribe.core.audio.backend import AudioSetupError
    from sessionscribe.core.audio.live import Meter
    from sessionscribe.core.audio.mic_check import analyze

    setup = resolve_setup(rig, device, host_api, channels, names)
    opener = _opener(setup)
    events: list[Any] = []
    meter = Meter(opener, len(setup.channels), setup.samplerate)
    total = (noise_s + 1 if noise_s else 0) + len(setup.ids) * (turn_s + 1) + 0.5
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    planned: dict[str, Any] = {}

    def prompts() -> None:
        planned["turns"] = _prompt_turns(setup, noise_s, turn_s, meter.audio_time, stop)
        stop.set()

    helper = threading.Thread(target=prompts, daemon=True)
    helper.start()
    # The stream runs in the main thread: ASIO drivers are COM objects and do not load
    # from a thread without COM (PortAudio: "Failed to load ASIO driver").
    try:
        meter.run(total, events.append, stop)
    except AudioSetupError as exc:
        stop.set()
        raise _open_failed(exc) from exc
    helper.join(timeout=STALL_S + 1)
    if not planned.get("turns"):
        raise _fail("mic check did not complete")
    noise, turns = planned["turns"]
    checks = analyze(events, setup.channels, setup.ids, turns, noise)
    if _print_checks(checks) == "FAIL":
        raise typer.Exit(1)


@record_app.command("start")
def record_start(
    session: Annotated[Path, typer.Option("--session", "-s", help="Session folder to create.")],
    rig: RigOpt = None,
    device: DeviceOpt = None,
    host_api: HostApiOpt = None,
    channels: ChannelsOpt = None,
    names: NamesOpt = None,
    hours: Annotated[float, typer.Option(help="Planned length (disk check).")] = 8.0,
    segment_min: Annotated[float | None, typer.Option(help="Segment length (min).")] = None,
    fmt: Annotated[str | None, typer.Option("--format", help="flac or rf64.")] = None,
    idle_warn_min: Annotated[float | None, typer.Option(help="Idle-channel warning (min).")] = None,
    check: Annotated[bool, typer.Option(help="Guided mic check at the start.")] = True,
    noise_s: Annotated[float, typer.Option(help="Mic check: silence seconds.")] = 5.0,
    turn_s: Annotated[float, typer.Option(help="Mic check: seconds per participant.")] = 5.0,
    force: Annotated[bool, typer.Option(help="Keep recording after a FAILed mic check.")] = False,
    seconds: Annotated[
        float | None, typer.Option(help="Stop by itself after this many seconds.")
    ] = None,
) -> None:
    """Record a session until Ctrl+C (starts with a guided mic check by default).

    The mic check is part of the recording, so its turns give guided coupling ranges;
    a FAILed check stops the recording unless --force.
    """
    from sessionscribe.core.audio.mic_check import analyze
    from sessionscribe.core.audio.power import inhibit_sleep
    from sessionscribe.core.audio.recorder import (
        LevelEvent,
        Recorder,
        RecorderAlarm,
        RecorderConfig,
        RecorderError,
        RecorderStatus,
    )

    setup = resolve_setup(
        rig,
        device,
        host_api,
        channels,
        names,
        segment_min=segment_min,
        fmt=fmt,
        idle_warn_min=idle_warn_min,
    )
    opener = _opener(setup)
    events: list[Any] = []
    state = {"line": ""}

    def sink(event: Any) -> None:
        if isinstance(event, LevelEvent):
            events.append(event)
            state["line"] = _level_line(event, setup, reserve=40)
        elif isinstance(event, RecorderAlarm):
            badge = "FAIL" if event.fatal else "WARN" if event.kind != "reconnected" else "INFO"
            typer.echo(f"\n{_badge(badge)} {event.message}")
        elif isinstance(event, RecorderStatus):
            left = "" if event.disk_left_s is None else f"  disk {event.disk_left_s / 3600:.1f} h"
            clock = time.strftime("%H:%M:%S", time.gmtime(event.time_s))
            _overwrite(
                f"{clock} buf {event.buffer_fill:4.0%}{left} gaps {event.gaps} {state['line']}"
            )

    def inhibit() -> Any:
        inh = inhibit_sleep()
        if not inh.active:
            typer.echo(f"{_badge('WARN')} cannot keep the computer awake: {inh.how}", err=True)
        return inh

    cfg = RecorderConfig(
        session_dir=session,
        channels=setup.channels,
        participants=setup.ids,
        names=setup.names,
        samplerate=setup.samplerate,
        blocksize=setup.blocksize,
        segment_s=setup.segment_min * 60,
        format=setup.format,
        planned_hours=hours,
        idle_warn_s=setup.idle_warn_min * 60,
    )
    try:
        recorder = Recorder(cfg, opener, sink=sink, inhibit=inhibit)
        recorder.check_disk()
    except RecorderError as exc:
        raise _fail(str(exc)) from exc
    from sessionscribe.core.audio.backend import AudioSetupError

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())

    def mic_check() -> None:
        # Recording time (frames written), so the turns are ranges in the recorded audio.
        planned = _prompt_turns(setup, noise_s, turn_s, recorder.audio_time, stop)
        if planned is None:
            return
        noise, turns = planned
        time.sleep(0.5)
        try:
            checks = analyze(list(events), setup.channels, setup.ids, turns, noise)
        except ValueError as exc:
            typer.echo(f"\n{_badge('FAIL')} mic check: {exc}")
            return
        verdict = _print_checks(checks)
        recorder.set_mic_check({t.participant: (t.start_s, t.end_s) for t in turns})
        if verdict == "FAIL" and not force:
            typer.echo(f"{_badge('FAIL')} stopping: fix the channels above or use --force")
            stop.set()

    typer.echo(
        f"Recording {len(setup.channels)} channel(s) from {opener.device_name} "
        f"({opener.host_api}) into {session}. Ctrl+C stops."
    )
    if check:
        threading.Thread(target=mic_check, daemon=True).start()
    # The stream runs in the main thread (ASIO needs it, see record check).
    try:
        manifest = recorder.run(stop, max_seconds=seconds)
    except AudioSetupError as exc:
        stop.set()
        raise _open_failed(exc) from exc
    except RecorderError as exc:
        stop.set()
        raise _fail(str(exc)) from exc
    typer.echo("")
    _summary(manifest)


def _summary(manifest: Any) -> None:
    m = manifest.metrics
    seconds = m.frames / manifest.samplerate
    typer.echo(
        f"{_badge('OK' if not manifest.gaps else 'WARN')} {manifest.state}: "
        f"{seconds / 60:.1f} min, {len(manifest.segments)} segment file(s), "
        f"{len(manifest.gaps)} gap(s), {m.dropped_frames} dropped frame(s), "
        f"{m.input_overflows} input overflow(s), max buffer {m.max_buffer_fill:.0%}"
    )


@record_app.command("verify")
def record_verify(
    session: Annotated[Path, typer.Option("--session", "-s", help="Session folder.", exists=True)],
) -> None:
    """Compare the manifest with the files: what exists and what is missing."""
    from sessionscribe.core.audio.recovery import verify

    try:
        report = verify(session)
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    bad = [s for s in report.segments if not s.ok]
    for s in bad:
        typer.echo(f"{_badge('FAIL')} {s.file} ({s.status}): {s.problem}")
    for f in report.unknown_files:
        typer.echo(f"{_badge('WARN')} not in the manifest: {f}")
    lengths = sorted(set(report.frames_per_channel.values()))
    typer.echo(
        f"{_badge('OK' if report.ok else 'FAIL')} {len(report.segments)} segment file(s), "
        f"{len(bad)} problem(s); state {report.state}; {report.gaps} gap(s) "
        f"({report.gap_seconds:.1f} s); channel lengths {lengths}"
    )
    if not report.ok:
        raise typer.Exit(1)


@record_app.command("recover")
def record_recover(
    session: Annotated[Path, typer.Option("--session", "-s", help="Session folder.", exists=True)],
) -> None:
    """After a crash: make open segments readable (originals kept as .partial)."""
    from sessionscribe.core.audio.recovery import recover

    try:
        manifest = recover(session)
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    counts: dict[str, int] = {}
    for s in manifest.segments:
        counts[s.status] = counts.get(s.status, 0) + 1
    typer.echo(
        f"{_badge('OK')} state {manifest.state}; segments "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
