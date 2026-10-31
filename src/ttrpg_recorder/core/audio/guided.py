"""Shared by the record CLI and the recorder child process (GUI): resolving a rig into a
setup, the timed mic-check prompts and the mic check that runs inside a recording.

Nothing here prints; prompts and results go to callbacks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ttrpg_recorder.core.audio.mic_check import ChannelCheck, Turn, analyze, overall, plan_turns

STALL_S = 3.0  # no audio time progress for this long: the stream failed


class SetupError(ValueError):
    """The device, channels or names do not add up."""


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
    """Rig values, overridden by explicit values; device and host API are required."""
    from ttrpg_recorder.core.audio.backend import parse_channel_spec
    from ttrpg_recorder.core.audio.manifest import RigConfig, read_rig

    profile: RigConfig | None = read_rig(rig) if rig else None
    dev = device or (profile.device if profile else None)
    api = host_api or (profile.host_api if profile else None)
    if not dev or not api:
        raise SetupError("name the device and its host API (--device, --host-api) or give --rig")
    if channels:
        chans = parse_channel_spec(channels)
    elif profile:
        chans = tuple(c.channel for c in profile.channels)
    else:
        raise SetupError("give --channels (or --rig)")
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
        raise SetupError(f"{len(chans)} channel(s) but {len(ids)} name(s)")
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


def open_device(setup: Setup) -> Any:
    """The PortAudio opener for a setup (raises ``AudioSetupError``)."""
    from ttrpg_recorder.core.audio.live import SoundDeviceOpener

    return SoundDeviceOpener(
        setup.device,
        setup.host_api,
        setup.channels,
        setup.samplerate,
        setup.blocksize,
        wasapi_exclusive=setup.wasapi_exclusive,
    )


def recorder_config(setup: Setup, session: Path, hours: float) -> Any:
    from ttrpg_recorder.core.audio.recorder import RecorderConfig

    return RecorderConfig(
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


@dataclass(frozen=True)
class Prompt:
    kind: Literal["noise", "turn", "stalled"]
    participant: str  # "" for noise / stalled
    label: str  # display name
    seconds: float
    until_s: float  # audio time when this step ends


def run_prompts(
    ids: Sequence[str],
    labels: Sequence[str],
    noise_s: float,
    turn_s: float,
    now: Callable[[], float],
    stop: threading.Event,
    on_prompt: Callable[[Prompt], None],
) -> tuple[tuple[float, float] | None, list[Turn]] | None:
    """Announce each step on time; ``now()`` is the audio time in seconds.

    Returns None when stopped or when the audio time stops advancing (a ``stalled``
    prompt is sent first).
    """
    noise, turns = plan_turns(ids, noise_s, turn_s)
    last = [time.monotonic(), now()]

    def wait_until(t: float) -> bool:
        while now() < t:
            if stop.is_set():
                return False
            current = now()
            if current > last[1]:
                last[0], last[1] = time.monotonic(), current
            elif time.monotonic() - last[0] > STALL_S:
                on_prompt(Prompt("stalled", "", "", 0.0, current))
                return False
            time.sleep(0.05)
        return True

    if noise:
        on_prompt(Prompt("noise", "", "", noise_s, noise[1]))
        if not wait_until(noise[1]):
            return None
    for t, label in zip(turns, labels, strict=True):
        if not wait_until(t.start_s - 0.5):
            return None
        on_prompt(Prompt("turn", t.participant, label, turn_s, t.end_s))
        if not wait_until(t.end_s):
            return None
    return noise, turns


@dataclass
class MicCheckResult:
    checks: list[ChannelCheck]
    verdict: str  # PASS | WARN | FAIL
    error: str = ""  # analysis failed (e.g. no level events)
    stopping: bool = False  # FAIL without force: the recording is being stopped


def check_in_recording(
    recorder: Any,
    levels: Callable[[], list[Any]],
    setup: Setup,
    noise_s: float,
    turn_s: float,
    stop: threading.Event,
    *,
    force: bool,
    on_prompt: Callable[[Prompt], None],
    on_result: Callable[[MicCheckResult], None],
) -> None:
    """The guided mic check inside a running recording (run it in a helper thread).

    Turns are timed in recording time (frames written), so they are saved as the
    session's ``[mic_check]`` ranges. A FAIL stops the recording unless ``force``.
    """
    planned = run_prompts(
        setup.ids, setup.names, noise_s, turn_s, recorder.audio_time, stop, on_prompt
    )
    if planned is None:
        return
    noise, turns = planned
    time.sleep(0.5)
    try:
        checks = analyze(levels(), setup.channels, setup.ids, turns, noise)
    except ValueError as exc:
        on_result(MicCheckResult([], "FAIL", error=str(exc)))
        return
    verdict = overall(checks)
    recorder.set_mic_check({t.participant: (t.start_s, t.end_s) for t in turns})
    stopping = verdict == "FAIL" and not force
    on_result(MicCheckResult(checks, verdict, stopping=stopping))
    if stopping:
        stop.set()
