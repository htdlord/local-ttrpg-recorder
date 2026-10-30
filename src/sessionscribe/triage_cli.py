"""Terminal front end for triage labelling: single keys, playback, autosave.

CLI layer only; the labelling logic lives in ``core.bench.triage``. The ASR text of a
unit is shown on the terminal as an aid and is never logged or written anywhere.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sessionscribe.core.bench.triage import (
    HELP,
    LABEL_NAMES,
    Labeller,
    TriageFile,
    TriageUnit,
    write_triage,
)
from sessionscribe.core.session import Session
from sessionscribe.core.timeline import TIMELINE_RATE, format_clock, rescale, seconds_to_samples

log = logging.getLogger(__name__)

CONTEXT_S = 0.3


class TerminalError(RuntimeError):
    pass


def getch() -> str:
    """One key press without Enter; arrows and function keys come back as ``\\x1b``."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # special key: swallow its second code
            msvcrt.getwch()
            return "\x1b"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # drain an escape sequence (arrow keys)
            while select.select([sys.stdin], [], [], 0.03)[0]:
                sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def require_tty() -> None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise TerminalError(
            "triage labelling needs an interactive terminal; use --export/--import for a "
            "spreadsheet instead"
        )


class Player:
    """sounddevice playback; disables itself (with one warning) when audio fails."""

    def __init__(self, enabled: bool = True) -> None:
        self.sd: Any = None
        if not enabled:
            return
        try:
            from sessionscribe.core.audio.backend import load_sounddevice

            self.sd = load_sounddevice()
        except Exception as exc:  # no PortAudio, no device
            typer.echo(f"[WARN] playback unavailable ({exc}); labelling without audio", err=True)

    def play(self, audio: np.ndarray, rate: int) -> None:
        if self.sd is None or audio.size == 0:
            return
        try:
            self.sd.stop()
            self.sd.play(audio, rate)
        except Exception as exc:
            typer.echo(f"[WARN] playback failed ({exc}); labelling without audio", err=True)
            self.sd = None

    def stop(self) -> None:
        if self.sd is not None:
            try:
                self.sd.stop()
            except Exception:  # pragma: no cover - device went away
                log.debug("stop failed", exc_info=True)


class AudioSource:
    """Unit audio on demand from the session's tracks (session time -> track frames)."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.tracks = {p.id: session.open_track(p) for p in session.participants}
        self.offsets = {p.id: p.offset_samples for p in session.participants}

    def _read(self, speaker: str, start: int, end: int) -> tuple[np.ndarray, int]:
        track = self.tracks[speaker]
        rate = track.samplerate
        a = rescale(max(0, start - self.offsets[speaker]), TIMELINE_RATE, rate)
        b = rescale(max(0, end - self.offsets[speaker]), TIMELINE_RATE, rate)
        return track.read(a, max(0, b - a)), rate

    def span(self, unit: TriageUnit) -> tuple[int, int]:
        pad = seconds_to_samples(CONTEXT_S)
        return max(0, seconds_to_samples(unit.start) - pad), seconds_to_samples(unit.end) + pad

    def own(self, unit: TriageUnit) -> tuple[np.ndarray, int]:
        return self._read(unit.speaker, *self.span(unit))

    def other(self, unit: TriageUnit) -> tuple[np.ndarray, int]:
        return self._read(unit.other_speaker or unit.speaker, *self.span(unit))

    def mix(self, unit: TriageUnit) -> tuple[np.ndarray, int]:
        parts = [self._read(s, *self.span(unit)) for s in self.tracks]
        rate = parts[0][1]
        same = [a for a, r in parts if r == rate]
        n = max(len(a) for a in same)
        out = np.zeros(n, np.float32)
        for a in same:
            out[: len(a)] += a
        peak = float(np.max(np.abs(out))) if n else 0.0
        return (out / peak * 0.9 if peak > 1.0 else out), rate


def describe(unit: TriageUnit, pos: int, total: int, names: Mapping[str, str]) -> str:
    clock = format_clock(seconds_to_samples(unit.start), millis=True)[:-2]
    head = (
        f"{pos}/{total} · {names.get(unit.speaker, unit.speaker)} · {clock} ({unit.duration:.1f} s)"
    )
    if unit.kind == "missed":
        head += " · possible missed speech"
    if unit.own_db is not None:
        head += f" · own {unit.own_db:+.0f} dB"
    if unit.other_speaker and unit.other_db is not None:
        head += f" / {names.get(unit.other_speaker, unit.other_speaker)} {unit.other_db:+.0f} dB"
    if unit.suggestion:
        src = (
            f"←{names.get(unit.suggestion_source, unit.suggestion_source)}"
            if unit.suggestion_source
            else ""
        )
        head += f" · suggestion: {unit.suggestion}{src} (Enter)"
    if unit.label:
        head += f" · labelled {unit.label}" + (f"←{unit.source}" if unit.source else "")
    return head


def run_labelling(
    triage: TriageFile,
    path: Path,
    labeller: Labeller,
    audio: AudioSource,
    texts: Mapping[str, str],
    names: Mapping[str, str],
    *,
    player: Player,
    read_key: Callable[[], str] = getch,
    echo: Callable[[str], None] = typer.echo,
) -> None:
    """Show a unit, play it, apply keys; save after every label."""
    total = len(triage.units)

    def show() -> None:
        unit = labeller.unit
        echo("")
        echo(describe(unit, labeller.order[labeller.pos] + 1, total, names))
        text = texts.get(unit.id)
        if text:
            echo(f"  {text}")
        player.play(*audio.own(unit))

    echo(
        "Keys: Enter accept · s b w v o n ? label · space replay · m mix · x other · "
        "u back · h help · q quit"
    )
    if labeller.done:
        echo("Nothing to label here.")
        return
    show()
    try:
        while True:
            action = labeller.key(read_key())
            if action == "moved":
                write_triage(path, triage)
                if labeller.done:
                    echo("\nAll units in this pass are labelled.")
                    break
                show()
            elif action == "need_source":
                choices = "  ".join(
                    f"{i}={names.get(s, s)}" for i, s in enumerate(triage.speakers, 1)
                )
                echo(
                    f"  {LABEL_NAMES[labeller.pending or '']}: source/speaker? {choices}  "
                    "(Enter = unknown, Esc = cancel)"
                )
            elif action == "play":
                player.play(*audio.own(labeller.unit))
            elif action == "play_mix":
                player.play(*audio.mix(labeller.unit))
            elif action == "play_other":
                player.play(*audio.other(labeller.unit))
            elif action == "help":
                echo(HELP)
            elif action == "quit":
                break
    finally:
        player.stop()
        write_triage(path, triage)
