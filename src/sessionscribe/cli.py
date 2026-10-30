"""Command-line interface. Thin layer over ``sessionscribe.core``."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from sessionscribe import __version__
from sessionscribe.core.audio.backend import (
    AudioSetupError,
    InputDevice,
    format_channel_spec,
    parse_channel_spec,
)
from sessionscribe.core.timeline import format_clock

if TYPE_CHECKING:
    from sessionscribe.core.config import Settings
    from sessionscribe.core.events import Event
    from sessionscribe.core.pipeline.run import RunStats
    from sessionscribe.core.session import Session

app = typer.Typer(
    name="sessionscribe",
    help="Local multi-track speech-to-text for tabletop RPG sessions.",
    no_args_is_help=True,
    add_completion=False,
)

_STATUS_COLOURS = {
    "SKIP": typer.colors.BRIGHT_BLACK,
    "INFO": typer.colors.CYAN,
    "OK": typer.colors.GREEN,
    "WARN": typer.colors.YELLOW,
    "FAIL": typer.colors.RED,
    "PASS": typer.colors.GREEN,
}

ChannelsOpt = Annotated[
    str | None,
    typer.Option("--channels", "-c", help="1-based input channels, e.g. 1-8 or 1,3,5-6."),
]
AudioDeviceOpt = Annotated[
    str | None, typer.Option("--audio-device", "-d", help="Input device index or name substring.")
]
HostApiOpt = Annotated[
    str | None,
    typer.Option("--host-api", help="asio, wasapi, mme, ds, wdmks, alsa, jack, pulse or a name."),
]
SampleRateOpt = Annotated[int, typer.Option("--samplerate", help="Sample rate in Hz.")]
ExclusiveOpt = Annotated[bool, typer.Option("--wasapi-exclusive", help="WASAPI exclusive mode.")]


def _badge(status: str) -> str:
    return typer.style(f"[{status:^4}]", fg=_STATUS_COLOURS.get(status), bold=True)


def _channels(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    try:
        return parse_channel_spec(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--channels") from exc


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sessionscribe {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    _setup_logging(verbose)


@app.command()
def doctor(
    device: Annotated[str, typer.Option(help="ASR device: auto, cuda or cpu.")] = "auto",
    compute_type: Annotated[str | None, typer.Option(help="ctranslate2 compute type.")] = None,
    model: Annotated[str, typer.Option(help="Model for the test inference.")] = "tiny",
    model_dir: Annotated[
        Path | None, typer.Option(help="Model download folder (default: Hugging Face cache).")
    ] = None,
    offline: Annotated[bool, typer.Option(help="Never download; use cached models only.")] = False,
    inference: Annotated[bool, typer.Option(help="Run a real short inference.")] = True,
    audio: Annotated[bool, typer.Option(help="Check audio devices and open a stream.")] = True,
    audio_device: AudioDeviceOpt = None,
    host_api: HostApiOpt = None,
    channels: ChannelsOpt = None,
    samplerate: SampleRateOpt = 48_000,
    wasapi_exclusive: ExclusiveOpt = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Check versions, CUDA, a real inference and the audio interface."""
    from sessionscribe.core.doctor import CheckResult, DoctorOptions, run_doctor

    opts = DoctorOptions(
        device=device,
        compute_type=compute_type,
        model=model,
        model_dir=model_dir,
        offline=offline,
        inference=inference,
        audio=audio,
        audio_device=audio_device,
        host_api=host_api,
        channels=_channels(channels),
        samplerate=samplerate,
        wasapi_exclusive=wasapi_exclusive,
    )

    def show(result: CheckResult) -> None:
        typer.echo(f"{_badge(result.status.name)} {result.name}: {result.summary}")
        for line in result.details:
            typer.echo(f"         {line}")

    report = run_doctor(opts, on_result=None if as_json else show)
    if as_json:
        data = [asdict(r) | {"status": r.status.name} for r in report.results]
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"\nOverall: {_badge(report.worst.name)}")
    raise typer.Exit(0 if report.ok else 1)


def _fmt_db(value: float) -> str:
    return f"{value:6.1f}" if math.isfinite(value) else "  -inf"


@app.command("record-probe")
def record_probe(
    channels: Annotated[
        str, typer.Option("--channels", "-c", help="1-based input channels, e.g. 1-8.")
    ] = "1-8",
    minutes: Annotated[float, typer.Option(help="Recording length in minutes.")] = 30.0,
    audio_device: AudioDeviceOpt = None,
    host_api: HostApiOpt = None,
    samplerate: SampleRateOpt = 48_000,
    blocksize: Annotated[int, typer.Option(help="Frames per block (0 = host decides).")] = 0,
    buffer_s: Annotated[float, typer.Option(help="Ring buffer length in seconds.")] = 10.0,
    out: Annotated[Path, typer.Option(help="Output folder for the FLAC files.")] = Path(
        "work/probe"
    ),
    write: Annotated[bool, typer.Option(help="Write FLAC files (--no-write only counts).")] = True,
    wasapi_exclusive: ExclusiveOpt = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Record raw multi-channel audio and report sample counts, overflows and levels."""
    from sessionscribe.core.audio.probe import ProbeConfig, ProbeProgress, Verdict, run_probe

    selected = _channels(channels)
    assert selected is not None
    config = ProbeConfig(
        channels=selected,
        duration_s=minutes * 60.0,
        device=audio_device,
        host_api=host_api,
        samplerate=samplerate,
        blocksize=blocksize,
        buffer_s=buffer_s,
        write=write,
        out_dir=out,
        wasapi_exclusive=wasapi_exclusive,
    )

    def progress(p: ProbeProgress) -> None:
        typer.echo(
            f"  {p.elapsed_s / 60:6.1f} min  {p.frames_received:>12} frames  "
            f"buffer {p.buffer_fill:5.1%}  overflows {p.input_overflows}  "
            f"dropped {p.dropped_blocks}",
            err=True,
        )

    def started(device: InputDevice, auto_selected: bool) -> None:
        how = "auto-selected" if auto_selected else "selected"
        typer.echo(typer.style(f"Device ({how}): {device.label}", bold=True))
        typer.echo(typer.style(f"Host API:      {device.host_api}", bold=True))
        typer.echo(
            f"Recording channels {format_channel_spec(selected)} for {minutes:g} min; "
            "Ctrl+C stops early."
        )

    try:
        report = run_probe(
            config,
            on_progress=None if as_json else progress,
            on_start=None if as_json else started,
        )
    except AudioSetupError as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(report.to_json())
    else:
        ppm = report.clock_deviation_ppm
        how = "auto-selected" if report.device_auto_selected else "selected"
        typer.echo(f"\nDevice:     [{report.device_index}] {report.device} ({how})")
        typer.echo(f"Host API:   {report.host_api}")
        typer.echo(
            f"Stream:     {report.actual_samplerate or 0:g} Hz, {report.dtype}, "
            f"latency {report.latency_s or 0:.3f} s"
        )
        typer.echo(
            f"Duration:   {report.wall_elapsed_s:.1f} s wall clock"
            + (" (interrupted)" if report.interrupted else "")
        )
        typer.echo(f"Frames:     received {report.frames_received} in {report.blocks} blocks")
        typer.echo(f"            per channel {report.frames_per_channel}")
        if report.frames_written is not None:
            typer.echo(f"            written {report.frames_written}")
        typer.echo(
            f"Status:     overflows {report.input_overflows}, underflows "
            f"{report.input_underflows}, other {report.other_status_flags}"
        )
        typer.echo(
            f"Buffer:     max fill {report.ring_max_fill_s:.2f} s "
            f"of {report.ring_capacity_s:.0f} s, dropped {report.ring_dropped_blocks} block(s), "
            f"max write {report.writer_max_write_s * 1000:.1f} ms"
        )
        typer.echo(
            "Timestamps: "
            + (
                f"{report.timestamp_discontinuities} discontinuities; jitter "
                f"median {report.timestamp_jitter_median_ms or 0:.2f} ms, "
                f"p99 {report.timestamp_jitter_p99_ms or 0:.2f} ms, "
                f"max {report.timestamp_jitter_max_ms or 0:.2f} ms"
                if report.timestamps_available
                else "not provided by host API"
            )
        )
        typer.echo(
            "Clock:      " + (f"{ppm:+.1f} ppm vs wall clock (info)" if ppm is not None else "n/a")
        )
        typer.echo("Levels:     ch   peak dBFS   RMS dBFS")
        for lv in report.levels:
            typer.echo(
                f"            {lv.channel:2d}   {_fmt_db(lv.peak_dbfs)}      {_fmt_db(lv.rms_dbfs)}"
            )
        if report.output_dir:
            typer.echo(f"Output:     {report.output_dir}")
        typer.echo(f"\nResult: {_badge(report.verdict.value)}")
        for reason in report.reasons:
            typer.echo(f"  - {reason}")
    raise typer.Exit(1 if report.verdict is Verdict.FAIL else 0)


# --- settings and single-file transcription -----------------------------------------------

config_app = typer.Typer(help="Show, create and validate settings files.", no_args_is_help=True)
presets_app = typer.Typer(help="Built-in and user presets.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(presets_app, name="presets")
session_app = typer.Typer(
    help="Session files: participants, sources, offsets.", no_args_is_help=True
)
app.add_typer(session_app, name="session")

PresetOpt = Annotated[
    str | None, typer.Option("--preset", "-p", help="Preset name (default: conversation).")
]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="Settings TOML file.", exists=True, dir_okay=False)
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a setting: section.field=value (repeatable)."),
]


def _load_settings(
    preset: str | None,
    config_path: Path | None,
    overrides: list[str] | None,
    cli: dict[str, Any] | None = None,
    *,
    session: Session | None = None,
) -> Settings:
    from pydantic import ValidationError

    from sessionscribe.core.config import DEFAULT_PRESET, load_settings

    session_preset = session.config.preset if session else None
    named = preset is not None or config_path is not None or session_preset is not None
    try:
        return load_settings(
            preset=preset if named else DEFAULT_PRESET,
            config_path=config_path,
            overrides=overrides or (),
            cli=cli,
            session=session.config.settings if session else None,
            session_preset=session_preset,
        )
    except ValidationError as exc:
        typer.echo(f"{_badge('FAIL')} invalid settings:", err=True)
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"]) or "settings"
            typer.echo(f"  - {where}: {err['msg']}", err=True)
        raise typer.Exit(2) from exc
    except (ValueError, OSError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(2) from exc


def _warn_mode_issues(settings: Settings) -> int:
    from sessionscribe.core.config import mode_issues

    issues = mode_issues(settings)
    for issue in issues:
        typer.echo(f"{_badge('WARN')} {issue.message}", err=True)
    if settings.decoding.language == "auto":
        typer.echo(
            f"{_badge('WARN')} decoding.language is auto; set it (e.g. --language hu) for "
            "reliable results on short utterances",
            err=True,
        )
    return len(issues)


def _cli_settings(**values: Any) -> dict[str, Any]:
    """Nested settings dict from flat ``section__field`` keyword arguments that are set."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        section, _, name = key.partition("__")
        out.setdefault(section, {})[name] = value
    return out


@config_app.command("show")
def config_show(preset: PresetOpt = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Print the resolved settings as TOML."""
    from sessionscribe.core.config import settings_to_toml

    settings = _load_settings(preset, config, set_)
    typer.echo(settings_to_toml(settings), nl=False)
    _warn_mode_issues(settings)


@config_app.command("init")
def config_init(
    path: Annotated[Path, typer.Argument(help="File to create.")],
    preset: PresetOpt = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing file.")] = False,
) -> None:
    """Write a settings file with all values of a preset."""
    from sessionscribe.core.config import write_settings

    if path.exists() and not force:
        typer.echo(f"{_badge('FAIL')} {path} exists; use --force to overwrite", err=True)
        raise typer.Exit(1)
    write_settings(_load_settings(preset, None, None), path)
    typer.echo(f"Wrote {path}")


@config_app.command("validate")
def config_validate(
    path: Annotated[Path, typer.Argument(help="Settings TOML file.", exists=True, dir_okay=False)],
) -> None:
    """Validate a settings file and report settings the chosen ASR mode ignores."""
    settings = _load_settings(None, path, None)
    warnings = _warn_mode_issues(settings)
    typer.echo(f"{_badge('WARN' if warnings else 'OK')} {path} (mode {settings.asr.mode})")


@presets_app.command("list")
def presets_list() -> None:
    """List presets (user presets live in the user config folder)."""
    from sessionscribe.core.config import list_presets, user_presets_dir

    for name, source in list_presets().items():
        typer.echo(f"{name:16} {source or 'built-in'}")
    typer.echo(f"\nUser presets: {user_presets_dir()}")


@session_app.command("init")
def session_init(
    folder: Annotated[
        Path, typer.Argument(help="Folder with one file or segment folder per participant.")
    ],
    out: Annotated[
        Path | None, typer.Option(help="Session file (default: FOLDER/session.toml).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing file.")] = False,
) -> None:
    """Suggest participants from file names (chNN_name.wav) and write session.toml."""
    from sessionscribe.core.session import SESSION_FILE, suggest_session, write_session

    path = out or folder / SESSION_FILE
    if path.exists() and not force:
        typer.echo(f"{_badge('FAIL')} {path} exists; use --force to overwrite", err=True)
        raise typer.Exit(1)
    try:
        config = suggest_session(folder)
    except (OSError, ValueError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc
    for participant in config.participants:  # sources are relative to the session file
        target = os.path.relpath(folder / participant.source, path.parent)
        participant.source = Path(target).as_posix()
    write_session(path, config)
    typer.echo(
        f"Wrote {path}: {len(config.participants)} participants; edit names and offsets there."
    )


SessionOpt = Annotated[
    Path | None,
    typer.Option("--session", "-s", help="session.toml (or its folder).", exists=True),
]


def _open_session(path: Path) -> Session:
    from pydantic import ValidationError

    from sessionscribe.core.session import Session

    try:
        return Session.load(path)
    except ValidationError as exc:
        typer.echo(f"{_badge('FAIL')} invalid session file:", err=True)
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"]) or "session"
            typer.echo(f"  - {where}: {err['msg']}", err=True)
        raise typer.Exit(2) from exc
    except (OSError, ValueError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(2) from exc


def _progress_sink() -> Callable[[Event], None]:
    from sessionscribe.core.events import ProgressEvent, WarningEvent

    last: dict[str, int] = {}
    started = time.monotonic()

    def sink(event: Event) -> None:
        if isinstance(event, WarningEvent):
            typer.echo(f"{_badge('WARN')} {event.message}", err=True)
        elif isinstance(event, ProgressEvent) and event.total and event.stage == "asr":
            pct = int(100 * event.done / event.total)
            if pct // 5 != last.get("asr", -1) // 5:
                last["asr"] = pct
                elapsed = time.monotonic() - started
                eta = (
                    f", ETA {format_clock(int(elapsed * (100 - pct) / pct), 1)}" if pct >= 5 else ""
                )
                typer.echo(f"  ASR {pct:3d} %{eta}", err=True)

    return sink


def _report(stats: RunStats, written: list[Path]) -> None:
    typer.echo(
        f"{stats.chunks} chunks: {stats.cached} from cache, {stats.transcribed} transcribed"
        + (f", {stats.crashes} worker crash(es) recovered" if stats.crashes else "")
        + (f"; {stats.refilled} segment(s) recovered by re-transcription" if stats.refilled else "")
        + (f"; {stats.missed} possible missed speech passage(s) to review" if stats.missed else "")
        + (
            f"; {stats.bleed_suppressed} bleed segment(s) suppressed"
            if stats.bleed_suppressed
            else ""
        )
        + (f"; {stats.duplicates} duplicate(s) removed" if stats.duplicates else ""),
        err=True,
    )
    for path in written:
        typer.echo(str(path))


@app.command()
def transcribe(
    file: Annotated[
        Path | None, typer.Argument(help="One audio file (one speaker).", exists=True)
    ] = None,
    session_path: SessionOpt = None,
    preset: PresetOpt = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    model: Annotated[str | None, typer.Option(help="Model name or folder.")] = None,
    language: Annotated[
        str | None, typer.Option("--language", "-l", help="hu, en or auto.")
    ] = None,
    mode: Annotated[str | None, typer.Option(help="ASR mode: quality or fast.")] = None,
    device: Annotated[str | None, typer.Option(help="auto, cuda or cpu.")] = None,
    compute_type: Annotated[str | None, typer.Option(help="ctranslate2 compute type.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Fast mode batch size.")] = None,
    initial_prompt: Annotated[str | None, typer.Option(help="Initial prompt text.")] = None,
    hotwords: Annotated[str | None, typer.Option(help="Hotwords text.")] = None,
    model_dir: Annotated[Path | None, typer.Option(help="Model download folder.")] = None,
    offline: Annotated[bool | None, typer.Option("--offline", help="Cached models only.")] = None,
    channel: Annotated[
        int | None, typer.Option(help="1-based channel of a multi-channel file.")
    ] = None,
    speaker: Annotated[str | None, typer.Option(help="Speaker id (default: file name).")] = None,
    name: Annotated[str | None, typer.Option(help="Speaker display name.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output folder.")] = None,
    formats: Annotated[
        str | None, typer.Option(help="Comma-separated: json,txt,srt,vtt,md.")
    ] = None,
    full_hash: Annotated[
        bool, typer.Option(help="Fingerprint whole files, not just their first/last MB.")
    ] = False,
) -> None:
    """Transcribe a session (--session) or one audio file; completed chunks are cached."""
    from sessionscribe.core.exporters.formats import write_session_outputs
    from sessionscribe.core.pipeline.asr import AsrError, Cancelled
    from sessionscribe.core.pipeline.run import process_session

    if (file is None) == (session_path is None):
        typer.echo(f"{_badge('FAIL')} give either FILE or --session", err=True)
        raise typer.Exit(2)
    if session_path is not None:
        session = _open_session(session_path)
    else:
        assert file is not None
        session = _file_session(file, speaker, name, channel)
    settings = _load_settings(
        preset,
        config,
        set_,
        _cli_settings(
            model__id=model,
            model__device=device,
            model__compute_type=compute_type,
            model__model_dir=model_dir,
            model__offline=offline,
            asr__mode=mode,
            asr__batch_size=batch_size,
            decoding__language=language,
            vocabulary__initial_prompt=initial_prompt,
            vocabulary__hotwords=hotwords,
            output__formats=formats.split(",") if formats else None,
        ),
        session=session,
    )
    _warn_mode_issues(settings)
    typer.echo(
        f"{session.name}: {len(session.participants)} participant(s); model "
        f"{settings.model.id}, {settings.asr.mode} mode",
        err=True,
    )
    try:
        result, stats = process_session(
            session, settings, sink=_progress_sink(), full_hash=full_hash
        )
        written = write_session_outputs(result, _out_dir(session, settings, out))
        _save_last_settings(session, settings)
    except KeyboardInterrupt as exc:
        typer.echo(f"{_badge('WARN')} cancelled; completed chunks stay cached", err=True)
        raise typer.Exit(130) from exc
    except (AsrError, Cancelled, ValueError, OSError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc
    _report(stats, written)


def _file_session(
    file: Path, speaker: str | None, name: str | None, channel: int | None
) -> Session:
    from sessionscribe.core.session import Participant, Session, SessionConfig

    pid = speaker or re.sub(r"[^A-Za-z0-9_-]+", "_", file.stem).strip("_-") or "speaker"
    config = SessionConfig(
        name=file.stem,
        participants=[Participant(id=pid, name=name or pid, source=file.name, channel=channel)],
    )
    return Session(file.parent, config)


LAST_SETTINGS = "last_settings.json"


def _save_last_settings(session: Session, settings: Settings) -> None:
    """Remember the settings of the last transcription so ``export`` finds its cache."""
    from sessionscribe.core.fileio import atomic_write_json

    work = session.work_dir(settings.output.work_dir)
    atomic_write_json(work / LAST_SETTINGS, settings.model_dump(mode="json"))


ASR_SECTIONS = ("model", "asr", "decoding", "vocabulary", "vad")


def _export_settings(
    session: Session,
    preset: str | None,
    config: Path | None,
    set_: list[str] | None,
    cli: dict[str, Any],
) -> Settings:
    """Fresh settings, with the ASR sections of the last transcription (export never runs ASR).

    Without this, a language or model given only on the transcribe command line would make
    every cache key miss. --preset or --config resolve everything afresh instead.
    """
    from pydantic import ValidationError

    from sessionscribe.core.config import Settings, deep_merge, parse_override

    settings = _load_settings(preset, config, set_, cli, session=session)
    last = session.work_dir(settings.output.work_dir) / LAST_SETTINGS
    if preset is not None or config is not None or not last.is_file():
        return settings
    previous = json.loads(last.read_text(encoding="utf-8"))
    data = settings.model_dump(mode="json")
    data.update({k: previous[k] for k in ASR_SECTIONS if k in previous})
    try:
        for item in set_ or ():
            data = deep_merge(data, parse_override(item))
        return Settings.model_validate(deep_merge(data, cli))
    except (ValidationError, ValueError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(2) from exc


def _out_dir(session: Session, settings: Settings, out: Path | None) -> Path:
    configured = out or settings.output.output_dir
    if configured is None:
        return session.folder / "output"
    return configured


@app.command()
def export(
    session_path: Annotated[
        Path, typer.Option("--session", "-s", help="session.toml (or its folder).", exists=True)
    ],
    preset: PresetOpt = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    out: Annotated[Path | None, typer.Option(help="Output folder.")] = None,
    formats: Annotated[
        str | None, typer.Option(help="Comma-separated: json,txt,srt,vtt,md.")
    ] = None,
) -> None:
    """Re-export from the cache without ASR (after renaming, offset or output changes).

    ASR settings come from the last ``transcribe`` run; output, quality and other
    settings are resolved afresh. --preset or --config resolve everything afresh.
    """
    from sessionscribe.core.exporters.formats import write_session_outputs
    from sessionscribe.core.pipeline.asr import AsrError
    from sessionscribe.core.pipeline.run import export_session

    session = _open_session(session_path)
    settings = _export_settings(
        session,
        preset,
        config,
        set_,
        _cli_settings(output__formats=formats.split(",") if formats else None),
    )
    try:
        result, stats = export_session(session, settings)
        written = write_session_outputs(result, _out_dir(session, settings, out))
    except (AsrError, ValueError, OSError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc
    _report(stats, written)


@app.command()
def reassign(
    session_path: Annotated[
        Path, typer.Option("--session", "-s", help="session.toml (or its folder).", exists=True)
    ],
    segment: Annotated[str, typer.Option(help="Segment id, e.g. gm-c003-0007.")],
    to: Annotated[str | None, typer.Option(help="Participant id to assign it to.")] = None,
    undo: Annotated[bool, typer.Option(help="Remove the manual assignment.")] = False,
) -> None:
    """Move a segment to another participant by hand (no re-ASR); then run export."""
    from sessionscribe.core.session import write_session

    if (to is None) == (not undo):
        typer.echo(f"{_badge('FAIL')} give --to PARTICIPANT or --undo", err=True)
        raise typer.Exit(2)
    session = _open_session(session_path)
    try:
        session.config.reassign(segment, None if undo else to)
    except ValueError as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(2) from exc
    write_session(session.folder / "session.toml", session.config)
    what = "removed" if undo else f"-> {to}"
    typer.echo(f"{segment} {what}; run 'sessionscribe export --session ...' to update the outputs")


@app.command()
def clean(
    session_path: Annotated[
        Path, typer.Option("--session", "-s", help="session.toml (or its folder).", exists=True)
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete the session's work folder (cache with transcript text); raw audio stays."""
    session = _open_session(session_path)
    settings = _load_settings(None, None, None, session=session)
    work = session.work_dir(settings.output.work_dir)
    if not work.exists():
        typer.echo(f"Nothing to delete: {work} does not exist")
        return
    if not yes and not typer.confirm(f"Delete {work} and everything in it?"):
        raise typer.Exit(1)
    shutil.rmtree(work)
    typer.echo(f"Deleted {work}")


from sessionscribe.cli_bench import bench_app  # noqa: E402
from sessionscribe.cli_record import record_app  # noqa: E402

app.add_typer(bench_app, name="bench")
app.add_typer(record_app, name="record")


if __name__ == "__main__":
    app()
