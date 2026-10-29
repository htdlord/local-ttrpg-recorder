"""Command-line interface. Thin layer over ``sessionscribe.core``."""

from __future__ import annotations

import json
import logging
import math
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

if TYPE_CHECKING:
    from sessionscribe.core.config import Settings

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
) -> Settings:
    from pydantic import ValidationError

    from sessionscribe.core.config import DEFAULT_PRESET, load_settings

    try:
        return load_settings(
            preset=preset if preset is not None else (None if config_path else DEFAULT_PRESET),
            config_path=config_path,
            overrides=overrides or (),
            cli=cli,
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


@app.command()
def transcribe(
    file: Annotated[Path, typer.Argument(help="Audio file (WAV, FLAC, RF64).", exists=True)],
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
    out: Annotated[Path | None, typer.Option(help="Output folder (default: output/).")] = None,
    formats: Annotated[str | None, typer.Option(help="Comma-separated: json,txt.")] = None,
    max_duration: Annotated[float, typer.Option(help="Longest accepted file in hours.")] = 2.0,
) -> None:
    """Transcribe one audio file (one speaker)."""
    from sessionscribe.core.audio.reader import AudioReadError, load_for_asr
    from sessionscribe.core.events import Event, ProgressEvent, WarningEvent
    from sessionscribe.core.exporters.single import Speaker, write_outputs
    from sessionscribe.core.pipeline.asr import AsrEngine, AsrError

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
    )
    _warn_mode_issues(settings)

    try:
        audio, info = load_for_asr(file, channel, max_duration * 3600.0)
    except AudioReadError as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{file.name}: {info.duration_s:.1f} s, {info.samplerate} Hz, {info.subtype}; "
        f"model {settings.model.id}, {settings.asr.mode} mode",
        err=True,
    )

    last = [-1]

    def sink(event: Event) -> None:
        if isinstance(event, WarningEvent):
            typer.echo(f"{_badge('WARN')} {event.message}", err=True)
        elif isinstance(event, ProgressEvent) and event.total:
            pct = int(100 * event.done / event.total)
            if pct // 10 != last[0] // 10:
                last[0] = pct
                typer.echo(f"  {pct:3d} %", err=True)

    speaker_id = speaker or file.stem
    try:
        engine = AsrEngine.load(settings)
        typer.echo(
            f"Model loaded on {engine.loaded.device}/{engine.loaded.compute_type} "
            f"in {engine.loaded.load_s:.1f} s",
            err=True,
        )
        result = engine.transcribe(audio, source_id=speaker_id, sink=sink)
    except (AsrError, ValueError) as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc

    out_dir = out or settings.output.output_dir or Path("output")
    who = Speaker(speaker_id, name or speaker_id, file.name)
    try:
        written = write_outputs(
            result, settings, who, out_dir, file.stem, tuple(settings.output.formats)
        )
    except ValueError as exc:
        typer.echo(f"{_badge('FAIL')} {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{len(result.segments)} segments, language {result.language}, "
        f"RTF {result.rtf:.3f}" + (f", batch {result.batch_size}" if result.batch_size else ""),
        err=True,
    )
    for path in written:
        typer.echo(str(path))


from sessionscribe.cli_bench import bench_app  # noqa: E402

app.add_typer(bench_app, name="bench")


if __name__ == "__main__":
    app()
