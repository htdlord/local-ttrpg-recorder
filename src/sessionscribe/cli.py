"""Command-line interface. Thin layer over ``sessionscribe.core``."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from sessionscribe import __version__
from sessionscribe.core.audio.backend import (
    AudioSetupError,
    InputDevice,
    format_channel_spec,
    parse_channel_spec,
)

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


if __name__ == "__main__":
    app()
