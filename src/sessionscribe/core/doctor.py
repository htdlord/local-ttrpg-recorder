"""Environment check (``sessionscribe doctor``).

Each check returns a :class:`CheckResult`; a crashing check becomes FAIL instead of
aborting the run. Order matters: CUDA libraries are registered before ctranslate2 is
imported, and the ASIO decision is made before sounddevice is imported.
"""

from __future__ import annotations

import logging
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sessionscribe.core.audio.backend import (
    InputDevice,
    canonical_host_api,
    format_channel_spec,
    is_asio,
    list_input_devices,
    load_sounddevice,
)
from sessionscribe.core.cuda_libs import INSTALL_HINT, prepare_cuda_libraries

logger = logging.getLogger(__name__)

PACKAGES = ("faster-whisper", "ctranslate2", "numpy", "sounddevice", "soundfile", "typer")


class Status(IntEnum):
    """Ordered by severity."""

    SKIP = 0
    INFO = 1
    OK = 2
    WARN = 3
    FAIL = 4


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    summary: str
    details: tuple[str, ...] = ()


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def worst(self) -> Status:
        return max((r.status for r in self.results), default=Status.OK)

    @property
    def ok(self) -> bool:
        return self.worst < Status.FAIL


@dataclass(frozen=True)
class DoctorOptions:
    device: str = "auto"
    compute_type: str | None = None
    model: str = "tiny"
    model_dir: Path | None = None
    offline: bool = False
    inference: bool = True
    audio: bool = True
    audio_device: str | None = None
    host_api: str | None = None
    channels: tuple[int, ...] | None = None
    samplerate: int = 48_000
    stream_seconds: float = 1.0
    wasapi_exclusive: bool = False


def check_platform() -> CheckResult:
    return CheckResult(
        "platform",
        Status.OK,
        f"Python {platform.python_version()} on {platform.system()} {platform.release()}",
        (f"executable: {sys.executable}", f"machine: {platform.machine()}"),
    )


def check_packages() -> CheckResult:
    details = []
    missing = []
    for name in (*PACKAGES, "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        try:
            details.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            if name.startswith("nvidia-"):
                details.append(f"{name}: not installed (optional 'cuda' extra)")
            else:
                missing.append(name)
    if missing:
        return CheckResult(
            "packages", Status.FAIL, f"missing: {', '.join(missing)}", tuple(details)
        )
    return CheckResult("packages", Status.OK, "required packages installed", tuple(details))


def check_cuda_libraries(cuda_devices: int | None) -> CheckResult:
    statuses = prepare_cuda_libraries()
    details: list[str] = []
    for s in statuses:
        details += s.locations
        details += [f"{name}: NOT FOUND" for name in s.missing]
        if s.error:
            details.append(f"{s.package}: load error: {s.error}")
    broken = [s.package for s in statuses if not s.ok]
    if not broken:
        return CheckResult(
            "cuda-libraries", Status.OK, "cuBLAS 12 and cuDNN 9 found", tuple(details)
        )
    status = Status.FAIL if cuda_devices else Status.INFO
    summary = f"missing or broken: {', '.join(broken)}; {INSTALL_HINT}"
    if not cuda_devices:
        summary += " (no CUDA device, CPU only)"
    return CheckResult("cuda-libraries", status, summary, tuple(details))


def check_compute_types() -> tuple[CheckResult, int]:
    from sessionscribe.core.config import available_compute_types, cuda_device_count

    count = cuda_device_count()
    details = [f"cpu: {', '.join(available_compute_types('cpu'))}"]
    if count:
        details.append(f"cuda: {', '.join(available_compute_types('cuda'))}")
        summary = f"{count} CUDA device(s) visible to ctranslate2"
        return CheckResult("compute", Status.OK, summary, tuple(details)), count
    summary = "no CUDA device visible to ctranslate2; CPU only"
    return CheckResult("compute", Status.WARN, summary, tuple(details)), 0


def check_inference(opts: DoctorOptions) -> CheckResult:
    from sessionscribe.core.inference_check import run_inference_check_subprocess

    r = run_inference_check_subprocess(
        opts.model, opts.device, opts.compute_type, opts.model_dir, opts.offline
    )
    where = f"{r.model} on {r.device}/{r.compute_type}"
    details = []
    if r.cached is not None:
        details.append("model cache: " + ("found" if r.cached else "not cached"))
    if r.network is not None:
        details.append("network: " + ("reachable" if r.network else "unreachable"))
    if not r.ok:
        return CheckResult("inference", Status.FAIL, f"{where}: {r.error}", tuple(details))
    details.append(f"load {r.load_s:.1f} s, 2 s audio decoded in {r.inference_s:.2f} s")
    return CheckResult("inference", Status.OK, f"{where} ran a real inference", tuple(details))


def check_audio_devices(opts: DoctorOptions) -> tuple[CheckResult, list[InputDevice]]:
    sd = load_sounddevice(enable_asio=is_asio(opts.host_api))
    apis = [api["name"] for api in sd.query_hostapis()]
    devices = list_input_devices(sd)
    needed = max(opts.channels) if opts.channels else None
    details = [f"PortAudio: {sd.get_portaudio_version()[1]}"]
    if opts.host_api is None:
        # Full lists get long (one device per channel pair and host API); summarise.
        for api in apis:
            count = sum(1 for d in devices if d.host_api == api)
            details.append(f"{api}: {count} input device(s)")
        details.append("use --host-api <name> to list the devices of one host API")
    else:
        wanted = canonical_host_api(opts.host_api).lower()
        for d in devices:
            if wanted not in d.host_api.lower():
                continue
            line = f"{d.label}: {d.max_input_channels} in, default {d.default_samplerate:.0f} Hz"
            if needed is not None and d.max_input_channels < needed:
                line += f"  - cannot open {needed} channels"
            details.append(line)
    if not devices:
        return CheckResult("audio-devices", Status.WARN, "no input devices", tuple(details)), []
    summary = f"{len(devices)} input device(s) on {len(apis)} host API(s)"
    return CheckResult("audio-devices", Status.OK, summary, tuple(details)), devices


def check_stream(opts: DoctorOptions, devices: list[InputDevice]) -> CheckResult:
    """Actually open the requested N-channel stream briefly."""
    from sessionscribe.core.audio.backend import (
        AudioSetupError,
        default_input_index,
        select_input_device,
    )
    from sessionscribe.core.audio.probe import ProbeConfig, Verdict, run_probe

    channels = opts.channels
    if channels is None:
        sd = load_sounddevice(enable_asio=is_asio(opts.host_api))
        try:
            default = default_input_index(sd, opts.host_api)
            dev = select_input_device(devices, opts.audio_device, opts.host_api, default)
        except AudioSetupError as exc:
            return CheckResult("audio-stream", Status.FAIL, str(exc))
        channels = tuple(range(1, dev.max_input_channels + 1))
    config = ProbeConfig(
        channels=channels,
        duration_s=opts.stream_seconds,
        device=opts.audio_device,
        host_api=opts.host_api,
        samplerate=opts.samplerate,
        write=False,
        wasapi_exclusive=opts.wasapi_exclusive,
    )
    name = f"audio-stream ({len(channels)} ch)"
    try:
        report = run_probe(config)
    except AudioSetupError as exc:
        return CheckResult(name, Status.FAIL, str(exc))
    details = (
        f"channels {format_channel_spec(channels)}, {report.frames_received} frames "
        f"in {report.blocks} blocks, latency {report.latency_s or 0:.3f} s",
        *report.reasons,
    )
    where = f"{report.device} ({report.host_api}) at {opts.samplerate} Hz"
    if report.verdict is Verdict.FAIL:
        return CheckResult(name, Status.FAIL, f"{where}: stream problems", details)
    status = Status.WARN if report.verdict is Verdict.WARN else Status.OK
    return CheckResult(name, status, f"{where}: opened and received audio", details)


def _guard(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    try:
        return fn()
    except Exception as exc:
        logger.debug("check %s crashed", name, exc_info=True)
        return CheckResult(name, Status.FAIL, f"check crashed: {exc}")


def run_doctor(
    opts: DoctorOptions, on_result: Callable[[CheckResult], None] | None = None
) -> DoctorReport:
    report = DoctorReport()

    def add(result: CheckResult) -> None:
        report.results.append(result)
        if on_result is not None:
            on_result(result)

    add(check_platform())
    add(_guard("packages", check_packages))
    # Registers the CUDA libraries; the device count is only known afterwards.
    cuda_libs = _guard("cuda-libraries", lambda: check_cuda_libraries(None))
    cuda_devices = 0
    try:
        compute, cuda_devices = check_compute_types()
    except Exception as exc:
        compute = CheckResult("compute", Status.FAIL, f"ctranslate2 unusable: {exc}")
    if cuda_devices and cuda_libs.status is Status.INFO:
        cuda_libs = check_cuda_libraries(cuda_devices)  # cached; re-rate with device count
    add(cuda_libs)
    add(compute)
    if opts.inference:
        add(_guard("inference", lambda: check_inference(opts)))
    else:
        add(CheckResult("inference", Status.SKIP, "skipped (--no-inference)"))
    if opts.audio:
        devices: list[InputDevice] = []
        try:
            result, devices = check_audio_devices(opts)
        except Exception as exc:
            result = CheckResult("audio-devices", Status.FAIL, f"audio backend unusable: {exc}")
        add(result)
        if devices:
            add(_guard("audio-stream", lambda: check_stream(opts, devices)))
    else:
        add(CheckResult("audio", Status.SKIP, "skipped (--no-audio)"))
    return report
