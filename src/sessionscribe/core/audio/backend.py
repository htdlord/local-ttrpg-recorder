"""sounddevice loading, device selection and input-stream planning.

``sounddevice`` is imported lazily through :func:`load_sounddevice`, because ASIO support
must be requested with ``SD_ENABLE_ASIO`` *before* the first import (SPEC §8.3).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HOST_API_ALIASES: dict[str, str] = {
    "asio": "ASIO",
    "wasapi": "Windows WASAPI",
    "mme": "MME",
    "directsound": "Windows DirectSound",
    "ds": "Windows DirectSound",
    "wdmks": "Windows WDM-KS",
    "alsa": "ALSA",
    "jack": "JACK Audio Connection Kit",
    "pulse": "PulseAudio",
}

SAMPLE_DTYPE = "int32"


class AudioSetupError(RuntimeError):
    """Audio backend, device or stream cannot be set up as requested."""


class ChannelCountError(AudioSetupError):
    """The device does not expose enough input channels."""


_asio_requested_at_import: bool | None = None


def is_asio(host_api: str | None) -> bool:
    return host_api is not None and canonical_host_api(host_api) == "ASIO"


def canonical_host_api(name: str) -> str:
    return HOST_API_ALIASES.get(name.strip().lower(), name.strip())


def load_sounddevice(*, enable_asio: bool = False) -> ModuleType:
    """Import sounddevice, enabling ASIO first if requested.

    Raises :class:`AudioSetupError` if ASIO is requested after sounddevice was already
    imported without it, or on a non-Windows platform.
    """
    global _asio_requested_at_import
    if enable_asio:
        if sys.platform != "win32":
            raise AudioSetupError("ASIO is only available on Windows")
        if "sounddevice" in sys.modules and not _asio_requested_at_import:
            raise AudioSetupError(
                "ASIO was requested after sounddevice had already been imported without it; "
                "SD_ENABLE_ASIO must be set before the first import"
            )
        os.environ["SD_ENABLE_ASIO"] = "1"
    if "sounddevice" not in sys.modules:
        _asio_requested_at_import = bool(os.environ.get("SD_ENABLE_ASIO"))
    import sounddevice

    if enable_asio and "ASIO" not in host_api_names(sounddevice):
        raise AudioSetupError(
            "ASIO host API is not available. It needs the pip-installed sounddevice package "
            "and no custom portaudio DLL earlier in PATH."
        )
    return sounddevice


def host_api_names(sd: Any) -> list[str]:
    return [api["name"] for api in sd.query_hostapis()]


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} ({self.host_api})"


def list_input_devices(sd: Any) -> list[InputDevice]:
    apis = sd.query_hostapis()
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=dev["name"],
                    host_api=apis[dev["hostapi"]]["name"],
                    max_input_channels=int(dev["max_input_channels"]),
                    default_samplerate=float(dev["default_samplerate"]),
                )
            )
    return devices


def default_input_index(sd: Any, host_api: str | None) -> int | None:
    """Default input device of ``host_api`` (or of the system when None)."""
    if host_api is None:
        index = sd.default.device[0]
        return int(index) if index is not None and index >= 0 else None
    wanted = canonical_host_api(host_api).lower()
    for api in sd.query_hostapis():
        if wanted in api["name"].lower():
            index = api.get("default_input_device", -1)
            return int(index) if index is not None and index >= 0 else None
    return None


def select_input_device(
    devices: list[InputDevice],
    device: str | None,
    host_api: str | None,
    default_index: int | None,
    min_channels: int = 1,
) -> InputDevice:
    """Pick an input device by index or name substring, optionally within a host API.

    Without ``device``: the default input (``default_index``) if it has ``min_channels``,
    otherwise the first candidate that has, otherwise the default (so the caller reports
    the channel shortage for a predictable device).
    """
    candidates = devices
    if host_api is not None:
        wanted = canonical_host_api(host_api).lower()
        candidates = [d for d in candidates if wanted in d.host_api.lower()]
        if not candidates:
            raise AudioSetupError(f"no input devices on host API {host_api!r}")
    if device is not None and device.strip().isdigit():
        index = int(device)
        for d in candidates:
            if d.index == index:
                return d
        raise AudioSetupError(f"input device {index} not found" + _in_api(host_api))
    if device is not None:
        needle = device.lower()
        matches = [d for d in candidates if needle in d.name.lower()]
        if not matches:
            raise AudioSetupError(f"no input device matching {device!r}" + _in_api(host_api))
        if len(matches) > 1:
            logger.info("several devices match %r, using %s", device, matches[0].label)
        return matches[0]
    if not candidates:
        raise AudioSetupError("no input devices found")
    default = next((d for d in candidates if d.index == default_index), candidates[0])
    if default.max_input_channels >= min_channels:
        return default
    big_enough = [d for d in candidates if d.max_input_channels >= min_channels]
    if big_enough:
        logger.info("default %s has too few channels, using %s", default.label, big_enough[0].label)
        return big_enough[0]
    return default


def _in_api(host_api: str | None) -> str:
    return f" on host API {host_api!r}" if host_api else ""


def parse_channel_spec(spec: str) -> tuple[int, ...]:
    """Parse 1-based channel numbers like ``"1-8"`` or ``"1,3,5-6"`` (sorted, unique)."""
    channels: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid channel range {part!r}")
            channels.update(range(lo, hi + 1))
        else:
            channels.add(int(part))
    if not channels:
        raise ValueError("no channels given")
    if min(channels) < 1:
        raise ValueError("channel numbers start at 1")
    return tuple(sorted(channels))


def format_channel_spec(channels: tuple[int, ...]) -> str:
    if channels == tuple(range(channels[0], channels[-1] + 1)):
        return f"{channels[0]}-{channels[-1]}" if len(channels) > 1 else str(channels[0])
    return ",".join(str(c) for c in channels)


@dataclass(frozen=True)
class StreamPlan:
    """How to open one input stream for the selected channels."""

    device: InputDevice
    channels: tuple[int, ...]  # 1-based, as selected by the user
    open_channels: int  # channel count passed to PortAudio
    columns: np.ndarray | None  # columns of indata to keep; None = all, in order
    extra_settings: Any


def plan_input_stream(
    sd: Any, device: InputDevice, channels: tuple[int, ...], *, wasapi_exclusive: bool = False
) -> StreamPlan:
    """Decide channel count, column mapping and host-API settings for ``channels``."""
    needed = max(channels)
    if needed > device.max_input_channels:
        hint = ""
        if "WASAPI" in device.host_api:
            hint = (
                " WASAPI may expose a multichannel interface as several stereo-pair devices;"
                " use ASIO (--host-api asio) for all channels in one stream."
            )
        raise ChannelCountError(
            f"{device.label} exposes {device.max_input_channels} input channel(s); "
            f"channels {format_channel_spec(channels)} need {needed}.{hint}"
        )
    if device.host_api == "ASIO":
        settings = sd.AsioSettings(channel_selectors=[c - 1 for c in channels])
        return StreamPlan(device, channels, len(channels), None, settings)
    settings = None
    if "WASAPI" in device.host_api:
        settings = sd.WasapiSettings(exclusive=wasapi_exclusive)
    columns = None
    if channels != tuple(range(1, needed + 1)):
        columns = np.asarray([c - 1 for c in channels], dtype=np.intp)
    return StreamPlan(device, channels, needed, columns, settings)


def check_stream_settings(sd: Any, plan: StreamPlan, samplerate: int) -> None:
    """Ask PortAudio whether the plan can be opened; raise :class:`AudioSetupError` if not."""
    try:
        sd.check_input_settings(
            device=plan.device.index,
            channels=plan.open_channels,
            dtype=SAMPLE_DTYPE,
            extra_settings=plan.extra_settings,
            samplerate=samplerate,
        )
    except Exception as exc:
        hint = ""
        if "WASAPI" in plan.device.host_api and samplerate != plan.device.default_samplerate:
            hint = (
                f" WASAPI shared mode runs at the device format "
                f"({plan.device.default_samplerate:.0f} Hz); set it to {samplerate} Hz in the "
                "Windows sound settings or use --wasapi-exclusive."
            )
        raise AudioSetupError(
            f"cannot open {plan.open_channels} channel(s) at {samplerate} Hz on "
            f"{plan.device.label}: {exc}.{hint}"
        ) from exc
