"""Opening the real input stream, and a write-free level meter.

The recorder never picks a device by itself (SPEC §8.1): the device is named and its
host API given, explicitly or from a saved ``rig.toml``. The index is only looked up,
because it can change after a restart or an unplug. PortAudio reads its device list
once at initialisation; ``refresh`` re-initialises it so a re-plugged interface is seen.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from sessionscribe.core.audio.backend import (
    SAMPLE_DTYPE,
    AudioSetupError,
    InputDevice,
    StreamPlan,
    canonical_host_api,
    check_stream_settings,
    is_asio,
    list_input_devices,
    load_sounddevice,
    plan_input_stream,
)
from sessionscribe.core.audio.recorder import CaptureEngine, ChannelLevels, LevelEvent


def find_device(devices: list[InputDevice], name: str, host_api: str) -> InputDevice:
    """Exact name (case-insensitive) or a unique part of it, within ``host_api``."""
    wanted = canonical_host_api(host_api).lower()
    in_api = [d for d in devices if wanted in d.host_api.lower()]
    if not in_api:
        raise AudioSetupError(f"no input devices on host API {host_api!r}")
    exact = [d for d in in_api if d.name.lower() == name.lower()]
    if len(exact) == 1:
        return exact[0]
    matches = exact or [d for d in in_api if name.lower() in d.name.lower()]
    if not matches:
        raise AudioSetupError(f"no input device named {name!r} on {host_api}")
    if len(matches) > 1:
        names = "; ".join(d.label for d in matches)
        raise AudioSetupError(
            f"{name!r} matches several devices on {host_api}: {names}. Give the full name."
        )
    return matches[0]


class SoundDeviceOpener:
    def __init__(
        self,
        device: str,
        host_api: str,
        channels: tuple[int, ...],
        samplerate: int,
        blocksize: int,
        *,
        wasapi_exclusive: bool = False,
    ) -> None:
        self.name_query = device
        self.host_api_query = host_api
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.wasapi_exclusive = wasapi_exclusive
        self.sd = load_sounddevice(enable_asio=is_asio(host_api))
        self._resolve()

    def _resolve(self) -> None:
        device = find_device(list_input_devices(self.sd), self.name_query, self.host_api_query)
        plan: StreamPlan = plan_input_stream(
            self.sd, device, self.channels, wasapi_exclusive=self.wasapi_exclusive
        )
        check_stream_settings(self.sd, plan, self.samplerate)
        self.plan = plan
        self.device_name = device.name
        self.device_index = device.index
        self.host_api = device.host_api
        self.columns = plan.columns

    def open(self, callback: Callable[..., None]) -> Any:
        try:
            return self.sd.InputStream(
                device=self.device_index,
                channels=self.plan.open_channels,
                samplerate=self.samplerate,
                dtype=SAMPLE_DTYPE,
                blocksize=self.blocksize,
                callback=callback,
                extra_settings=self.plan.extra_settings,
            )
        except Exception as exc:
            raise AudioSetupError(
                f"cannot open {self.plan.open_channels} channel(s) on {self.plan.device.label}: "
                f"{exc}"
            ) from exc

    def refresh(self) -> None:
        """Re-read PortAudio's device list (sounddevice: terminate + initialize)."""
        self.sd._terminate()
        self.sd._initialize()
        self._resolve()


class Meter:
    """Level events from a stream without writing anything (meters, mic check)."""

    def __init__(self, opener: Any, n_channels: int, samplerate: int, interval_s: float = 0.25):
        self.opener = opener
        self.engine = CaptureEngine(n_channels, samplerate, samplerate * 5, opener.columns)
        self.levels = ChannelLevels(n_channels, samplerate)
        self.interval_s = interval_s

    def run(
        self,
        seconds: float | None,
        on_level: Callable[[LevelEvent], None],
        stop: threading.Event | None = None,
    ) -> None:
        stop = stop or threading.Event()
        stream = self.opener.open(self.engine.callback)
        t0 = time.monotonic()
        last = t0
        stream.start()
        try:
            while not stop.is_set():
                time.sleep(0.02)
                data = self.engine.ring.read()
                if data.shape[0]:
                    self.levels.add(data)
                now = time.monotonic()
                if now - last >= self.interval_s:
                    last = now
                    event = self.levels.take_event(round(now - t0, 3))
                    if event:
                        on_level(event)
                if self.engine.fatal:
                    raise AudioSetupError(self.engine.fatal)
                if seconds is not None and now - t0 >= seconds:
                    break
        finally:
            stream.stop()
            stream.close()


def bar(dbfs: float, width: int = 20, floor: float = -60.0) -> str:
    filled = round(width * min(1.0, max(0.0, (dbfs - floor) / -floor)))
    return "#" * filled + "." * (width - filled)
