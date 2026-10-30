"""Keep the computer awake while recording (owner request, M5).

Windows: ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` from the
recording thread, cleared on release. Linux: ``systemd-inhibit --what=sleep:idle`` as a
child process for as long as the recording runs. Elsewhere, or when neither works, the
inhibitor is inactive and says why, so the CLI can warn.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class SleepInhibitor:
    def __init__(self, release: Callable[[], None] | None, active: bool, how: str) -> None:
        self._release = release
        self.active = active
        self.how = how

    def release(self) -> None:
        if self._release is not None:
            try:
                self._release()
            except Exception:  # never fail a recording over this
                log.warning("could not release the sleep inhibitor", exc_info=True)
            self._release = None
        self.active = False


def _windows_state() -> Callable[[int], int]:
    import ctypes

    # getattr: windll exists only on Windows, and mypy also checks this on Linux.
    kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009
    fn = kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32
    return lambda flags: int(fn(flags))


def inhibit_sleep(
    *,
    platform: str = sys.platform,
    set_state: Callable[[int], int] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    popen: Callable[..., Any] = subprocess.Popen,
) -> SleepInhibitor:
    if platform == "win32":
        try:
            setter = set_state or _windows_state()
            if setter(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
                return SleepInhibitor(None, False, "SetThreadExecutionState failed")
        except Exception as exc:
            return SleepInhibitor(None, False, f"SetThreadExecutionState unavailable: {exc}")

        def clear() -> None:
            setter(ES_CONTINUOUS)

        return SleepInhibitor(clear, True, "SetThreadExecutionState")
    if platform.startswith("linux"):
        exe = which("systemd-inhibit")
        if exe is None:
            return SleepInhibitor(None, False, "systemd-inhibit not found")
        try:
            proc = popen(
                [
                    exe,
                    "--what=sleep:idle",
                    "--who=sessionscribe",
                    "--why=recording a session",
                    "--mode=block",
                    "sleep",
                    "infinity",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            return SleepInhibitor(None, False, f"systemd-inhibit failed: {exc}")

        def stop() -> None:
            proc.terminate()
            proc.wait(timeout=5)

        return SleepInhibitor(stop, True, "systemd-inhibit")
    return SleepInhibitor(None, False, f"no sleep inhibitor for {platform}")
