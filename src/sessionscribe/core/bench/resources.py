"""Peak RAM of this process and GPU memory sampling via nvidia-smi (no extra packages)."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
from ctypes import wintypes


def peak_rss_mb() -> float | None:
    """Peak resident memory of the current process in MiB."""
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 2**20 if ok else None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    except (ImportError, OSError):
        return None


def _smi(args: list[str]) -> list[str] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *args, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines() if out.returncode == 0 else None


def gpu_used_mb() -> float | None:
    lines = _smi(["--query-gpu=memory.used"])
    try:
        return float(lines[0]) if lines else None
    except ValueError:
        return None


def process_gpu_mb(pid: int) -> float | None:
    """Per-process usage; None when the driver does not report it (common on WDDM)."""
    for line in _smi(["--query-compute-apps=pid,used_memory"]) or []:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


class GpuSampler:
    """Polls nvidia-smi while a child process runs; keeps the peak.

    Method ``per-process`` when the driver reports the child's usage, otherwise
    ``device-delta`` (device total minus the baseline before the child started).
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.baseline = gpu_used_mb()
        self.peak_process: float | None = None
        self.peak_device: float | None = None
        self._stop = threading.Event()
        self._pid: int | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def available(self) -> bool:
        return self.baseline is not None

    def start(self, pid: int) -> None:
        self._pid = pid
        if self.available:
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pid is not None:
                proc = process_gpu_mb(self._pid)
                if proc is not None:
                    self.peak_process = max(self.peak_process or 0.0, proc)
            dev = gpu_used_mb()
            if dev is not None:
                self.peak_device = max(self.peak_device or 0.0, dev)
            self._stop.wait(self.interval_s)

    def stop(self) -> tuple[float | None, str | None]:
        """Peak MiB and the method used."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if self.peak_process is not None:
            return round(self.peak_process), "per-process"
        if self.peak_device is not None and self.baseline is not None:
            return round(self.peak_device - self.baseline), "device-delta"
        return None, None
