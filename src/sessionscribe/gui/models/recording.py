"""The Record tab as data (no Qt): rig rows, recorder arguments, meters, alarm severity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from sessionscribe.core.audio.manifest import RigChannel, RigConfig

Severity = Literal["info", "warning", "error"]
# Alarms that end or interrupt the recording; "reconnected" is good news.
ERROR_ALARMS = frozenset({"device_lost", "disk_full", "fatal"})
INFO_ALARMS = frozenset({"reconnected"})
METER_FLOOR_DB = -60.0


@dataclass
class RigRow:
    channel: str = ""
    id: str = ""
    name: str = ""


@dataclass
class RigValidation:
    rig: RigConfig | None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rig is not None


def rows_from_rig(rig: RigConfig) -> list[RigRow]:
    return [RigRow(str(c.channel), c.id, c.name) for c in rig.channels]


def build_rig(
    device: str, host_api: str, rows: list[RigRow], base: RigConfig | None = None
) -> RigValidation:
    """A ``RigConfig`` from the table, keeping the other values of ``base``."""
    errors: list[str] = []
    if not device or not host_api:
        errors.append("choose a device")
    channels: list[RigChannel] = []
    for i, row in enumerate(rows, start=1):
        if not (row.channel.strip() or row.id.strip() or row.name.strip()):
            continue  # an empty row is ignored
        try:
            channels.append(
                RigChannel(
                    channel=int(row.channel.strip()), id=row.id.strip(), name=row.name.strip()
                )
            )
        except (ValueError, ValidationError):
            errors.append(f"row {i}: channel must be a number >= 1 and the id is required")
    if not channels:
        errors.append("assign at least one channel")
    if errors:
        return RigValidation(None, errors)
    data = base.model_dump() if base is not None else {}
    data.update(device=device, host_api=host_api, channels=[c.model_dump() for c in channels])
    try:
        return RigValidation(RigConfig.model_validate(data))
    except ValidationError as exc:
        return RigValidation(
            None, [str(e["msg"]).removeprefix("Value error, ") for e in exc.errors()]
        )


def child_args(rig_path: Path, *, hours: float, check: bool, force: bool) -> list[str]:
    args = ["--rig", str(rig_path), "--hours", f"{hours:g}"]
    if not check:
        args.append("--no-check")
    if force:
        args.append("--force")
    return args


@dataclass(frozen=True)
class Meter:
    participant: str
    name: str
    peak_db: float
    rms_db: float
    clipped: bool

    def fraction(self, db: float) -> float:
        """Position on the meter (0 at the floor, 1 at full scale)."""
        return min(1.0, max(0.0, (db - METER_FLOOR_DB) / -METER_FLOOR_DB))


def meters(status: dict[str, Any]) -> list[Meter]:
    ids = list(status.get("participants") or [])
    names = list(status.get("names") or ids)
    levels = status.get("levels") or {}
    peak = list(levels.get("peak") or [])
    rms = list(levels.get("rms") or [])
    clipped = list(levels.get("clipped") or [])
    out = []
    for i, pid in enumerate(ids):
        out.append(
            Meter(
                pid,
                names[i] if i < len(names) else pid,
                float(peak[i]) if i < len(peak) else METER_FLOOR_DB,
                float(rms[i]) if i < len(rms) else METER_FLOOR_DB,
                bool(clipped[i]) if i < len(clipped) else False,
            )
        )
    return out


def alarm_severity(event: dict[str, Any]) -> Severity:
    kind = str(event.get("kind", ""))
    if event.get("fatal") or kind in ERROR_ALARMS:
        return "error"
    return "info" if kind in INFO_ALARMS else "warning"


def format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
