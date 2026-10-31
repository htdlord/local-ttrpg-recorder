"""Recording manifest (SPEC §8.2) and recording profile (``rig.toml``).

The manifest lives in the session file as ``[recording]``: device and stream settings,
every segment file with its start sample, frame count, hash and status, every gap
(lost or padded audio) and every alarm. It is rewritten atomically after each closed
segment, gap or state change, so it always describes what exists on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ttrpg_recorder.core.config import read_toml
from ttrpg_recorder.core.fileio import atomic_write_text

RecordingState = Literal["recording", "stopped", "interrupted"]
SegmentStatus = Literal["open", "closed", "recovered", "damaged"]
FileFormat = Literal["flac", "rf64"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentRecord(_Model):
    channel: int  # 1-based interface channel
    participant: str
    index: int
    file: str  # relative to the session folder, posix
    start_sample: int  # on the recording timeline (frames at the stream rate)
    frames: int = 0
    sha256: str | None = None
    status: SegmentStatus = "open"


class GapRecord(_Model):
    start_sample: int
    length_samples: int  # 0 = unknown length
    reason: str  # buffer_overflow | input_overflow | device_lost | process_killed
    padded: bool = False  # the files contain silence for this gap
    estimated: bool = False  # length is an estimate (from timestamps or wall time)


class AlarmRecord(_Model):
    time_s: float  # recording time
    kind: str  # no_audio | device_lost | idle_channel | disk_low | ...
    channel: int | None = None
    message: str = ""


class RecordingMetrics(_Model):
    frames: int = 0
    blocks: int = 0
    input_overflows: int = 0
    dropped_frames: int = 0
    padded_frames: int = 0
    max_buffer_fill: float = 0.0
    max_write_s: float = 0.0
    timestamp_discontinuities: int = 0
    reconnects: int = 0


class RecordingManifest(_Model):
    device_name: str
    device_index: int
    host_api: str
    channels: list[int]  # interface channels in file order
    participants: list[str]  # participant id per channel
    samplerate: int
    actual_samplerate: float | None = None
    dtype: str = "int32"
    subtype: str = "PCM_24"
    format: FileFormat = "flac"
    blocksize: int = 0
    latency_s: float | None = None
    started_utc: str = ""
    segment_s: float = 300.0
    state: RecordingState = "recording"
    metrics: RecordingMetrics = Field(default_factory=RecordingMetrics)
    segments: list[SegmentRecord] = Field(default_factory=list)
    gaps: list[GapRecord] = Field(default_factory=list)
    alarms: list[AlarmRecord] = Field(default_factory=list)

    def segments_for(self, channel: int) -> list[SegmentRecord]:
        return [s for s in self.segments if s.channel == channel]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- recording profile ----------------------------------------------------------------------


class RigChannel(_Model):
    channel: int = Field(ge=1)
    id: str
    name: str = ""


class RigConfig(_Model):
    """A saved interface setup: device, host API and channel -> participant assignment."""

    schema_version: int = 1
    device: str = Field(description="Device name (exact, or a unique part of it).")
    host_api: str
    samplerate: int = 48_000
    blocksize: int = 480
    wasapi_exclusive: bool = False
    segment_min: float = Field(5.0, gt=0)
    format: FileFormat = "flac"
    idle_warn_min: float = Field(10.0, gt=0)
    channels: list[RigChannel] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> RigConfig:
        nums = [c.channel for c in self.channels]
        ids = [c.id for c in self.channels]
        if len(set(nums)) != len(nums):
            raise ValueError("each interface channel may be assigned once")
        if len(set(ids)) != len(ids):
            raise ValueError("participant ids must be unique")
        return self


def read_rig(path: Path) -> RigConfig:
    return RigConfig.model_validate(read_toml(path))


def write_rig(path: Path, rig: RigConfig) -> None:
    atomic_write_text(path, tomli_w.dumps(rig.model_dump(mode="json", exclude_none=True)))
