"""Reference clip folder: ``clip.toml`` with language and track -> speaker mapping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ttrpg_recorder.core.audio.reader import track_info
from ttrpg_recorder.core.config import read_toml
from ttrpg_recorder.core.fileio import atomic_write_text

if TYPE_CHECKING:
    from ttrpg_recorder.core.session import Session

CLIP_FILE = "clip.toml"
TRACK_NAME = re.compile(r"^ch(?P<channel>\d+)_(?P<speaker>[A-Za-z0-9_]+)$")
SPEAKER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


class ClipTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="Path relative to the clip folder.")
    speaker: str
    name: str = Field("", description="Display name (personal data; stays in the clip folder).")
    channel: int | None = Field(None, description="Interface channel, for importing old data.")


class ClipConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    language: str
    duration_s: float = Field(gt=0)
    tracks: list[ClipTrack] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> ClipConfig:
        speakers = [t.speaker for t in self.tracks]
        if len(set(speakers)) != len(speakers):
            raise ValueError("each speaker must have exactly one track")
        for s in speakers:
            if not SPEAKER_ID.match(s):
                raise ValueError(f"speaker id {s!r}: use letters, digits and _ (not leading _)")
        return self

    @property
    def speakers(self) -> list[str]:
        return [t.speaker for t in self.tracks]

    def track_for_channel(self, channel: int) -> ClipTrack | None:
        return next((t for t in self.tracks if t.channel == channel), None)


class Clip:
    def __init__(self, folder: Path, config: ClipConfig) -> None:
        self.folder = folder
        self.config = config

    @classmethod
    def load(cls, folder: Path) -> Clip:
        path = folder / CLIP_FILE
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; create it with 'bench clip init'")
        return cls(folder, ClipConfig.model_validate(read_toml(path)))

    def track_path(self, track: ClipTrack) -> Path:
        return self.folder / track.file

    @property
    def draft_dir(self) -> Path:
        return self.folder / "draft"

    @property
    def labels_dir(self) -> Path:
        return self.folder / "labels"

    @property
    def reference_path(self) -> Path:
        return self.folder / "reference.json"


def init_clip(folder: Path, language: str, names: dict[int, str] | None = None) -> ClipConfig:
    """Build ``clip.toml`` from ``tracks/chNN_<speaker>.wav|flac`` file names."""
    tracks: list[ClipTrack] = []
    durations: list[float] = []
    files = sorted(
        p for p in (folder / "tracks").glob("*") if p.suffix.lower() in {".wav", ".flac"}
    )
    if not files:
        raise FileNotFoundError(f"no WAV/FLAC files in {folder / 'tracks'}")
    for path in files:
        m = TRACK_NAME.match(path.stem)
        if not m:
            raise ValueError(f"{path.name}: expected chNN_<speaker>.wav (e.g. ch01_gm.wav)")
        channel = int(m["channel"])
        tracks.append(
            ClipTrack(
                file=path.relative_to(folder).as_posix(),
                speaker=m["speaker"],
                name=(names or {}).get(channel, ""),
                channel=channel,
            )
        )
        durations.append(track_info(path).duration_s)
    config = ClipConfig(
        id=folder.name, language=language, duration_s=round(max(durations), 3), tracks=tracks
    )
    write_clip(folder, config)
    return config


def write_clip(folder: Path, config: ClipConfig) -> None:
    atomic_write_text(
        folder / CLIP_FILE, tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )


ACTIVE_FRAME_S = 0.03
ACTIVE_THRESHOLD_DBFS = -40.0


def active_seconds(path: Path) -> float:
    """Cheap workload estimate: seconds of 30 ms frames above -40 dBFS RMS (bleed included).

    Used to balance tracks across parallel workers; not a speech detector.
    """
    import numpy as np

    from ttrpg_recorder.core.audio.reader import read_blocks

    info = track_info(path)
    frame = max(1, round(ACTIVE_FRAME_S * info.samplerate))
    threshold = 10 ** (ACTIVE_THRESHOLD_DBFS / 20)
    active = 0
    rest = np.zeros(0, np.float32)
    for block in read_blocks(path, None if info.channels == 1 else 1, block_frames=frame * 4096):
        data = np.concatenate([rest, block])
        n = len(data) // frame
        frames = data[: n * frame].reshape(n, frame).astype(np.float64)
        active += int(np.count_nonzero(np.sqrt(np.mean(frames**2, axis=1)) > threshold))
        rest = data[n * frame :]
    return active * frame / info.samplerate


def cut_clip(
    session: Session,
    folder: Path,
    start_s: float,
    duration_s: float,
    language: str,
) -> ClipConfig:
    """Cut ``[start_s, start_s + duration_s)`` of session time from every participant's
    track (segment folders and offsets resolved) into a new clip folder.

    Tracks are written as 24-bit FLAC ``tracks/chNN_<participant>.flac`` in participant
    order; stretches outside a track (offset, end of recording) are written as silence so
    every track covers the same time. The originals are only read.
    """
    import numpy as np
    import soundfile as sf

    if duration_s <= 0 or start_s < 0:
        raise ValueError("need start >= 0 and a positive duration")
    tracks_dir = folder / "tracks"
    if tracks_dir.exists() and any(tracks_dir.iterdir()):
        raise FileExistsError(f"{tracks_dir} is not empty")
    tracks_dir.mkdir(parents=True, exist_ok=True)
    names: dict[int, str] = {}
    for i, participant in enumerate(session.participants, start=1):
        track = session.open_track(participant)
        rate = track.samplerate
        frames = round(duration_s * rate)
        first = round((start_s - participant.offset_s) * rate)  # track-local
        lead = min(frames, max(0, -first))
        path = tracks_dir / f"ch{i:02d}_{participant.id}.flac"
        with sf.SoundFile(
            str(path), "w", samplerate=rate, channels=1, format="FLAC", subtype="PCM_24"
        ) as out:
            if lead:
                out.write(np.zeros(lead, np.float32))
            written = lead
            for block in track.blocks(max(0, first), frames - lead):
                out.write(block)
                written += len(block)
            if written < frames:
                out.write(np.zeros(frames - written, np.float32))
        names[i] = participant.name
    return init_clip(folder, language, names)
