"""Reference clip folder: ``clip.toml`` with language and track -> speaker mapping."""

from __future__ import annotations

import re
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sessionscribe.core.audio.reader import track_info
from sessionscribe.core.config import read_toml
from sessionscribe.core.fileio import atomic_write_text

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
