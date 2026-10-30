"""Session description (``session.toml``): participants, their sources and offsets."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sessionscribe.core.audio.manifest import RecordingManifest
from sessionscribe.core.audio.track import AUDIO_SUFFIXES, VirtualTrack, audio_files
from sessionscribe.core.config import read_toml
from sessionscribe.core.fileio import atomic_write_text
from sessionscribe.core.timeline import TIMELINE_RATE, rescale, seconds_to_samples

if TYPE_CHECKING:
    from sessionscribe.core.merge import RecordingGap

SESSION_FILE = "session.toml"
SESSION_SCHEMA_VERSION = 1
MAX_PARTICIPANTS = 8
PARTICIPANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# chNN_name.wav from the recorder / bench clips; the channel number is dropped.
CHANNEL_NAME = re.compile(r"^ch(?P<channel>\d+)[_-](?P<name>.+)$", re.IGNORECASE)


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    source: str = Field(description="File or folder of segment files, relative to the session.")
    channel: int | None = Field(None, ge=1, description="1-based channel of a multi-channel file.")
    offset_s: float = Field(0.0, ge=0, description="Track start on the session timeline.")
    color: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.id

    @property
    def offset_samples(self) -> int:
        return seconds_to_samples(self.offset_s, TIMELINE_RATE)


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SESSION_SCHEMA_VERSION
    name: str = ""
    date: str | None = None
    preset: str | None = None
    settings: dict[str, Any] = Field(
        default_factory=dict, description="Settings overrides, layered after --config."
    )
    participants: list[Participant] = Field(min_length=1, max_length=MAX_PARTICIPANTS)
    # Bleed handling (SPEC §5.2): guided coupling from a mic check (participant ->
    # [start_s, end_s] of session time where only they speak), or a manual matrix in
    # participant order (C[i][j] dB: j's voice on mic i relative to mic j).
    mic_check: dict[str, tuple[float, float]] = Field(default_factory=dict)
    coupling_db: list[list[float]] | None = None
    recording: RecordingManifest | None = None  # written by the recorder (SPEC §8.2)

    @model_validator(mode="after")
    def _check(self) -> SessionConfig:
        if self.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported session schema_version {self.schema_version}")
        ids = [p.id for p in self.participants]
        for pid in ids:
            if not PARTICIPANT_ID.match(pid):
                raise ValueError(f"participant id {pid!r}: use letters, digits, _ and -")
        if len(set(ids)) != len(ids):
            raise ValueError("participant ids must be unique")
        for pid, (start, end) in self.mic_check.items():
            if pid not in ids:
                raise ValueError(f"mic_check names unknown participant {pid!r}")
            if not 0 <= start < end:
                raise ValueError(f"mic_check range of {pid!r} must satisfy 0 <= start < end")
        if self.coupling_db is not None and (
            len(self.coupling_db) != len(ids) or any(len(r) != len(ids) for r in self.coupling_db)
        ):
            raise ValueError(f"coupling_db must be a {len(ids)} x {len(ids)} matrix")
        return self


class Session:
    def __init__(self, folder: Path, config: SessionConfig) -> None:
        self.folder = Path(folder)
        self.config = config

    @classmethod
    def load(cls, path: Path) -> Session:
        path = Path(path)
        if path.is_dir():
            path = path / SESSION_FILE
        return cls(path.parent, SessionConfig.model_validate(read_toml(path)))

    @property
    def name(self) -> str:
        return self.config.name or self.folder.name

    @property
    def participants(self) -> list[Participant]:
        return self.config.participants

    def source_path(self, participant: Participant) -> Path:
        return self.folder / participant.source

    def open_track(self, participant: Participant, *, full_hash: bool = False) -> VirtualTrack:
        return VirtualTrack.open(
            self.source_path(participant), participant.channel, full_hash=full_hash
        )

    def gaps(self) -> list[RecordingGap]:
        """Recording gaps on every track, in track-local timeline samples."""
        from sessionscribe.core.merge import RecordingGap

        rec = self.config.recording
        if rec is None:
            return []
        out = []
        for g in rec.gaps:
            out.append(
                RecordingGap(
                    rescale(g.start_sample, rec.samplerate, TIMELINE_RATE),
                    rescale(g.length_samples, rec.samplerate, TIMELINE_RATE),
                    g.reason,
                )
            )
        return out

    def work_dir(self, configured: Path | None = None) -> Path:
        if configured is None:
            return self.folder / "work"
        return configured if configured.is_absolute() else self.folder / configured


def write_session(path: Path, config: SessionConfig) -> None:
    atomic_write_text(path, tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)))


def _participant_id(stem: str) -> tuple[str, str]:
    """(id, name) suggested from a file or folder name."""
    m = CHANNEL_NAME.match(stem)
    name = m["name"] if m else stem
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    pid = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_-") or "p"
    return pid, name


def suggest_session(folder: Path) -> SessionConfig:
    """Participants from the audio files or segment subfolders of ``folder`` (workflow A)."""
    folder = Path(folder)
    tracks = folder / "tracks"
    if tracks.is_dir():  # bench clip layout
        config = _with_prefix(suggest_session(tracks), "tracks/")
        config.name = folder.name
        return config
    sources: list[Path] = [
        p
        for p in sorted(folder.iterdir(), key=lambda p: p.name)
        if (p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES) or (p.is_dir() and audio_files(p))
    ]
    if not sources:
        raise FileNotFoundError(f"no audio files or segment folders in {folder}")
    participants: list[Participant] = []
    seen: set[str] = set()
    for src in sources:
        pid, name = _participant_id(src.stem if src.is_file() else src.name)
        base, n = pid, 2
        while pid in seen:
            pid, n = f"{base}_{n}", n + 1
        seen.add(pid)
        participants.append(Participant(id=pid, name=name, source=src.name))
    return SessionConfig(name=folder.name, participants=participants)


def _with_prefix(config: SessionConfig, prefix: str) -> SessionConfig:
    for p in config.participants:
        p.source = prefix + p.source
    return config
