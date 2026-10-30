"""Everything the exporters need about one processed session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sessionscribe.core.config import Settings
from sessionscribe.core.merge import RecordingGap, Transcript
from sessionscribe.core.pipeline.asr import SpeechInterval
from sessionscribe.core.session import Participant


@dataclass(frozen=True)
class ChunkInfo:
    participant: str
    index: int
    owned: tuple[int, int]  # track-local samples
    context: tuple[int, int]
    cache_key: str
    cached: bool
    transcribe_s: float
    segments: int


@dataclass
class SessionResult:
    name: str
    date: str | None
    participants: list[Participant]
    settings: Settings
    transcript: Transcript
    duration: int  # session timeline samples
    model: dict[str, Any] = field(default_factory=dict)
    languages: dict[str, str | None] = field(default_factory=dict)
    batch_sizes: list[int] = field(default_factory=list)
    chunks: list[ChunkInfo] = field(default_factory=list)
    vad: dict[str, list[SpeechInterval] | None] = field(default_factory=dict)
    gaps: list[RecordingGap] = field(default_factory=list)
    processing_events: list[dict[str, Any]] = field(default_factory=list)
    bleed_mode_applied: str = "off"
    coupling_db: list[list[float]] | None = None  # participant order, C[i][j] dB

    def name_of(self, participant_id: str) -> str:
        for p in self.participants:
            if p.id == participant_id:
                return p.display_name
        return participant_id
