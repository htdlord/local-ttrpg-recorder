"""Events between pipeline stages and sinks (SPEC §10, live-readiness).

Batch processing emits only ``final`` segments; the ``provisional`` status and the
revision counter exist so a later live mode can replace text without new types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from sessionscribe.core.merge import TranscriptSegment
    from sessionscribe.core.pipeline.asr import Segment


@dataclass(frozen=True)
class AudioBlock:
    """Samples of one source starting at ``start_sample`` on the session timeline."""

    source_id: str
    start_sample: int
    samplerate: int
    samples: np.ndarray


@dataclass(frozen=True)
class SegmentEvent:
    source_id: str
    segment: Segment | TranscriptSegment  # track-local from ASR, session time after merge
    status: Literal["provisional", "final"] = "final"
    revision: int = 0


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    done: int
    total: int
    unit: str = "samples"


@dataclass(frozen=True)
class WarningEvent:
    stage: str
    message: str


Event = SegmentEvent | ProgressEvent | WarningEvent
EventSink = Callable[[Event], None]


def null_sink(_event: Event) -> None:
    return None
