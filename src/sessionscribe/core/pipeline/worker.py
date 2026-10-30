"""ASR work for one chunk, run in a child process that survives nothing (SPEC §5.7).

The parent sends jobs over a pipe; a long-lived child loads the model once, reads the
chunk audio itself (no large arrays between processes), transcribes, writes the result
to the cache atomically and replies. If the child dies (non-zero exit code, Windows
NTSTATUS such as 0xC0000005, a signal, or a closed pipe), the chunk's partial files are
discarded and the chunk is re-run once in a fresh child; a second death is a clear error.
``InlineRunner`` runs the same job function in-process (tests, debugging).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sessionscribe.core.audio.resample import resample
from sessionscribe.core.audio.track import VirtualTrack
from sessionscribe.core.cache import Cache, CacheKey
from sessionscribe.core.config import Settings
from sessionscribe.core.events import EventSink, ProgressEvent, WarningEvent, null_sink
from sessionscribe.core.pipeline.asr import (
    AsrEngine,
    AsrError,
    Cancelled,
    Segment,
    SpeechInterval,
    TranscriptionResult,
    Word,
)
from sessionscribe.core.pipeline.chunks import Chunk
from sessionscribe.core.pipeline.refill import refill_holes
from sessionscribe.core.timeline import TIMELINE_RATE, WHISPER_RATE, rescale

log = logging.getLogger(__name__)

POLL_S = 0.5
STOP_TIMEOUT_S = 10.0


class Engine(Protocol):
    def transcribe(
        self,
        audio: np.ndarray,
        *,
        source_id: str,
        sink: EventSink = ...,
        cancel: Callable[[], bool] | None = ...,
    ) -> TranscriptionResult: ...


EngineFactory = Callable[[Settings], Engine]


def default_engine_factory(settings: Settings) -> Engine:
    return AsrEngine.load(settings)


# --- jobs and results ---------------------------------------------------------------------


@dataclass(frozen=True)
class AsrJob:
    participant: str
    source: str
    channel: int | None
    chunk: Chunk
    key: CacheKey

    def describe(self) -> str:
        a, b = self.chunk.owned_start / TIMELINE_RATE, self.chunk.owned_end / TIMELINE_RATE
        return f"{self.participant} chunk {self.chunk.index} ({a:.1f}-{b:.1f} s track time)"


@dataclass
class ChunkResult:
    segments: list[Segment]
    vad_intervals: list[SpeechInterval]
    language: str | None = None
    language_probability: float | None = None
    batch_size: int | None = None
    transcribe_s: float = 0.0
    vad_s: float = 0.0
    model: dict[str, Any] = field(default_factory=dict)
    # VAD speech still without words after re-transcription (track-local samples).
    missed: list[tuple[int, int]] = field(default_factory=list)
    refill_holes: int = 0


def encode_result(result: ChunkResult) -> dict[str, Any]:
    return {
        "segments": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
                "temperature": s.temperature,
                "words": [[w.start, w.end, w.text, w.probability] for w in s.words],
                "refilled": s.refilled,
            }
            for s in result.segments
        ],
        "missed": [list(m) for m in result.missed],
        "refill_holes": result.refill_holes,
        "vad_intervals": [[v.start, v.end] for v in result.vad_intervals],
        "language": result.language,
        "language_probability": result.language_probability,
        "batch_size": result.batch_size,
        "transcribe_s": result.transcribe_s,
        "vad_s": result.vad_s,
        "model": result.model,
    }


def decode_result(data: dict[str, Any]) -> ChunkResult:
    return ChunkResult(
        segments=[
            Segment(
                index=s["index"],
                start=s["start"],
                end=s["end"],
                text=s["text"],
                avg_logprob=s["avg_logprob"],
                no_speech_prob=s["no_speech_prob"],
                compression_ratio=s["compression_ratio"],
                temperature=s["temperature"],
                words=tuple(Word(a, b, t, p) for a, b, t, p in s["words"]),
                refilled=s.get("refilled", False),
            )
            for s in data["segments"]
        ],
        vad_intervals=[SpeechInterval(a, b) for a, b in data["vad_intervals"]],
        language=data.get("language"),
        language_probability=data.get("language_probability"),
        batch_size=data.get("batch_size"),
        transcribe_s=data.get("transcribe_s", 0.0),
        vad_s=data.get("vad_s", 0.0),
        model=data.get("model", {}),
        missed=[(a, b) for a, b in data.get("missed", [])],
        refill_holes=data.get("refill_holes", 0),
    )


def chunk_audio(track: VirtualTrack, chunk: Chunk) -> np.ndarray:
    """Context range of a chunk as 16 kHz mono float32, resampled once."""
    rate = track.samplerate
    start = rescale(chunk.ctx_start, TIMELINE_RATE, rate)
    end = rescale(chunk.ctx_end, TIMELINE_RATE, rate)
    return resample(track.read(start, end - start), rate, WHISPER_RATE)


def _shift(seg: Segment, offset: int) -> Segment:
    return replace(
        seg,
        start=seg.start + offset,
        end=seg.end + offset,
        words=tuple(replace(w, start=w.start + offset, end=w.end + offset) for w in seg.words),
    )


def run_job(
    engine: Engine,
    job: AsrJob,
    cache: Cache,
    settings: Settings,
    sink: EventSink = null_sink,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Transcribe one chunk (plus refills of word-less speech) and store it; small stats only."""
    track = VirtualTrack.open(Path(job.source), job.channel)
    audio = chunk_audio(track, job.chunk)
    result = engine.transcribe(audio, source_id=job.participant, sink=sink, cancel=cancel)
    offset = job.chunk.ctx_start
    owned = (job.chunk.owned_start - offset, job.chunk.owned_end - offset)
    refill = refill_holes(
        engine, audio, result, settings, source_id=job.participant, owned=owned, cancel=cancel
    )
    chunk_result = ChunkResult(
        segments=[_shift(s, offset) for s in [*result.segments, *refill.segments]],
        vad_intervals=[
            SpeechInterval(v.start + offset, v.end + offset) for v in result.vad_intervals
        ],
        language=result.language,
        language_probability=result.language_probability,
        batch_size=result.batch_size,
        transcribe_s=result.transcribe_s + refill.transcribe_s,
        vad_s=result.vad_s,
        model=asdict(result.model),
        missed=[(a + offset, b + offset) for a, b in refill.missed],
        refill_holes=refill.holes,
    )
    cache.store_json(job.key, encode_result(chunk_result))
    return {
        "segments": len(chunk_result.segments),
        "refilled": len(refill.segments),
        "transcribe_s": round(chunk_result.transcribe_s, 3),
    }


# --- exit codes ---------------------------------------------------------------------------

NTSTATUS = {
    0xC0000005: "access violation",
    0xC0000017: "out of memory",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC0000374: "heap corruption",
    0xC0000409: "stack buffer overrun",
}


def describe_exit(code: int | None) -> str:
    if code is None:
        return "no exit code (process vanished)"
    if -128 < code < 0:  # POSIX: killed by a signal
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    unsigned = code & 0xFFFFFFFF
    if unsigned in NTSTATUS:
        return f"exit code 0x{unsigned:08X} ({NTSTATUS[unsigned]})"
    return f"exit code {code}"


class WorkerDied(RuntimeError):
    def __init__(self, exitcode: int | None) -> None:
        super().__init__(describe_exit(exitcode))
        self.exitcode = exitcode


class WorkerCrashed(AsrError):
    """The same chunk killed its worker twice."""

    def __init__(self, job: AsrJob, exitcode: int | None) -> None:
        self.job = job
        self.exitcode = exitcode
        super().__init__(
            f"ASR worker crashed twice on {job.describe()}: {describe_exit(exitcode)}. "
            "Likely cause: GPU memory at its limit or a driver/library fault. Completed chunks "
            "are cached; re-running resumes after them. Try a smaller model or compute type "
            "(e.g. int8_float16), quality mode, or close other GPU programs."
        )


# --- runners ------------------------------------------------------------------------------


class InlineRunner:
    """Runs jobs in this process; a crash here takes the whole program down."""

    def __init__(self, settings: Settings, cache: Cache, factory: EngineFactory) -> None:
        self.settings, self.cache, self.factory = settings, cache, factory
        self._engine: Engine | None = None

    def run(
        self, job: AsrJob, sink: EventSink, cancel: Callable[[], bool] | None
    ) -> dict[str, Any]:
        if self._engine is None:
            self._engine = self.factory(self.settings)
        return run_job(self._engine, job, self.cache, self.settings, sink, cancel)

    def close(self) -> None:
        self._engine = None


def _worker_main(conn: Any, settings_json: str, cache_root: str, factory: EngineFactory) -> None:
    settings = Settings.model_validate_json(settings_json)
    cache = Cache(Path(cache_root))
    try:
        engine = factory(settings)
    except Exception as exc:
        conn.send(("error", type(exc).__name__, str(exc)))
        return
    conn.send(("ready",))
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return

        def sink(event: Any) -> None:
            if isinstance(event, ProgressEvent):
                conn.send(("progress", event.done, event.total))
            elif isinstance(event, WarningEvent):
                conn.send(("warning", event.stage, event.message))

        try:
            conn.send(("done", run_job(engine, job, cache, settings, sink)))
        except Exception as exc:
            conn.send(("error", type(exc).__name__, str(exc)))


class ProcessRunner:
    """One long-lived child process holding the model."""

    def __init__(self, settings: Settings, cache: Cache, factory: EngineFactory) -> None:
        self.settings, self.cache, self.factory = settings, cache, factory
        self._ctx = mp.get_context("spawn")
        self._proc: Any = None
        self._conn: Any = None

    def _start(self, cancel: Callable[[], bool] | None) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child, self.settings.model_dump_json(), str(self.cache.root), self.factory),
            daemon=True,
            name="sessionscribe-asr",
        )
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent
        msg = self._receive(null_sink, cancel)
        if msg[0] == "error":
            self.close()
            raise AsrError(f"ASR worker could not load the model: {msg[2]}")

    def _receive(self, sink: EventSink, cancel: Callable[[], bool] | None) -> tuple[Any, ...]:
        """Next non-progress message; raises WorkerDied when the child is gone."""
        while True:
            if cancel is not None and cancel():
                self.kill()
                raise Cancelled("transcription cancelled")
            try:
                ready = self._conn.poll(POLL_S)
            except (EOFError, OSError):
                ready = False
                self._proc.join(STOP_TIMEOUT_S)
            if ready:
                try:
                    msg = self._conn.recv()
                except (EOFError, OSError):
                    self._proc.join(STOP_TIMEOUT_S)
                    raise WorkerDied(self._proc.exitcode) from None
                if msg[0] == "progress":
                    sink(ProgressEvent("asr", msg[1], msg[2]))
                elif msg[0] == "warning":
                    sink(WarningEvent(msg[1], msg[2]))
                else:
                    return tuple(msg)
            elif not self._proc.is_alive():
                if self._conn.poll(0):  # a last message may still be in the pipe
                    continue
                raise WorkerDied(self._proc.exitcode)

    def run(
        self, job: AsrJob, sink: EventSink, cancel: Callable[[], bool] | None
    ) -> dict[str, Any]:
        if self._proc is None or not self._proc.is_alive():
            self._start(cancel)
        try:
            self._conn.send(job)
        except (BrokenPipeError, OSError):
            self._proc.join(STOP_TIMEOUT_S)
            raise WorkerDied(self._proc.exitcode) from None
        msg = self._receive(sink, cancel)
        if msg[0] == "done":
            stats: dict[str, Any] = msg[1]
            return stats
        name, text = msg[1], msg[2]
        if name == "Cancelled":
            raise Cancelled(text)
        raise AsrError(text)

    def kill(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(STOP_TIMEOUT_S)
        self._proc = None

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            with contextlib.suppress(OSError):
                self._conn.send(None)
            self._proc.join(STOP_TIMEOUT_S)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(STOP_TIMEOUT_S)
        self._proc = None


Runner = InlineRunner | ProcessRunner


@dataclass(frozen=True)
class CrashEvent:
    """Recorded in the output JSON (``processing_events``); never contains text."""

    kind: str  # "worker_crash" | "worker_retry_ok" | "worker_crash_fatal"
    participant: str
    chunk: int
    detail: str
    time: float = field(default_factory=time.time)


def run_with_retry(
    runner: Runner,
    job: AsrJob,
    cache: Cache,
    sink: EventSink,
    cancel: Callable[[], bool] | None,
    events: list[CrashEvent],
) -> dict[str, Any]:
    """Run a job; after a worker death, discard partial files and re-run it once."""
    try:
        return runner.run(job, sink, cancel)
    except WorkerDied as first:
        removed = cache.discard_partial(job.key)
        log.warning("ASR worker died on %s: %s; retrying once", job.describe(), first)
        events.append(CrashEvent("worker_crash", job.participant, job.chunk.index, str(first)))
        sink(WarningEvent("asr", f"ASR worker died on {job.describe()} ({first}); retrying once"))
        if removed:
            log.info("discarded %d partial cache files", removed)
    try:
        stats = runner.run(job, sink, cancel)
    except WorkerDied as second:
        cache.discard_partial(job.key)
        events.append(
            CrashEvent("worker_crash_fatal", job.participant, job.chunk.index, str(second))
        )
        raise WorkerCrashed(job, second.exitcode) from second
    events.append(
        CrashEvent("worker_retry_ok", job.participant, job.chunk.index, "re-run succeeded")
    )
    return stats
