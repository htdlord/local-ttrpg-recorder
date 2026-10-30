"""Session processing: features -> chunk plan -> ASR (cached) -> merge -> flags.

``process_session`` runs ASR for chunks missing from the cache; ``export_session`` uses
the cache only and never transcribes. Neither writes output files; see
``exporters.formats.write_session_outputs``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from sessionscribe.core.audio.track import VirtualTrack
from sessionscribe.core.cache import Cache, CacheKey, make_key
from sessionscribe.core.config import Settings, resolve_runtime
from sessionscribe.core.events import (
    Event,
    EventSink,
    ProgressEvent,
    SegmentEvent,
    WarningEvent,
    null_sink,
)
from sessionscribe.core.exporters.single import _library_versions
from sessionscribe.core.merge import RecordingGap, TrackInput, merge_tracks
from sessionscribe.core.pipeline import PROCESSING_VERSION
from sessionscribe.core.pipeline.asr import (
    AsrError,
    Segment,
    SpeechInterval,
    resolve_model_path,
    transcribe_kwargs,
)
from sessionscribe.core.pipeline.chunks import Chunk, plan_chunks, resolve_chunks
from sessionscribe.core.pipeline.features import frame_samples, track_envelope
from sessionscribe.core.pipeline.worker import (
    AsrJob,
    ChunkResult,
    CrashEvent,
    EngineFactory,
    InlineRunner,
    ProcessRunner,
    decode_result,
    default_engine_factory,
    run_with_retry,
)
from sessionscribe.core.result import ChunkInfo, SessionResult
from sessionscribe.core.session import Participant, Session
from sessionscribe.core.timeline import TIMELINE_RATE, rescale

log = logging.getLogger(__name__)

ModelIdentity = Callable[[Settings], dict[str, Any]]
Isolation = Literal["process", "inline"]


def default_model_identity(settings: Settings) -> dict[str, Any]:
    """What identifies the model in cache keys: id, Hub revision, device, compute type."""
    runtime = resolve_runtime(settings)
    m = settings.model
    _, revision, _ = resolve_model_path(m.id, m.model_dir, m.revision, m.offline)
    return {
        "id": m.id,
        "revision": revision,
        "device": runtime.device,
        "compute_type": runtime.compute_type,
    }


def asr_key(
    track: VirtualTrack, chunk: Chunk, settings: Settings, model: dict[str, Any]
) -> CacheKey:
    """SPEC §5.7 asr layer. Participant id, name and offset are deliberately absent."""
    rate = track.samplerate
    start = rescale(chunk.ctx_start, TIMELINE_RATE, rate)
    end = rescale(chunk.ctx_end, TIMELINE_RATE, rate)
    decoding = {k: v for k, v in transcribe_kwargs(settings).items() if k != "vad_parameters"}
    fast = settings.asr.mode == "fast"
    return make_key(
        "asr",
        {
            "fingerprint": track.fingerprint(start, end - start),
            "samplerate": rate,
            "context": [chunk.ctx_start, chunk.ctx_end],
            "model": model,
            "mode": settings.asr.mode,
            "batch_size": settings.asr.batch_size if fast else None,
            "decoding": decoding,
            "vad": settings.vad.model_dump(mode="json"),
            "libraries": _library_versions(),
            "processing_version": PROCESSING_VERSION,
        },
    )


@dataclass
class RunStats:
    chunks: int = 0
    cached: int = 0
    transcribed: int = 0
    asr_s: float = 0.0
    crashes: int = 0
    refilled: int = 0  # segments recovered by re-transcribing word-less VAD speech
    missed: int = 0  # VAD stretches still without words (flagged for review)
    missing: list[str] = field(default_factory=list)


class MissingResults(AsrError):
    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        shown = ", ".join(self.missing[:5]) + (" ..." if len(self.missing) > 5 else "")
        super().__init__(
            f"{len(self.missing)} chunk(s) have no cached ASR result ({shown}); "
            "run 'transcribe --session' first"
        )


@dataclass
class _Planned:
    participant: Participant
    track: VirtualTrack
    length: int
    chunks: list[Chunk]
    keys: list[CacheKey]


def _plan(
    session: Session,
    settings: Settings,
    cache: Cache,
    model: dict[str, Any],
    sink: EventSink,
    full_hash: bool,
) -> list[_Planned]:
    frame_ms = settings.bleed.frame_ms
    frame = frame_samples(frame_ms)
    planned: list[_Planned] = []
    for p in session.participants:
        track = session.open_track(p, full_hash=full_hash)
        env = track_envelope(track, cache, frame_ms, source_id=p.id, sink=sink)
        length = rescale(track.frames, track.samplerate, TIMELINE_RATE)
        chunks = plan_chunks(
            length,
            env,
            frame=frame,
            owned_s=settings.asr.chunk_owned_s,
            context_s=settings.asr.chunk_context_s,
            search_s=settings.asr.chunk_boundary_search_s,
        )
        keys = [asr_key(track, c, settings, model) for c in chunks]
        planned.append(_Planned(p, track, length, chunks, keys))
    return planned


def _without(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in keys}


def _clip_vad(intervals: Sequence[SpeechInterval], chunk: Chunk) -> list[SpeechInterval]:
    out = []
    for v in intervals:
        a, b = max(v.start, chunk.owned_start), min(v.end, chunk.owned_end)
        if b > a:
            out.append(SpeechInterval(a, b))
    return out


def _run_session(
    session: Session,
    settings: Settings,
    *,
    transcribe: bool,
    sink: EventSink,
    cancel: Callable[[], bool] | None,
    isolation: Isolation,
    factory: EngineFactory,
    model_identity: ModelIdentity,
    gaps: Sequence[RecordingGap],
    full_hash: bool,
) -> tuple[SessionResult, RunStats]:
    cache = Cache(session.work_dir(settings.output.work_dir) / "cache")
    model = model_identity(settings)
    planned = _plan(session, settings, cache, model, sink, full_hash)
    stats = RunStats(chunks=sum(len(p.chunks) for p in planned))
    events: list[dict[str, Any]] = []
    crash_events: list[CrashEvent] = []

    def recording_sink(event: Event) -> None:
        if isinstance(event, WarningEvent):
            events.append({"kind": "warning", "stage": event.stage, "detail": event.message})
        sink(event)

    # Chunk-major order: chunk 0 of every track, then chunk 1, ...
    order = sorted(((c.index, i) for i, p in enumerate(planned) for c in p.chunks), key=lambda t: t)
    results: dict[tuple[int, int], ChunkResult] = {}
    todo: list[tuple[int, int]] = []
    for ci, ti in order:
        data = cache.load_json(planned[ti].keys[ci])
        if data is not None:
            results[(ti, ci)] = decode_result(data)
            stats.cached += 1
        else:
            todo.append((ti, ci))
    if todo and not transcribe:
        raise MissingResults([f"{planned[ti].participant.id}-c{ci:03d}" for ti, ci in todo])

    total = sum(
        planned[ti].chunks[ci].ctx_end - planned[ti].chunks[ci].ctx_start for ti, ci in todo
    )
    done = 0
    runner = (
        ProcessRunner(settings, cache, factory)
        if isolation == "process"
        else InlineRunner(settings, cache, factory)
    )
    try:
        for ti, ci in todo:
            p, chunk = planned[ti], planned[ti].chunks[ci]
            base = done

            def job_sink(event: Event, base: int = base) -> None:
                if isinstance(event, ProgressEvent):
                    sink(ProgressEvent("asr", base + event.done, total))
                else:
                    recording_sink(event)

            job = AsrJob(
                p.participant.id,
                str(session.source_path(p.participant)),
                p.participant.channel,
                chunk,
                p.keys[ci],
            )
            run_with_retry(runner, job, cache, job_sink, cancel, crash_events)
            data = cache.load_json(p.keys[ci])
            if data is None:
                raise AsrError(f"ASR result for {job.describe()} missing after a successful run")
            results[(ti, ci)] = decode_result(data)
            stats.transcribed += 1
            done += chunk.ctx_end - chunk.ctx_start
            sink(ProgressEvent("asr", done, total))
    finally:
        runner.close()
        stats.crashes = sum(e.kind != "worker_retry_ok" for e in crash_events)
        events.extend(
            {"kind": e.kind, "participant": e.participant, "chunk": e.chunk, "detail": e.detail}
            for e in crash_events
        )

    pending = set(todo)
    tracks: list[TrackInput] = []
    chunk_infos: list[ChunkInfo] = []
    vad: dict[str, list[SpeechInterval] | None] = {}
    languages: dict[str, str | None] = {}
    batch_sizes: list[int] = []
    model_info: dict[str, Any] = {}
    for ti, p in enumerate(planned):
        chunk_results = [results[(ti, c.index)] for c in p.chunks]
        segments: list[list[Segment]] = [r.segments for r in chunk_results]
        kept = resolve_chunks(p.chunks, segments)
        pid = p.participant.id
        vad[pid] = (
            [
                v
                for c, r in zip(p.chunks, chunk_results, strict=True)
                for v in _clip_vad(r.vad_intervals, c)
            ]
            if settings.vad.enabled
            else None
        )
        missed = [
            (c.index, max(a, c.owned_start), min(b, c.owned_end))
            for c, r in zip(p.chunks, chunk_results, strict=True)
            for a, b in r.missed
            if min(b, c.owned_end) > max(a, c.owned_start)
        ]
        stats.missed += len(missed)
        stats.refilled += sum(k.segment.refilled for k in kept if not k.boundary_duplicate)
        tracks.append(TrackInput(pid, p.participant.offset_samples, kept, vad[pid], missed))
        langs = Counter(r.language for r in chunk_results if r.language)
        languages[pid] = langs.most_common(1)[0][0] if langs else None
        for c, r, key in zip(p.chunks, chunk_results, p.keys, strict=True):
            stats.asr_s += r.transcribe_s
            if r.batch_size:
                batch_sizes.append(r.batch_size)
            if r.model and not model_info:
                model_info = {
                    k: r.model.get(k) for k in ("model_id", "revision", "device", "compute_type")
                }
            chunk_infos.append(
                ChunkInfo(
                    pid,
                    c.index,
                    c.owned,
                    (c.ctx_start, c.ctx_end),
                    key.short,
                    (ti, c.index) not in pending,
                    r.transcribe_s,
                    len(r.segments),
                )
            )

    transcript = merge_tracks(tracks, settings.quality, gaps)
    for seg in transcript.segments:
        sink(SegmentEvent(seg.speaker, seg))
    duration = max((p.participant.offset_samples + p.length for p in planned), default=0)
    result = SessionResult(
        name=session.name,
        date=session.config.date,
        participants=list(session.participants),
        settings=settings,
        transcript=transcript,
        duration=duration,
        model=model_info or {"model_id": model.get("id"), **_without(model, "id")},
        languages=languages,
        batch_sizes=batch_sizes,
        chunks=chunk_infos,
        vad=vad,
        gaps=list(gaps),
        processing_events=events,
    )
    return result, stats


def process_session(
    session: Session,
    settings: Settings,
    *,
    sink: EventSink = null_sink,
    cancel: Callable[[], bool] | None = None,
    isolation: Isolation = "process",
    factory: EngineFactory = default_engine_factory,
    model_identity: ModelIdentity = default_model_identity,
    gaps: Sequence[RecordingGap] = (),
    full_hash: bool = False,
) -> tuple[SessionResult, RunStats]:
    """Transcribe what the cache lacks, then merge; completed chunks survive any failure."""
    return _run_session(
        session,
        settings,
        transcribe=True,
        sink=sink,
        cancel=cancel,
        isolation=isolation,
        factory=factory,
        model_identity=model_identity,
        gaps=gaps,
        full_hash=full_hash,
    )


def export_session(
    session: Session,
    settings: Settings,
    *,
    sink: EventSink = null_sink,
    model_identity: ModelIdentity = default_model_identity,
    gaps: Sequence[RecordingGap] = (),
    full_hash: bool = False,
) -> tuple[SessionResult, RunStats]:
    """Merge from the cache only (re-export, rename, offset change); raises MissingResults."""
    return _run_session(
        session,
        settings,
        transcribe=False,
        sink=sink,
        cancel=None,
        isolation="inline",
        factory=default_engine_factory,
        model_identity=model_identity,
        gaps=gaps,
        full_hash=full_hash,
    )
