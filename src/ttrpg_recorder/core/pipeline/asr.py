"""ASR stage on faster-whisper: ``quality`` (WhisperModel.transcribe) and ``fast`` (batched).

Only settings that the chosen mode honours are passed (see ``core.config.mode_table``).
Library results are converted to integer samples on the track-local timeline once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ttrpg_recorder.core.config import (
    DEFAULT_FAST_BATCH_SIZE,
    Settings,
    VadSettings,
    resolve_runtime,
)
from ttrpg_recorder.core.events import (
    EventSink,
    ProgressEvent,
    SegmentEvent,
    WarningEvent,
    null_sink,
)
from ttrpg_recorder.core.timeline import TIMELINE_RATE, WHISPER_RATE, rescale, seconds_to_samples

logger = logging.getLogger(__name__)

MODEL_HOST = "huggingface.co"


class AsrError(RuntimeError):
    pass


class ModelUnavailableError(AsrError):
    pass


class Cancelled(AsrError):
    pass


@dataclass(frozen=True)
class Word:
    start: int  # track-local samples at TIMELINE_RATE
    end: int
    text: str
    probability: float


@dataclass(frozen=True)
class Segment:
    index: int
    start: int  # track-local samples at TIMELINE_RATE
    end: int
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float | None
    words: tuple[Word, ...] = ()
    refilled: bool = False  # from re-transcribing a VAD stretch that got no words


@dataclass(frozen=True)
class SpeechInterval:
    start: int
    end: int


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    path: str
    revision: str | None
    device: str
    compute_type: str
    cached: bool
    load_s: float


@dataclass
class TranscriptionResult:
    source_id: str
    model: LoadedModel
    mode: str
    batch_size: int | None
    language: str | None
    language_probability: float | None
    duration: int  # samples at TIMELINE_RATE
    transcribe_s: float
    vad_s: float
    segments: list[Segment] = field(default_factory=list)
    vad_intervals: list[SpeechInterval] = field(default_factory=list)

    @property
    def audio_s(self) -> float:
        return self.duration / TIMELINE_RATE

    @property
    def rtf(self) -> float:
        return self.transcribe_s / self.audio_s if self.duration else float("nan")


# --- model loading ------------------------------------------------------------------------


def _has_network(host: str = MODEL_HOST) -> bool:
    from ttrpg_recorder.core.inference_check import has_network

    return has_network(host)


def resolve_model_path(
    model_id: str, model_dir: Path | None, revision: str | None, offline: bool
) -> tuple[str, str | None, bool]:
    """Local folder of the model, its Hub revision (if known) and whether it was cached."""
    if Path(model_id).is_dir():
        return str(Path(model_id)), None, True
    from faster_whisper.utils import download_model

    cache_dir = str(model_dir) if model_dir else None
    try:
        path = download_model(
            model_id, cache_dir=cache_dir, local_files_only=True, revision=revision
        )
        cached = True
    except Exception as exc:
        logger.debug("model not available locally: %r", exc)
        if offline:
            raise ModelUnavailableError(
                f"model {model_id!r} is not in the local cache and offline mode is set"
            ) from exc
        if not _has_network():
            raise ModelUnavailableError(
                f"model {model_id!r} is not in the local cache and {MODEL_HOST} is not "
                "reachable (no network); connect once to download it"
            ) from exc
        path = download_model(
            model_id, cache_dir=cache_dir, local_files_only=False, revision=revision
        )
        cached = False
    folder = Path(path)
    resolved_revision = folder.name if folder.parent.name == "snapshots" else revision
    return str(folder), resolved_revision, cached


def load_model(settings: Settings) -> tuple[Any, LoadedModel]:
    runtime = resolve_runtime(settings)
    m = settings.model
    path, revision, cached = resolve_model_path(m.id, m.model_dir, m.revision, m.offline)
    from ttrpg_recorder.core.cuda_libs import prepare_cuda_libraries

    prepare_cuda_libraries()
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(
        path,
        device=runtime.device,
        compute_type=runtime.compute_type,
        cpu_threads=runtime.cpu_threads,
        num_workers=m.num_workers,
    )
    info = LoadedModel(
        model_id=m.id,
        path=path,
        revision=revision,
        device=runtime.device,
        compute_type=runtime.compute_type,
        cached=cached,
        load_s=time.perf_counter() - t0,
    )
    return model, info


# --- parameter mapping (pure) -------------------------------------------------------------


def vad_options(vad: VadSettings) -> Any:
    from faster_whisper.vad import VadOptions

    # Always an object: the batched pipeline overrides max_speech_duration_s for dicts.
    return VadOptions(
        threshold=vad.threshold,
        min_speech_duration_ms=vad.min_speech_duration_ms,
        max_speech_duration_s=vad.max_speech_duration_s,
        min_silence_duration_ms=vad.min_silence_duration_ms,
        speech_pad_ms=vad.speech_pad_ms,
    )


def transcribe_kwargs(settings: Settings) -> dict[str, Any]:
    """Keyword arguments for the mode's ``transcribe()``; batch_size is added by the engine."""
    d = settings.decoding
    kwargs: dict[str, Any] = {
        "language": None if d.language == "auto" else d.language,
        "task": d.task,
        "beam_size": d.beam_size,
        "word_timestamps": d.word_timestamps,
        "initial_prompt": settings.vocabulary.initial_prompt,
        "hotwords": settings.vocabulary.hotwords,
        "vad_filter": settings.vad.enabled,
        "vad_parameters": vad_options(settings.vad) if settings.vad.enabled else None,
    }
    if settings.asr.mode == "quality":
        kwargs |= {
            "best_of": d.best_of,
            "temperature": list(d.temperature),
            "condition_on_previous_text": d.condition_on_previous_text,
            "no_speech_threshold": d.no_speech_threshold,
            "log_prob_threshold": d.log_prob_threshold,
            "compression_ratio_threshold": d.compression_ratio_threshold,
            "hallucination_silence_threshold": d.hallucination_silence_threshold,
        }
    else:
        # The batched pipeline samples with temperature[0] only and never retries.
        kwargs["temperature"] = [d.temperature[0]]
    return kwargs


def batch_sizes(settings: Settings) -> tuple[int, ...]:
    """Starting batch size, then halved down to 1 unless the OOM fallback is off."""
    size = settings.asr.batch_size or DEFAULT_FAST_BATCH_SIZE
    if not settings.asr.oom_fallback:
        return (size,)
    sizes = [size]
    while sizes[-1] > 1:
        sizes.append(sizes[-1] // 2)
    return tuple(sizes)


def is_out_of_memory(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def convert_segment(index: int, seg: Any) -> Segment:
    words = tuple(
        Word(
            seconds_to_samples(w.start),
            seconds_to_samples(w.end),
            w.word,
            float(w.probability),
        )
        for w in (seg.words or ())
    )
    return Segment(
        index=index,
        start=seconds_to_samples(seg.start),
        end=seconds_to_samples(seg.end),
        text=seg.text.strip(),
        avg_logprob=float(seg.avg_logprob),
        no_speech_prob=float(seg.no_speech_prob),
        compression_ratio=float(seg.compression_ratio),
        temperature=None if seg.temperature is None else float(seg.temperature),
        words=words,
    )


def speech_intervals(audio: np.ndarray, vad: VadSettings) -> list[SpeechInterval]:
    from faster_whisper.vad import get_speech_timestamps

    chunks = get_speech_timestamps(audio, vad_options(vad))
    return [
        SpeechInterval(
            rescale(c["start"], WHISPER_RATE, TIMELINE_RATE),
            rescale(c["end"], WHISPER_RATE, TIMELINE_RATE),
        )
        for c in chunks
    ]


# --- engine -------------------------------------------------------------------------------


class AsrEngine:
    """One loaded model; transcribes 16 kHz mono float32 arrays."""

    def __init__(
        self,
        settings: Settings,
        model: Any,
        loaded: LoadedModel,
        pipeline_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.loaded = loaded
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None

    @classmethod
    def load(cls, settings: Settings) -> AsrEngine:
        model, loaded = load_model(settings)
        return cls(settings, model, loaded)

    def _batched(self) -> Any:
        if self._pipeline is None:
            factory = self._pipeline_factory
            if factory is None:
                from faster_whisper import BatchedInferencePipeline

                factory = BatchedInferencePipeline
            self._pipeline = factory(self.model)
        return self._pipeline

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        source_id: str,
        sink: EventSink = null_sink,
        cancel: Callable[[], bool] | None = None,
    ) -> TranscriptionResult:
        settings = self.settings
        duration = rescale(len(audio), WHISPER_RATE, TIMELINE_RATE)
        t0 = time.perf_counter()
        intervals = speech_intervals(audio, settings.vad) if settings.vad.enabled else []
        vad_s = time.perf_counter() - t0
        kwargs = transcribe_kwargs(settings)

        mode = settings.asr.mode
        used_batch: int | None = None
        t0 = time.perf_counter()
        if mode == "quality":
            segments, info = self._run(
                self.model.transcribe(audio, **kwargs), duration, sink, cancel
            )
        else:
            sizes = batch_sizes(settings)
            for i, size in enumerate(sizes):
                try:
                    out = self._batched().transcribe(audio, batch_size=size, **kwargs)
                    segments, info = self._run(out, duration, sink, cancel)
                    used_batch = size
                    break
                except Cancelled:
                    raise
                except Exception as exc:
                    if is_out_of_memory(exc) and i + 1 < len(sizes):
                        sink(
                            WarningEvent(
                                "asr",
                                f"out of GPU memory at batch size {size}; "
                                f"retrying with {sizes[i + 1]}",
                            )
                        )
                        continue
                    if is_out_of_memory(exc):
                        how = "even at" if len(sizes) > 1 or size == 1 else "at"
                        hint = "" if settings.asr.oom_fallback else " (OOM fallback is off)"
                        raise AsrError(
                            f"out of GPU memory {how} batch size {size}{hint}; "
                            "use a smaller model or compute type, or quality mode"
                        ) from exc
                    raise
        elapsed = time.perf_counter() - t0

        result = TranscriptionResult(
            source_id=source_id,
            model=self.loaded,
            mode=mode,
            batch_size=used_batch,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=duration,
            transcribe_s=elapsed,
            vad_s=vad_s,
            segments=segments,
            vad_intervals=intervals,
        )
        for seg in segments:
            sink(SegmentEvent(source_id, seg))
        return result

    @staticmethod
    def _run(
        out: tuple[Iterable[Any], Any],
        duration: int,
        sink: EventSink,
        cancel: Callable[[], bool] | None,
    ) -> tuple[list[Segment], Any]:
        """Consume the lazy segment generator (that is where decoding happens)."""
        raw_segments, info = out
        segments: list[Segment] = []
        for raw in raw_segments:
            if cancel is not None and cancel():
                raise Cancelled("transcription cancelled")
            seg = convert_segment(len(segments), raw)
            segments.append(seg)
            sink(ProgressEvent("asr", min(seg.end, duration), duration))
        sink(ProgressEvent("asr", duration, duration))
        return segments, info
