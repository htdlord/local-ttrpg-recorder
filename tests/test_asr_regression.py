"""Which settings each ASR mode actually honours, on the pinned faster-whisper (1.2.1).

Observes the real calls into ctranslate2 and the real TranscriptionOptions with the
tiny model. If this fails after a library bump, update ``core.config`` mode metadata
and ``test_settings.test_mode_table_is_pinned`` together.
"""

import numpy as np
import pytest

from ttrpg_recorder.core.config import load_settings
from ttrpg_recorder.core.pipeline.asr import transcribe_kwargs, vad_options

pytestmark = pytest.mark.slow

RATE = 16_000


class GenerateSpy:
    """Wraps ctranslate2's Whisper model and records every ``generate`` call."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._inner.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def whisper():
    from faster_whisper import WhisperModel

    try:
        return WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as exc:  # pragma: no cover - needs the model once
        pytest.skip(f"tiny model not available: {exc}")


@pytest.fixture
def spy(whisper):
    original = whisper.model
    spy = GenerateSpy(original)
    whisper.model = spy
    yield spy
    whisper.model = original


def noise(seconds=8.0):
    # Broadband noise decodes to low-confidence text, so fallback thresholds trigger.
    return (np.random.default_rng(1).normal(0, 0.1, int(seconds * RATE))).astype(np.float32)


@pytest.fixture
def full_speech(monkeypatch):
    """Make the VAD report the whole clip as speech, recording the options it got."""
    import faster_whisper.transcribe as fwt

    seen = []

    def fake(audio, vad_options=None, **_):
        seen.append(vad_options)
        return [{"start": 0, "end": len(audio)}]

    monkeypatch.setattr(fwt, "get_speech_timestamps", fake)
    return seen


def strict(mode):
    """Thresholds that every decode fails, so quality mode must walk the temperatures."""
    preset = "fast" if mode == "fast" else "conversation"
    return load_settings(
        preset=preset,
        cli={
            "model": {"id": "tiny"},
            "decoding": {
                "language": "en",
                "log_prob_threshold": 0.0,
                "no_speech_threshold": None,  # else noise counts as silence: no retry
                "compression_ratio_threshold": 0.01,
                "temperature": [0.0, 0.5, 1.0],
                "best_of": 2,
            },
        },
    )


def test_quality_mode_runs_temperature_fallback(whisper, spy, full_speech):
    settings = strict("quality")
    segments, _ = whisper.transcribe(noise(), **transcribe_kwargs(settings))
    list(segments)
    temps = [c.get("sampling_temperature", 0.0) for c in spy.calls]
    assert {0.5, 1.0} <= set(temps)  # retried with higher temperatures
    sampled = [c for c in spy.calls if c.get("sampling_temperature", 0.0) > 0]
    assert all(c.get("num_hypotheses") == 2 for c in sampled)  # best_of honoured


def test_fast_mode_uses_first_temperature_only_and_no_best_of(whisper, spy, full_speech):
    from faster_whisper import BatchedInferencePipeline

    settings = strict("fast")
    raw = dict(transcribe_kwargs(settings), temperature=[0.0, 0.5, 1.0], best_of=2)
    segments, _ = BatchedInferencePipeline(whisper).transcribe(noise(), batch_size=2, **raw)
    list(segments)
    assert spy.calls, "batched pipeline did not decode"
    assert {c.get("sampling_temperature") for c in spy.calls} == {0.0}
    assert all("num_hypotheses" not in c for c in spy.calls)


def test_fast_mode_overrides_options(whisper, monkeypatch, full_speech):
    """condition_on_previous_text and hallucination_silence_threshold are forced off."""
    from faster_whisper import BatchedInferencePipeline

    captured = []
    pipeline = BatchedInferencePipeline(whisper)
    original = pipeline._batched_segments_generator

    def record(features, tokenizer, chunks, batch_size, options, log_progress):
        captured.append(options)
        return original(features, tokenizer, chunks, batch_size, options, log_progress)

    monkeypatch.setattr(pipeline, "_batched_segments_generator", record)
    segments, _ = pipeline.transcribe(
        noise(),
        language="en",
        batch_size=2,
        condition_on_previous_text=True,
        hallucination_silence_threshold=2.0,
        word_timestamps=True,
    )
    list(segments)
    (options,) = captured
    assert options.condition_on_previous_text is False
    assert options.hallucination_silence_threshold is None


def test_fast_mode_keeps_vad_options_object_but_overrides_dict(whisper, full_speech):
    from faster_whisper import BatchedInferencePipeline

    settings = load_settings(preset="fast", cli={"model": {"id": "tiny"}})
    pipeline = BatchedInferencePipeline(whisper)
    ours = vad_options(settings.vad)
    list(pipeline.transcribe(noise(), language="en", batch_size=2, vad_parameters=ours)[0])
    as_dict = {"min_silence_duration_ms": 800, "max_speech_duration_s": 10.0}
    list(pipeline.transcribe(noise(), language="en", batch_size=2, vad_parameters=as_dict)[0])
    obj_opts, dict_opts = full_speech
    assert obj_opts is ours and obj_opts.min_silence_duration_ms == 800
    assert dict_opts.max_speech_duration_s == 30  # library replaced our 10 s


def test_fast_mode_library_defaults_differ(whisper, full_speech):
    """Batched defaults: VAD on with min_silence 160 ms; we always pass our own options."""
    from faster_whisper import BatchedInferencePipeline

    list(BatchedInferencePipeline(whisper).transcribe(noise(), language="en", batch_size=2)[0])
    (opts,) = full_speech
    assert opts.min_silence_duration_ms == 160


def test_quality_mode_honours_condition_and_hallucination_options(whisper, monkeypatch):
    captured = []
    original = whisper.generate_segments

    def record(features, tokenizer, options, *args, **kwargs):
        captured.append(options)
        return original(features, tokenizer, options, *args, **kwargs)

    monkeypatch.setattr(whisper, "generate_segments", record)
    settings = load_settings(
        cli={"model": {"id": "tiny"}, "decoding": {"language": "en"}},
        overrides=[
            "decoding.condition_on_previous_text=true",
            "decoding.hallucination_silence_threshold=2.0",
            "vad.enabled=false",
        ],
    )
    list(whisper.transcribe(noise(), **transcribe_kwargs(settings))[0])
    (options,) = captured
    assert options.condition_on_previous_text is True
    assert options.hallucination_silence_threshold == 2.0
    assert options.word_timestamps is True


def test_real_transcribe_end_to_end(whisper):
    """Engine on a real model: runs, returns integer sample positions."""
    from ttrpg_recorder.core.pipeline.asr import AsrEngine, LoadedModel

    settings = load_settings(cli={"model": {"id": "tiny"}, "decoding": {"language": "en"}})
    loaded = LoadedModel("tiny", "", None, "cpu", "int8", True, 0.0)
    result = AsrEngine(settings, whisper, loaded).transcribe(noise(4.0), source_id="x")
    assert result.duration == 4 * 48_000
    assert all(isinstance(s.start, int) for s in result.segments)
