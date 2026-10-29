"""ASR engine with fake models: parameter mapping, conversion, OOM fallback, outputs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sessionscribe.core.config import load_settings
from sessionscribe.core.events import ProgressEvent, SegmentEvent, WarningEvent
from sessionscribe.core.exporters.single import Speaker, to_json, to_txt, write_outputs
from sessionscribe.core.pipeline import asr
from sessionscribe.core.pipeline.asr import (
    AsrEngine,
    AsrError,
    Cancelled,
    LoadedModel,
    transcribe_kwargs,
)

LOADED = LoadedModel("fake", "/m", "abc123", "cpu", "int8", True, 0.0)


def fw_segment(start, end, text, words=()):
    return SimpleNamespace(
        start=start,
        end=end,
        text=f" {text}",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        compression_ratio=1.3,
        temperature=0.0,
        words=[SimpleNamespace(start=a, end=b, word=w, probability=0.9) for a, b, w in words],
    )


SEGMENTS = [
    fw_segment(0.5, 1.25, "Árvíztűrő tükörfúrógép", [(0.5, 0.9, " Árvíztűrő"), (0.9, 1.25, " ő")]),
    fw_segment(2.0, 3.0, "ű"),
]


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return iter(SEGMENTS), SimpleNamespace(language="hu", language_probability=0.98)


class FakeBatched:
    def __init__(self, oom_above):
        self.oom_above = oom_above
        self.sizes = []

    def transcribe(self, audio, batch_size, **kwargs):
        self.sizes.append(batch_size)
        self.kwargs = kwargs

        def gen():
            if batch_size > self.oom_above:
                raise RuntimeError("CUDA failed with error out of memory")
            yield from SEGMENTS

        return gen(), SimpleNamespace(language="hu", language_probability=0.9)


@pytest.fixture(autouse=True)
def no_vad(monkeypatch):
    monkeypatch.setattr(asr, "speech_intervals", lambda audio, vad: [asr.SpeechInterval(0, 96_000)])


def audio(seconds=4.0):
    return np.zeros(int(seconds * 16_000), np.float32)


def test_quality_kwargs():
    s = load_settings(
        cli={
            "decoding": {"language": "hu"},
            "vocabulary": {"initial_prompt": "Bloodwell", "hotwords": "Balthazar"},
        }
    )
    kw = transcribe_kwargs(s)
    assert kw["language"] == "hu" and kw["condition_on_previous_text"] is False
    assert kw["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert kw["initial_prompt"] == "Bloodwell" and kw["hotwords"] == "Balthazar"
    assert kw["vad_filter"] is True
    vad = kw["vad_parameters"]
    assert (vad.min_silence_duration_ms, vad.max_speech_duration_s, vad.speech_pad_ms) == (
        800,
        30.0,
        400,
    )
    for key in ("best_of", "log_prob_threshold", "hallucination_silence_threshold"):
        assert key in kw


def test_fast_kwargs_leave_out_ignored_settings():
    s = load_settings(preset="fast-draft", overrides=["decoding.temperature=[0.2, 0.4]"])
    kw = transcribe_kwargs(s)
    assert kw["temperature"] == [0.2]
    for key in (
        "best_of",
        "condition_on_previous_text",
        "hallucination_silence_threshold",
        "log_prob_threshold",
        "no_speech_threshold",
        "compression_ratio_threshold",
    ):
        assert key not in kw
    assert kw["vad_parameters"] is not None and not isinstance(kw["vad_parameters"], dict)


def test_auto_language_is_none():
    assert transcribe_kwargs(load_settings())["language"] is None


def test_quality_transcribe_converts_to_samples():
    events = []
    model = FakeModel()
    engine = AsrEngine(load_settings(), model, LOADED)
    result = engine.transcribe(audio(), source_id="gm", sink=events.append)
    first = result.segments[0]
    assert (first.start, first.end) == (24_000, 60_000)
    assert first.text == "Árvíztűrő tükörfúrógép"
    assert [(w.start, w.end) for w in first.words] == [(24_000, 43_200), (43_200, 60_000)]
    assert result.duration == 192_000 and result.language == "hu" and result.batch_size is None
    assert [type(e) for e in events if isinstance(e, SegmentEvent)] == [SegmentEvent] * 2
    assert all(e.status == "final" for e in events if isinstance(e, SegmentEvent))
    assert isinstance(events[-1], SegmentEvent)
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    assert progress[-1].done == progress[-1].total == 192_000


def test_fast_mode_halves_batch_on_oom():
    events = []
    fake = FakeBatched(oom_above=4)
    engine = AsrEngine(load_settings(preset="fast-draft"), object(), LOADED, lambda m: fake)
    result = engine.transcribe(audio(), source_id="p1", sink=events.append)
    assert fake.sizes == [16, 8, 4]
    assert result.batch_size == 4 and len(result.segments) == 2
    assert sum(isinstance(e, WarningEvent) for e in events) == 2


def test_fast_mode_gives_up_with_clear_error():
    fake = FakeBatched(oom_above=0)
    engine = AsrEngine(load_settings(preset="fast-draft"), object(), LOADED, lambda m: fake)
    with pytest.raises(AsrError, match="even at batch size 1"):
        engine.transcribe(audio(), source_id="p1")


def test_fixed_batch_size_is_not_probed():
    fake = FakeBatched(oom_above=4)
    s = load_settings(preset="fast-draft", overrides=["asr.batch_size=8"])
    with pytest.raises(AsrError):
        AsrEngine(s, object(), LOADED, lambda m: fake).transcribe(audio(), source_id="p1")
    assert fake.sizes == [8]


def test_cancel():
    engine = AsrEngine(load_settings(), FakeModel(), LOADED)
    with pytest.raises(Cancelled):
        engine.transcribe(audio(), source_id="gm", cancel=lambda: True)


def test_outputs(tmp_path):
    s = load_settings(cli={"decoding": {"language": "hu"}})
    result = AsrEngine(s, FakeModel(), LOADED).transcribe(audio(), source_id="gm")
    who = Speaker("gm", "Mesélő", "ch01_gm.wav")
    data = to_json(result, s, who)
    assert data["schema_version"] == 1 and data["model"]["revision"] == "abc123"
    seg = data["segments"][0]
    assert (
        seg["id"] == "gm-c000-0000" and seg["start"] == 0.5 and seg["audio_start_sample"] == 24_000
    )
    assert seg["words"][1] == {"start": 0.9, "end": 1.25, "word": " ő", "prob": 0.9}
    assert data["vad_intervals"] == [{"start": 0.0, "end": 2.0}]
    assert to_txt(result, who).splitlines() == [
        "[00:00:00] Mesélő: Árvíztűrő tükörfúrógép",
        "[00:00:02] Mesélő: ű",
    ]
    paths = write_outputs(result, s, who, tmp_path, "ch01_gm", ("json", "txt"))
    raw = paths[0].read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert json.loads(raw.decode("utf-8"))["segments"][1]["text"] == "ű"
    with pytest.raises(ValueError, match="not available yet"):
        write_outputs(result, s, who, tmp_path, "x", ("srt",))
