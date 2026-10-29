"""Session pipeline end to end with fake engines (8 synthetic tracks)."""

import shutil

import numpy as np
import pytest
import soundfile as sf

from sessionscribe.core.config import load_settings
from sessionscribe.core.events import SegmentEvent, WarningEvent
from sessionscribe.core.pipeline import asr
from sessionscribe.core.pipeline.run import MissingResults, export_session, process_session
from sessionscribe.core.pipeline.worker import WorkerCrashed
from sessionscribe.core.session import Participant, Session, SessionConfig, write_session
from tests.fakes import FakeFactory, OomFactory

R = 48_000
IDENTITY = {"id": "fake", "revision": "r1", "device": "cpu", "compute_type": "int8"}
MILLIS = 'output.timestamp_style="hh:mm:ss.mmm"'


def identity(settings):
    return dict(IDENTITY)


def burst_track(path, seconds, bursts, length=0.8):
    x = np.zeros(int(seconds * R), np.float32)
    t = np.arange(int(length * R)) / R
    for start in bursts:
        a = int(start * R)
        part = x[a : a + len(t)]
        part[:] = (0.4 * np.sin(2 * np.pi * 500 * t))[: len(part)]
    sf.write(str(path), x, R, subtype="FLOAT")


def make_session(folder, n=8, seconds=12.0):
    folder.mkdir(parents=True, exist_ok=True)
    participants = []
    for i in range(n):
        name = f"ch{i + 1:02d}_p{i}.wav"
        burst_track(folder / name, seconds, [1.0 + 0.3 * i, 7.0 + 0.2 * i])
        participants.append(Participant(id=f"p{i}", name=f"Név {i}", source=name))
    write_session(folder / "session.toml", SessionConfig(name="teszt", participants=participants))
    return Session.load(folder)


def settings(*extra):
    base = ["asr.chunk_owned_s=4", "asr.chunk_context_s=1", "asr.chunk_boundary_search_s=1"]
    return load_settings(overrides=[*base, *extra])


def calls(path):
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def run(session, s, **kw):
    kw.setdefault("isolation", "inline")
    kw.setdefault("factory", FakeFactory())
    kw.setdefault("model_identity", identity)
    return process_session(session, s, **kw)


def test_eight_tracks_merge_and_rerun_without_asr(tmp_path):
    session = make_session(tmp_path / "s")
    log = tmp_path / "calls.txt"
    events = []
    result, stats = run(session, settings(), factory=FakeFactory(calls=log), sink=events.append)
    assert stats.chunks == 24 and stats.transcribed == 24 and stats.cached == 0
    assert len(calls(log)) == 24
    segs = result.transcript.segments
    assert len(segs) == 16 and len({s.id for s in segs}) == 16
    assert [s.start for s in segs] == sorted(s.start for s in segs)
    assert segs[0].speaker == "p0" and segs[0].start == pytest.approx(R, abs=R // 50)
    assert sum(isinstance(e, SegmentEvent) for e in events) == 16
    ids = [s.id for s in segs]

    # rename + offset change: same ASR cache, shifted session times, same ids
    cfg = session.config
    cfg.participants[3].name = "Új név"
    cfg.participants[3].offset_s = 2.5
    write_session(session.folder / "session.toml", cfg)
    again = Session.load(session.folder)
    result2, stats2 = run(again, settings(), factory=FakeFactory(calls=log))
    assert stats2.transcribed == 0 and stats2.cached == 24 and len(calls(log)) == 24
    assert sorted(s.id for s in result2.transcript.segments) == sorted(ids)
    p3 = [s for s in result2.transcript.segments if s.speaker == "p3"]
    old = [s for s in segs if s.speaker == "p3"]
    assert [s.start - o.start for s, o in zip(p3, old, strict=True)] == [int(2.5 * R)] * 2
    assert result2.name_of("p3") == "Új név"

    # export only, from the cache; a copied folder (new mtimes) still hits the cache
    exported, _ = export_session(again, settings(), model_identity=identity)
    assert [s.id for s in exported.transcript.segments] == [
        s.id for s in result2.transcript.segments
    ]
    copy = tmp_path / "copy"
    shutil.copytree(session.folder, copy)
    _, stats3 = export_session(Session.load(copy), settings(), model_identity=identity)
    assert stats3.cached == 24


def test_settings_change_invalidates_and_export_reports_missing(tmp_path):
    session = make_session(tmp_path / "s", n=2)
    run(session, settings())
    with pytest.raises(MissingResults, match="6 chunk"):
        export_session(session, settings("decoding.beam_size=2"), model_identity=identity)
    with pytest.raises(MissingResults):
        export_session(session, settings(), model_identity=lambda s: {**IDENTITY, "revision": "2"})
    # output-only settings never invalidate
    export_session(session, settings(MILLIS), model_identity=identity)


def test_chunk_boundary_word_is_not_lost_or_doubled(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    burst_track(folder / "a.wav", 10.0, [2.5], length=3.0)  # 2.5 .. 5.5 s, across 4 s
    write_session(
        folder / "session.toml", SessionConfig(participants=[Participant(id="a", source="a.wav")])
    )
    s = settings("asr.chunk_boundary_search_s=0", "asr.chunk_context_s=2")
    result, _ = run(Session.load(folder), s)
    words = [w for seg in result.transcript.segments for w in seg.words]
    starts = [w.start for w in words]
    assert len(starts) == len(set(starts))  # no word twice
    assert sum(w.end - w.start for w in words) >= int(2.9 * R)  # the whole burst is there


def test_worker_crash_in_session_is_retried_and_recorded(tmp_path):
    session = make_session(tmp_path / "s", n=2, seconds=5.0)
    state = tmp_path / "crashes"
    state.write_text("1", encoding="utf-8")
    result, stats = run(
        session,
        settings("asr.chunk_owned_s=10"),
        isolation="process",
        factory=FakeFactory(crash_state=state, crash_on="p1"),
    )
    assert stats.transcribed == 2 and stats.crashes == 1
    kinds = [e["kind"] for e in result.processing_events]
    assert "worker_crash" in kinds and "worker_retry_ok" in kinds
    assert all("ő" not in str(e) for e in result.processing_events)  # no transcript text


def test_second_crash_keeps_completed_chunks(tmp_path):
    session = make_session(tmp_path / "s", n=2, seconds=5.0)
    state = tmp_path / "crashes"
    state.write_text("2", encoding="utf-8")
    log = tmp_path / "calls.txt"
    s = settings("asr.chunk_owned_s=10")
    with pytest.raises(WorkerCrashed, match="p1 chunk 0"):
        run(
            session,
            s,
            isolation="process",
            factory=FakeFactory(calls=log, crash_state=state, crash_on="p1"),
        )
    assert calls(log).count("p0") == 1
    # resumed run: p0 comes from the cache, only p1 is transcribed
    _, stats = run(session, s, isolation="process", factory=FakeFactory(calls=log))
    assert stats.cached == 1 and stats.transcribed == 1
    assert calls(log).count("p0") == 1


def test_fast_preset_oom_fallback_in_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "speech_intervals", lambda audio, vad: [])
    session = make_session(tmp_path / "s", n=1, seconds=3.0)
    sizes: list[int] = []
    warnings = []

    def sink(event):
        if isinstance(event, WarningEvent):
            warnings.append(event)

    result, _ = run(session, load_settings(preset="fast"), factory=OomFactory(1, sizes), sink=sink)
    assert sizes == [4, 2, 1] and len(warnings) == 2
    assert [e["kind"] for e in result.processing_events] == ["warning", "warning"]
    assert result.batch_sizes == [1]
