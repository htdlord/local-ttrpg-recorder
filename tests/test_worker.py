import os

import numpy as np
import pytest
import soundfile as sf

from tests.fakes import FakeFactory
from ttrpg_recorder.core.cache import Cache, make_key
from ttrpg_recorder.core.config import load_settings
from ttrpg_recorder.core.pipeline.asr import AsrError, Cancelled
from ttrpg_recorder.core.pipeline.chunks import Chunk
from ttrpg_recorder.core.pipeline.worker import (
    AsrJob,
    InlineRunner,
    ProcessRunner,
    WorkerCrashed,
    decode_result,
    describe_exit,
    run_with_retry,
)

R = 48_000
CRASH_TEXT = "0xC0000005 (access violation)" if os.name == "nt" else "SIGSEGV"


def make_track(path, seconds=6.0):
    x = np.zeros(int(seconds * R), np.float32)
    t = np.arange(R) / R
    x[R : 2 * R] = 0.5 * np.sin(2 * np.pi * 500 * t)
    x[4 * R : 5 * R] = 0.5 * np.sin(2 * np.pi * 500 * t)
    sf.write(str(path), x, R, subtype="FLOAT")
    return path


@pytest.fixture
def job(tmp_path):
    src = make_track(tmp_path / "a.wav")
    chunk = Chunk(0, 3 * R, 6 * R, 2 * R, 6 * R)
    return AsrJob("anna", str(src), None, chunk, make_key("asr", {"t": 1}))


def test_describe_exit():
    assert describe_exit(0xC0000005) == "exit code 0xC0000005 (access violation)"
    assert describe_exit(0xC0000005 - (1 << 32)) == "exit code 0xC0000005 (access violation)"
    assert describe_exit(-11) == "killed by SIGSEGV"
    assert describe_exit(3) == "exit code 3"
    assert "vanished" in describe_exit(None)


def test_inline_job_shifts_to_track_time(tmp_path, job):
    cache = Cache(tmp_path / "cache")
    runner = InlineRunner(load_settings(), cache, FakeFactory())
    stats = runner.run(job, lambda e: None, None)
    assert stats["segments"] == 1
    result = decode_result(cache.load_json(job.key))
    (seg,) = result.segments
    assert seg.start == 4 * R and seg.end == 5 * R  # chunk audio starts at 2 s
    assert seg.words[0].text == " ő0" and result.model["model_id"] == "fake"


def test_crash_once_is_retried(tmp_path, job):
    state = tmp_path / "crashes"
    state.write_text("1", encoding="utf-8")
    cache = Cache(tmp_path / "cache")
    runner = ProcessRunner(load_settings(), cache, FakeFactory(crash_state=state, crash_on="anna"))
    events, warnings = [], []
    try:
        stats = run_with_retry(runner, job, cache, warnings.append, None, events)
    finally:
        runner.close()
    assert stats["segments"] == 1 and cache.load_json(job.key) is not None
    assert [e.kind for e in events] == ["worker_crash", "worker_retry_ok"]
    assert CRASH_TEXT in events[0].detail and warnings


def test_second_crash_is_a_clear_error(tmp_path, job):
    state = tmp_path / "crashes"
    state.write_text("2", encoding="utf-8")
    cache = Cache(tmp_path / "cache")
    (tmp_path / "cache" / "asr").mkdir(parents=True)
    stale = tmp_path / "cache" / "asr" / f".{job.key.digest}.json.x.tmp"
    stale.write_text("half", encoding="utf-8")
    runner = ProcessRunner(load_settings(), cache, FakeFactory(crash_state=state, crash_on="anna"))
    events = []
    try:
        with pytest.raises(WorkerCrashed) as err:
            run_with_retry(runner, job, cache, lambda e: None, None, events)
    finally:
        runner.close()
    text = str(err.value)
    assert "anna chunk 0 (3.0-6.0 s track time)" in text and CRASH_TEXT in text
    assert "smaller model" in text
    assert [e.kind for e in events] == ["worker_crash", "worker_crash_fatal"]
    assert not cache.has(job.key) and not stale.exists()


def test_model_load_failure(tmp_path, job):
    cache = Cache(tmp_path / "cache")
    runner = ProcessRunner(load_settings(), cache, FakeFactory(fail_load=True))
    with pytest.raises(AsrError, match="could not load the model: no model here"):
        runner.run(job, lambda e: None, None)
    runner.close()


def test_cancel_leaves_no_result(tmp_path, job):
    cache = Cache(tmp_path / "cache")
    runner = ProcessRunner(load_settings(), cache, FakeFactory())
    with pytest.raises(Cancelled):
        runner.run(job, lambda e: None, lambda: True)
    runner.close()
    assert not cache.has(job.key)
