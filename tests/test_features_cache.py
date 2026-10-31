import numpy as np
import pytest
import soundfile as sf

from ttrpg_recorder.core.audio.track import VirtualTrack
from ttrpg_recorder.core.cache import Cache, make_key
from ttrpg_recorder.core.pipeline import features
from ttrpg_recorder.core.pipeline.features import (
    SILENCE_DB,
    envelope,
    frame_samples,
    frames_db,
    track_envelope,
)


def tone(seconds, rate=48_000, amp=1.0, freq=500.0):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_frames_db_levels():
    x = np.concatenate([tone(0.1), np.zeros(4800, np.float32), tone(0.1, amp=0.1)])
    db = frames_db(x, 960)
    assert len(db) == 15
    assert db[0] == pytest.approx(-3.01, abs=0.05)
    assert db[6] == SILENCE_DB
    assert db[12] == pytest.approx(-23.01, abs=0.05)


def test_envelope_is_block_invariant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, 48_000 * 2 + 123).astype(np.float32)
    whole = frames_db(x, 960)
    for size in (1, 500, 960, 7777):
        blocks = [x[i : i + size] for i in range(0, len(x), size)]
        np.testing.assert_allclose(envelope(blocks, 960), whole, atol=1e-4)


def test_frame_samples():
    assert frame_samples(20) == 960
    assert frame_samples(20, 44_100) == 882
    with pytest.raises(ValueError):
        frame_samples(1, 22_050)


def test_track_envelope_cached_in_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "FEATURE_BLOCK_S", 1)
    path = tmp_path / "a.wav"
    sf.write(str(path), tone(2.5, amp=0.5), 48_000, subtype="FLOAT")
    track = VirtualTrack.open(path)
    cache = Cache(tmp_path / "cache")
    env = track_envelope(track, cache, 20)
    assert len(env) == 125
    assert len(list((tmp_path / "cache" / "features").glob("*.npy"))) == 3
    calls = []
    monkeypatch.setattr(features, "envelope", lambda *a: calls.append(a))
    np.testing.assert_array_equal(track_envelope(track, cache, 20), env)
    assert calls == []


def test_cache_json_round_trip_and_corruption(tmp_path):
    cache = Cache(tmp_path)
    key = make_key("asr", {"fingerprint": "abc", "range": [0, 10]})
    assert make_key("asr", {"range": [0, 10], "fingerprint": "abc"}) == key  # order-free
    assert make_key("asr", {"fingerprint": "abd", "range": [0, 10]}).digest != key.digest
    assert cache.load_json(key) is None
    cache.store_json(key, {"segments": ["ő ű á"]})
    assert cache.load_json(key) == {"segments": ["ő ű á"]}
    assert "segments" not in (tmp_path / "asr" / f"{key.digest}.key.json").read_text("utf-8")
    cache.path(key, "json").write_text("{ broken", encoding="utf-8")
    assert cache.load_json(key) is None


def test_cache_discards_partial_files(tmp_path):
    cache = Cache(tmp_path)
    key = make_key("asr", {"x": 1})
    (tmp_path / "asr").mkdir()
    tmp = tmp_path / "asr" / f".{key.digest}.json.abc123.tmp"
    tmp.write_text("half", encoding="utf-8")
    assert not cache.has(key)  # a temp file never counts as done
    assert cache.discard_partial(key) == 1 and not tmp.exists()
