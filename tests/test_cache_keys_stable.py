"""Cache keys must not change unless processing changes (e.g. not by renaming the package).

The expected digests were computed before the project was renamed; if one of them
changes, existing caches are invalidated, which needs a deliberate `processing_version`
bump instead.
"""

import numpy as np
import soundfile as sf

from sessionscribe.core.config import Settings
from sessionscribe.core.pipeline import features, ownership, run
from sessionscribe.core.pipeline.chunks import Chunk
from sessionscribe.core.session import Session, SessionConfig, write_session

R = 48_000
EXPECTED = {
    "features": "c834c36acc1c7f792073953d2ba9082a6b004167267fd0520c257fc5c358dfe6",
    "ownership": "867e2c4799ddb29bb89730e4b35856072f24f6984f6c58be2f0d6057b7e46de7",
    "asr": "75d9069472f0482e7a9ce5576241a5c9e56b4bd958939049bfcd72b8e170d875",
}


def make_session(folder):
    folder.mkdir()
    rng = np.random.default_rng(7)
    for name in ("gm", "anna"):
        x = (rng.normal(0, 0.05, 6 * R) * (np.arange(6 * R) % R < R // 2)).astype(np.float32)
        sf.write(str(folder / f"{name}.flac"), x, R, subtype="PCM_24")
    config = SessionConfig.model_validate(
        {
            "participants": [
                {"id": "gm", "source": "gm.flac"},
                {"id": "anna", "source": "anna.flac", "offset_s": 0.5},
            ]
        }
    )
    write_session(folder / "session.toml", config)
    return Session(folder, config)


def capture(monkeypatch, module):
    seen = []
    original = module.make_key

    def spy(layer, fields):
        key = original(layer, fields)
        seen.append(key)
        return key

    monkeypatch.setattr(module, "make_key", spy)
    return seen


def test_cache_keys_are_stable(tmp_path, monkeypatch):
    session = make_session(tmp_path / "s")
    settings = Settings()
    track = session.open_track(session.participants[0])

    feature_keys = capture(monkeypatch, features)
    features.track_envelope(track, None, settings.bleed.frame_ms)
    ownership_keys = capture(monkeypatch, ownership)
    ownership.compute_ownership(session, settings, None)
    monkeypatch.setattr(
        run, "_library_versions", lambda: {"faster-whisper": "1", "ctranslate2": "1", "av": "1"}
    )
    asr = run.asr_key(
        track,
        Chunk(0, 0, 6 * R, 0, 6 * R),
        settings,
        {"id": "large-v3", "revision": None, "device": "cuda", "compute_type": "float16"},
    )
    got = {
        "features": feature_keys[0].digest,
        "ownership": ownership_keys[0].digest,
        "asr": asr.digest,
    }
    assert got == EXPECTED
