"""Real tiny model through the session pipeline in a worker process (slow)."""

import numpy as np
import pytest
import soundfile as sf

from sessionscribe.core.config import load_settings
from sessionscribe.core.pipeline.run import export_session, process_session
from sessionscribe.core.session import Participant, Session, SessionConfig, write_session

pytestmark = pytest.mark.slow
R = 48_000


def test_tiny_model_three_chunks_and_cache(tmp_path):
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(12 * R)).astype(np.float32)
    sf.write(str(tmp_path / "a.wav"), x, R, subtype="FLOAT")
    write_session(
        tmp_path / "session.toml",
        SessionConfig(participants=[Participant(id="a", source="a.wav")]),
    )
    session = Session.load(tmp_path)
    settings = load_settings(
        overrides=["asr.chunk_owned_s=4", "asr.chunk_context_s=1"],
        cli={
            "model": {"id": "tiny", "device": "cpu", "compute_type": "int8"},
            "decoding": {"language": "en"},
        },
    )
    try:
        result, stats = process_session(session, settings, isolation="process")
    except Exception as exc:  # no network and no cached tiny model
        if "not in the local cache" in str(exc):
            pytest.skip(str(exc))
        raise
    assert stats.chunks == 3 and stats.transcribed == 3
    assert [c.index for c in result.chunks] == [0, 1, 2]
    assert result.model["model_id"] == "tiny"
    _, again = export_session(session, settings)
    assert again.cached == 3
