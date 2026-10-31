"""Bleed handling end to end on synthetic audio with known coupling (fake ASR engine)."""

import numpy as np
import soundfile as sf

from tests.fakes import FakeFactory
from ttrpg_recorder.core.config import load_settings
from ttrpg_recorder.core.pipeline.run import process_session
from ttrpg_recorder.core.session import Participant, Session, SessionConfig, write_session

R = 48_000
IDENTITY = {"id": "fake", "revision": "r1", "device": "cpu", "compute_type": "int8"}
BLEED = 10 ** (-18 / 20)  # each voice reaches the other mics 18 dB down


def make(tmp_path, turns, seconds=20.0, gains=(0.0, -6.0, -3.0), offsets=(0.0, 0.0, 0.0)):
    """turns[j] = [(start_s, end_s), ...] in each participant's own track time."""
    rng = np.random.default_rng(1)
    n = len(turns)
    voices = np.zeros((n, int(seconds * R)), np.float32)
    for j, spans in enumerate(turns):
        for a, b in spans:
            t = np.arange(int((b - a) * R)) / R
            voices[j, int(a * R) : int(b * R)] = 0.5 * np.sin(2 * np.pi * (300 + 70 * j) * t)
    names = ["gm", "anna", "bela"][:n]
    parts = []
    for i in range(n):
        heard = sum(voices[j] * (1.0 if i == j else BLEED) for j in range(n))
        x = heard * 10 ** (gains[i] / 20) + rng.normal(0, 10 ** (-70 / 20), heard.size)
        sf.write(str(tmp_path / f"{names[i]}.wav"), x.astype(np.float32), R, subtype="FLOAT")
        parts.append(Participant(id=names[i], source=f"{names[i]}.wav", offset_s=offsets[i]))
    write_session(tmp_path / "session.toml", SessionConfig(participants=parts))
    return Session.load(tmp_path)


def settings(*extra):
    base = ["bleed.coupling_min_frames=50", 'decoding.language="hu"']
    return load_settings(overrides=[*base, *extra])


def run(session, s, log=None):
    return process_session(
        session,
        s,
        isolation="inline",
        factory=FakeFactory(calls=log),
        model_identity=lambda _: IDENTITY,
    )


def test_bleed_suppressed_with_source_and_coupling_reported(tmp_path):
    session = make(tmp_path, [[(1, 3)], [(6, 8)], [(11, 13)]])
    result, stats = run(session, settings())
    kept = [(s.speaker, round(s.start / R)) for s in result.transcript.segments]
    assert kept == [("gm", 1), ("anna", 6), ("bela", 11)]
    bleed = sorted((s.speaker, s.bleed_source) for s in result.transcript.suppressed)
    assert bleed == [
        ("anna", "bela"),
        ("anna", "gm"),
        ("bela", "anna"),
        ("bela", "gm"),
        ("gm", "anna"),
        ("gm", "bela"),
    ]
    assert stats.bleed_suppressed == 6 and result.bleed_mode_applied == "post"
    c = np.array(result.coupling_db)
    assert abs(c[1, 0] - (-18 - 6)) < 1.5  # acoustic -18 dB plus anna's -6 dB gain
    assert all(s.ownership and s.ownership > 0.9 for s in result.transcript.segments)


def test_off_mode_keeps_everything(tmp_path):
    session = make(tmp_path, [[(1, 3)], [(6, 8)], [(11, 13)]])
    result, _ = run(session, settings('bleed.mode="off"'))
    assert len(result.transcript.segments) == 9 and not result.transcript.suppressed
    assert result.coupling_db is None and result.bleed_mode_applied == "off"


def test_offsets_are_respected(tmp_path):
    # bela's recorder started 2 s late: his track-local 9-11 s is session time 11-13 s
    session = make(tmp_path, [[(1, 3)], [(6, 8)], [(9, 11)]], offsets=(0.0, 0.0, 2.0))
    # re-make bela's track without bleed: in track time it would not line up with the others
    rng = np.random.default_rng(2)
    t = np.arange(20 * R) / R
    x = np.zeros(20 * R, np.float32)
    x[9 * R : 11 * R] = 0.5 * np.sin(2 * np.pi * 440 * t[: 2 * R])
    sf.write(
        str(tmp_path / "bela.wav"),
        (x + rng.normal(0, 3e-4, x.size)).astype(np.float32),
        R,
        subtype="FLOAT",
    )
    result, _ = run(session, settings())
    bela = [s for s in result.transcript.segments if s.speaker == "bela"]
    assert bela and round(bela[0].start / R) == 11 and bela[0].track_start == 9 * R


def test_post_parameters_rerun_without_asr(tmp_path):
    session = make(tmp_path, [[(1, 3)], [(6, 8)], [(11, 13)]])
    log = tmp_path / "calls.txt"
    run(session, settings(), log)
    first = len(log.read_text(encoding="utf-8").splitlines())
    _, stats = run(
        session, settings("bleed.ownership_margin_db=3", "bleed.suppress_below=0.2"), log
    )
    assert len(log.read_text(encoding="utf-8").splitlines()) == first  # no ASR
    assert stats.transcribed == 0 and stats.bleed_suppressed == 6


def test_bleed_across_a_chunk_boundary(tmp_path):
    # gm talks over the 6 s chunk boundary; his bleed on anna's track spans it too
    session = make(tmp_path, [[(5, 7.5)], [(12, 14)]], gains=(0.0, -6.0))
    s = settings("asr.chunk_owned_s=6", "asr.chunk_context_s=1", "asr.chunk_boundary_search_s=0")
    result, _ = run(session, s)
    assert {x.speaker for x in result.transcript.segments} == {"gm", "anna"}
    anna_kept = [x for x in result.transcript.segments if x.speaker == "anna"]
    assert [round(x.start / R) for x in anna_kept] == [12]
    bleed = sorted(
        (x.speaker, round(x.start / R), x.bleed_source)
        for x in result.transcript.suppressed
        if x.suppression_reason == "bleed"
    )
    assert [b for b in bleed if b[0] == "gm"] == [("gm", 12, "anna")]
    assert [b for b in bleed if b[0] == "anna"] and all(
        b[2] == "gm" and 5 <= b[1] <= 7 for b in bleed if b[0] == "anna"
    )
    gm_words = sum(len(x.words) for x in result.transcript.segments if x.speaker == "gm")
    assert gm_words >= 6  # 2.5 s of gm in 0.4 s fake words, across the boundary


def test_pre_mode_attenuates_before_asr_with_its_own_cache_keys(tmp_path):
    session = make(tmp_path, [[(1, 3)], [(6, 8)], [(11, 13)]])
    log = tmp_path / "calls.txt"
    post, _ = run(session, settings(), log)
    after_post = len(log.read_text(encoding="utf-8").splitlines())
    pre, stats = run(session, settings('bleed.mode="pre"', "bleed.pre_attenuation_db=18"), log)
    assert stats.transcribed == stats.chunks  # new ASR keys: the audio differs
    assert len(log.read_text(encoding="utf-8").splitlines()) > after_post
    # bleed (-18 dB) attenuated another 18 dB falls below what the fake engine hears
    assert len(pre.transcript.segments) == 3 and not pre.transcript.suppressed
    assert pre.bleed_mode_applied == "pre"
    assert [s.speaker for s in pre.transcript.segments] == [
        s.speaker for s in post.transcript.segments
    ]
    _, stats2 = run(session, settings('bleed.mode="pre"', "bleed.pre_attenuation_db=18"), log)
    assert stats2.transcribed == 0  # cached under the pre-mode key
