import numpy as np
import soundfile as sf

from sessionscribe.core.cache import Cache
from sessionscribe.core.config import load_settings
from sessionscribe.core.pipeline.coupling import estimate_coupling
from sessionscribe.core.pipeline.ownership import (
    compute_ownership,
    hold_forward,
    ownership_masks,
    track_floors,
)
from sessionscribe.core.session import Participant, Session, SessionConfig, write_session
from tests import synth_bleed as sb

FPS = sb.FPS


def masks_for(turns, gains, noise, seconds, ac=None, margin=6.0):
    n = len(gains)
    ac = sb.acoustic(n) if ac is None else ac
    lv = sb.levels(turns, ac, gains, noise, seconds)
    c = estimate_coupling([str(i) for i in range(n)], lv, min_frames=250, default_db=-30)
    return ownership_masks(lv, c.matrix, track_floors(lv, 10.0), margin), lv


def frac(mask, a, b):
    return float(mask[int(a * FPS) : int(b * FPS)].mean())


def test_relative_floor_with_different_gains_and_noise():
    # same voices, mics at 0 / -15 / -20 dB gain with different noise floors
    turns = [[(0, 20)], [(20, 40)], [(40, 60)]]
    m, lv = masks_for(turns, [0.0, -15.0, -20.0], [-75.0, -65.0, -90.0], 80)  # 60-80 s quiet
    for i, (a, b) in enumerate([(0, 20), (20, 40), (40, 60)]):
        assert frac(m[i], a + 0.5, b - 0.5) > 0.95, i  # own speech owned on every mic
        others = [j for j in range(3) if j != i]
        assert all(frac(m[j], a + 0.5, b - 0.5) < 0.02 for j in others)  # bleed is not
    floors = track_floors(lv, 10.0)
    assert floors[1] > floors[2] + 15  # per track: the noisy mic has a higher floor
    # a common absolute floor would miss the -20 dB mic's own speech entirely
    assert float(np.median(lv[2, 40 * FPS : 60 * FPS])) < -38


def test_quiet_speaker_next_to_loud_one():
    ac = sb.acoustic(2, -18.0)
    turns = [[(0, 30)], [(10, 12), (30, 40)]]
    lv = sb.levels(turns, ac, [0.0, 0.0], [-80, -80], 50, voice_db=[-20.0, -30.0])
    c = estimate_coupling(["loud", "quiet"], lv, min_frames=250, default_db=-30)
    m = ownership_masks(lv, c.matrix, track_floors(lv, 10.0), 6.0)
    # the quiet remark is 9 dB below the loud neighbour's absolute level, but above the
    # bleed predicted from it, so it is owned
    assert float(np.median(lv[0, 505:595])) - float(np.median(lv[1, 505:595])) > 8
    assert frac(m[1], 10.2, 11.8) > 0.8
    assert frac(m[1], 2, 8) < 0.05


def test_true_overlap_both_own():
    m, _ = masks_for([[(0, 30), (40, 45)], [(30, 60), (40, 45)]], [0, 0], [-80, -80], 60)
    assert frac(m[0], 40.5, 44.5) > 0.9 and frac(m[1], 40.5, 44.5) > 0.9


def test_hold_forward():
    mask = np.zeros(10, bool)
    mask[2] = True
    assert hold_forward(mask, 3).nonzero()[0].tolist() == [2, 3, 4, 5]
    assert hold_forward(mask, 0).nonzero()[0].tolist() == [2]


def write_session_folder(tmp_path, offsets=(0.0, 0.0)):
    rate = 48_000
    t = np.arange(10 * rate) / rate
    a = np.zeros(20 * rate, np.float32)
    b = np.zeros(20 * rate, np.float32)
    a[: 10 * rate] = 0.3 * np.sin(2 * np.pi * 300 * t)
    b[: 10 * rate] = 0.03 * np.sin(2 * np.pi * 300 * t)
    b[10 * rate :] = 0.3 * np.sin(2 * np.pi * 450 * t)
    a[10 * rate :] = 0.03 * np.sin(2 * np.pi * 450 * t)
    sf.write(str(tmp_path / "a.wav"), a, rate, subtype="FLOAT")
    sf.write(str(tmp_path / "b.wav"), b, rate, subtype="FLOAT")
    cfg = SessionConfig(
        participants=[
            Participant(id="a", source="a.wav", offset_s=offsets[0]),
            Participant(id="b", source="b.wav", offset_s=offsets[1]),
        ]
    )
    write_session(tmp_path / "session.toml", cfg)
    return Session.load(tmp_path)


def test_compute_ownership_cached_and_invalidated(tmp_path):
    session = write_session_folder(tmp_path)
    cache = Cache(tmp_path / "cache")
    s = load_settings(overrides=["bleed.coupling_min_frames=100"])
    own = compute_ownership(session, s, cache)
    assert own.owned("a", 48_000, 9 * 48_000) > 0.95 and own.owned("b", 48_000, 9 * 48_000) < 0.05
    assert own.coupling.matrix[1, 0] < -15  # a's voice on b's mic is ~20 dB down
    again = compute_ownership(session, s, cache)
    assert again.key == own.key and np.array_equal(again.masks, own.masks)
    other_margin = compute_ownership(
        session,
        load_settings(overrides=["bleed.coupling_min_frames=100", "bleed.ownership_margin_db=9"]),
        cache,
    )
    assert other_margin.key != own.key
    shifted = write_session_folder(tmp_path, offsets=(0.0, 1.0))
    moved = compute_ownership(shifted, s, cache)
    assert moved.key != own.key
    assert moved.owned("b", 12 * 48_000, 19 * 48_000) > 0.9  # b's speech now at 11-21 s


def test_manual_and_guided_warnings(tmp_path):
    session = write_session_folder(tmp_path)
    s = load_settings(overrides=['bleed.coupling="guided"'])
    own = compute_ownership(session, s, None)
    assert any("no [mic_check]" in w for w in own.warnings)
    session.config.coupling_db = [[0, -20], [-20, 0]]
    manual = compute_ownership(session, load_settings(overrides=['bleed.coupling="manual"']), None)
    assert manual.coupling.matrix[0, 1] == -20
