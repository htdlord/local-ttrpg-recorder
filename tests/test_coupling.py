import numpy as np
import pytest

from sessionscribe.core.pipeline.coupling import (
    align,
    apply_guided,
    estimate_coupling,
    manual_coupling,
    noise_level,
    speech_level,
)
from tests import synth_bleed as sb

SPEAKERS = ["gm", "anna", "bela"]
GAINS = [0.0, -15.0, -20.0]  # mic gains differ like on the real interface
NOISE = [-75.0, -70.0, -80.0]
TURNS = [[(0, 20), (60, 70)], [(20, 40)], [(40, 60)]]


def test_auto_coupling_recovers_known_matrix():
    ac = sb.acoustic(3, -18.0)
    ac[1][0] = -10.0  # anna sits right next to the GM
    lv = sb.levels(TURNS, ac, GAINS, NOISE, 80)
    c = estimate_coupling(SPEAKERS, lv, min_frames=250, default_db=-30)
    np.testing.assert_allclose(c.matrix, sb.expected_coupling(ac, GAINS), atol=1.0)
    assert c.weak_pairs() == []
    assert c.to_json()[0][0] == 0.0


def test_too_few_frames_fall_back_to_default():
    lv = sb.levels([[(0, 30)], [(30, 32)], []], sb.acoustic(3), GAINS, NOISE, 40)
    c = estimate_coupling(SPEAKERS, lv, min_frames=250, default_db=-30)
    assert ("gm", "anna") in c.weak_pairs() and ("gm", "bela") in c.weak_pairs()
    assert c.matrix[0, 1] == -30 and c.source[(1, 0)] == "auto"


def test_guided_overrides_auto_for_checked_speakers():
    ac = sb.acoustic(3, -18.0)
    lv = sb.levels([[(0, 5)], [(5, 10)], [(10, 15)]], ac, GAINS, NOISE, 20)
    c = estimate_coupling(SPEAKERS, lv, min_frames=250, default_db=-30)
    assert c.weak_pairs()  # 5 s each is too little for auto
    c = apply_guided(
        c, lv, {"gm": (0, 250), "anna": (250, 500), "bela": (500, 750)}, min_frames=250
    )
    np.testing.assert_allclose(c.matrix, sb.expected_coupling(ac, GAINS), atol=1.0)
    assert {c.source[(1, 0)], c.source[(0, 1)]} == {"guided"}


def test_manual_coupling_shape_checked():
    c = manual_coupling(["a", "b"], [[5, -20], [-22, 3]])
    assert c.matrix.tolist() == [[0, -20], [-22, 0]]
    with pytest.raises(ValueError, match="2 x 2"):
        manual_coupling(["a", "b"], [[0, 1, 2]])


def test_levels_and_align():
    env = np.array([-120, -80, -70, -20, -18, -22], np.float32)
    assert speech_level(env) == pytest.approx(-18.2, abs=0.1)
    assert noise_level(env) == pytest.approx(-77, abs=1.5)
    grid = align([np.zeros(3, np.float32), np.ones(2, np.float32)], [0, 2])
    assert grid.shape == (2, 4) and grid[1, 2] == 1 and grid[1, 0] == -120
