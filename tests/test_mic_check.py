"""Mic check alarms on synthetic level events (rises relative to each channel's noise)."""

import numpy as np

from sessionscribe.core.audio.mic_check import analyze, overall, plan_turns
from sessionscribe.core.audio.recorder import LevelEvent

PARTS = ["gm", "anna", "bela"]
CHANNELS = [1, 2, 3]


def events(noise, speech, *, bleed=-25.0, peak_extra=6.0, swap=None, dt=0.25):
    """noise[c] dBFS; speech[c] dBFS on c's own mic during c's turn; bleed relative."""
    noise_w, turns = plan_turns(PARTS, 5.0, 5.0)
    out = []
    t = 0.0
    end = turns[-1].end_s + 1
    rng = np.random.default_rng(0)
    while t <= end:
        level = np.array(noise, dtype=float) + rng.normal(0, 0.5, 3)
        for turn in turns:
            if turn.start_s <= t <= turn.end_s:
                spk = turn.channel_index
                src = swap.get(spk, spk) if swap else spk
                for c in range(3):
                    own = speech[c] if c == src else speech[c] + bleed
                    level[c] = max(
                        level[c],
                        own if c == src else speech[src] + bleed + (noise[c] - noise[src]) * 0,
                    )
        out.append(LevelEvent(round(t, 2), tuple(level + peak_extra), tuple(level), (False,) * 3))
        t += dt
    return out, noise_w, turns


def kinds(checks):
    return [[k for k, _ in c.alarms] for c in checks]


def run(evts):
    e, noise_w, turns = evts
    return analyze(e, CHANNELS, PARTS, turns, noise_w)


def test_all_good_with_very_different_gains():
    # gains differ by 20 dB: absolute levels -20 / -35 / -40, all rise > 25 dB over noise
    checks = run(events([-70, -75, -80], [-20, -35, -40]))
    assert kinds(checks) == [[], [], []] and overall(checks) == "PASS"
    assert checks[1].rise_db > 30


def test_no_signal_and_digital_silence():
    checks = run(events([-70, -75, -60], [-20, -35, -58]))  # bela's mic: no rise
    assert kinds(checks)[2] == ["no_signal"] and overall(checks) == "FAIL"
    checks = run(events([-70, -75, -130], [-20, -35, -130]))  # unplugged: digital zero
    assert "no_signal" in kinds(checks)[2]


def test_quiet_absolute_and_relative():
    checks = run(events([-90, -75, -80], [-55, -35, -40]))  # gm: rises 35 dB but -55 dBFS
    assert kinds(checks)[0] == ["quiet"]
    checks = run(events([-70, -75, -80], [-20, -35, -66]))  # bela rises 14 dB, others ~45
    assert kinds(checks)[2] == ["quiet"]


def test_swapped_assignment_uses_rise_not_level():
    # anna's voice lands on channel 3 and bela's on 2; channel 3 has 20 dB less gain, so its
    # absolute level stays lower, but its rise above its own noise is larger
    checks = run(events([-70, -60, -90], [-20, -30, -45], swap={1: 2, 2: 1}))
    assert "wrong_channel" in kinds(checks)[1]


def test_clipping():
    checks = run(events([-70, -75, -80], [-3, -35, -40], peak_extra=3.0))
    assert "clipping" in kinds(checks)[0] and overall(checks) == "WARN"


def test_plan_turns():
    noise, turns = plan_turns(PARTS, 5, 5)
    assert noise == (0.5, 5) and [(t.start_s, t.end_s) for t in turns] == [
        (6.5, 11),
        (12.5, 17),
        (18.5, 23),
    ]
    assert plan_turns(PARTS, 0, 4)[0] is None
