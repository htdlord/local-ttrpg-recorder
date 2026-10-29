"""Needs a real audio input device. Excluded from CI."""

import pytest

from sessionscribe.core.audio.probe import ProbeConfig, Verdict, run_probe


@pytest.mark.audio_hw
def test_default_device_short_probe():
    report = run_probe(ProbeConfig(channels=(1,), duration_s=1.0, write=False))
    assert report.frames_received > 0
    assert report.verdict is not Verdict.FAIL, report.reasons
