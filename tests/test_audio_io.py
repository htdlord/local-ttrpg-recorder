import numpy as np
import pytest
import soundfile as sf

from ttrpg_recorder.core.audio.reader import AudioReadError, load_for_asr, read_blocks, track_info
from ttrpg_recorder.core.audio.resample import StreamResampler, resample
from ttrpg_recorder.core.timeline import (
    format_clock,
    rescale,
    samples_to_seconds,
    seconds_to_samples,
)


def tone(freq, seconds, rate=48_000, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_timeline_conversions():
    assert seconds_to_samples(1.5) == 72_000
    assert samples_to_seconds(72_001) == 1.5
    assert rescale(3, 16_000, 48_000) == 9
    assert rescale(10, 48_000, 16_000) == 3  # 3.33 -> 3
    assert rescale(11, 48_000, 16_000) == 4  # 3.67 -> 4
    assert rescale(-11, 48_000, 16_000) == -4
    assert format_clock(seconds_to_samples(3723.9)) == "01:02:03"
    assert format_clock(seconds_to_samples(3723.9), millis=True) == "01:02:03.900"


def test_resampler_is_aligned():
    impulse = np.zeros(48_000, np.float32)
    impulse[24_000] = 1.0
    out = resample(impulse, 48_000, 16_000)
    assert len(out) == 16_000
    assert int(np.argmax(np.abs(out))) == 8_000


def test_resampler_block_invariant():
    x = tone(1000, 3.0)
    one = resample(x, 48_000, 16_000)
    rs = StreamResampler(48_000, 16_000)
    parts = [rs.process(x[i : i + 1234]) for i in range(0, len(x), 1234)]
    blocks = np.concatenate([*parts, rs.flush()])
    np.testing.assert_array_equal(one, blocks)


def test_resampler_passband_and_aliasing():
    passband = resample(tone(7000, 2.0), 48_000, 16_000)[1000:-1000]
    alias = resample(tone(10_000, 2.0), 48_000, 16_000)[1000:-1000]
    ref = rms_db(tone(7000, 2.0))
    assert abs(rms_db(passband) - ref) < 1.0
    assert rms_db(alias) - ref < -60


def test_resampler_other_rates_and_passthrough():
    assert len(resample(tone(440, 1.0, rate=44_100), 44_100, 16_000)) == 16_000
    x = tone(440, 0.5, rate=16_000)
    np.testing.assert_array_equal(resample(x, 16_000, 16_000), x)


def test_reader_mono_and_channels(tmp_path):
    mono = tmp_path / "mono.flac"
    sf.write(mono, tone(440, 1.0), 48_000, subtype="PCM_24")
    info = track_info(mono)
    assert (info.samplerate, info.channels, info.frames) == (48_000, 1, 48_000)
    assert sum(len(b) for b in read_blocks(mono, block_frames=10_000)) == 48_000
    audio, _ = load_for_asr(mono)
    assert audio.dtype == np.float32 and len(audio) == 16_000

    stereo = tmp_path / "st.wav"
    sf.write(stereo, np.stack([tone(440, 1.0), np.zeros(48_000, np.float32)], axis=1), 48_000)
    with pytest.raises(AudioReadError, match="--channel"):
        load_for_asr(stereo)
    silent, _ = load_for_asr(stereo, channel=2)
    assert np.max(np.abs(silent)) == 0
    with pytest.raises(AudioReadError, match="no channel 3"):
        load_for_asr(stereo, channel=3)


def test_reader_duration_limit_and_bad_file(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, tone(440, 2.0), 48_000)
    with pytest.raises(AudioReadError, match="at most"):
        load_for_asr(path, max_duration_s=1.0)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(AudioReadError, match="cannot read"):
        track_info(bad)
