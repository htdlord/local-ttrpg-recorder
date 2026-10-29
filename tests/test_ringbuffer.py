import numpy as np
import pytest

from sessionscribe.core.audio.ringbuffer import RingBuffer


def frames(start: int, n: int, channels: int = 2) -> np.ndarray:
    base = np.arange(start, start + n, dtype=np.int32)[:, None]
    return base * 10 + np.arange(channels, dtype=np.int32)[None, :]


def test_write_read_roundtrip():
    rb = RingBuffer(16, 2)
    assert rb.write(frames(0, 5))
    np.testing.assert_array_equal(rb.read(), frames(0, 5))
    assert rb.available() == 0


def test_wraparound_preserves_order():
    rb = RingBuffer(8, 2)
    rb.write(frames(0, 6))
    rb.read(4)
    assert rb.write(frames(6, 5))  # wraps
    np.testing.assert_array_equal(rb.read(), frames(4, 7))


def test_overflow_drops_whole_block_and_counts():
    rb = RingBuffer(8, 2)
    assert rb.write(frames(0, 6))
    assert not rb.write(frames(6, 3))
    assert rb.dropped_blocks == 1
    assert rb.dropped_frames == 3
    np.testing.assert_array_equal(rb.read(), frames(0, 6))


def test_max_fill_tracked():
    rb = RingBuffer(8, 1)
    rb.write(np.zeros((3, 1), np.int32))
    rb.write(np.zeros((4, 1), np.int32))
    rb.read()
    rb.write(np.zeros((2, 1), np.int32))
    assert rb.max_fill_frames == 7


def test_column_selection_with_wrap():
    rb = RingBuffer(6, 2)
    cols = np.asarray([1, 3], dtype=np.intp)
    rb.write(frames(0, 4, channels=4), cols)
    rb.read(3)
    rb.write(frames(4, 4, channels=4), cols)  # wraps
    np.testing.assert_array_equal(rb.read(), frames(3, 5, channels=4)[:, [1, 3]])


def test_read_limit_and_empty():
    rb = RingBuffer(8, 1)
    assert rb.read().shape == (0, 1)
    rb.write(np.ones((5, 1), np.int32))
    assert rb.read(2).shape == (2, 1)
    assert rb.available() == 3


def test_invalid_size():
    with pytest.raises(ValueError):
        RingBuffer(0, 1)
