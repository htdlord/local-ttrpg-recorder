import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ttrpg_recorder.core.audio import backend
from ttrpg_recorder.core.audio.backend import (
    AudioSetupError,
    ChannelCountError,
    InputDevice,
    format_channel_spec,
    parse_channel_spec,
    plan_input_stream,
    select_input_device,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("1-8", (1, 2, 3, 4, 5, 6, 7, 8)),
        ("3", (3,)),
        ("1,3,5-6", (1, 3, 5, 6)),
        ("2,1,2", (1, 2)),
    ],
)
def test_parse_channel_spec(spec, expected):
    assert parse_channel_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "0-2", "5-3", "a"])
def test_parse_channel_spec_invalid(spec):
    with pytest.raises(ValueError):
        parse_channel_spec(spec)


def test_format_channel_spec():
    assert format_channel_spec((1, 2, 3)) == "1-3"
    assert format_channel_spec((4,)) == "4"
    assert format_channel_spec((1, 3)) == "1,3"


class FakeSD:
    @staticmethod
    def AsioSettings(channel_selectors):
        return ("asio", tuple(channel_selectors))

    @staticmethod
    def WasapiSettings(exclusive=False):
        return ("wasapi", exclusive)


def dev(api: str, channels: int, index: int = 0, name: str = "UMC") -> InputDevice:
    return InputDevice(index, name, api, channels, 48000.0)


def test_plan_asio_uses_channel_selectors():
    plan = plan_input_stream(FakeSD, dev("ASIO", 10), (2, 4, 5))
    assert plan.open_channels == 3
    assert plan.columns is None
    assert plan.extra_settings == ("asio", (1, 3, 4))


def test_plan_non_asio_opens_up_to_max_and_selects_columns():
    plan = plan_input_stream(FakeSD, dev("MME", 8), (2, 4))
    assert plan.open_channels == 4
    np.testing.assert_array_equal(plan.columns, [1, 3])
    assert plan.extra_settings is None


def test_plan_contiguous_from_one_needs_no_columns():
    plan = plan_input_stream(FakeSD, dev("Windows WASAPI", 8), tuple(range(1, 9)))
    assert plan.columns is None
    assert plan.extra_settings == ("wasapi", False)


def test_plan_wasapi_too_few_channels_has_clear_message():
    with pytest.raises(ChannelCountError) as err:
        plan_input_stream(FakeSD, dev("Windows WASAPI", 2, name="IN 01-02"), tuple(range(1, 9)))
    msg = str(err.value)
    assert "IN 01-02" in msg and "exposes 2" in msg and "need 8" in msg
    assert "stereo-pair" in msg and "asio" in msg


DEVICES = [
    dev("MME", 2, 0, "Sound Mapper"),
    dev("ASIO", 2, 5, "Realtek ASIO"),
    dev("ASIO", 10, 6, "UMC ASIO Driver"),
    dev("Windows WASAPI", 2, 9, "IN 01-02 (UMC)"),
]


def test_select_by_name_and_api():
    assert select_input_device(DEVICES, "umc", "asio", None).index == 6


def test_select_by_index():
    assert select_input_device(DEVICES, "9", None, None).index == 9


def test_select_prefers_default_with_enough_channels():
    assert select_input_device(DEVICES, None, "asio", 6, min_channels=8).index == 6


def test_select_falls_back_to_device_with_enough_channels():
    assert select_input_device(DEVICES, None, "asio", 5, min_channels=8).index == 6


def test_select_keeps_default_when_none_is_big_enough():
    chosen = select_input_device(DEVICES, None, "wasapi", None, min_channels=8)
    assert chosen.index == 9  # the caller then reports the channel shortage


def test_select_unknown_api_or_name():
    with pytest.raises(AudioSetupError):
        select_input_device(DEVICES, None, "jack", None)
    with pytest.raises(AudioSetupError):
        select_input_device(DEVICES, "nope", None, None)


def test_asio_after_plain_import_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())
    monkeypatch.setattr(backend, "_asio_requested_at_import", False)
    monkeypatch.delenv("SD_ENABLE_ASIO", raising=False)
    with pytest.raises(AudioSetupError, match="before the first import"):
        backend.load_sounddevice(enable_asio=True)


def test_asio_rejected_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(AudioSetupError, match="only available on Windows"):
        backend.load_sounddevice(enable_asio=True)
