"""record CLI with the fake stream: rig profiles, start with mic check, verify, recover."""

import time

import pytest
from typer.testing import CliRunner

from tests.fake_stream import FakeOpener, Scenario
from ttrpg_recorder import cli_record
from ttrpg_recorder.cli import app
from ttrpg_recorder.core.audio.manifest import read_rig
from ttrpg_recorder.core.session import Session

runner = CliRunner()


def test_rig_save_and_flag_precedence(tmp_path):
    rig = tmp_path / "rig.toml"
    args = [
        "record",
        "rig",
        "save",
        str(rig),
        "--device",
        "UMC ASIO Driver",
        "--host-api",
        "asio",
        "--channels",
        "1-3",
        "--names",
        "gm,anna,bela",
        "--labels",
        "Mesélő,Anna,Béla",
    ]
    assert runner.invoke(app, args).exit_code == 0
    assert [c.name for c in read_rig(rig).channels] == ["Mesélő", "Anna", "Béla"]
    setup = cli_record.resolve_setup(rig, None, None, None, None)
    assert setup.device == "UMC ASIO Driver" and setup.channels == (1, 2, 3)
    assert setup.ids == ("gm", "anna", "bela") and setup.names[0] == "Mesélő"
    # explicit flags win; channels picked from the rig keep their participants
    over = cli_record.resolve_setup(rig, None, "wasapi", "1,3", None, segment_min=2)
    assert over.host_api == "wasapi" and over.ids == ("gm", "bela") and over.segment_min == 2
    assert runner.invoke(app, args).exit_code == 1  # exists without --force


def test_device_is_never_auto_selected():
    with pytest.raises(Exception) as err:
        cli_record.resolve_setup(None, None, None, "1-2", "a,b")
    assert getattr(err.value, "exit_code", None) == 2


@pytest.fixture
def fake_device(monkeypatch):
    def opener(setup):
        n = len(setup.channels)
        turn = 0.6  # seconds per mic-check phase at 480-frame blocks
        blocks_per_phase = int(turn * 1.0 * 100) + 20

        def amps(b):
            phase = b // blocks_per_phase
            out = [0.0] * n
            if 1 <= phase <= n:
                out[phase - 1] = 0.3
            return out

        return FakeOpener([Scenario(blocks=100 * 8, channels=n, block_sleep=0.01, amps_fn=amps)])

    monkeypatch.setattr(cli_record, "_opener", opener)


def test_start_without_check_then_verify_and_recover(tmp_path, fake_device):
    session = tmp_path / "rec1"
    args = [
        "record",
        "start",
        "--session",
        str(session),
        "--device",
        "x",
        "--host-api",
        "asio",
        "--channels",
        "1-2",
        "--names",
        "gm,anna",
        "--no-check",
        "--hours",
        "0.01",
        "--segment-min",
        "0.02",
        "--seconds",
        "3",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "stopped" in result.output
    s = Session.load(session)
    assert [p.source for p in s.participants] == ["rec/ch01_gm", "rec/ch02_anna"]
    assert runner.invoke(app, ["record", "verify", "--session", str(session)]).exit_code == 0
    recovered = runner.invoke(app, ["record", "recover", "--session", str(session)])
    assert recovered.exit_code == 0 and "interrupted" in recovered.output


def test_start_with_mic_check_saves_ranges(tmp_path, fake_device):
    session = tmp_path / "rec2"
    args = [
        "record",
        "start",
        "--session",
        str(session),
        "--device",
        "x",
        "--host-api",
        "asio",
        "--channels",
        "1-2",
        "--names",
        "gm,anna",
        "--noise-s",
        "0.6",
        "--turn-s",
        "0.6",
        "--hours",
        "0.01",
        "--force",
        "--seconds",
        "7",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Mic check:" in result.output and "please speak now" in result.output
    s = Session.load(session)
    assert set(s.config.mic_check) == {"gm", "anna"}


def test_open_failure_is_a_clear_error_not_a_hang(tmp_path, monkeypatch):
    from ttrpg_recorder.core.audio.backend import AudioSetupError

    class Broken(FakeOpener):
        def open(self, callback):
            raise AudioSetupError("Failed to load ASIO driver")

    monkeypatch.setattr(cli_record, "_opener", lambda setup: Broken([Scenario()]))
    base = ["--device", "x", "--host-api", "asio", "--channels", "1-2", "--names", "gm,anna"]
    t0 = time.monotonic()
    result = runner.invoke(
        app, ["record", "start", "--session", str(tmp_path / "s"), "--hours", "0.1", *base]
    )
    assert result.exit_code == 1 and "another program using the ASIO driver" in result.output
    check = runner.invoke(app, ["record", "check", *base])
    assert check.exit_code == 1 and "Failed to load ASIO driver" in check.output
    assert time.monotonic() - t0 < 15
    assert not (tmp_path / "s" / "session.toml").exists()  # nothing half-written


def test_level_line_fits_the_terminal(monkeypatch):
    import os

    from ttrpg_recorder.core.audio.recorder import LevelEvent

    monkeypatch.setattr(
        cli_record.shutil, "get_terminal_size", lambda d: os.terminal_size((80, 24))
    )
    setup = cli_record.resolve_setup(None, "x", "asio", "1-8", "gm,p1,p2,p3,p4,p5,p6,p7")
    event = LevelEvent(0.0, (-10.0,) * 8, (-20.0,) * 8, (False,) * 8)
    line = cli_record._level_line(event, setup)
    assert len(line) <= 79 and "p7" in line
