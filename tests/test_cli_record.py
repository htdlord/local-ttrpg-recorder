"""record CLI with the fake stream: rig profiles, start with mic check, verify, recover."""

import pytest
from typer.testing import CliRunner

from sessionscribe import cli_record
from sessionscribe.cli import app
from sessionscribe.core.audio.manifest import read_rig
from sessionscribe.core.session import Session
from tests.fake_stream import FakeOpener, Scenario

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
