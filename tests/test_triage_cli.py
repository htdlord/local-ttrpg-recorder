"""Triage terminal loop and `bench ref triage` with scripted keys and a fake player."""

import csv
import json

import pytest
from typer.testing import CliRunner

from sessionscribe.cli import app
from sessionscribe.core.bench import triage as t
from sessionscribe.triage_cli import AudioSource, run_labelling
from tests.fakes import FakeFactory
from tests.test_triage import IDENTITY, make_clip

runner = CliRunner()


class FakePlayer:
    def __init__(self):
        self.played = []

    def play(self, audio, rate):
        self.played.append((len(audio), rate))

    def stop(self):
        pass


@pytest.fixture
def fake_pipeline(monkeypatch):
    real = t.clip_result

    def clip_result(clip, **kw):
        kw.pop("sink", None)
        kw.setdefault("model_identity", lambda s: IDENTITY)
        if kw.get("transcribe", True):
            kw.update(isolation="inline", factory=FakeFactory())
        return real(clip, **kw)

    monkeypatch.setattr(t, "clip_result", clip_result)


def test_loop_saves_after_every_label(tmp_path):
    clip = make_clip(tmp_path)
    kw = {"isolation": "inline", "factory": FakeFactory(), "model_identity": lambda s: IDENTITY}
    session, result = t.clip_result(clip, **kw)
    triage = t.new_triage(clip, session, result)
    path = t.triage_path(clip)
    keys = iter(["\r", " ", "m", "x", "b", "1", "q"])
    saved = []

    def read_key():
        saved.append(path.exists() and t.read_triage(path).first_unlabelled())
        return next(keys)

    out = []
    player = FakePlayer()
    labeller = t.Labeller(triage, t.visit_order(triage, None, None))
    run_labelling(
        triage,
        path,
        labeller,
        AudioSource(session),
        t.texts_by_id(result),
        {"gm": "Mesélő", "anna": "Anna"},
        player=player,
        read_key=read_key,
        echo=out.append,
    )
    final = t.read_triage(path)
    assert [(u.label, u.source) for u in final.units] == [("s", None), ("b", "gm"), (None, None)]
    assert saved[1] == 1  # written right after the first label
    assert len(player.played) == 6  # unit 1, unit 2, replay, mix, other, unit 3
    assert player.played[0] == (int(1.6 * 48_000), 48_000)  # 1 s unit + 2 x 0.3 s context
    assert any("Mesélő" in line and "suggestion: s (Enter)" in line for line in out)
    assert any(line.startswith("  ő0") for line in out)  # text on the terminal only


def test_triage_command_export_import_stats(tmp_path, fake_pipeline):
    clip = make_clip(tmp_path)
    csv_path = tmp_path / "t.csv"
    # first run outside a terminal: units are built and saved, labelling needs a TTY
    first = runner.invoke(app, ["bench", "ref", "triage", str(clip.folder)])
    assert first.exit_code == 1 and "interactive terminal" in first.output
    assert t.triage_path(clip).is_file()
    exported = runner.invoke(
        app, ["bench", "ref", "triage", str(clip.folder), "--export", str(csv_path)]
    )
    assert exported.exit_code == 0, exported.output
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert list(rows[0]) == [
        "id",
        "speaker",
        "kind",
        "start",
        "end",
        "suggestion",
        "label",
        "source",
    ]
    assert len(rows) == 3
    rows[0]["label"] = "s"
    rows[1]["label"], rows[1]["source"] = "b", "gm"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    imported = runner.invoke(
        app, ["bench", "ref", "triage", str(clip.folder), "--import", str(csv_path)]
    )
    assert imported.exit_code == 0, imported.output
    assert "2 labels imported" in imported.output
    stats = runner.invoke(app, ["bench", "ref", "triage", str(clip.folder), "--stats"])
    assert "2/3 labelled (draft); s 1, b 1" in stats.output
    rebuilt = runner.invoke(
        app, ["bench", "ref", "triage", str(clip.folder), "--rebuild", "--stats"]
    )
    assert "2 labels carried over" in rebuilt.output, rebuilt.output
    data = json.loads(t.triage_path(clip).read_text(encoding="utf-8"))
    assert [u["label"] for u in data["units"]] == ["s", "b", None]


@pytest.mark.audio_hw
def test_real_playback_of_a_short_tone():
    import time

    import numpy as np

    from sessionscribe.triage_cli import Player

    player = Player()
    assert player.sd is not None
    player.play(
        (0.1 * np.sin(np.arange(9600) * 2 * np.pi * 440 / 48_000)).astype(np.float32), 48_000
    )
    time.sleep(0.3)
    player.stop()
