"""Manual reassignment: session file, merge, export without ASR, CLI."""

import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.fakes import FakeFactory
from tests.test_attribution import seg
from tests.test_pipeline import IDENTITY, make_session
from ttrpg_recorder.cli import app
from ttrpg_recorder.core.config import load_settings
from ttrpg_recorder.core.exporters.formats import to_json, to_txt
from ttrpg_recorder.core.merge import REASSIGNED, Transcript, apply_reassignments
from ttrpg_recorder.core.pipeline.run import export_session, process_session
from ttrpg_recorder.core.session import Session, SessionConfig, write_session

runner = CliRunner()


def test_apply_reassignments_keeps_text_and_reports_unknown_ids():
    t = Transcript(segments=[seg("gm", 1, 2, "egy", sid="a"), seg("anna", 3, 4, "kettő", sid="b")])
    out, missing = apply_reassignments(t, {"a": "anna", "zzz": "gm"}, ["gm", "anna"])
    a = next(s for s in out.segments if s.id == "a")
    assert (a.speaker, a.reassigned_from, a.text) == ("anna", "gm", "egy")
    assert REASSIGNED in a.quality_flags and missing == ["zzz"]


def test_session_validation_and_reassign_method():
    cfg = SessionConfig(
        participants=[{"id": "gm", "source": "a.wav"}, {"id": "anna", "source": "b.wav"}]
    )
    cfg.reassign("gm-c000-0001", "anna")
    cfg.reassign("gm-c000-0001", "gm")  # replaces, not duplicates
    assert [(r.segment, r.to) for r in cfg.reassignments] == [("gm-c000-0001", "gm")]
    cfg.reassign("gm-c000-0001", None)
    assert cfg.reassignments == []
    with pytest.raises(ValueError, match="unknown participant"):
        cfg.reassign("x", "nobody")
    with pytest.raises(ValidationError, match="unknown participant"):
        SessionConfig(
            participants=[{"id": "gm", "source": "a.wav"}],
            reassignments=[{"segment": "x", "to": "anna"}],
        )


def test_reassignment_reexports_without_asr(tmp_path):
    session = make_session(tmp_path / "s", n=2)
    log = tmp_path / "calls.txt"
    s = load_settings(overrides=["asr.chunk_owned_s=4", 'bleed.mode="off"'])
    result, _ = process_session(
        session,
        s,
        isolation="inline",
        factory=FakeFactory(calls=log),
        model_identity=lambda x: IDENTITY,
    )
    calls = log.read_text(encoding="utf-8")
    target = result.transcript.segments[0]
    session.config.reassign(target.id, "p1")
    write_session(session.folder / "session.toml", session.config)
    again, stats = export_session(
        Session.load(session.folder), s, model_identity=lambda x: IDENTITY
    )
    assert stats.transcribed == 0 and log.read_text(encoding="utf-8") == calls
    moved = next(x for x in again.transcript.segments if x.id == target.id)
    assert moved.speaker == "p1" and moved.reassigned_from == target.speaker
    data = json.loads(json.dumps(to_json(again)))
    js = next(x for x in data["segments"] if x["id"] == target.id)
    assert js["reassigned_from"] == target.speaker and "reassigned" in js["quality_flags"]
    assert "Név 1:" in to_txt(again).splitlines()[0]


def test_reassign_cli(tmp_path):
    session = make_session(tmp_path / "s", n=2)
    folder = str(session.folder)
    ok = runner.invoke(
        app, ["reassign", "--session", folder, "--segment", "p0-c000-0000", "--to", "p1"]
    )
    assert ok.exit_code == 0, ok.output
    assert Session.load(session.folder).config.reassignments[0].to == "p1"
    bad = runner.invoke(app, ["reassign", "--session", folder, "--segment", "x", "--to", "nobody"])
    assert bad.exit_code == 2
    undo = runner.invoke(
        app, ["reassign", "--session", folder, "--segment", "p0-c000-0000", "--undo"]
    )
    assert undo.exit_code == 0 and Session.load(session.folder).config.reassignments == []
    both = runner.invoke(app, ["reassign", "--session", folder, "--segment", "x"])
    assert both.exit_code == 2
