from sessionscribe.core.bench import triage as t
from sessionscribe.core.bench.bleed import (
    Variant,
    default_variants,
    results_markdown,
    run_bleed_matrix,
)
from tests.fakes import FakeFactory
from tests.test_triage import IDENTITY, make_clip


def test_default_variants():
    names = [v.name for v in default_variants()]
    assert names[0] == "off" and "post m6 s0.5" in names and "pre a18" in names


def test_matrix_on_a_labelled_clip(tmp_path):
    clip = make_clip(tmp_path)
    kw = {"isolation": "inline", "factory": FakeFactory(), "model_identity": lambda s: IDENTITY}
    session, result = t.clip_result(clip, **kw)
    triage = t.new_triage(clip, session, result)
    for u, label in zip(triage.units, ["s", "b", "b", "s"], strict=True):
        u.label = label
    t.write_triage(t.triage_path(clip), triage)
    variants = [Variant("off", ('bleed.mode="off"',)), Variant("post", ('bleed.mode="post"',))]
    rows = run_bleed_matrix(clip, tmp_path / "run", variants, **kw)
    off, post = (r["score"] for r in rows)
    assert off["residual_duplicates"] == 1.0 and off["speech_recall"] == 1.0
    assert post["residual_duplicates"] == 0.0 and post["speech_recall"] == 1.0
    assert rows[1]["transcribed"] == 0  # post mode reuses the ASR cache
    md = (tmp_path / "run" / "results.md").read_text(encoding="utf-8")
    assert md.startswith("# Bleed matrix: clip") and "| post | 100.0 |" in md
    assert results_markdown("x", rows).count("\n|") == 4
