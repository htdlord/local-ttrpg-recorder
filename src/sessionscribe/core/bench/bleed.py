"""Bleed-handling matrix on a triage-labelled clip (M3): off vs post/pre variants.

Post variants reuse the ASR cache (attribution only); pre variants transcribe again.
Results are numbers only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sessionscribe.core.bench.clip import Clip
from sessionscribe.core.bench.triage import (
    TriageFile,
    clip_session,
    read_triage,
    triage_path,
)
from sessionscribe.core.bench.triage_metrics import score_triage
from sessionscribe.core.config import load_settings
from sessionscribe.core.exporters.formats import to_json
from sessionscribe.core.fileio import atomic_write_json, atomic_write_text


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: tuple[str, ...]


def default_variants() -> list[Variant]:
    out = [Variant("off", ('bleed.mode="off"',))]
    for margin in (3, 6, 9):
        for below in (0.3, 0.5):
            out.append(
                Variant(
                    f"post m{margin} s{below:g}",
                    (
                        'bleed.mode="post"',
                        f"bleed.ownership_margin_db={margin}",
                        f"bleed.suppress_below={below}",
                    ),
                )
            )
    for att in (12, 18):
        out.append(Variant(f"pre a{att}", ('bleed.mode="pre"', f"bleed.pre_attenuation_db={att}")))
    return out


def run_variant(clip: Clip, triage: TriageFile, variant: Variant, **kwargs: Any) -> dict[str, Any]:
    from sessionscribe.core.pipeline.run import process_session

    settings = load_settings(
        overrides=[f'decoding.language="{clip.config.language}"', *variant.overrides]
    )
    t0 = time.perf_counter()
    result, stats = process_session(clip_session(clip), settings, **kwargs)
    score = score_triage(triage, to_json(result))
    return {
        "variant": variant.name,
        "overrides": list(variant.overrides),
        "seconds": round(time.perf_counter() - t0, 1),
        "transcribed": stats.transcribed,
        "bleed_suppressed": stats.bleed_suppressed,
        "duplicates": stats.duplicates,
        "score": score.to_dict(),
        "coupling_db": result.coupling_db,
    }


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}"


def results_markdown(clip_id: str, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        f"# Bleed matrix: {clip_id}",
        "",
        "Scored against the triage reference. Recall / false suppression over s/w/v/o, "
        "residual duplicates over b (flagged = kept with a review flag).",
        "",
        "| variant | recall % | recall by time % | false suppr. % | absent | residual dup. % | "
        "residual flagged % | no-speech kept % | w attributed % | bleed suppr. | dup. removed "
        "| ASR chunks | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["score"]
        lines.append(
            f"| {r['variant']} | {_pct(s['speech_recall'])} | {_pct(s['speech_recall_s'])} | "
            f"{_pct(s['false_suppression'])} | {s['speech_absent']} | "
            f"{_pct(s['residual_duplicates'])} | {_pct(s['residual_flagged'])} | "
            f"{_pct(s['hallucination_residue'])} | {_pct(s['attribution'])} | "
            f"{r['bleed_suppressed']} | {r['duplicates']} | {r['transcribed']} | {r['seconds']} |"
        )
    return "\n".join(lines) + "\n"


def run_bleed_matrix(
    clip: Clip, out_dir: Path, variants: Sequence[Variant] | None = None, **kwargs: Any
) -> list[dict[str, Any]]:
    triage = read_triage(triage_path(clip))
    rows = [run_variant(clip, triage, v, **kwargs) for v in (variants or default_variants())]
    atomic_write_json(out_dir / "results.json", {"clip": clip.config.id, "rows": rows})
    atomic_write_text(out_dir / "results.md", results_markdown(clip.config.id, rows))
    return rows
