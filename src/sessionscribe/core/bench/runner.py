"""Benchmark matrix: run cells in child processes, score them, write numbers-only reports."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sessionscribe.core.bench.clip import CLIP_FILE, Clip
from sessionscribe.core.bench.metrics import TrackScore, score_track, segments_from_json, summarize
from sessionscribe.core.bench.reference import Reference, read_reference
from sessionscribe.core.bench.resources import GpuSampler
from sessionscribe.core.fileio import atomic_write_json, atomic_write_text

# The default mode stays "quality" if, on the default configuration, its projected time
# for 10 h of speech is within this budget and its WER is at most this much worse.
PROJECTION_SPEECH_H = 10.0
QUALITY_BUDGET_MIN = 60.0
WER_TOLERANCE = 0.01  # 1 percentage point
DEFAULT_CONFIG = ("large-v3", "float16", True)


@dataclass(frozen=True)
class Cell:
    model: str
    compute_type: str
    mode: str
    batch_size: int | None = None
    word_timestamps: bool = True

    @property
    def name(self) -> str:
        parts = [self.model, self.compute_type, self.mode]
        if self.batch_size:
            parts.append(f"b{self.batch_size}")
        parts.append("wts" if self.word_timestamps else "nowts")
        return "_".join(parts)


def _matrix_m1() -> list[Cell]:
    cells: list[Cell] = []
    for model in ("large-v3", "large-v3-turbo"):
        for ct in ("float16", "int8_float16"):
            for wts in (True, False):
                cells.append(Cell(model, ct, "quality", None, wts))
                cells.extend(Cell(model, ct, "fast", b, wts) for b in (4, 8, 16))
    return cells


MATRICES: dict[str, Callable[[], list[Cell]]] = {
    "m1": _matrix_m1,
    # Smoke test of the harness: tiny model, both modes.
    "quick": lambda: [
        Cell("tiny", "int8", "quality", None, True),
        Cell("tiny", "int8", "fast", 4, True),
    ],
}


@dataclass
class ClipRef:
    clip: Clip
    reference: Reference
    path: Path

    @property
    def verified(self) -> bool:
        return self.reference.status == "verified"


def discover_clips(
    refs: Path, names: list[str] | None = None, *, allow_draft: bool = False
) -> tuple[list[ClipRef], list[str]]:
    """Clip folders with a reference; returns the usable clips and skip messages."""
    found: list[ClipRef] = []
    skipped: list[str] = []
    for folder in sorted(p for p in refs.iterdir() if (p / CLIP_FILE).is_file()):
        if names and folder.name not in names:
            continue
        clip = Clip.load(folder)
        path = clip.reference_path
        if not path.is_file() and allow_draft:
            path = clip.draft_dir / "import.json"
        if not path.is_file():
            skipped.append(f"{folder.name}: no reference.json (build it with 'bench ref build')")
            continue
        found.append(ClipRef(clip, read_reference(path), path))
    return found, skipped


def run_cell(cell: Cell, clips: list[ClipRef], cell_dir: Path, timeout_s: float) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "sessionscribe.core.bench.cell",
        "--out",
        str(cell_dir),
        "--model",
        cell.model,
        "--compute-type",
        cell.compute_type,
        "--mode",
        cell.mode,
        "--word-timestamps",
        "1" if cell.word_timestamps else "0",
    ]
    for c in clips:
        cmd += ["--clip", str(c.clip.folder)]
    if cell.batch_size:
        cmd += ["--batch-size", str(cell.batch_size)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    sampler = GpuSampler()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env
    )
    sampler.start(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        info: dict[str, Any] = {"ok": False, "error": f"timed out after {timeout_s:.0f} s"}
    else:
        tail = (stderr.strip().splitlines() or ["no output"])[-1]
        info = {"ok": False, "error": f"exit code {proc.returncode}: {tail}"}
        for line in reversed(stdout.strip().splitlines()):
            if line.startswith("{"):
                with contextlib.suppress(json.JSONDecodeError):
                    info = json.loads(line)
                break
    vram, method = sampler.stop()
    info |= {
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_vram_mb": vram,
        "vram_method": method,
    }
    return info


def score_cell(cell_dir: Path, clips: list[ClipRef]) -> dict[str, Any]:
    all_scores: list[TrackScore] = []
    per_clip: dict[str, Any] = {}
    for c in clips:
        scores = []
        for track in c.clip.config.tracks:
            path = cell_dir / c.clip.config.id / f"{track.speaker}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            scores.append(score_track(track.speaker, c.reference, segments_from_json(data)))
        per_clip[c.clip.config.id] = summarize(scores).to_dict()
        all_scores += scores
    return {"summary": summarize(all_scores).to_dict(), "clips": per_clip}


def derived(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if info.get("ok") and info.get("audio_s"):
        out["rtf"] = round(info["transcribe_s"] / info["audio_s"], 4)
        if info.get("speech_s"):
            speech_rtf = info["transcribe_s"] / info["speech_s"]
            out["speech_rtf"] = round(speech_rtf, 4)
            out["projected_min"] = round(speech_rtf * PROJECTION_SPEECH_H * 60, 1)
    return out


def recommend_mode(cells: list[dict[str, Any]], clip_seconds: float) -> dict[str, Any]:
    """Provisional default-mode decision on the default configuration (DECISIONS M1)."""
    model, ct, wts = DEFAULT_CONFIG

    def default(c: dict[str, Any]) -> bool:
        s = c["cell"]
        return bool(
            c["run"].get("ok")
            and c.get("metrics")
            and (s["model"], s["compute_type"], s["word_timestamps"]) == (model, ct, wts)
        )

    quality = next((c for c in cells if default(c) and c["cell"]["mode"] == "quality"), None)
    fast = [c for c in cells if default(c) and c["cell"]["mode"] == "fast"]
    if quality is None or not fast:
        return {"decision": None, "reason": "default configuration not measured in both modes"}
    best_fast = min(fast, key=lambda c: c["metrics"]["summary"]["wer"] or 1.0)
    q_wer = quality["metrics"]["summary"]["wer"] or 0.0
    f_wer = best_fast["metrics"]["summary"]["wer"] or 0.0
    projected = quality["derived"].get("projected_min")
    keep = (
        projected is not None and projected <= QUALITY_BUDGET_MIN and q_wer <= f_wer + WER_TOLERANCE
    )
    return {
        "decision": "quality" if keep else "revisit",
        "provisional": True,
        "clip_seconds": round(clip_seconds, 1),
        "quality_wer": q_wer,
        "best_fast_wer": f_wer,
        "best_fast_cell": best_fast["name"],
        "quality_projected_min": projected,
        "budget_min": QUALITY_BUDGET_MIN,
        "wer_tolerance": WER_TOLERANCE,
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def _num(x: Any, fmt: str = "{}") -> str:
    return "-" if x is None else fmt.format(x)


def results_markdown(results: dict[str, Any]) -> str:
    lines = [f"# Benchmark {results['run']}", ""]
    for c in results["clips"]:
        status = c["status"]
        warn = "" if status == "verified" else "  **(UNVERIFIED reference: numbers provisional)**"
        lines.append(
            f"- clip {c['id']}: {c['seconds']:.1f} s, {c['tracks']} tracks, {status}{warn}"
        )
    lines += [
        "",
        "Speech RTF = processing time / VAD speech time; projection = 10 h of speech.",
        "Silent tracks: tracks without reference speech; their output is hallucination/bleed.",
        "",
        "| cell | WER % | CER % | recall % | outside words | silent-track words / segs | "
        "ts start/end ms | RTF | speech RTF | 10 h speech min | load s | VRAM MiB | RAM MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in results["cells"]:
        run, d = c["run"], c.get("derived", {})
        if not run.get("ok"):
            lines.append(f"| {c['name']} | failed: {run.get('error')} |" + " |" * 11)
            continue
        s = c["metrics"]["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    c["name"],
                    _pct(s["wer"]),
                    _pct(s["cer"]),
                    _pct(s["speech_recall"]),
                    str(s["hyp_words_outside"]),
                    f"{s['silent_track_words']} / {s['silent_track_segments']}",
                    f"{_num(s['timestamp_start_median_ms'])} / "
                    f"{_num(s['timestamp_end_median_ms'])}",
                    _num(d.get("rtf"), "{:.3f}"),
                    _num(d.get("speech_rtf"), "{:.3f}"),
                    _num(d.get("projected_min"), "{:.0f}"),
                    _num(run.get("load_s"), "{:.1f}"),
                    _num(run.get("peak_vram_mb"))
                    + ("*" if run.get("vram_method") == "device-delta" else ""),
                    _num(run.get("peak_ram_mb")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "\\* VRAM as device total minus baseline (driver gives no per-process value).",
        "",
    ]
    rec = results.get("recommendation") or {}
    if rec.get("decision"):
        lines.append(
            f"Default mode (provisional, {rec['clip_seconds']} s of clips): "
            f"**{rec['decision']}** - "
            f"quality WER {_pct(rec['quality_wer'])} % vs best fast {_pct(rec['best_fast_wer'])} % "
            f"({rec['best_fast_cell']}), quality projection {rec['quality_projected_min']} min "
            f"(budget {rec['budget_min']:.0f} min, tolerance {100 * rec['wer_tolerance']:.0f} pp)."
        )
    elif rec:
        lines.append(f"Default mode: no decision ({rec.get('reason')}).")
    return "\n".join(lines) + "\n"


def write_results(run_dir: Path, results: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "results.json", results)
    atomic_write_text(run_dir / "results.md", results_markdown(results))


def clip_summaries(clips: list[ClipRef]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.clip.config.id,
            "seconds": c.clip.config.duration_s,
            "tracks": len(c.clip.config.tracks),
            "status": c.reference.status if c.path.name == "reference.json" else "draft-import",
            "reference": c.path.name,
        }
        for c in clips
    ]


def score_run(run_dir: Path, clips: list[ClipRef]) -> dict[str, Any]:
    """(Re)score every cell of a run from its saved outputs; no ASR."""
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    for c in results["cells"]:
        if c["run"].get("ok"):
            c["metrics"] = score_cell(run_dir / "cells" / c["name"], clips)
    results["clips"] = clip_summaries(clips)
    seconds = sum(c.clip.config.duration_s for c in clips)
    results["recommendation"] = recommend_mode(results["cells"], seconds)
    write_results(run_dir, results)
    return results


def run_matrix(
    cells: list[Cell],
    clips: list[ClipRef],
    run_dir: Path,
    *,
    timeout_s: float = 3600.0,
    on_cell: Callable[[int, Cell, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "run": run_dir.name,
        "clips": clip_summaries(clips),
        "cells": [],
    }
    for i, cell in enumerate(cells):
        info = run_cell(cell, clips, run_dir / "cells" / cell.name, timeout_s)
        entry: dict[str, Any] = {"name": cell.name, "cell": asdict(cell), "run": info}
        entry["derived"] = derived(info)
        if info.get("ok"):
            entry["metrics"] = score_cell(run_dir / "cells" / cell.name, clips)
        results["cells"].append(entry)
        write_results(run_dir, results)  # partial results survive an interruption
        if on_cell:
            on_cell(i, cell, entry)
    seconds = sum(c.clip.config.duration_s for c in clips)
    results["recommendation"] = recommend_mode(results["cells"], seconds)
    write_results(run_dir, results)
    return results
