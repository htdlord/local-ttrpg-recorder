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
from typing import Any, Literal

from ttrpg_recorder.core.bench.clip import CLIP_FILE, Clip, active_seconds
from ttrpg_recorder.core.bench.metrics import TrackScore, score_track, segments_from_json, summarize
from ttrpg_recorder.core.bench.reference import Reference, read_reference
from ttrpg_recorder.core.bench.resources import GpuSampler
from ttrpg_recorder.core.fileio import atomic_write_json, atomic_write_text

# The default mode stays "quality" unless, on the default configuration, the best fast
# cell beats its WER by more than this (owner decision: accuracy decides; a session may
# take several hours). Processing time is reported for information only.
PROJECTION_SPEECH_H = 10.0
WER_TOLERANCE = 0.01  # 1 percentage point
DEFAULT_CONFIG = ("large-v3", "float16", True)


@dataclass(frozen=True)
class Cell:
    model: str
    compute_type: str
    mode: str
    batch_size: int | None = None
    word_timestamps: bool = True
    # Tracks processed at once: threads share one model (num_workers), processes each
    # load their own copy.
    parallel: int = 1
    parallel_kind: Literal["threads", "processes"] = "threads"
    # Chunked like the session pipeline (owned range in s); None = one piece per track.
    owned_s: float | None = None
    # Re-transcribe word-less VAD speech as the session pipeline does (M2).
    refill: bool = False

    @property
    def name(self) -> str:
        parts = [self.model, self.compute_type, self.mode]
        if self.batch_size:
            parts.append(f"b{self.batch_size}")
        parts.append("wts" if self.word_timestamps else "nowts")
        if self.parallel > 1:
            parts.append(
                f"p{self.parallel}{'threads' if self.parallel_kind == 'threads' else 'procs'}"
            )
        if self.owned_s:
            parts.append(f"own{self.owned_s:g}")
        if self.refill:
            parts.append("refill")
        return "_".join(parts)


def _parallel_cells() -> list[Cell]:
    """Quality mode, default configuration: 1, 2 and 3 tracks at once, threads vs processes."""
    return [
        Cell("large-v3", "float16", "quality", None, True, n, kind)
        for kind in ("threads", "processes")
        for n in (2, 3)
    ]


def _matrix_m1() -> list[Cell]:
    cells: list[Cell] = []
    for model in ("large-v3", "large-v3-turbo"):
        for ct in ("float16", "int8_float16"):
            for wts in (True, False):
                cells.append(Cell(model, ct, "quality", None, wts))
                cells.extend(Cell(model, ct, "fast", b, wts) for b in (4, 8, 16))
    return cells + _parallel_cells()


MATRICES: dict[str, Callable[[], list[Cell]]] = {
    "m1": _matrix_m1,
    "parallel": lambda: [Cell("large-v3", "float16", "quality"), *_parallel_cells()],
    # M2: chunk boundaries must not cost recall (owned 30/60 s vs whole tracks).
    "chunking": lambda: [
        Cell("large-v3", "float16", "quality", owned_s=s, refill=r)
        for r in (False, True)
        for s in (None, 30.0, 60.0)
    ],
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


def _cell_cmd(
    cell: Cell, clips: list[ClipRef], cell_dir: Path, workers: int, tracks: list[str] | None
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "ttrpg_recorder.core.bench.cell",
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
        "--workers",
        str(workers),
    ]
    for c in clips:
        cmd += ["--clip", str(c.clip.folder)]
    for t in tracks or ():
        cmd += ["--track", t]
    if cell.batch_size:
        cmd += ["--batch-size", str(cell.batch_size)]
    if cell.owned_s:
        cmd += ["--owned-s", str(cell.owned_s)]
    if cell.refill:
        cmd.append("--refill")
    return cmd


def partition_tracks(clips: list[ClipRef], groups: int) -> list[list[str]]:
    """Longest-processing-time-first split of all tracks into ``groups`` balanced lists."""
    loads = sorted(
        (
            (active_seconds(c.clip.track_path(t)), f"{c.clip.config.id}/{t.speaker}")
            for c in clips
            for t in c.clip.config.tracks
        ),
        reverse=True,
    )
    buckets: list[tuple[float, list[str]]] = [(0.0, []) for _ in range(groups)]
    for load, key in loads:
        i = min(range(groups), key=lambda k: buckets[k][0])
        buckets[i] = (buckets[i][0] + load, [*buckets[i][1], key])
    return [b for _, b in buckets if b]


def _child_info(proc: subprocess.Popen[str], stdout: str, stderr_path: Path) -> dict[str, Any]:
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    tail = (stderr.strip().splitlines() or ["no output"])[-1]
    info: dict[str, Any] = {"ok": False, "error": f"exit code {proc.returncode}: {tail}"}
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                info = json.loads(line)
            break
    return info


def merge_child_infos(infos: list[dict[str, Any]]) -> dict[str, Any]:
    """One cell's figures from several processes: times span all, memory adds up."""
    if len(infos) == 1:
        return infos[0]
    failed = [i for i in infos if not i.get("ok")]
    if failed:
        return {**failed[0], "processes": len(infos)}

    def total(key: str) -> float:
        return round(sum(i.get(key) or 0.0 for i in infos), 3)

    start = min(i["proc_start"] for i in infos)
    end = max(i["proc_end"] for i in infos)
    return {
        "ok": True,
        "error": None,
        "oom": False,
        "processes": len(infos),
        "workers": 1,
        "load_s": max(i.get("load_s") or 0.0 for i in infos),
        "warmup_s": max(i.get("warmup_s") or 0.0 for i in infos),
        "device": infos[0].get("device"),
        "compute_type": infos[0].get("compute_type"),
        "revision": infos[0].get("revision"),
        "tracks": int(total("tracks")),
        "proc_start": start,
        "proc_end": end,
        "elapsed_s": round(end - start, 3),
        "transcribe_s": total("transcribe_s"),
        "audio_s": total("audio_s"),
        "speech_s": total("speech_s"),
        "batch_sizes": [],
        "peak_ram_mb": int(total("peak_ram_mb")),
    }


def run_cell(cell: Cell, clips: list[ClipRef], cell_dir: Path, timeout_s: float) -> dict[str, Any]:
    if cell.parallel > 1 and cell.parallel_kind == "processes":
        cmds = [
            _cell_cmd(cell, clips, cell_dir, 1, group)
            for group in partition_tracks(clips, cell.parallel)
        ]
    else:
        cmds = [_cell_cmd(cell, clips, cell_dir, cell.parallel, None)]
    cell_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    sampler = GpuSampler()
    t0 = time.perf_counter()
    procs: list[tuple[subprocess.Popen[str], Path]] = []
    for i, cmd in enumerate(cmds):
        # stderr to a file: several children must never block on a full pipe.
        err_path = cell_dir / f"child{i}.stderr.log"
        with err_path.open("w", encoding="utf-8") as err:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", env=env
            )
        procs.append((proc, err_path))
    sampler.start([p.pid for p, _ in procs])
    infos: list[dict[str, Any]] = []
    deadline = t0 + timeout_s
    for proc, err_path in procs:
        try:
            stdout, _ = proc.communicate(timeout=max(1.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            for other, _ in procs:
                other.kill()
            proc.communicate()
            infos.append({"ok": False, "error": f"timed out after {timeout_s:.0f} s"})
            continue
        infos.append(_child_info(proc, stdout, err_path))
    vram, method = sampler.stop()
    info = merge_child_infos(infos)
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
        # Wall time of the processing phase (all tracks, after model load and warm-up), so
        # parallel cells are compared fairly; older runs only have the per-track sum.
        busy = info.get("elapsed_s") or info["transcribe_s"]
        out["rtf"] = round(busy / info["audio_s"], 4)
        if info.get("speech_s"):
            speech_rtf = busy / info["speech_s"]
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
            and not s.get("owned_s")
            and not s.get("refill")
        )

    qualities = [c for c in cells if default(c) and c["cell"]["mode"] == "quality"]
    fast = [c for c in cells if default(c) and c["cell"]["mode"] == "fast"]
    if not qualities or not fast:
        return {"decision": None, "reason": "default configuration not measured in both modes"}
    best_fast = min(fast, key=lambda c: c["metrics"]["summary"]["wer"] or 1.0)
    # Parallel variants decode the same way; the fastest one sets the projection.
    sequential = [c for c in qualities if c["cell"].get("parallel", 1) == 1] or qualities
    q_wer = max(c["metrics"]["summary"]["wer"] or 0.0 for c in sequential)
    f_wer = best_fast["metrics"]["summary"]["wer"] or 0.0
    timed = [c for c in qualities if c["derived"].get("projected_min") is not None]
    quality = min(timed, key=lambda c: c["derived"]["projected_min"]) if timed else qualities[0]
    projected = quality["derived"].get("projected_min")
    keep = q_wer <= f_wer + WER_TOLERANCE
    return {
        "decision": "quality" if keep else "revisit",
        "provisional": True,
        "clip_seconds": round(clip_seconds, 1),
        "quality_wer": q_wer,
        "best_fast_wer": f_wer,
        "best_fast_cell": best_fast["name"],
        "quality_cell": quality["name"],
        "quality_projected_min": projected,
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
        "proc s = wall time of the processing phase (after model load); wall s = whole cell.",
        "RTF and speech RTF use proc s; projection = 10 h of speech.",
        "Parallel cells: VRAM/RAM cover all workers; processes load one model each.",
        "Silent tracks: tracks without reference speech; their output is hallucination/bleed.",
        "",
        "| cell | WER % | CER % | recall % | outside words | silent-track words / segs | "
        "ts start/end ms | RTF | speech RTF | 10 h speech min | proc s | wall s | load s | "
        "VRAM MiB | RAM MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in results["cells"]:
        run, d = c["run"], c.get("derived", {})
        if not run.get("ok"):
            lines.append(f"| {c['name']} | failed: {run.get('error')} |" + " |" * 13)
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
                    _num(run.get("elapsed_s"), "{:.1f}"),
                    _num(run.get("wall_s"), "{:.1f}"),
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
            f"({rec.get('quality_cell', 'quality')}) "
            f"(information only; WER decides, tolerance {100 * rec['wer_tolerance']:.0f} pp)."
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
