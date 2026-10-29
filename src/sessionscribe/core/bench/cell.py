"""One benchmark cell in a child process: ``python -m sessionscribe.core.bench.cell``.

A separate process per cell gives clean peak RAM/VRAM figures and survives GPU OOM or
a crashing CUDA library. Transcripts go to ``--out``; stdout carries one JSON line
with timings only (no text).

``--owned-s S`` transcribes each track in chunks of S s (owned range), as the session
pipeline does, to check that chunk boundaries cost no recall.

``--workers N`` transcribes N tracks at once from threads sharing one model loaded with
``num_workers=N``. ``--track clip/speaker`` restricts the child to some tracks; the
runner uses it to split one cell over several processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def run(args: argparse.Namespace) -> dict[str, Any]:
    from sessionscribe.core.audio.reader import load_for_asr
    from sessionscribe.core.bench.clip import Clip, ClipTrack, active_seconds
    from sessionscribe.core.bench.resources import peak_rss_mb
    from sessionscribe.core.config import Settings, load_settings
    from sessionscribe.core.exporters.single import Speaker, write_outputs
    from sessionscribe.core.inference_check import synthetic_audio
    from sessionscribe.core.pipeline.asr import (
        AsrEngine,
        TranscriptionResult,
        is_out_of_memory,
        load_model,
    )
    from sessionscribe.core.timeline import TIMELINE_RATE

    info: dict[str, Any] = {"ok": False, "error": None, "oom": False}
    clips = [Clip.load(Path(p)) for p in args.clip]
    wanted = set(args.track or ())

    def settings_for(language: str) -> Settings:
        return load_settings(
            preset=args.preset,
            cli={
                "model": {
                    "id": args.model,
                    "compute_type": args.compute_type,
                    "num_workers": args.workers,
                },
                # Fixed batch sizes: an OOM must show as a failed cell, not a quiet fallback.
                "asr": {"mode": args.mode, "batch_size": args.batch_size, "oom_fallback": False},
                "decoding": {"language": language, "word_timestamps": args.word_timestamps},
            },
        )

    jobs: list[tuple[Clip, ClipTrack]] = [
        (clip, track)
        for clip in clips
        for track in clip.config.tracks
        if not wanted or f"{clip.config.id}/{track.speaker}" in wanted
    ]
    if args.workers > 1:  # longest first, so the pool finishes evenly
        jobs.sort(key=lambda j: -active_seconds(j[0].track_path(j[1])))

    try:
        base = settings_for(clips[0].config.language)
        model, loaded = load_model(base)
        info |= {
            "load_s": round(loaded.load_s, 3),
            "device": loaded.device,
            "compute_type": loaded.compute_type,
            "revision": loaded.revision,
        }
        t0 = time.perf_counter()
        warm, _ = model.transcribe(synthetic_audio(), language="en", beam_size=1, vad_filter=False)
        list(warm)
        info["warmup_s"] = round(time.perf_counter() - t0, 3)

        engines = {
            clip.config.id: AsrEngine(settings_for(clip.config.language), model, loaded)
            for clip in clips
        }

        def one(job: tuple[Clip, ClipTrack]) -> TranscriptionResult:
            clip, track = job
            engine = engines[clip.config.id]
            if args.owned_s:
                result = chunked(engine, clip.track_path(track), track.speaker, args.owned_s)
            else:
                audio, _ = load_for_asr(clip.track_path(track), max_duration_s=None)
                result = engine.transcribe(audio, source_id=track.speaker)
            who = Speaker(track.speaker, track.speaker, Path(track.file).name)
            out_dir = Path(args.out) / clip.config.id
            write_outputs(result, engine.settings, who, out_dir, track.speaker, ("json",))
            return result

        proc_start = time.time()
        t0 = time.perf_counter()
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(one, jobs))
        else:
            results = [one(job) for job in jobs]
        elapsed = time.perf_counter() - t0
        info |= {
            "ok": True,
            "tracks": len(jobs),
            "workers": args.workers,
            "proc_start": proc_start,
            "proc_end": proc_start + elapsed,
            "elapsed_s": round(elapsed, 3),
            "transcribe_s": round(sum(r.transcribe_s + r.vad_s for r in results), 3),
            "audio_s": round(sum(r.audio_s for r in results), 3),
            "speech_s": round(
                sum(sum(v.end - v.start for v in r.vad_intervals) for r in results) / TIMELINE_RATE,
                3,
            ),
            "batch_sizes": sorted({r.batch_size for r in results if r.batch_size}),
        }
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["oom"] = is_out_of_memory(exc)
    try:
        ram = peak_rss_mb()
    except (OSError, ValueError, TypeError):  # never lose the results over a statistic
        ram = None
    info["peak_ram_mb"] = None if ram is None else round(ram)
    return info


def chunked(engine: Any, path: Path, speaker: str, owned_s: float) -> Any:
    """Transcribe one track chunk by chunk, as the session pipeline does (no cache)."""
    from dataclasses import replace

    from sessionscribe.core.audio.track import VirtualTrack
    from sessionscribe.core.pipeline.asr import SpeechInterval
    from sessionscribe.core.pipeline.chunks import plan_chunks, resolve_chunks
    from sessionscribe.core.pipeline.features import envelope, frame_samples
    from sessionscribe.core.pipeline.worker import _shift, chunk_audio
    from sessionscribe.core.timeline import TIMELINE_RATE, rescale

    s = engine.settings
    track = VirtualTrack.open(path)
    frame_ms = s.bleed.frame_ms
    env = envelope(track.blocks(), frame_samples(frame_ms, track.samplerate))
    length = rescale(track.frames, track.samplerate, TIMELINE_RATE)
    chunks = plan_chunks(
        length,
        env,
        frame=frame_samples(frame_ms),
        owned_s=owned_s,
        context_s=s.asr.chunk_context_s,
        search_s=s.asr.chunk_boundary_search_s,
    )
    parts = [engine.transcribe(chunk_audio(track, c), source_id=speaker) for c in chunks]
    shifted = [
        [_shift(seg, c.ctx_start) for seg in r.segments] for c, r in zip(chunks, parts, strict=True)
    ]
    kept = [k for k in resolve_chunks(chunks, shifted) if not k.boundary_duplicate]
    vad = [
        SpeechInterval(
            max(v.start + c.ctx_start, c.owned_start), min(v.end + c.ctx_start, c.owned_end)
        )
        for c, r in zip(chunks, parts, strict=True)
        for v in r.vad_intervals
        if min(v.end + c.ctx_start, c.owned_end) > max(v.start + c.ctx_start, c.owned_start)
    ]
    segments = [replace(k.segment, index=i) for i, k in enumerate(kept)]
    return replace(
        parts[0],
        duration=length,
        transcribe_s=sum(r.transcribe_s for r in parts),
        vad_s=sum(r.vad_s for r in parts),
        segments=segments,
        vad_intervals=vad,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sessionscribe.core.bench.cell")
    parser.add_argument("--clip", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--mode", choices=("quality", "fast"), required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--word-timestamps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--preset", default="conversation")
    parser.add_argument("--workers", type=int, default=1, help="Parallel tracks (threads).")
    parser.add_argument("--track", action="append", help="clip_id/speaker to process.")
    parser.add_argument("--owned-s", type=float, help="Chunk like the session pipeline.")
    args = parser.parse_args(argv)
    args.word_timestamps = bool(args.word_timestamps)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    info = run(args)
    sys.stdout.write(json.dumps(info) + "\n")
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
