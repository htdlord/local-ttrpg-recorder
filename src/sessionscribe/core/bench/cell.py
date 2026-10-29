"""One benchmark cell in a child process: ``python -m sessionscribe.core.bench.cell``.

A separate process per cell gives clean peak RAM/VRAM figures and survives GPU OOM or
a crashing CUDA library. Transcripts go to ``--out``; stdout carries one JSON line
with timings only (no text).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


def run(args: argparse.Namespace) -> dict[str, Any]:
    from sessionscribe.core.audio.reader import load_for_asr
    from sessionscribe.core.bench.clip import Clip
    from sessionscribe.core.bench.resources import peak_rss_mb
    from sessionscribe.core.config import load_settings
    from sessionscribe.core.exporters.single import Speaker, write_outputs
    from sessionscribe.core.inference_check import synthetic_audio
    from sessionscribe.core.pipeline.asr import AsrEngine, is_out_of_memory, load_model
    from sessionscribe.core.timeline import TIMELINE_RATE

    info: dict[str, Any] = {"ok": False, "error": None, "oom": False}
    clips = [Clip.load(Path(p)) for p in args.clip]

    def settings_for(language: str):
        return load_settings(
            preset=args.preset,
            cli={
                "model": {"id": args.model, "compute_type": args.compute_type},
                "asr": {"mode": args.mode, "batch_size": args.batch_size},
                "decoding": {"language": language, "word_timestamps": args.word_timestamps},
            },
        )

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

        transcribe_s = audio_s = speech_s = 0.0
        batch_sizes: set[int] = set()
        for clip in clips:
            settings = settings_for(clip.config.language)
            engine = AsrEngine(settings, model, loaded)
            for track in clip.config.tracks:
                audio, _ = load_for_asr(clip.track_path(track), max_duration_s=None)
                result = engine.transcribe(audio, source_id=track.speaker)
                transcribe_s += result.transcribe_s + result.vad_s
                audio_s += result.audio_s
                speech_s += sum(v.end - v.start for v in result.vad_intervals) / TIMELINE_RATE
                if result.batch_size:
                    batch_sizes.add(result.batch_size)
                who = Speaker(track.speaker, track.speaker, Path(track.file).name)
                write_outputs(
                    result,
                    settings,
                    who,
                    Path(args.out) / clip.config.id,
                    track.speaker,
                    ("json",),
                )
        info |= {
            "ok": True,
            "transcribe_s": round(transcribe_s, 3),
            "audio_s": round(audio_s, 3),
            "speech_s": round(speech_s, 3),
            "batch_sizes": sorted(batch_sizes),
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
    args = parser.parse_args(argv)
    args.word_timestamps = bool(args.word_timestamps)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    info = run(args)
    sys.stdout.write(json.dumps(info) + "\n")
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
