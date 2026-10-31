"""Real short inference for ``doctor``.

Runs in a child process (``python -m ttrpg_recorder.core.inference_check``) because
ctranslate2 may terminate the whole process when a CUDA/cuDNN library is missing.
The child prints one JSON line on stdout; the parent turns crashes into a result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_RATE = 16_000
MODEL_HOST = "huggingface.co"


@dataclass
class InferenceCheckResult:
    ok: bool
    model: str
    device: str
    compute_type: str | None
    cached: bool | None = None  # model found locally before loading
    network: bool | None = None  # only checked when the model is not cached
    load_s: float | None = None
    inference_s: float | None = None
    error: str | None = None


def has_network(host: str = MODEL_HOST, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def synthetic_audio(seconds: float = 2.0) -> np.ndarray:
    """Short 16 kHz test signal: silence, a tone, light noise. Deterministic."""
    n = int(seconds * WHISPER_RATE)
    t = np.arange(n) / WHISPER_RATE
    tone = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    tone[: n // 4] = 0.0
    noise = np.random.default_rng(0).normal(0.0, 0.003, n)
    return (tone + noise).astype(np.float32)


def run_inference_check(
    model: str,
    device: str,
    compute_type: str | None,
    model_dir: Path | None = None,
    offline: bool = False,
) -> InferenceCheckResult:
    """Load ``model`` and transcribe 2 s of synthetic audio, in this process."""
    from ttrpg_recorder.core.config import resolve_device, validate_compute_type

    result = InferenceCheckResult(ok=False, model=model, device=device, compute_type=compute_type)
    try:
        result.device = resolve_device(device)
        result.compute_type = validate_compute_type(result.device, compute_type)
    except Exception as exc:
        result.error = str(exc)
        return result

    from faster_whisper import WhisperModel

    kwargs: dict[str, Any] = dict(
        device=result.device,
        compute_type=result.compute_type,
        download_root=str(model_dir) if model_dir else None,
    )
    t0 = time.perf_counter()
    try:
        whisper = WhisperModel(model, local_files_only=True, **kwargs)
        result.cached = True
    except Exception as exc:
        result.cached = False
        logger.debug("model not available locally: %r", exc)
        if offline:
            result.error = f"model {model!r} is not in the local cache and offline mode is set"
            return result
        result.network = has_network()
        if not result.network:
            result.error = (
                f"model {model!r} is not in the local cache and {MODEL_HOST} is not reachable "
                "(no network); connect once to download it"
            )
            return result
        try:
            t0 = time.perf_counter()
            whisper = WhisperModel(model, local_files_only=False, **kwargs)
        except Exception as exc2:
            result.error = f"model load failed: {exc2}"
            return result
    result.load_s = time.perf_counter() - t0

    try:
        t0 = time.perf_counter()
        segments, _info = whisper.transcribe(
            synthetic_audio(),
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        for _ in segments:  # the generator does the actual decoding
            pass
        result.inference_s = time.perf_counter() - t0
    except Exception as exc:
        result.error = f"inference failed: {exc}"
        return result
    result.ok = True
    return result


def run_inference_check_subprocess(
    model: str,
    device: str,
    compute_type: str | None,
    model_dir: Path | None = None,
    offline: bool = False,
    timeout_s: float = 900.0,
) -> InferenceCheckResult:
    """Run :func:`run_inference_check` in a child process and survive its crashes."""
    cmd = [sys.executable, "-m", "ttrpg_recorder.core.inference_check", model, device]
    if compute_type:
        cmd += ["--compute-type", compute_type]
    if model_dir:
        cmd += ["--model-dir", str(model_dir)]
    if offline:
        cmd.append("--offline")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_s, env=env
        )
    except subprocess.TimeoutExpired:
        return InferenceCheckResult(
            False, model, device, compute_type, error=f"timed out after {timeout_s:.0f} s"
        )
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return InferenceCheckResult(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                break
    tail = " | ".join(proc.stderr.strip().splitlines()[-5:])
    return InferenceCheckResult(
        False,
        model,
        device,
        compute_type,
        error=f"inference process exited with code {proc.returncode}: {tail or 'no output'}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ttrpg_recorder.core.inference_check")
    parser.add_argument("model")
    parser.add_argument("device")
    parser.add_argument("--compute-type")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    result = run_inference_check(
        args.model, args.device, args.compute_type, args.model_dir, args.offline
    )
    sys.stdout.write(json.dumps(asdict(result)) + "\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
