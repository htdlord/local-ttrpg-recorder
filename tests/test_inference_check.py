import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from ttrpg_recorder.core import inference_check
from ttrpg_recorder.core.inference_check import run_inference_check_subprocess, synthetic_audio


def test_synthetic_audio_is_deterministic_16k_float32():
    a, b = synthetic_audio(), synthetic_audio()
    assert a.dtype == np.float32 and a.shape == (32_000,)
    np.testing.assert_array_equal(a, b)
    assert np.abs(a).max() < 1.0


def fake_run(stdout="", stderr="", returncode=0):
    return lambda *a, **k: SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_subprocess_result_is_parsed(monkeypatch):
    payload = {
        "ok": True,
        "model": "tiny",
        "device": "cuda",
        "compute_type": "float16",
        "cached": True,
        "network": None,
        "load_s": 1.0,
        "inference_s": 0.2,
        "error": None,
    }
    monkeypatch.setattr(subprocess, "run", fake_run(stdout="noise\n" + json.dumps(payload)))
    result = run_inference_check_subprocess("tiny", "auto", None)
    assert result.ok and result.device == "cuda"


def test_subprocess_crash_is_reported(monkeypatch):
    stderr = "Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!"
    monkeypatch.setattr(subprocess, "run", fake_run(stderr=stderr, returncode=-1073740791))
    result = run_inference_check_subprocess("tiny", "cuda", None)
    assert not result.ok
    assert "cudnn_ops64_9.dll" in result.error and "-1073740791" in result.error


def test_subprocess_timeout(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(subprocess, "run", boom)
    assert "timed out" in run_inference_check_subprocess("tiny", "cpu", None).error


@pytest.mark.slow
def test_real_inference_cpu():
    result = inference_check.run_inference_check("tiny", "cpu", "int8")
    assert result.ok, result.error


@pytest.mark.gpu
def test_real_inference_cuda_subprocess():
    result = run_inference_check_subprocess("tiny", "cuda", "float16")
    assert result.ok, result.error
