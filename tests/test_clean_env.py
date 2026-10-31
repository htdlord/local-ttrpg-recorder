"""The cuda extra alone is enough for CUDA inference (SPEC §13 M6); local only."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "clean_env_check", ROOT / "scripts" / "clean_env_check.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_system_cuda_locations_are_hidden():
    script = _script()
    sep = os.pathsep
    env = {
        "PATH": sep.join(
            [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
                r"C:\Windows\System32",
                r"C:\Program Files\NVIDIA\CUDNN\v9.27\bin",
            ]
        ),
        "CUDA_PATH": "x",
        "CUDA_PATH_V12_8": "x",
        "HOME": "h",
    }
    out, removed = script.clean_environment(env)
    assert out["PATH"] == r"C:\Windows\System32" and "CUDA_PATH" not in out
    assert out["HOME"] == "h" and len(removed) == 4


@pytest.mark.gpu
@pytest.mark.slow
def test_cuda_extra_alone_runs_cuda_inference_without_system_cuda():
    summary = _script().check(cuda_extra=True)
    assert summary["passed"], summary
    assert set(summary["library_sources"].values()) <= {"pip", "ctranslate2 wheel"}
