"""Does the `cuda` extra alone make CUDA inference work? (SPEC §13 M6)

Builds a fresh virtual environment from uv.lock in a temporary folder, removes every
system CUDA/cuDNN location from the environment of the check (PATH entries, CUDA_PATH*,
CUDNN*, LD_LIBRARY_PATH entries) and runs `ttrpg doctor --device cuda` there.

    uv run python scripts/clean_env_check.py                 # with the cuda extra: must pass
    uv run python scripts/clean_env_check.py --no-cuda-extra # control: what is missing then

Passes when the real CUDA inference succeeds and cuBLAS/cuDNN were all loaded from the
environment itself ("pip" or "ctranslate2 wheel"), not from a system install. Prints only
check names, statuses and library locations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CUDA = re.compile(r"cuda|cudnn|gpu computing toolkit|nsight", re.IGNORECASE)
OWN_SOURCES = ("pip", "ctranslate2 wheel")


def clean_environment(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """``env`` without system CUDA/cuDNN locations; also returns what was removed."""
    out = dict(env)
    removed = []
    for key in list(out):
        if key.upper().startswith(("CUDA_PATH", "CUDNN", "CUDA_HOME")):
            removed.append(key)
            del out[key]
    for key in ("PATH", "LD_LIBRARY_PATH"):
        if key not in out:
            continue
        kept = []
        for entry in out[key].split(os.pathsep):
            if SYSTEM_CUDA.search(entry):
                removed.append(entry)
            else:
                kept.append(entry)
        out[key] = os.pathsep.join(kept)
    return out, removed


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def build_env(folder: Path, cuda_extra: bool, python: str) -> Path:
    """A venv with exactly the locked runtime packages (+ cuda extra) and ttrpg_recorder."""
    requirements = folder / "requirements.txt"
    export = ["uv", "export", "--locked", "--no-dev", "--no-hashes", "--no-emit-project"]
    if cuda_extra:
        export += ["--extra", "cuda"]
    _run([*export, "-o", str(requirements)], cwd=ROOT)
    venv = folder / "venv"
    _run(["uv", "venv", str(venv), "--python", python, "--quiet"])
    target = ["--python", str(venv)]
    _run(["uv", "pip", "install", *target, "--quiet", "-r", str(requirements)])
    _run(["uv", "pip", "install", *target, "--quiet", "--no-deps", str(ROOT)])
    scripts = venv / ("Scripts" if sys.platform == "win32" else "bin")
    return scripts / "ttrpg"


def check(cuda_extra: bool = True, python: str = "3.12", keep: bool = False) -> dict[str, Any]:
    folder = Path(tempfile.mkdtemp(prefix="ttrpg-recorder-clean-"))
    try:
        exe = build_env(folder, cuda_extra, python)
        env, removed = clean_environment(dict(os.environ))
        env.pop("VIRTUAL_ENV", None)
        done = subprocess.run(
            [str(exe), "doctor", "--device", "cuda", "--no-audio", "--json"],
            text=True,
            capture_output=True,
            env=env,
            cwd=folder,  # not the checkout: nothing imported from the source tree
        )
        results = json.loads(done.stdout) if done.stdout.strip().startswith("[") else []
        by_name = {r["name"]: r for r in results}
        libs = by_name.get("cuda-libraries", {}).get("details", [])
        sources = {}
        for line in libs:
            name, _, where = line.partition(": ")
            sources[name] = where.split(" (")[0]
        inference = by_name.get("inference", {})
        own = bool(sources) and all(s in OWN_SOURCES for s in sources.values())
        return {
            "cuda_extra": cuda_extra,
            "removed_from_environment": len(removed),
            "checks": {r["name"]: r["status"] for r in results},
            "inference": inference.get("summary", done.stderr.strip()[-300:]),
            "library_sources": sources,
            "passed": inference.get("status") == "OK" and own,
        }
    finally:
        if keep:
            print(f"kept {folder}")
        else:
            shutil.rmtree(folder, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-cuda-extra", action="store_true", help="control run without it")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--keep", action="store_true", help="keep the temporary environment")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()
    summary = check(not args.no_cuda_extra, args.python, args.keep)
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(
        f"cuda extra: {'yes' if summary['cuda_extra'] else 'no'}; "
        f"{summary['removed_from_environment']} system CUDA location(s) hidden"
    )
    for name, status in summary["checks"].items():
        print(f"  {status:5} {name}")
    for name, where in summary["library_sources"].items():
        print(f"  {name}: {where}")
    print(f"inference: {summary['inference']}")
    print("PASS" if summary["passed"] else "FAIL")
    if not summary["passed"] and summary["cuda_extra"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
