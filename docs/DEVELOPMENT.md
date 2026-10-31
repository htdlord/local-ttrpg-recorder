# Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. Read [SPEC.md](SPEC.md) before
changing behaviour and record non-trivial decisions in [DECISIONS.md](DECISIONS.md).

```bash
uv sync --extra gui          # add --extra cuda for pip-installed cuBLAS/cuDNN 9
uv run pre-commit install
uv run ruff check . && uv run ruff format .
uv run mypy
uv run pytest -m "not gpu and not slow and not audio_hw"   # fast suite (= CI)
uv run pytest -m "gpu or slow or audio_hw"                 # local: GPU, real model, audio interface
```

Test markers: `gpu` (needs CUDA), `slow` (loads a real model or runs long), `audio_hw`
(needs an audio interface). CI runs the fast suite on Windows and Ubuntu, Python 3.11
and 3.12; GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`).

## Hardware checks

```bash
uv run sessionscribe doctor                                   # versions, CUDA, real inference, audio
uv run sessionscribe doctor --host-api asio --channels 1-8    # also opens an 8-channel ASIO stream
uv run sessionscribe record-probe --host-api asio --channels 1-8 --minutes 30
```

`record-probe` writes one 24-bit FLAC per channel to `work/probe/<timestamp>/` (use
`--no-write` to only count) and reports frame counts, overflows, timestamp continuity,
clock deviation and per-channel peak/RMS levels.

## Benchmarks

Benchmarks run against local reference clips, which are personal data and git-ignored
(`bench/refs/`). [REFERENCE_GUIDE.md](REFERENCE_GUIDE.md) describes how to prepare one.

```bash
uv run sessionscribe bench run --refs bench/refs --matrix m1   # writes work/bench/<run>/results.md
uv run sessionscribe bench score work/bench/<run>              # re-score without ASR
uv run sessionscribe bench clip cut bench/refs/clip02 --session recordings/2026-11-05 --start 1:20:00
uv run sessionscribe bench ref triage bench/refs/clip01        # label segments by ear (M3)
uv run sessionscribe bench triage score bench/refs/clip01 out.json
uv run sessionscribe bench bleed bench/refs/clip01             # off vs post/pre on the triage
```

## Release checks

```bash
uv run python scripts/clean_env_check.py                  # the cuda extra alone runs CUDA inference
uv run python scripts/clean_env_check.py --no-cuda-extra  # control: what is missing without it
uv run python scripts/third_party_notices.py              # regenerate THIRD_PARTY_NOTICES.md
```

The `Install` workflow (`.github/workflows/install.yml`) repeats [INSTALL.md](INSTALL.md)
on clean Windows and Ubuntu runners. [RELEASING.md](RELEASING.md) lists the release steps.

## Translations

GUI strings go through `tr()`. After changing them:

```bash
uv run pyside6-lupdate <gui .py files> -ts src/sessionscribe/gui/i18n/sessionscribe_hu.ts
# translate the new entries (Qt Linguist or by hand), then
uv run pyside6-lrelease src/sessionscribe/gui/i18n/sessionscribe_hu.ts -qm src/sessionscribe/gui/i18n/sessionscribe_hu.qm
```

A test checks that every message has a finished Hungarian translation. Settings tooltips
(`SettingsTab.field_tips`) repeat the `description` of each field in `core.config`; a
test keeps the two equal.
