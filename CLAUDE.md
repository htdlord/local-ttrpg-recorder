# CLAUDE.md

Guidance for Claude Code when working in this repository. Read `docs/SPEC.md` before starting any milestone; it is the source of truth for requirements. Record non-trivial design decisions in `docs/DECISIONS.md`.

## Project

**SessionScribe** (working title): a local, multi-track speech-to-text app built on faster-whisper. Primary use case: tabletop RPG sessions recorded with one lavalier mic per participant (up to 8 participants, sessions of 5–6.5 hours, designed for 8). Each participant's track is transcribed separately, cross-track bleed is handled, and all segments are merged into one speaker-labelled, timestamped transcript.

- Reference machine: Ryzen 9 5900X (12C/24T), RTX 4070 Ti Super (16 GB VRAM), Behringer UMC1820.
- Platforms: Windows 10/11 and Linux.
- Primary language: Hungarian; English must work equally well. Language is always a setting.
- Fully usable offline after models are downloaded.

## Guiding principle

**Losing real speech is worse than showing a duplicate.** Filters (VAD, bleed, dedup, hallucination checks) are conservative by default, non-destructive where possible, and every removal is recorded in the JSON output with a reason. Raw audio is never modified. When in doubt: keep and flag.

## Tech stack

- Python 3.11+ with **uv** (`pyproject.toml`, `uv.lock`); pin faster-whisper and ctranslate2 versions.
- ASR: `faster-whisper` – `WhisperModel.transcribe()` (`quality` mode, default) and `BatchedInferencePipeline` (`fast` mode).
- Audio: `sounddevice` (PortAudio), `soundfile` (FLAC, RF64 WAV), `numpy`.
- Config: `pydantic` v2, TOML read via `tomllib`, write via `tomli-w`.
- CLI: `typer`. GUI: `PySide6` (incl. QtMultimedia for playback).
- Dedup similarity: `rapidfuzz`. Benchmarks (dev only): `jiwer`.
- Tooling: `ruff`, `pytest`, `pre-commit`, `mypy` (non-strict at first), GitHub Actions on standard runners (public repo).

Do not add runtime dependencies without stating why in the plan and getting approval.

## Commands

```bash
uv sync
uv run pre-commit install
uv run ruff check . && uv run ruff format .
uv run pytest -m "not gpu and not slow and not audio_hw"   # fast suite = CI
uv run pytest -m gpu                                       # local, needs CUDA
uv run sessionscribe doctor                                # environment check
uv run sessionscribe record-probe --channels 1-8 --minutes 30
uv run sessionscribe bench run --refs bench/refs           # local reference set
uv run sessionscribe --help
uv run sessionscribe-gui
```

## Architecture rules

```
src/sessionscribe/
  core/                # no Qt imports, ever
    config.py          # settings, presets, schema_version + migrations, mode-aware param metadata
    timeline.py        # integer-sample session timeline, offsets, conversions
    audio/             # readers (incl. segmented virtual tracks), recorder, ring buffer
    pipeline/          # stages: features, coupling, ownership, vad, asr, attribution
    merge.py           # sort, dedup (evidence-based), overlap marking
    quality.py         # quality flags
    exporters/         # txt, srt, vtt, json, md
    events.py          # AudioBlock, SegmentEvent(provisional|final), progress/warning/error
    cache.py           # layered cache, atomic writes
    bench/             # reference tooling, metrics, matrix runner
  cli.py
  gui/                 # PySide6; uses core public API + events only
```

1. **`core` never depends on the UI.** CLI and GUI are thin layers.
2. **Chunked processing; never load a whole session into RAM.** Chunks are defined in *track-local* time with an owned range + context. Keep only segments whose midpoint is in the owned range.
3. **Time is integer samples** on the session timeline (`timeline_rate` = 48 000) internally. Float seconds only in public JSON and exporters, rounded once.
4. **Layered cache** (`features`, `coupling/ownership`, `asr`) with separate keys as in SPEC §5.7. Writes are atomic (temp file + `os.replace`). Incomplete results never count as done. Keys include model revision, library versions and `processing_version` – bump `processing_version` when algorithm output changes.
5. **Bleed handling is non-destructive by default** (`post` mode: attribution after ASR). Pre-ASR processing attenuates, never mutes, unless the expert `mute` option is explicitly set.
6. **Dedup needs evidence beyond text similarity** (SPEC §5.5). Uncertain → keep + flag.
7. **Settings know which ASR mode they apply to.** Validation rejects or warns on settings ignored by the chosen mode; device/compute-type validity is checked in `core.config` using `ctranslate2.get_supported_compute_types`.
8. **Recorder:** one input stream for all channels; the callback only copies into a preallocated bounded ring buffer and records status flags – no I/O, no allocation, no blocking. A writer thread writes segmented files and updates the manifest atomically. Any data loss is surfaced to the user and recorded as a gap.
9. **Live-readiness:** stages consume `AudioBlock`s and emit events; `SegmentEvent` has `provisional|final` status. Do not implement a live source yet.
10. Long jobs run off the GUI thread and are cancellable with the cache left consistent.

## Coding conventions

- Type hints on public functions; dataclasses/pydantic models across module boundaries, no loose dicts.
- `logging` with module loggers; no `print` outside the CLI. Logs never contain transcript text unless debug logging of content is explicitly enabled.
- `pathlib` everywhere; think about both Windows and Linux paths.
- GUI strings via `tr()`; English source, Hungarian first translation.
- Pure, small functions for timeline, ownership, dedup and merge logic, tested with synthetic arrays.

## Testing

- Synthetic audio fixtures (sines, noise bursts, silence, simulated bleed with known coupling). No large binaries in the repo.
- Markers: `gpu`, `slow` (real model), `audio_hw` (needs an audio interface). Default/CI run excludes all three.
- Required test cases for bleed/merge: single speaker with bleed, quiet speaker next to a loud one, true overlap, two people saying the same short word, near-duplicate text across tracks, chunk-boundary words, offsets.
- Regression test for which parameters each ASR mode actually honours (pinned library version).
- **Real recordings and reference transcripts live in `bench/refs/` which is git-ignored – they are personal data. Never commit them, never print their content into issues or logs.**

## Workflow

- Work milestone by milestone (SPEC §13). Start each in plan mode with files to touch and acceptance criteria.
- One logical change per commit; conventional commits.
- Run ruff + fast tests before declaring a step done; run `gpu`/`audio_hw` tests locally when the change touches those areas.
- Ambiguity → ask; record the answer in `docs/DECISIONS.md`.
- Don't implement "Later features" (SPEC §14) unless asked, but don't design them out.

## Domain glossary

- **Track**: one participant's audio (possibly made of several recorded segment files).
- **Bleed**: a participant's voice picked up faintly by another participant's mic. Suppress.
- **Crosstalk / overlap**: two participants genuinely speaking at once. Keep and mark.
- **Coupling `C[i][j]`**: typical level (dB) of speaker j's voice on mic i relative to mic j.
- **Ownership**: a track owns a frame when its energy exceeds the bleed predicted from all other tracks by a margin.
- **Quality flag**: marker on a segment worth human review; never a deletion.
- **Gap**: a stretch of lost recording data, known by sample position.

## Known gotchas

- Current ctranslate2 needs CUDA 12 + cuDNN 9. `doctor` must report versions and run a real short inference, not just detect the GPU.
- `BatchedInferencePipeline` does not run the normal temperature fallback and ignores some anti-hallucination options; its `condition_on_previous_text` default also differs from `WhisperModel.transcribe()`. Don't expose ignored options as if they worked.
- Library VAD defaults (min_silence 2000 ms, pad 400 ms, max_speech ∞) are not ideal for conversation; use the "Conversation" preset values from SPEC §3.4.
- Windows ASIO: set `SD_ENABLE_ASIO` before `import sounddevice`; it only works with the pip-installed package and no custom `portaudio.dll` in PATH. WASAPI is fallback and must be tested, not assumed.
- Classic WAV is limited to 4 GB → recordings are segmented FLAC (default) or RF64.
- `.en` / distil English models must not be used with non-English languages.
- Whisper expects 16 kHz mono; resample once per chunk.
- PySide6 is LGPLv3: no static linking, ship licence texts in builds.
