# Decision log

Short entries: date, decision, reason, alternatives considered.

## 2026-10-29 – initial plan (v0.1)

- **Multi-track transcription instead of diarization.** One lavalier mic and track per participant; transcribe per track and merge. Diarization of mixed audio is a later feature.
- **Local-first, offline.** faster-whisper on CUDA (reference RTX 4070 Ti Super) with CPU fallback.
- **Platforms: Windows and Linux.**
- **Design for 8 tracks × up to 8 hours** → chunked pipeline, resumable cache, FLAC/RF64 recording.
- **Live transcription later, architecture prepared now.**
- **GUI: PySide6**, UI-independent core, typer CLI.
- **Deferred:** diarization, campaign glossary, session summary.

## 2026-10-29 – revision after independent review (v0.2)

Accepted:
- **Guiding principle:** losing real speech is worse than a duplicate; filters keep and flag when uncertain; every removal is recorded with a reason.
- **Two ASR modes** (`quality` = `WhisperModel.transcribe()`, `fast` = `BatchedInferencePipeline`) because the batched pipeline does not honour the same anti-hallucination parameters. Settings are mode-aware; a regression test pins the behaviour.
- **Never mute by default**; attenuation only in `pre` mode; raw audio never modified.
- **Coupling-based ownership** instead of a plain "within X dB of the loudest track" gate (the latter can drop a quiet speaker next to a loud one).
- **Evidence-based dedup**: text similarity alone is never enough; short utterances need strong energy evidence; uncertain → keep + flag.
- **Quality flags** for suspicious segments instead of more deletion; review view with click-to-play.
- **Layered cache** with separate keys, atomic writes, model/library/processing versions in keys.
- **Chunk owned range + context**, integer-sample internal timeline.
- **Recorder:** single stream, callback only fills a preallocated ring buffer, visible data-loss alarms, **segmented FLAC files** with manifest and recovery check, full-duration disk check.
- **ASIO via `SD_ENABLE_ASIO` before import** (confirmed in sounddevice docs); WASAPI tested, not assumed.
- **Early hardware validation:** `record-probe` CLI and extended `doctor` in M0.
- **Measurable QA:** reference clips + metrics (WER/CER, speech recall, speaker accuracy, false suppression, residual duplicates, timestamp error, RTF/VRAM); bleed handling always compared with `off`.
- **CI from M0** (lint + fast tests, public repo, standard runners). Resolves the v0.1 contradiction ("CI optional" vs M6 criterion). GPU/hardware tests stay local.
- **Minimal `initial_prompt` / `hotwords` pass-through moved to M1**; full glossary remains later.
- **TOML via tomllib + tomli-w**, config `schema_version` + migrations; `.en` model/language validation.
- **Privacy and licence notes** (consent responsibility, offline mode, clean command, no transcript text in logs, PySide6 LGPL obligations, model licence check before bundling).
- **Full 8 h recording test** as M5 acceptance, plus fault injection (unplug, low disk, slow disk, kill).

Adjusted / differing from the review:
- **Default bleed mode is `post` (non-destructive attribution after ASR)**, not pre-ASR attenuation. Reason: it fully follows the guiding principle and makes bleed parameter tuning free (no re-transcription). Pre-ASR attenuation remains as `pre` mode ("Strong" profile). Trade-off: more ASR compute and Whisper may transcribe loud bleed – acceptable because attribution removes it from the text outputs. To be verified in M3.
- **Coupling is estimated automatically from the session by default**; the guided 30 s mic check is optional and improves it, rather than being required.
- **Default ASR mode `quality`**, on the expectation that VAD-trimmed tracks are mostly silence so sequential decoding is fast enough. To be confirmed by the M1 benchmark.
- The review's numeric example (Anna −27 / Béla −12) would not actually drop Anna, because Béla's bleed raises Anna's mic level too. The real failure case is a quiet speaker with *low* bleed next to a loud one. Conclusion (plain loudest-track gate is unsafe) stands.

Added beyond the review:
- **Real reference recordings are personal data**: kept in git-ignored `bench/refs/`, never committed; public tests use synthetic audio.
- **Track-local chunking**, so offset changes never invalidate ASR results in `post` mode.
- **Single input stream keeps channels aligned through dropouts**; gaps are recorded by sample position and carried into the transcript.
