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

## 2026-10-29 – M0 skeleton

- **Pinned ASR stack:** `faster-whisper==1.2.1`, `ctranslate2==4.8.4`; everything else has a lower bound and is locked by `uv.lock`.
- **Python:** 3.12 locally (`.python-version`), CI tests 3.11 and 3.12 on Ubuntu and Windows. On 3.11 uv resolves an older numpy (numpy ≥ 2.5 needs 3.12); accepted.
- **Runtime dependencies are added per milestone**: M0 needs faster-whisper, ctranslate2, numpy, sounddevice, soundfile, typer. pydantic, tomli-w, rapidfuzz, PySide6 come in with the milestone that uses them.
- **CUDA libraries:** optional `cuda` extra (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12` 9.x). Before `ctranslate2` is imported, `core.cuda_libs` registers the pip-installed library folders: Windows via `os.add_dll_directory` (and `PATH`, because cuDNN loads its sub-libraries by name); Linux by preloading the `.so` files with `ctypes` (`RTLD_GLOBAL`). `doctor` reports missing libraries on both platforms. Alternative (rely on a system CUDA toolkit) rejected as default: varies per install; still detected as "system" if present.
- **Real inference in `doctor` runs in a subprocess**, because ctranslate2 can terminate the process outright when a cuDNN sub-library is missing; the parent reports the exit code and stderr tail instead of dying.
- **Model download:** faster-whisper's default cache (Hugging Face hub cache) unless `--model-dir` is given; the project-local `models/` folder is git-ignored. If the model is not cached and there is no network, `doctor` says so explicitly.
- **`record-probe` PASS criterion** (owner decision): PASS = no input overflow or other error status flag, no blocks dropped by the ring buffer, all channels have the same frame count, written frames == received frames (checked by reading the files back), and callback block timestamps (`inputBufferAdcTime`) are continuous. "Continuous" (owner-approved, refined): the residual between ADC time and sample count, smoothed with a running median over 200 ms, shows no shift of more than **half a block** (5 ms at 10 ms blocks) against a slowly tracking baseline that **persists ≥ 100 ms**. One lost block shifts the residual by a full block, i.e. twice the threshold, so a single lost block is detected (dedicated tests, also under ±4 ms jitter and 500 ppm drift); single late timestamps and shifts shorter than 100 ms are ignored; the baseline absorbs slow clock drift. Reason: WASAPI timestamps jitter by several ms with 10 ms blocks, a plain per-block check would fail falsely. The analysis runs in the writer thread; the callback only queues (ADC time, frame count) into a preallocated buffer. The report shows jitter statistics (median, p99, max of |residual − smoothed residual|). The wall-clock deviation is reported in ppm for information only; > 200 ppm → WARN, never FAIL, because the interface clock legitimately drifts against the PC clock. If the host API provides no ADC timestamps, continuity cannot be verified → WARN.
- **Sample format:** callback dtype `int32`; files written by soundfile as FLAC `PCM_24` (one file per channel, default `work/probe/<timestamp>/`, `--no-write` only counts). Free disk space is checked before starting (uncompressed 24-bit size + 10 %).
- **Report includes per-channel peak and RMS (dBFS)** so channel order can be verified by tapping each mic.
- **Channel selection:** `--channels` takes 1-based numbers (`1-8`, `1,3,5-6`). On ASIO it is implemented with `AsioSettings(channel_selectors=…)`; on other host APIs the stream opens channels `1..max(selection)` and the callback copies the selected columns into the ring buffer without allocating (`np.take(..., out=, mode="clip")`).
- **WASAPI:** the UMC1820 appears as several stereo-pair / single-channel devices, but also as an 8-channel `IN 01-08` endpoint, which opens 8 channels at 48 kHz (verified on the reference machine). **The M0 WASAPI acceptance criterion is therefore the full criterion** (30 min, 8 ch, PASS as above), the same as ASIO. Independently, `doctor` and `record-probe` must say clearly when a device cannot open N channels (device name, host API, available vs. requested, hint to use ASIO).
- **Device auto-selection (probe/doctor only):** without `--audio-device`, the default input of the host API is used if it has enough channels, otherwise the first device that has. The chosen device, its index, host API and whether it was auto-selected are shown prominently before recording starts and in the report/JSON. The M5 recorder must not auto-select (SPEC §8.1).
- **CUDA extra stays** although the reference machine works without it (system CUDA 12.8 + cuDNN 9.27 on PATH; ctranslate2 wheel ships `cudnn64_9.dll`). M6 task: verify on a clean environment without system CUDA in PATH that the extra alone is sufficient (SPEC §13).
- **CI triggers:** `push` only on `main`; every other branch is tested via `pull_request`; concurrency group per ref with `cancel-in-progress: true`.
- **Known upstream warning:** sounddevice 0.5.6 sets `ndarray.shape`, deprecated in numpy 2.5; filtered in the pytest config by message and module only.
