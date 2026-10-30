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

## 2026-10-29 – M0 hardware acceptance (reference machine)

30-minute `record-probe`, 8 channels, 48 kHz, int32 → FLAC PCM_24, Windows 10, UMC1820. Both host APIs meet the full criterion; **M0 hardware acceptance passed.**

| | ASIO (`UMC ASIO Driver`) | WASAPI shared (`IN 01-08`) |
|---|---|---|
| Result | PASS | PASS |
| Duration (wall clock) | 1800.0 s | 1799.9 s |
| Frames received = per channel = written | 86 403 072 (all 8) | 86 398 080 (all 8) |
| Blocks / latency | 42 189 / 0.044 s | 179 996 / 0.022 s |
| Overflow / underflow / other flags | 0 / 0 / 0 | 0 / 0 / 0 |
| Ring buffer max fill / dropped | 0.09 s of 10 s / 0 | 0.02 s of 10 s / 0 |
| Max file write | 103.7 ms | 6.1 ms |
| Timestamp discontinuities | 0 | 0 |
| Jitter median / p99 / max | 0.33 / 1.33 / 1.33 ms | 0.51 / 1.70 / 2.78 ms |
| Clock vs wall clock (info) | +1.2 ppm | +1.1 ppm |

Notes:
- The largest single write (103.7 ms, ASIO with ~2048-frame blocks) is ~1 % of the 10 s ring buffer; the buffer size is ample. Revisit headroom for the M5 recorder under slow-disk fault injection.
- Clock deviation is far below the 200 ppm WARN threshold on both APIs.
- Levels were informational (no defined input signal): channels 2–3 at about −90…−96 dBFS RMS look unconnected/at minimum gain; channel 1 peaked at −0.0 dBFS on the ASIO run (clipping on a transient). Neither affects the probe verdict; gain staging is a setup task, and a clipping indicator belongs in the M5 recorder level meters.
- Device was auto-selected on both runs (allowed for probe only, see above).

## 2026-10-29 – M1 plan and implementation

- **Dependencies (owner-approved):** `pydantic` and `tomli-w` at runtime; `jiwer` dev-only (the `bench` commands import it lazily and say so if it is missing). `av` (PyAV) is now declared directly because we import it; it was already installed through faster-whisper.
- **Resampler: PyAV (libswresample), not our own FIR.** Evaluated first as requested. Measured on the reference machine (av 19.0.2):
  - alignment: an impulse at 24 000 (48 kHz) lands at exactly 8 000 (16 kHz) once flushed;
  - block invariance: output is identical for 4 800- and 1 024-frame blocks;
  - 10 kHz alias: −99 dB; 7 kHz passband: −0.4 dB;
  - speed: 600 s of audio resamples in 0.12 s.

  That leaves nothing a custom resampler would add. Consequence: the `asr` cache key (M2) must include the `av` version.
- **Settings model** (`core.config`, pydantic v2, `schema_version = 1`, migration registry):
  - Layers: defaults → preset → config file → `--set section.field=value` (TOML-parsed value) → CLI flags.
  - Built-in presets: `conversation` (the default: quality, large-v3, SPEC §3.4 VAD) and `fast-draft` (fast, large-v3-turbo). User presets live in `<config dir>/presets/*.toml`. Config dir: `%APPDATA%\sessionscribe` on Windows; `$XDG_CONFIG_HOME/sessionscribe` (or `~/.config/sessionscribe`) on Linux.
  - Language default is `auto` with a CLI warning; the bench takes the language from `clip.toml`.
  - `.en` and distil models force `en`; any other language is a validation error.
  - `hallucination_silence_threshold` needs word timestamps.
  - Fast mode needs VAD.
  - Device and compute type are checked at run time (`resolve_runtime`), so a settings file stays loadable on a machine without a GPU.
  - The `bleed`, `dedup` and `output` sections already exist with SPEC defaults; the stages that use them come later.
- **Mode metadata and what faster-whisper 1.2.1 actually does.** Pinned by `tests/test_asr_regression.py` (`slow`, tiny model; it records the real ctranslate2 `generate` calls and the `TranscriptionOptions` that are built) and by `test_mode_table_is_pinned` plus a version pin in the fast suite. In `BatchedInferencePipeline`:
  - only `temperature[0]` is used, with no fallback;
  - `best_of` is not used;
  - `condition_on_previous_text` is forced to `False` and `hallucination_silence_threshold` to `None`;
  - `log_prob`, `no_speech` and `compression_ratio` thresholds never trigger anything;
  - a VAD *dict* gets `max_speech_duration_s` replaced by 30 s, while a `VadOptions` object is used as given;
  - the library's own default VAD has min_silence 160 ms.

  Consequences:
  - Fast mode passes only the parameters it honours.
  - VAD settings always go as a `VadOptions` object.
  - `mode_issues` warns only about values the user set explicitly that the chosen mode ignores; default values are documented per mode instead.
  - In quality mode the fallback and `best_of` demonstrably work.
  - Note for tests: the `no_speech` check deliberately turns a retry into "silence".
- **Fast mode keeps the library's `without_timestamps=True`.** Segments are then VAD chunks of up to 30 s; word timestamps still give word-level times. The exporters may split long segments at word gaps in M2.
- **M1 single-file limit:** a track is loaded whole at 16 kHz (about 230 MB per hour), with a default maximum of 2 h (`--max-duration`). Chunking with owned and context ranges comes in M2, as SPEC §5.1 describes.
- **VAD intervals** (SPEC §3.4) come from our own `get_speech_timestamps` call with the same options. The VAD therefore runs twice per track (ours and the library's), a small cost that keeps both sets of intervals identical.
- **txt timestamps** are truncated to the second (`[00:00:02]` for 2.9 s).
- **Bench code lives in `core/bench/`** (a package instead of a single `bench.py`): `clip`, `reference`, `labels`, `draft`, `diff`, `text`, `metrics`, `runner`, `cell`, `resources`.
- **Reference workflow (owner decisions):**
  - Hand correction is an Audacity label round-trip; the draft starts from the imported existing transcript, plus our own large-v3 draft and a disagreement list.
  - The clip folder layout is `bench/refs/<clip>/`.
  - Label files are UTF-8 without BOM, written by us and tolerated (BOM stripped) on input.
  - Label text is `speaker | id [tags] text`. The speaker sits in the text because Audacity's label export drops track names. A `_` prefix marks a line as not that speaker's speech (kept in `excluded` with a reason), and labels deleted anyway are kept as "deleted while editing", so nothing disappears without a trace.
  - `reference.json` is canonical and generated; `--verified` forbids leftover `?` draft hints.
  - Details: `docs/REFERENCE_GUIDE.md`.
- **clip01:** 7 tracks (ch02 is unused) of 109.5 s, Hungarian. 4 tracks (`seat_2`, `seat_4`, `seat_5`, `seat_6`) contain no reference speech, which suits the silent-track metric. The import gave 64 segments: 60 utterances and 4 bleed candidates. The large-v3 diff shows 38 matching passages and 33 to check.
- **Metrics (M1):**
  - **WER/CER** count only hypothesis words whose midpoint lies within ±0.3 s of the speaker's own reference utterances, so bleed is not counted against recognition (that is M3).
  - **Speech recall:** an utterance counts as found when at least 50 % of its words are hits.
  - **Outside words** (on tracks with speech) and the **silent-track** output (words, segments and seconds on tracks without any reference speech, i.e. hallucination or bleed) are reported separately (owner request).
  - **Timestamp error:** median |start| and |end| error of found utterances.
  - **Averaging:** micro-averaged over tracks and clips.
  - **Without word timestamps,** word times are spread evenly over the segment, and the report flags it.
  - **Normalisation:** NFC, lower case, `[…]` removed, punctuation removed (except `-` and `'` inside words), accents kept.
- **Benchmark runs:**
  - Every matrix cell runs in its own process for clean peak RAM/VRAM figures and to survive OOM.
  - The model load and a warm-up are measured separately and are not part of the RTF.
  - Peak RAM comes from `GetProcessMemoryInfo` or `getrusage`. VRAM comes from `nvidia-smi`: per process where the driver reports it, otherwise device total minus baseline, marked `*` (Windows WDDM).
  - Speech RTF = processing time / VAD speech time; projection = 10 h of speech.
  - Fixed batch sizes are used in the matrix, so an OOM shows as a failed cell instead of a silent fallback.
  - Reports contain numbers only. Transcripts stay in the git-ignored `work/bench/<run>/cells/`.
- **Default-mode criterion (owner):** quality stays the default if, on the default configuration (large-v3, float16, word timestamps), its projected time for 10 h of speech is ≤ 60 min **and** its WER is at most **1 percentage point** worse than the best fast cell; otherwise SPEC §3.2 is reopened. **A decision based on the 109 s clip is provisional:** repeat it with a ≥ 5 min clip (SPEC §11.1) before the M3 tolerances are fixed.

## 2026-10-29 – M1 benchmark, first run (draft reference, provisional)

32-cell `m1` matrix on clip01 (7 tracks × 109.5 s = 766 s of audio; 202.5 s of VAD speech, bleed included), RTX 4070 Ti Super. The reference is still the **unverified import draft** (`draft/import.json`), so WER mostly measures agreement with the old tool's transcript. Rescore with `bench score work/bench/20261029-212243` once `reference.json` is verified; no ASR rerun is needed.

| configuration | WER % | recall % | speech RTF | 10 h speech (min) | VRAM MiB* |
|---|---|---|---|---|---|
| large-v3 float16 quality wts | **21.9** | 86.7 | 0.181 | 109 | 4335 |
| large-v3 float16 fast b4 wts | 25.1 | 85.0 | 0.075 | 45 | 5130 |
| large-v3 float16 fast b8/b16 wts | 25.6 | 83.3 | 0.068 | 40 | 5390–5503 |
| large-v3 int8_float16 quality wts | 26.8 | 78.3 | 0.230 | 138 | 2556 |
| large-v3 int8_float16 fast b8 wts | 40.4 | 61.7 | 0.077 | 46 | 3534 |
| large-v3-turbo float16 quality wts | 36.2 | 71.7 | 0.050 | 30 | 2345 |
| large-v3-turbo float16 fast b4 wts | 38.7 | 68.3 | 0.025 | 15 | 2858 |

\* device total minus baseline (no per-process value under WDDM). The full table is in the run's `results.md`.

Observations:
- **Silent tracks:** every cell produced 0 words on the 4 silent tracks; VAD + large-v3/turbo did not hallucinate there.
- **Compute type:** int8_float16 costs large-v3 about 5 pp in quality mode and about 15 pp in fast mode, and is not faster here. float16 stays the default.
- **Batch size:** 4, 8 and 16 differ by less than 1 pp WER and barely in speed on a clip this short. Too few chunks to tell; re-measure on the longer clip.
- **Turbo:** 2.5–3.5× faster than large-v3, but about 14 pp worse WER on Hungarian.
- **`nowts` cells are penalised by the metric:** without word timestamps, words are spread evenly over segments (in fast mode, VAD chunks of up to 30 s), so many land outside the reference windows. Their WER and recall are not comparable to the `wts` cells. Word timestamps stay on by default anyway.
- **Criterion result: `revisit`.** Quality has the better WER (21.9 % vs 25.1 %, well beyond the 1 pp tolerance), but its projected 109 min for 10 h of speech exceeds the 60 min budget. Per-track overhead weighs more on a 109 s clip, so the projection is likely pessimistic.

**Provisional decision:** the default stays `quality`, unchanged until the verified reference and a ≥ 5 min clip are measured. On these numbers, quality is the only configuration within reach of the best accuracy. Fast b4 (+3.2 pp WER, 45 min) is the candidate for a "faster" preset or for the later hybrid mode (SPEC §14). The SPEC §3.2 rationale (finishing a session in "well under an hour") is not confirmed and is reopened for that re-measurement.

## 2026-10-29 – M1 benchmark: parallel tracks in quality mode (owner request)

Matrix `parallel` (also part of `m1`): large-v3 float16 quality with word timestamps, the same clip01 and the same draft reference. Tracks run either from threads sharing one model with `num_workers = N`, or as N separate processes each loading the model, with tracks split between them longest-first by an activity estimate. "proc s" is the wall time of the processing phase after model load and warm-up; "wall s" is the whole cell. VRAM is the device total minus the baseline (about 1.8 GB baseline on a 16 GB card).

| configuration | proc s | wall s | 10 h speech (min) | VRAM MiB | RAM MiB |
|---|---|---|---|---|---|
| sequential | 35.1 | 39.6 | 104 | 4299 | 3299 |
| 2 threads | 32.3 | 37.1 | 96 | 4720 | 3298 |
| 3 threads | 32.3 | 37.2 | 96 | 5018 | 3299 |
| 2 processes | 33.4 | 41.1 | 99 | 8564 | 5860 |
| 3 processes | 33.3 | 41.2 | 99 | 12881 | 8469 |

- **Little to gain from parallelism, about 8 %.** WER and the transcripts are identical in every variant. The per-track times add up to about 64 s across workers, so concurrent tracks just share a GPU that one quality-mode track already keeps busy: throughput is GPU-bound, not limited by sequential scheduling.
- **Threads cost little memory** (+0.4 GB for 2, +0.7 GB for 3). **Processes cost one full model each** (+4.3 GB VRAM and +2.5 GB RAM per process), take longer to load, and are no faster.
- **3 processes crashed once:** a child exited with 0xC0000005 (access violation) and no stderr output. The repeat run succeeded. The total was about 14.5 GB of 16 GB, so this is treated as a near-limit GPU memory failure that ctranslate2 does not report cleanly as OOM.
- **Conclusion:** parallel tracks do not bring quality mode under the 60 min budget (best: 96 min for 10 h of speech). If anything is used, it is 2 threads (`num_workers = 2`), cheap and slightly faster. Multi-process parallelism is rejected for this GPU class. Re-measure with the ≥ 5 min clip, where several speakers being active at the same time may change the picture.
- `recommend_mode` now takes the quality projection from the fastest quality variant of the default configuration, parallel ones included; WER still comes from the sequential cell.

## Open tasks (M1)

- [x] ~~Re-score with the verified clip01 reference, then merge PR #2.~~ Superseded on 2026-10-30, see "M1 closed on the unverified reference" below.
- [ ] Repeat the M1 benchmark (incl. `parallel`) with a ≥ 5 min clip (SPEC §11.1) and settle the default ASR mode; required before the M3 tolerances are fixed.

## 2026-10-29 – Default ASR mode decided by WER (owner decision)

- **`quality` stays the default mode.** This supersedes the M1 default-mode criterion above. Processing that takes several hours is acceptable to the owner, so the 60 min budget is **no longer a criterion**. WER alone decides: quality stays the default unless the best fast cell on the default configuration (large-v3, float16, word timestamps) beats its WER by more than 1 percentage point. `recommend_mode` implements this; projected processing time is still reported, for information only. On the first run (draft reference), quality has 21.9 % WER vs 25.1 % for fast b4, so the decision is `quality`.
- **Faster optional preset `fast`:** large-v3, fast mode, batch size 4. In the M1 run it was about 2.4× faster (45 vs 104–109 min for 10 h of speech) at +3.2 pp WER. It replaces `fast-draft` (large-v3-turbo), which was about 14 pp worse on Hungarian. A fixed batch size of 4 means no OOM probing; b4 needs about 5 GB VRAM.
- SPEC §3.2 rationale updated accordingly. The re-measurement on the verified reference and on a ≥ 5 min clip (open tasks above) still applies: it confirms the WER comparison; the speed figures there are informational only.

## 2026-10-29 – M2 task: survive crashing worker processes (owner request)

Prompted by the intermittent 0xC0000005 crash of a child process near the VRAM limit (parallel measurement above). Requirement for M2, also in SPEC §5.7 and §13:

- The pipeline runs ASR work in child processes. It detects an unexpected child exit: a non-zero exit code, a Windows NTSTATUS code such as 0xC0000005, a signal on Linux, or a missing result. Such an exit is a failure of that chunk, not of the session.
- **The interrupted chunk is re-run once from the cache state.** Completed chunks are kept: they were written atomically and their cache keys are valid. Partial or temporary files of the failed chunk are discarded, never used.
- If the same chunk fails again, processing stops with a clear error that names the track, the chunk range (track-local time), the exit code (decoded where known, e.g. "access violation"), the likely cause (e.g. GPU memory near its limit) and a hint (smaller model or compute type, fewer parallel workers, quality mode). The cache stays consistent, so a later run resumes after the completed chunks.
- Every crash and retry is recorded in the run log and in the output JSON (`processing_events`), with no transcript text.
- Tests: a fake worker that crashes once (the chunk is retried and the run succeeds) and one that crashes twice (clear error, cache consistent, the resumed run does not redo completed chunks).

## 2026-10-29 – OOM fallback also for a fixed batch size (owner request)

- `asr.batch_size` is the **starting** batch size (default 16). On GPU OOM, fast mode halves it down to 1 and then fails with a clear error. This also applies to the `fast` preset (4 → 2 → 1), so it works on smaller GPUs. Non-powers of two halve with rounding down (6 → 3 → 1).
- New setting `asr.oom_fallback` (default `true`, fast mode only) switches the fallback off. Benchmark cells set it to `false`, so a fixed batch size that does not fit shows up as a failed cell instead of quietly measuring a smaller batch.

## 2026-10-30 – M1 closed on the unverified reference; triage reference before M3 (owner decision)

- **The verbatim correction of clip01 is not done; PR #2 is merged with `reference.json` absent and the import draft as the reference.** All M1 WER and recall figures above were measured against the **old tool's transcript** (`draft/import.json`, status `draft`). They are **informational only**: they show how far each configuration is from that transcript, not its true error rate.
- **The mode choice does not depend on them.** It rests on the ranking: on the same reference, quality beats fast b4 by 3.2 pp (21.9 % vs 25.1 %), beyond the 1 pp tolerance, and it is the best cell overall. The default stays `quality`, and the `fast` preset stays optional. Re-checking the ranking on the ≥ 5 min clip (open task below) is still planned.
- **Verbatim reference transcripts move to SPEC §14 (later).** WER against a hand-corrected text is not needed until we tune recognition itself.

### Triage reference (M3 prerequisite)

M3 measures what bleed handling removes or keeps, not how words are recognised. For that, every transcribed segment only needs a judgement, not a corrected text. Planned workflow, `sessionscribe bench ref triage CLIP`:

- **Units:** our own per-track segments with bleed handling off (`ref draft`, i.e. exactly what M3 filters). Each unit has an id, a track, start and end. No text is edited.
- **Labels, one key each:**
  - `s` real speech by the track's owner;
  - `b` bleed: another participant's voice that their own track also has (a duplicate), with an optional source digit `1`–`8`;
  - `w` wrong speaker: real speech, the only copy, but on the wrong track; a digit gives the true speaker (suppressing it would lose speech);
  - `o` own speech plus audible bleed in the same segment;
  - `n` no speech (noise, hallucination);
  - `?` unsure, excluded from scoring.
- **Playback in the terminal** via `sounddevice` (already a dependency, so no new packages): the unit plays automatically from the owner's track, with 0.3 s of context. Further keys:
  - `space` replays it;
  - `m` plays all tracks mixed;
  - `x` plays the loudest other track at the same time;
  - `u` undoes the last label;
  - `q` saves and quits.
- **Suggestions to accept with Enter:** a level-ratio heuristic against the other tracks, plus our draft's text similarity, propose `s` or `b`. Most units should then need only Enter. The ASR text is shown in the terminal as an aid and is never written to logs.
- **Storage:** after every key, the labels are written atomically to `bench/refs/<clip>/triage.json` (git-ignored, schema-versioned, `status: draft|complete`). A later run resumes at the first unlabelled unit. If the draft is regenerated, labels carry over by time overlap and speaker; units that no longer match are listed.
- **Table fallback:** `ref triage --export triage.csv` / `--import triage.csv` (id, track, start, end, suggestion, label), with the Audacity label files from `ref labels` for listening. This is for anyone who prefers a spreadsheet.
- **Metrics from a triage reference** (SPEC §11.2), each compared with `bleed_mode=off`:
  - speech recall and false suppression over `s`, `w` and `o`;
  - residual duplicate rate over `b`;
  - speaker attribution over `w` and `b` with a source;
  - silent-track output over `n`.
  - WER and CER are reported only where a verbatim reference exists.
- **Effort estimate:** about 2–4 s per unit, so 5 min for clip01 (about 60–70 units) and 15–20 min for a 5 min clip.

Built at the start of M3, before any M3 measurement. M3 acceptance needs complete triage references for clip01 and the ≥ 5 min clip(s).

## Open tasks (after M1)

- [x] ~~Re-score M1 with a verified verbatim reference before merging PR #2.~~ Dropped: verbatim reference moved to SPEC §14, and PR #2 was merged on the unverified reference (above).
- [ ] Repeat the M1 benchmark (incl. `parallel`) with a ≥ 5 min clip (SPEC §11.1) and confirm the quality-vs-fast WER ranking. Required before the M3 tolerances are fixed.
- [ ] Triage reference (`bench ref triage`) and complete triage of clip01 and the ≥ 5 min clip(s): M3 prerequisite.

## 2026-10-30 – M2 implementation decisions

- **Session file:** `session.toml` holds participants (id, name, source file or segment folder, channel, offset, colour), an optional preset and an optional `[settings]` table.
  - The settings layers are: preset → `--config` → session `[settings]` → `--set` → CLI flags. An explicit `--preset` beats the session's preset, which beats the config file's.
  - `session init FOLDER` suggests participants from `chNN_name` file names or segment subfolders. Ids are ASCII (`Ádám` → `Adam`), and display names keep the original spelling.
- **`offset_s ≥ 0`:** offsets can always be chosen so the earliest track starts at 0, and this keeps the session timeline non-negative.
- **Fingerprint without mtime (owner request):** size plus SHA-256 of the first and last MB of each file a range touches, plus the file's position in the virtual track.
  - Copying or moving a session folder keeps its cache valid.
  - Risk: an edit in the middle of a file that keeps its size goes unnoticed. `--full-hash` hashes whole files. Raw audio is never modified by the app, so this risk is accepted.
- **Features in fixed blocks:** energy envelopes (20 ms dBFS) are cached in nominal 10 min blocks of the file, independent of chunk boundaries, so boundary placement does not invalidate features. This avoids a circular dependency.
- **Chunk plan:**
  - The owned range is 600 s and the context 5 s.
  - Each nominal boundary moves to the quietest 0.5 s-smoothed point within ±15 s (at most a quarter of the owned range); on ties the earliest point wins.
  - A last chunk shorter than a quarter of the owned range is merged into the previous one.
  - A track shorter than the owned range is one chunk, and its output is identical to M1 single-file output (verified on clip01: all 7 tracks segment-for-segment identical).
- **Boundary rule:**
  - Segments are kept by midpoint.
  - A kept segment's words past the owned range are dropped only if the neighbouring chunk's own (inside) words cover at least 50 % of each such word.
  - A segment whose every word is covered goes to `suppressed` with `suppression_reason = "chunk_boundary_duplicate"`, so nothing disappears without a trace.
  - `boundary_trimmed_words` counts what was trimmed.
- **ASR cache key:**
  - track fingerprint and sample rate, context range;
  - model id, revision, device, compute type;
  - mode and the batch size (fast mode only);
  - the decoding parameters the mode honours, VAD settings, library versions and `processing_version`.
  - Participant id, name and offset, as well as the output, quality and bleed settings, are **not** in the key, so renaming, offset changes and re-export never re-run ASR (tested with 8 synthetic tracks).
- **Stable segment ids:** `{participant}-c{chunk:03d}-{raw ASR index:04d}`. The index is the position in the chunk's raw ASR output, so filtering never shifts ids.
- **One long-lived worker process** (spawn) holds the model and reads chunk audio itself. Parallel tracks are not used (M1: about 8 % gain).
  - The crash handling follows the M2 task above. Exit codes are decoded, including signed and unsigned NTSTATUS and POSIX signals.
  - `InlineRunner` exists for tests and debugging.
- **`audio_start_sample` is track-local** (48 kHz timeline samples from the start of that participant's source), so a passage can be found in the original file whatever the offset. `start`/`end` are session time.
- **`export` and the work folder:**
  - `transcribe` stores its resolved settings in `<work>/last_settings.json`.
  - `export` takes the ASR sections (model, asr, decoding, vocabulary, vad) from there and resolves the rest afresh, so a language given only on the transcribe command line still finds the cache.
  - The work folder defaults to `<session folder>/work/`. It contains transcript text, and `clean` deletes it.
- **M2 bleed handling:** none yet. The JSON says `bleed_mode_applied: "off"`, and the CLI says so once per run.
- **Quality flags calibrated on clip01:**
  - `no_speech_prob` and `avg_logprob` are per 30 s decoding window, shared by every segment in it. A no-speech flag alone marked a third of all segments, so it now also needs `avg_logprob < -0.5`.
  - `too_long_for_duration` uses 35 chars/s, counting at least 1 s: Hungarian words with tight word timestamps reached 25–30 chars/s.
  - Result on clip01: 0 of 69 segments flagged.
- **`exporters/single.py` stays** for the bench's single-track tools (`ref draft`, matrix cells). `transcribe FILE` now goes through the session pipeline as a one-participant session.

## 2026-10-30 – M2 measurement: chunk boundaries on clip01 (needs a decision)

`bench run --matrix chunking` measured large-v3 float16 quality on clip01 against the draft reference (WER informational only):

| owned range | WER % | recall % | VAD speech without any word (holes ≥ 1.5 s) |
|---|---|---|---|
| whole track (= default 600 s on this clip) | 21.9 | 86.7 | 25.5 s of 202.5 s, 10 holes |
| 60 s | 26.0 | 83.3 | 29.4 s, 12 holes |
| 30 s | 31.9 | 81.7 | 39.5 s, 11 holes |

- **Cause:** not the boundary logic; the kept, trimmed and suppressed words are correct. Whisper in quality mode sometimes skips speech after a window start. VAD marks the stretch as speech, but decoding starts 3–14 s later.
  - Short chunks start many more windows, so they lose more.
  - Longer context does not fix it reliably: 5/10/15/30 s of context gave 0.1/4.0/8.2/0.4 word-seconds in the 15 s after one boundary, against 6.9 unchunked.
- **The same dropout exists without chunking:** 12 % of VAD speech on clip01 has no word at all. Some of it may be bleed or laughter, and the triage reference will tell.
- **At the default 600 s owned range** boundaries are 20× sparser than in the 30 s cell. The extra loss is estimated at about 2 s per boundary, under 1 % of speech.
- **Proposed remedy (decided and implemented; see the next entry):** after a chunk's ASR, re-transcribe every VAD speech stretch of at least 1.5 s that got no words, as its own short window with a little padding.
  - Insert the result flagged `refilled`, and cache it with the chunk.
  - This addresses dropout everywhere, not only at boundaries. The cost is extra ASR on the uncovered share (about 12 % here).

## 2026-10-30 – Re-transcribe word-less VAD speech (owner decision)

The owner chose the remedy proposed above, with two additions. Implemented in M2:

- After each chunk's ASR, every VAD speech stretch of at least `asr.refill_min_s` (1.5 s) that overlaps the owned range and has no recognised word is transcribed again as its own window.
  - The window gets `asr.refill_pad_s` (0.5 s) of extra audio on each side, but only words whose midpoint lies inside the stretch are kept, because the padding was recognised already.
  - Recovered segments get raw indices after the chunk's own segments, so ids stay stable. They are cached with the chunk; `processing_version` is now 2.
- **Owner addition 1:** recovered segments carry the `refilled` flag (always shown, even with quality flags off) **and go through every other quality flag**. The model hallucinates more easily on silence or laughter.
- **Owner addition 2:** a stretch that still has no words after re-transcription is reported as `possible_missed_speech`, for review. It appears in the JSON `possible_missed_speech` list and as an md line ("⚠ possible missed speech, N s"); the CLI prints the count.
- `asr.refill_holes = false` switches the step off. It needs VAD.

Measured on clip01 (large-v3 float16 quality; WER against the draft reference, informational only). "Word-less" counts VAD stretches of at least 1.5 s with no recognised word.

| owned range | refill | WER % | recall % | word-less VAD speech | RTF |
|---|---|---|---|---|---|
| whole track | off | 21.9 | 86.7 | 25.5 s (10) | 0.047 |
| whole track | **on** | **20.7** | **88.3** | 14.9 s (7) | 0.053 |
| 60 s | off | 26.0 | 83.3 | 29.4 s (12) | 0.052 |
| 60 s | on | 22.1 | 86.7 | 15.2 s (8) | 0.060 |
| 30 s | off | 31.9 | 81.7 | 39.5 s (11) | 0.062 |
| 30 s | on | 29.9 | 88.3 | 23.7 s (6) | 0.069 |

- **Refill improves WER and recall at every chunk length**, for about 13 % more processing time. Even 30 s chunks reach the unchunked recall. A first version that kept the words in the padding duplicated neighbouring words (WER 24.8 % on whole tracks); it was fixed before these numbers were taken.
- **Session run on clip01:**
  - 74 segments, 5 of them `refilled`;
  - 2 `high_no_speech` flags, both on refilled segments;
  - 7 `possible_missed_speech` passages totalling 14.9 s, all on the two speaking tracks (`gm`, `seat_3`), likely bleed or laughter. The triage reference will tell.
- Chunk-length effects that remain at 30 s mostly come from re-windowing, and are much smaller at the default 600 s. Repeat on the ≥ 5 min clip (open task).

## 2026-10-30 – Open tasks after M2 (owner notes)

- [ ] `export`: warn when the ASR settings resolved now (preset, config, session, `--set`) differ from the last `transcribe` run's (`last_settings.json`), instead of silently using the stored ones. Do it the next time the export path is touched.
- [x] Short-chunk WER loss recorded in SPEC §14 as a known risk for live transcription.

## 2026-10-30 – M3 plan; triage tool implemented (phase A)

M3 order (owner decision): the triage tool first, in its own PR, so labelling can start early. Then the ≥ 5 min clip (M1 re-run, triage). Then coupling, ownership, post/pre, dedup and overlap, tuned on the triage data. `rapidfuzz>=3.9` is now a runtime dependency (approved), for dedup and triage suggestions.

- **Labels (owner addition):** the earlier draft's `o` = "own speech plus bleed" is split in two.
  - `v` (mixed): own speech and bleed in one segment.
  - `o`: true overlap.
  - Final set: `s` own, `b` bleed (the source track has it too), `w` wrong speaker (only copy, reassign), `v` mixed, `o` overlap, `n` no speech, `?` unsure.
  - Recall and false suppression use `s/w/v/o`; residual duplicates use `b` only; `?` is not scored.
  - The help text and REFERENCE_GUIDE explain `b` vs `w` with an example (owner addition).
- **Units:**
  - the segments and possible-missed-speech stretches of the clip's own session output, produced with the default settings, `bleed.mode = off` and the clip's language;
  - keyed by the stable segment id plus speaker and time;
  - `triage.json` holds no text; the ASR text is shown on the terminal only, read from the cache at start.
- **Suggestions (Enter to accept):**
  - **Relative levels:** each track's level is measured relative to its own typical speech level (95th percentile of frames above −60 dBFS). Raw levels were useless on clip01: mic gains differ by 15–20 dB, so on most GM units another track was "louder".
  - **Rules:**
    - `b` needs the louder track (relative, ≥ 6 dB) to have a segment there itself, or ≥ 80 % similar text on a louder track;
    - `s` when the own track is ≥ 6 dB louder, even if another track echoes the text, because the louder copy is the original;
    - `o` when both tracks are within 10 dB of their speech level and the texts differ;
    - `n` for quiet missed-speech stretches.
  - **Contradictions:** overlapping units that would name each other as the bleed source get no suggestion.
  - **Result on clip01:** 16 of 81 units get a suggestion. Level cues are weak before coupling is estimated (M3); the rest is decided by ear. `--stats` will show the accepted share and seconds per unit after the first real labelling.
- **Saving:** `triage.json` is saved atomically after every label; a run resumes at the first unlabelled unit. `--rebuild` carries labels over by id, else by the same speaker and ≥ 50 % mutual overlap, and lists the rest.
- **Matching for scoring:** id first (post mode keeps the ASR); otherwise the same speaker and ≥ 50 % mutual time overlap (pre mode re-runs ASR). A unit is kept, suppressed or absent.
- **Terminal:**
  - single keys via `msvcrt` (Windows) or `termios` (Linux); outside a TTY there is a clear error pointing to `--export/--import`;
  - playback via `sounddevice`, which disables itself with one warning if audio fails.

## 2026-10-30 – M3 phase C: bleed handling (provisional until clip02)

Owner decision: developed and tested on synthetic data and measured on the clip01 triage reference without waiting for clip02. **Every default and tolerance below is provisional**; they are re-measured and fixed when clip02 (≥ 5 min, triage-labelled) arrives.

### clip01 triage labels

81 units: 29 `s`, 34 `b`, 3 `w`, 7 `v`, 7 `n`, 1 `?`, 0 `o`.

- Suggestion acceptance was 0 % at 15.8 s mean (5.1 s median) per unit.
- 13 of the 34 `b` units name a source with no words on its own track: `seat_4` 8, `seat_2` 5, `seat_1` 5. Those mics are silent or very low gain.
- There are no overlap labels, so the overlap criterion is only checked synthetically here.

### Algorithm, as measured

- **Coupling `C[i][j]`:** the lowest significant peak (≥ 1 s of frames) of `E_i − E_j` on the frames where mic j is ≥ 15 dB above its own noise. On those frames the difference clusters at `C[i][j]` (j speaks), `−C[j][i]` (i speaks) and third-party values in between.
  - Two earlier designs failed on clip01:
    - dominance of relative levels: every pair fell back to the default, because a silent track's "speech level" is its bleed;
    - a low percentile: about 5 dB too low, so masks owned bleed.
  - Pair sums are clamped to `C[i][j] + C[j][i] ≤ −20 dB` (two acoustic paths; the gains cancel). Clamped or default pairs are reported as weak; clip01 has 30 of 42, mostly silent mics.
  - Guided coupling comes from the session's `[mic_check]`, manual coupling from `coupling_db`.
- **Ownership:**
  - `E_i ≥ floor_i` and `E_i ≥ max_j(E_j + C[i][j]) + margin`, held for `hold_ms`.
  - `floor_i` is the track's noise level (10th percentile of non-silent frames) + 10 dB, **per track** (owner request, because mic gains differ by 15–20 dB). The absolute `bleed.floor_db` is gone (settings schema 2, migration drops it).
  - A cap at speech level − 20 dB was tried and removed: a silent, noisy mic then owned 99 % of frames.
  - Cached in an `ownership` layer, keyed by track fingerprints, offsets, coupling inputs and bleed parameters.
- **Post mode: a segment is bleed only with four pieces of evidence**, otherwise it is kept:
  1. ownership < `suppress_below` (0.5), or < 0.2 for utterances under `min_words`;
  2. another track owns ≥ 50 % of its frames;
  3. that track's kept words cover ≥ 50 % of it;
  4. its text matches those words (partial ratio ≥ `bleed.text_similarity`, 70).
  - **When evidence is missing:**
    - without 3, the segment is the only copy: `possible_wrong_speaker` plus `probable_speaker`;
    - without 4 (e.g. a short own interjection while the other speaks) it is kept;
    - a segment with ≥ 0.5 s of owned words (own speech + bleed, triage `v`) is never suppressed;
    - ownership < 0.6 is flagged `low_ownership`.
  - Text evidence was added after clip01: ownership alone dropped a 0.4 s own utterance.
- **Dedup:**
  - It removes a cross-track text duplicate (ratio ≥ 85, ≤ 1 s apart) only with ownership evidence: < 0.5 vs ≥ 0.5, or < 0.2 vs > 0.8 for short ones.
  - It never removes a mixed segment.
  - If both copies own their frames, a short word means two people said it and both are kept; a longer sentence means a third person heard on two mics (clip01's `seat_4` case), and both are kept and flagged `possible_duplicate`.
- **Overlap:** different speakers ≥ 0.3 s together, both owning the shared frames, and different words (ratio < 60). With ownership alone, 51 of 70 clip01 segments were marked; now 23. This is unverified until clip02 has `o` labels.
- **Pre mode:** non-owned frames are attenuated by `pre_attenuation_db` before ASR (10 ms ramps, falling back over `release_ms`; silence only with `mute`). The ASR key includes the ownership key. Presets: `conservative` = post (the default), `strong` = pre, 15 dB.
- **Triage suggestions now come from ownership:** `s` ≥ 0.8; `b` (or `w` without source words) < 0.2 with another owner. On clip01, 15 units got a suggestion and 12 of them matched the label, against 5 of 16 for the level rules.
- **Cache:** adding the `pre` field to the ASR key invalidated earlier ASR caches once.

### clip01 matrix (`bench bleed`)

Speech = s/w/v/o, 39 units; bleed = b, 34 units.

| variant | recall % | false suppr. % | residual dup. % | of which flagged % |
|---|---|---|---|---|
| off | 100.0 | 0.0 | 100.0 | 0.0 |
| post m3 s0.5 | 100.0 | 0.0 | 97.1 | 32.4 |
| post m6 s0.3 | 100.0 | 0.0 | 91.2 | 38.2 |
| **post m6 s0.5 (default)** | **100.0** | **0.0** | **88.2** | **35.3** |
| post m9 s0.5 | 94.9 | 5.1 | 79.4 | – |
| pre a12 | 94.9 | 2.6 | 97.1 | 35.3 |
| pre a18 | 97.4 | 0.0 | 94.1 | 26.5 |

### Provisional tolerances (vs `off`)

- speech recall drop ≤ 2 pp;
- false suppression ≤ 3 %;
- no `w` unit suppressed;
- residual duplicates at least halved.

On clip01 one speech unit is 2.6 pp, so the default may lose none. **The default meets the first three, but not the fourth: residual duplicates drop only from 100 % to 88 %.** About a third of the residue is flagged for review.

The main reasons:
- 13 of 34 bleed units come from participants whose own track has no words, so they are protected as possible only copies;
- the GM and `seat_3` mics are strongly coupled both ways.

More aggressive settings (margin 9) cut more but lose speech. Following "losing speech is worse than a duplicate", the provisional default stays post m6 s0.5.

### Open tasks (M3)

- [ ] clip02: triage, `bench bleed`, fix tolerances and defaults (including overlap, which clip01 cannot test).
- [x] ~~Bleed from participants whose own mic records nothing.~~ Decided and implemented, see "Silent mics" below.
- [ ] The weak-pair warning is noisy for sessions with silent mics: count pairs whose voice actually speaks.

## 2026-10-30 – Silent mics (owner decisions after M3 phase C)

1. **The residual-duplicate metric is split by the labelled source mic:** working vs silent or near-silent. Tolerance 4 (residual duplicates at least halved) is measured on **working source mics only**.
   - A mic is *silent* when its speech level (95th percentile of active frames) is within `bleed.silent_mic_range_db` (10 dB) of its noise level.
   - On clip01 the silent mics are `seat_2` and `seat_4`. `seat_1` (speech −50 dBFS, noise −64) counts as working: quiet, but its wearer's own speech is on it.
   - Silent mics are reported as a warning and in the JSON (`silent_mics`).
2. **Speech of a silent mic's wearer (owner decision):** if the same text is on two other tracks, one copy stays, on the louder track, flagged `possible_wrong_speaker` with `probable_speaker = "unknown"`; the other becomes its duplicate.
   - Loudness cannot identify the wearer, so there is **no automatic reassignment** to the silent mic's owner. The M4 GUI makes it reassignable by hand.
   - As implemented, the rule applies when all of these hold:
     - the session has a silent mic;
     - the text is longer (≥ `min_words`), and the two copies are on two non-silent tracks;
     - the ownership evidence rules leave the pair unresolved;
     - **both copies are at least 20 dB below their tracks' speech levels**, i.e. neither wearer is speaking normally.
   - "Louder" is the level relative to the track's own speech level, which is independent of gain.
   - The −20 dB limit comes from clip01:
     - with −6 dB, the rule removed a quiet own utterance (−15 dB relative);
     - own speech (`s`) lies between −18.3 and +5 dB;
     - bleed copies of silent sources lie between −27 and −16 dB;
     - provisional until clip02.
3. **M5 mic check:** it raises an alarm when an assigned channel has no signal, or a suspiciously quiet one, compared with its noise and with the other channels (SPEC §8.1, §13 M5).

### clip01 after these changes

`bench bleed`; recall and false suppression over 39 speech units; bleed: 21 units from working sources, 13 from silent ones.

| variant | recall % | false suppr. % | residual, working source % | residual, silent source % |
|---|---|---|---|---|
| off | 100.0 | 0.0 | 100.0 | 100.0 |
| post m6 s0.3 | 100.0 | 0.0 | 81.0 | 92.3 |
| **post m6 s0.5 (default)** | **100.0** | **0.0** | **76.2** | **92.3** |
| post m9 s0.5 | 94.9 | 5.1 | 61.9 | 92.3 |
| pre a18 | 97.4 | 0.0 | 95.2 | 92.3 |

- **The default still does not meet tolerance 4 on working sources** (76 % instead of ≤ 50 %).
- The remaining residue is mostly the strongly coupled GM ↔ `seat_3` pair; margin 9 dB reaches 62 % but loses two speech units.
- The silent-source residue barely moves: only one pair satisfies the conservative third-party rule.
- Re-measure on clip02.
## 2026-10-30 – M5 recorder (CLI first)

- **CLI first** (`sessionscribe record …`), so the next session can already be recorded with it. The M4 GUI Record tab will use the same `core.audio.recorder` API and its events (`LevelEvent`, `RecorderAlarm`, `RecorderStatus`).
- **Device selection:**
  - never automatic;
  - the device is given by name and host API: exactly, or by a unique part of the name, otherwise an error;
  - **`rig.toml` (owner request)** saves device, host API, channel → participant id/name, segment length, format and the idle-warning time; explicit flags override it;
  - the device index is only looked up, because it can change between restarts or after an unplug.
- **Files and manifest:**
  - each channel is written to `rec/chNN_<participant>/NNNN.flac` (or RF64 `.wav`), so the M2 `VirtualTrack` reads it directly;
  - the manifest is `[recording]` in `session.toml`: device, stream settings, segments (start sample, frames, SHA-256, status), gaps, alarms and metrics;
  - it is rewritten atomically after every closed segment, gap, alarm and state change.
- **Segments are flushed every second**, so a crash or kill loses at most about one second of the open segment.
- **Lost audio is padded with silence and recorded as a gap; the timeline never shifts.** Real audio is never modified; padding only fills time that has no audio.

  | gap reason | length | how it is detected |
  |---|---|---|
  | `buffer_overflow` | exact | ring-buffer drops queued by the callback with their position |
  | `input_overflow` | estimated | a jump in ADC timestamps, `TimestampMonitor`; no jump: no padding |
  | `device_lost` | estimated | wall time until the reconnect |
  | `process_killed` | unknown | resuming an interrupted or stopped session; the gap is only marked |

- **Recording gaps reach the pipeline.** `Session.gaps()` feeds `merge_tracks`, so segments get the `overlaps_gap` flag and the md output lists the gaps.
- **Watchdog and reconnect:**
  - no callback for 1 s, or an inactive stream, counts as a lost device: the open segment is closed and an alarm is raised;
  - with `reconnect` (the default), the recorder tries every 2 s. **Before each try PortAudio is re-initialised** (`sd._terminate()` + `sd._initialize()`, owner request), because PortAudio reads its device list only once; the device is then found again by name and host API;
  - whether this works under ASIO with the UMC1820 has to be shown by the `audio_hw` test and an unplug test on the reference machine. If it does not, the alarm should advise restarting the recorder.
- **Disk:**
  - at start, the planned length (8 h) as uncompressed 24-bit + 10 % must fit;
  - while recording, a warning below 30 min of remaining time and a clean stop below 2 min.
- **Idle channel (owner request):** an assigned channel that stays within 10 dB of its noise floor for `idle_warn_min` (10) minutes while another channel is active gets one warning, which is repeated only after the channel has been active again.
  - The noise floor is the quietest 1 s level seen so far, since conversation always has pauses.
  - Nothing is reported in the first 2 minutes.
- **No sleep while recording (owner request):** `SetThreadExecutionState` on Windows (in the recording thread), `systemd-inhibit` on Linux. If neither is available, the CLI warns.
- **Mic check (owner request: alarms):**
  - silence for the noise level, then each participant speaks in turn;
  - all comparisons use a channel's **rise above its own noise**, not absolute levels, because gains differ by 15–20 dB (owner request).
  - **Alarms** (provisional thresholds):

    | alarm | when | severity |
    |---|---|---|
    | `no_signal` | rise < 10 dB during its own turn, or digital silence | FAIL |
    | `wrong_channel` | another channel rises more during this participant's turn | FAIL |
    | `quiet` | own speech < −50 dBFS, or its rise > 20 dB below the others' median rise | WARN |
    | `clipping` | peak ≥ −1 dBFS | WARN |

  - `record start` runs the check **inside the recording**, so its turns are saved as `[mic_check]` ranges in recording time, ready for guided coupling. A FAIL stops the recording unless `--force`.
  - `record check` runs the same check without writing anything.
- **Recovery:** `record recover` copies the decodable frames of `open` segments into a clean file of the same name and keeps the original as `.partial`. It cuts the channels of a segment to the same length and marks the run `interrupted`.
  - `record verify` checks every manifest entry against the disk.
  - Tested by killing a recorder child process.
- **Stopping:** Ctrl+C or `--seconds`. There is no separate `q` key, because Ctrl+C already stops cleanly.

- **ASIO only opens from the main thread** (found in the owner's first test): the UMC ASIO driver is a COM object, and `Pa_OpenStream` from a worker thread without COM fails with "Failed to load ASIO driver". The record CLI therefore runs the stream in the main thread (`Recorder.run`, `Meter.run`); mic check prompts and analysis run in a helper thread. Prompts stop on Ctrl+C or when no audio arrives for 3 s; an open error is reported at once (with a hint about another program holding the driver), and no session file is written. The GUI (M4) must do the same: audio on the thread that initialised PortAudio, or COM initialised on its own.

### Acceptance still to do (reference machine, owner)

- [ ] `pytest -m audio_hw` with `SESSIONSCRIBE_TEST_DEVICE`, `SESSIONSCRIBE_TEST_HOST_API` and `SESSIONSCRIBE_TEST_CHANNELS` set, on ASIO and WASAPI.
- [ ] Unplug test: pull the UMC1820 during `record start`, plug it back in, and check that recording continues, with a `device_lost` gap.
- [ ] `record check` with one channel turned down and one muted: the alarms fire.
- [ ] An 8 h, 8-channel `record start` followed by `record verify`, without loss.
