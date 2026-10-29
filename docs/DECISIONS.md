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

- [ ] Re-score `work/bench/20261029-212243` and `work/bench/20261029-214701` with the hand-corrected, verified clip01 reference (`bench score`); record the numbers, then merge PR #2.
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
