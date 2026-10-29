# SessionScribe – Specification

Status: draft v0.2 (2026-10-29) – revised after independent review. Owner decides; reviewers comment. Changes go into `DECISIONS.md`.

## 0. Guiding principle

**Losing a sentence that was actually spoken is worse than showing a duplicate.** Every filtering step (VAD, bleed handling, dedup, hallucination checks) must be conservative by default, non-destructive where possible, and leave a trace in the JSON output of what it removed and why. Raw audio is never modified.

## 1. Goal

Turn a multi-track recording of a conversation (primarily a tabletop RPG session) into an accurate, speaker-labelled, timestamped transcript, running locally on consumer hardware, with a pleasant GUI and a scriptable CLI. Every questionable result must be traceable back to the original audio.

### 1.1 Constraints and sizing

| Item | Value |
|---|---|
| Participants / tracks | up to 8 (design limit; 7 typical max) |
| Session length | typically 5 h, up to 6.5 h; design and test for 8 h |
| Recording format | 48 kHz, 24-bit, mono per track |
| Reference hardware | Ryzen 9 5900X, RTX 4070 Ti Super 16 GB, Behringer UMC1820 |
| Platforms | Windows 10/11, Linux |
| Languages | any Whisper language; Hungarian and English are first-class |

Storage: 8 ch × 48 kHz × 24 bit ≈ 4.15 GB/h uncompressed (≈ 27 GB for 6.5 h, ≈ 33 GB for 8 h); FLAC roughly halves this.

### 1.2 Non-goals (for v1)

- Speaker diarization of a single mixed recording.
- Cloud processing of any kind.
- Live transcription (architecture must allow it, see §10).
- Neural source separation / denoising.
- Automatic session summary.
- Transcript editor (v1 has a review view with audio playback only, see §9).

## 2. User workflows

**A. Process existing files.** Pick a session folder or files → assign each file to a participant (auto-suggested from filename) → optional per-track offset → choose preset → run → outputs + a list of flagged passages to review.

**B. Record then process.** Pick device → browse input channels with live meters → assign channels to participants → optional 30 s guided mic check (§5.2) → record → stop → session opens in workflow A pre-filled.

**C. Headless.** `sessionscribe transcribe --session session.toml` or flags; same engine, same outputs.

## 3. Settings

A pydantic v2 model with `schema_version`; read with `tomllib`, written with `tomli-w`; explicit migration functions between schema versions. Presets are named bundles (built-ins + user presets in the user config dir).

### 3.1 Model and device

- `model`: `tiny`, `base`, `small`, `medium` (+ `.en` variants), `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v3`, or a local path.
- `.en` and distil English models force `language=en`; choosing another language with them is a validation error.
- `device`: `cuda` | `cpu` | `auto`.
- `compute_type`: offered list comes from `ctranslate2.get_supported_compute_types(device)` at runtime, intersected with the known-good list:
  - cuda: `float16` (default), `int8_float16`, `bfloat16`, `int8`, `float32`
  - cpu: `int8` (default), `int8_float32`, `float32`
- `cpu_threads` (default: physical cores), `num_workers`, `model_dir`, `offline` (load models with local files only, no network).

### 3.2 ASR mode

Two inference modes, because they do not honour the same parameters:

| | `quality` (default) | `fast` |
|---|---|---|
| Backend | `WhisperModel.transcribe()` | `BatchedInferencePipeline` |
| Temperature fallback | full list | first value only (verify on pinned version) |
| `condition_on_previous_text` | configurable (default `False`) | not applicable |
| `hallucination_silence_threshold` | supported (needs word timestamps) | not applied (verify) |
| `log_prob` / `compression_ratio` thresholds | drive retries | reported only; our own post-checks use them |
| `batch_size` | – | auto-probed: 16 → 8 → 4 → 2 → 1 on OOM, then fail with a clear error |

Each setting declares which modes it applies to; the GUI disables (with tooltip) settings that the selected mode ignores, and the CLI warns. A regression test pins this behaviour against the locked faster-whisper version.

Rationale for `quality` as default (owner decision after the M1 benchmark): accuracy decides, and processing a session in several hours is acceptable. The default stays `quality` unless `fast` is better by more than 1 percentage point WER on the default configuration; processing time is reported for information only. M1 measured about 1.8 h for 10 h of speech in `quality` mode (large-v3 float16, reference GPU). Processing tracks in parallel gains only about 8 %. The built-in `fast` preset (large-v3, batch 4) is the optional faster choice: about 2.4× faster at +3 pp WER.

Later option (not v1): hybrid – `fast` first pass, `quality` re-run on flagged passages only.

### 3.3 Language, decoding, vocabulary

- `language` (ISO code or `auto`), `task=transcribe`.
- `beam_size`, `best_of`, `temperature` list (default 0.0–1.0 in 0.2 steps), `condition_on_previous_text=False`.
- `no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4` (reference values for `quality` mode).
- `hallucination_silence_threshold` (quality mode, default off; benchmark 2.0).
- `word_timestamps=True`.
- **Vocabulary (minimal, M1):** `initial_prompt` and `hotwords` as plain text fields passed through. Campaign glossary and post-correction list remain later features.

### 3.4 VAD (Silero via faster-whisper)

Library defaults are threshold 0.5, min_speech 0 ms, min_silence 2000 ms, speech_pad 400 ms, max_speech ∞. Built-in "Conversation" preset starting point (to be tuned in M1):

| Parameter | Start value | Notes |
|---|---|---|
| `threshold` | 0.5 | test 0.45–0.50 |
| `min_speech_duration_ms` | 0 | never filter short words like "igen"/"nem" aggressively (≤ 100) |
| `min_silence_duration_ms` | 800 | test 700–1000 |
| `speech_pad_ms` | 400 | test 300–400 |
| `max_speech_duration_s` | 30 | test 25–30 |

VAD speech intervals are stored per track/chunk so passages dropped by VAD can be inspected later.

### 3.5 Bleed handling

See §6.2 for the algorithm.

- `bleed_mode`: `post` (default, non-destructive) | `pre` | `off`.
- `coupling`: `auto` (estimated from the session, default) | `guided` (from mic check) | `manual`.
- `ownership_margin_db` (default to tune in M3; start 6).
- `pre_attenuation_db` (pre mode only; default 15, test 12–18). `mute` is available only as an explicit expert option.
- `frame_ms` (20), `hold_ms`, `release_ms`, `floor_db`.
- Two built-in profiles: **Conservative** (default: post mode, low ownership threshold) and **Strong** (pre mode, attenuation).

### 3.6 Dedup and overlap

- `dedup_enabled`, `dedup_similarity` (default 85), `dedup_max_time_gap_s` (1.0), `dedup_min_words` (3; shorter utterances require strong energy evidence).
- `mark_overlaps` (on).

### 3.7 Session and output

- Participants: `{id, name, source, offset_s, color}`. Renaming a participant only re-exports.
- `input_dir`, `output_dir`, `work_dir`.
- `formats`: `txt`, `srt`, `vtt`, `json`, `md`.
- `timestamp_style` for txt/md: `hh:mm:ss` | `hh:mm:ss.mmm`. SRT/VTT always use their own millisecond formats.
- `merge_same_speaker_gap_s` (txt/md only), filename template.

## 4. Time model

- Internally all positions are **integer sample indices** on a session timeline at `timeline_rate` (48 000 Hz). Tracks at other rates are resampled for gating; ASR works on 16 kHz copies and its seconds are converted back once.
- Float seconds appear only in the public JSON and exporters (rounded once, at export).
- Per-track `offset_s` is converted to samples and applied when mapping track-local positions to the session timeline.

## 5. Processing pipeline

```
Track readers (raw, untouched) ─► energy envelopes ─► coupling estimate ─► ownership masks
        │                                                                    │
        └─► [pre mode: attenuate non-owned] ─► 16 kHz ─► VAD ─► ASR ─► segments + words
                                                                             │
             attribution (post mode) ─► dedup ─► overlap mark ─► quality flags ─► exporters
```

### 5.1 Chunking

- Chunks are defined in **track-local** time: an **owned range** (default 10 min) plus **context** on both sides (default 5 s). ASR sees owned + context; only segments whose midpoint lies in the owned range are kept. Words straddling the boundary are resolved by word timestamps; boundary behaviour has dedicated tests.
- Owned-range boundaries are nudged to a nearby low-energy point (deterministic: same input → same boundaries).
- Track-local chunking means changing an offset never invalidates ASR results in `post` mode.
- Memory budget: at most one chunk of all tracks in RAM at once.

### 5.2 Coupling estimation

The coupling `C[i][j]` (dB) says how loud speaker *j*'s voice typically appears on mic *i* relative to mic *j*.

- `auto`: from frames where exactly one track is clearly dominant, regress other tracks' energy against it; robust statistics (median), per mic pair. Needs no user action.
- `guided`: 30 s mic check before recording – each participant speaks ~5 s in turn. Optional, improves `auto`.
- Stored in the session manifest; shown as a small matrix in Advanced.

### 5.3 Ownership

Per frame, track *i* **owns** the frame if
`E_i ≥ floor` and `E_i ≥ max_j≠i (E_j + C[i][j]) + margin`,
i.e. it is louder than the bleed we would expect from everyone else. Smoothed with hold/release. If two tracks own the same frame, both speakers are talking → overlap, both kept.

This avoids the weakness of comparing against the single loudest track: a quiet speaker next to a loud one still owns their frames as long as they exceed the *predicted bleed*, not the neighbour's absolute level.

### 5.4 Bleed modes

- **post (default):** ASR runs on raw audio (with VAD). Each word gets an ownership ratio from the mask; segments whose words are mostly not owned are marked `bleed` and excluded from txt/srt/vtt/md but kept in JSON with `suppression_reason="bleed"`. Changing bleed parameters only re-runs attribution – no re-transcription.
- **pre:** non-owned frames are attenuated (not muted) before VAD/ASR. Less compute, fewer bleed hallucinations, but parameter changes require re-ASR.
- **off:** baseline; used as the reference in benchmarks.

### 5.5 Dedup

A segment is removed as a duplicate only if **all** hold:
1. text similarity ≥ `dedup_similarity` and time distance ≤ `dedup_max_time_gap_s`;
2. energy/ownership evidence indicates one of them is bleed of the other (its frames are not owned, consistent with `C`);
3. there is no evidence of two independent speakers (both tracks owning the frames → never dedup).

Utterances shorter than `dedup_min_words` (e.g. two people shouting "Nem!") need strong energy evidence. Uncertain cases are **kept** and flagged `possible_duplicate`. Removed items go to `suppressed_duplicates` with `duplicate_of` and `suppression_reason`.

Cross-correlation / spectral similarity between tracks may be explored in M3 as *additional* evidence only, never as a sole criterion.

### 5.6 Quality flags

Segments are flagged (not deleted) when: low `avg_logprob`; high `no_speech_prob`; high compression ratio / repeated n-grams; text far too long for the voiced duration; segment outside VAD speech; possible duplicate; low ownership; overlaps a recording gap. Flags appear in JSON, in md as a marker, and in the GUI review list.

### 5.7 Cache

Layered, each layer with its own key; files written atomically (temp file + rename); incomplete results never count as done.

| Layer | Key includes | Invalidated by |
|---|---|---|
| `features` (energy envelopes) | file fingerprint, track-local chunk range, frame_ms, `processing_version` | audio change |
| `coupling` + `ownership` | all feature keys, offsets, coupling/ownership params | offsets, bleed params |
| `asr` | file fingerprint, track-local chunk + context range, model id + revision, compute type, ASR mode + params, VAD params, faster-whisper/ctranslate2 versions, `processing_version`; in `pre` mode also the ownership key | model/ASR/VAD change (and bleed params in pre mode) |
| merge / export | not cached (cheap) | – |

File fingerprint: size + mtime + hash of first/last MB (fast), full hash optional.

**Crashing workers.** ASR runs in child processes. An unexpected child exit (non-zero exit code, e.g. Windows 0xC0000005, a signal, or a missing result) fails only that chunk: completed chunks stay valid in the cache, partial files of the failed chunk are discarded, and the chunk is re-run **once**. A second failure stops processing with a clear error: track, chunk range, decoded exit code, likely cause and a hint. The cache stays consistent, so a later run resumes after the completed chunks. Crashes and retries are recorded in the log and in `processing_events` in the JSON output.

## 6. Data model (JSON output)

```jsonc
{
  "schema_version": 1,
  "app_version": "…", "processing_version": 1,
  "model": { "id": "large-v3", "revision": "…", "compute_type": "float16", "asr_mode": "quality" },
  "session": { "name": "…", "date": "2026-10-29", "duration_s": 19842.5, "timeline_rate": 48000 },
  "settings": { /* resolved */ },
  "participants": [ { "id": "p1", "name": "Anna", "source": "anna/", "offset_s": 0.0 } ],
  "coupling_db": [[0, -24.1], [-26.3, 0]],
  "recording_gaps": [ { "start_sample": 0, "length_samples": 0, "reason": "input_overflow" } ],
  "segments": [
    {
      "id": "p1-c003-0007",            // stable: speaker-chunk-index
      "speaker": "p1", "source_chunk_id": "p1-c003",
      "start": 12.34, "end": 15.80, "audio_start_sample": 592320,
      "text": "…", "avg_logprob": -0.21, "no_speech_prob": 0.02, "compression_ratio": 1.4,
      "ownership": 0.97, "overlap": false, "quality_flags": [],
      "words": [ { "start": 12.34, "end": 12.61, "word": "…", "prob": 0.97 } ]
    }
  ],
  "suppressed": [ /* same shape + "suppression_reason": "bleed"|"duplicate", "duplicate_of" */ ]
}
```

## 7. Output formats

- **txt**: `[01:02:03] Anna: text`.
- **md**: title, metadata (date, length, participants, model, mode), lines; overlap `⟂`, flagged lines `⚠`; summary of recording gaps at top.
- **srt / vtt**: one cue per segment, `Anna: text`, millisecond timing.
- **json**: §6.

## 8. Recording

### 8.1 Stream and threads

- **One input stream** covering all needed channels of the interface (keeps channels mutually sample-aligned even through dropouts).
- **Explicit device selection:** the recorder never auto-selects a device. The user picks device and host API explicitly; device name, index, host API, channel selection and stream settings are saved in the manifest. (Auto-selection exists only in the M0 `record-probe`/`doctor` tools.)
- The PortAudio callback only copies the block into a **preallocated, bounded ring buffer** and records status flags. No file I/O, no allocation, no locks that can block.
- A writer thread drains the buffer, converts, and writes files.
- Exceptions in the callback or writer are caught, set a fatal state and are surfaced in the GUI immediately; a watchdog checks `stream.active` and that frames keep arriving.
- Metrics shown live and stored in the manifest: buffer fill level, input overflows, dropped blocks, writer latency.
- Any data loss is a **visible alarm** in the GUI, recorded as a gap (sample position + estimated length) in the manifest, and later in the transcript JSON/md.

### 8.2 Files

- **Segmented files:** each channel is written as consecutive, individually closed FLAC files (default 5 min). WAV option uses RF64.
- Manifest (`session.toml`), updated atomically after every closed segment: per segment start/end sample, frame count, file hash, status (`open`/`closed`/`damaged`).
- On restart the app verifies segments against the manifest and reports missing/damaged parts. Readers expose a channel's segments as one virtual track.
- Verify the stream's actual sample format and rate; record them in the manifest (24-bit file ≠ guaranteed 24-bit end-to-end).
- Before recording, check free space for the planned duration (default 8 h) + 10 % reserve; warn during recording when remaining time drops below 30 min.

### 8.3 Host APIs

- Windows: if ASIO is requested, set `SD_ENABLE_ASIO` **before importing sounddevice** (decided in a startup config step). WASAPI is the fallback, but its multichannel behaviour is tested, not assumed.
- Linux: ALSA, PulseAudio/PipeWire, JACK – whatever PortAudio exposes.
- At startup and in `doctor`: list host APIs, device channel counts, supported rates, and actually open the requested N-channel stream briefly.

## 9. GUI (PySide6)

Tabs: **Session**, **Record**, **Settings** (Basic / Advanced, mode-aware disabling), **Run** (progress, ETA, log, cancel/resume), **Result**.

Result view: colour-coded transcript; filter for flagged / overlap / suppressed items; clicking a line plays that time range from the original track (QtMultimedia); recording gaps shown inline. No text editing in v1.

Dark/light theme; English + Hungarian UI; "System check" runs `doctor`.

## 10. Live-readiness (design requirement, not v1)

Concepts defined now, implementations minimal:

- `AudioBlock`: samples + `start_sample` on the session timeline + `source_id`.
- `ProcessingStage`: stateful or stateless step consuming `AudioBlock`s.
- `SegmentEvent` with `status: provisional | final` and a revision id, so a later live mode can replace provisional text.
- Sinks (exporters, GUI) consume events.

Not now: `LiveSource`, partial-result stabilisation, real-time scheduling.

## 11. Quality assurance

### 11.1 Reference set

- 3–4 clips of 5 min from real sessions, chosen to contain: normal conversation, quiet remarks, true overlap, loud bleed, laughter, repeated short answers, fantasy names.
- Hand-corrected reference transcript per clip (speaker, start, end, text). A draft is generated by the app and corrected by hand.
- **Real recordings are personal data: stored locally, git-ignored, never committed.** Public tests use synthetic audio only.

### 11.2 Metrics (`sessionscribe bench`)

| Metric | Checks |
|---|---|
| WER / CER | recognition accuracy |
| Speech recall | reference utterances present in output |
| Speaker attribution accuracy | right person |
| False suppression rate | real utterances removed as bleed/duplicate |
| Residual duplicate rate | bleed left in final transcript |
| Timestamp error | start/end deviation |
| RTF, peak VRAM, peak RAM | performance |

Bleed handling is always compared against `bleed_mode=off` on the same clips.

## 12. Privacy and licensing

- Fully local; `offline` setting guarantees no network use after model download.
- Docs: participants' voices are personal data; whoever records is responsible for informing participants and obtaining consent where required.
- `sessionscribe clean <session>` deletes work/cache files; option to delete raw audio separately (with confirmation).
- Logs contain no transcript text by default.
- Own code MIT. PySide6 is LGPLv3: keep it as a separately replaceable library (no static linking; one-dir builds), ship its licence texts. Model weights: check each model's licence before bundling any in an installer (default: download on first use).

## 13. Milestones and acceptance criteria

| # | Scope | Done when |
|---|---|---|
| M0 | Skeleton: uv, ruff, pytest, pre-commit, **CI (lint + fast tests)**, docs; `doctor` (versions, supported compute types, real short inference, audio host APIs, open N-ch stream); **`record-probe` CLI**: 8-ch raw recording for N minutes, reports sample counts/overflows | CI green; `doctor` passes on reference machine; 30-min 8-ch probe via ASIO and WASAPI with zero overflow and matching sample counts |
| M1 | Single-file ASR CLI, full settings model, presets, both ASR modes with mode-aware params, VAD preset, initial_prompt/hotwords pass-through; bench harness; reference clips prepared | Regression test confirms batched-mode parameter behaviour; benchmark matrix (large-v3 / turbo × float16 / int8_float16 × batch 4/8/16 × word ts × mode) recorded on a HU clip; default mode confirmed |
| M2 | Multi-track: participants, offsets, time model, chunking with owned/context ranges, layered cache, merge, all exporters, quality flags | Chunk-boundary tests pass; re-export / rename / offset change (post mode) without re-ASR; stable segment ids; 8 synthetic tracks + real clips merge correctly; a crashing worker process (simulated) is detected, its chunk re-run once from the cache, a second crash gives a clear error with the cache left consistent (§5.7) |
| M3 | Coupling (auto + guided), ownership, post/pre modes, dedup with evidence rules, overlap marking | On the reference set vs `off`: speech recall does not drop beyond an agreed tolerance, no hand-marked overlap is lost silently, residual duplicates substantially reduced; tolerances fixed from first measurements and written to DECISIONS |
| M4 | GUI: Session, Settings, Run, Result (review + playback), i18n, themes | Full workflow A from GUI; cancel + resume; flagged items reviewable |
| M5 | Recorder: device/channel browser with explicit device + host API selection saved in the manifest (§8.1), meters, guided mic check, segmented writer, manifest, recovery | 8 h, 8-ch recording on reference machine without loss; fault injection: device unplug, low disk, slowed disk writes, process kill → manifest correctly reports what exists and what is missing; restart recovery works |
| M6 | Release: README (EN + HU summary), supported-configuration matrix, licence notices, install tests on Windows and Linux, optional one-dir build | Fresh install on both OSes following README succeeds; on a clean environment **without system CUDA/cuDNN in PATH** the `cuda` extra alone is sufficient (`doctor` passes incl. real CUDA inference) |

## 14. Later features (do not design out)

Live transcription; campaign glossary + post-correction list; hybrid fast/quality re-run; session summary; diarization for mixed recordings; cross-correlation evidence; transcript editor.

## 15. Open questions

- Default ownership margin and attenuation – measured in M3.
- Default ASR mode – `quality` (decided by WER). Confirm the WER comparison on the verified reference and a ≥ 5 min clip (§11.1) before the M3 tolerances are fixed.
- Recording segment length (5 min default) – confirm in M5.
- Project name.
