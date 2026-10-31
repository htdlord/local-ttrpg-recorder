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
| `batch_size` | – | starting size (default 16; `fast` preset 4), halved on OOM down to 1, then a clear error; `oom_fallback=false` keeps it fixed (benchmarks) |

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
- `ownership_margin_db` (6), `suppress_below` (0.5), `low_ownership_below` (0.6), `text_similarity` (70) – provisional defaults from clip01, to be fixed on clip02 (DECISIONS 2026-10-30).
- `pre_attenuation_db` (pre mode only; default 15, test 12–18). `mute` is available only as an explicit expert option.
- `frame_ms` (20), `hold_ms` (ownership kept after a frame), `release_ms` (pre-mode gain ramp), `floor_above_noise_db` (10: per-track floor relative to the track's noise level, because mic gains differ by 15–20 dB), `coupling_min_frames`, `coupling_default_db`.
- Two built-in profiles (presets): **`conservative`** (default: post mode) and **`strong`** (pre mode, 15 dB).
- Session file: `[mic_check]` (participant → `[start_s, end_s]` where only they speak) for `guided`, `coupling_db` matrix for `manual`.

### 3.6 Dedup and overlap

- `dedup_enabled`, `dedup_similarity` (default 85), `dedup_max_time_gap_s` (1.0), `dedup_min_words` (3; shorter utterances require strong energy evidence).
- `mark_overlaps` (on).

### 3.7 Session and output

- Participants: `{id, name, source, offset_s, color}`. Renaming a participant only re-exports.
- `input_dir`, `output_dir`, `work_dir`.
- `formats`: `txt`, `srt`, `vtt`, `json`, `md`.
- `timestamp_style` for txt/md: `hh:mm:ss` | `hh:mm:ss.mmm`. SRT/VTT always use their own millisecond formats.
- `merge_same_speaker_gap_s` (txt/md only), filename template.
- Session file `session.toml`: `name`, `date`, optional `preset` and `[settings]` (layered after `--config`), `[[participants]]` with `id`, `name`, `source` (file or folder of segment files, relative to the session), `channel`, `offset_s` (≥ 0), `color`; optional `[[reassignments]]` (`segment` id → participant `to`, set by hand in the GUI or with `reassign`). `session init FOLDER` suggests one from file names.
- Chunking (`asr.chunk_owned_s` 600, `asr.chunk_context_s` 5, `asr.chunk_boundary_search_s` 15) and quality-flag thresholds (`[quality]`) are settings; see §5.1 and §5.6.

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
- Owned-range boundaries are nudged to a nearby low-energy point (deterministic: same input → same boundaries): the quietest 0.5 s-smoothed point within ±`chunk_boundary_search_s`.
- Words of a kept segment that lie past the owned range are dropped only when the neighbouring chunk's own words cover them; a segment duplicated entirely goes to `suppressed` (`chunk_boundary_duplicate`).
- Track-local chunking means changing an offset never invalidates ASR results in `post` mode.
- **Re-transcription of word-less speech:** Whisper sometimes skips speech after a window start. After each chunk's ASR, every VAD speech stretch ≥ `asr.refill_min_s` (1.5 s) without any recognised word is transcribed again as its own window (`asr.refill_pad_s` padding, only words inside the stretch kept). Recovered segments are flagged `refilled` and pass through all quality flags; stretches still without words are reported as `possible_missed_speech`. Switch: `asr.refill_holes`.
- Memory budget: at most one chunk of all tracks in RAM at once.

### 5.2 Coupling estimation

The coupling `C[i][j]` (dB) says how loud speaker *j*'s voice typically appears on mic *i* relative to mic *j*.

- `auto`: per ordered pair, the lowest significant peak of the level difference `E_i − E_j` on frames where mic *j* is clearly above its own noise (*j*'s voice; *i*'s voice and third parties form higher peaks). Pair sums `C[i][j] + C[j][i]` are kept ≤ −20 dB; pairs without enough data use a conservative default and are reported. Needs no user action.
- `guided`: 30 s mic check before recording – each participant speaks ~5 s in turn. Optional, improves `auto`. The M5 mic check also **alarms when an assigned channel has no signal or a suspiciously quiet one** (vs its own noise floor and the other channels), so a muted or unplugged lavalier is caught before the session.
- **Silent mics:** a track whose speech level stays within `silent_mic_range_db` (10) of its noise records (almost) nothing of its wearer; it is reported (`silent_mics` in JSON). If the same longer text appears on two other tracks, unresolved by ownership evidence and with both copies ≥ 20 dB below their tracks' speech levels, one copy is kept on the louder track as `possible_wrong_speaker` with `probable_speaker = "unknown"` and the other becomes its duplicate. There is no automatic reassignment to the silent mic's owner; the GUI (M4) allows it by hand.
- Stored in the session manifest; shown as a small matrix in Advanced.

### 5.3 Ownership

Per frame, track *i* **owns** the frame if
`E_i ≥ floor_i` and `E_i ≥ max_j≠i (E_j + C[i][j]) + margin`,
i.e. it is louder than the bleed we would expect from everyone else. `floor_i` is the track's own noise level + `floor_above_noise_db`. Ownership is held for `hold_ms`. If two tracks own the same frame, both may be talking; overlap marking (§5.5) also requires different words.

This avoids the weakness of comparing against the single loudest track: a quiet speaker next to a loud one still owns their frames as long as they exceed the *predicted bleed*, not the neighbour's absolute level.

### 5.4 Bleed modes

- **post (default):** ASR runs on raw audio (with VAD). Each word gets an ownership ratio from the mask. A segment is suppressed as bleed (excluded from txt/srt/vtt/md, kept in JSON with `suppression_reason="bleed"` and `bleed_source`) only if: its ownership < `suppress_below` (< 0.2 for utterances under `min_words`), another track owns its frames, that track has words there, and its text matches those words (≥ `text_similarity`). Otherwise it is kept: as `possible_wrong_speaker` (with `probable_speaker`) when the other track has no words – the only copy –, and never when it contains ≥ 0.5 s of owned words (own speech mixed with bleed). Ownership < `low_ownership_below` is flagged `low_ownership`. Changing bleed parameters only re-runs attribution – no re-transcription.
- **pre:** non-owned frames are attenuated (not muted) before VAD/ASR. Less compute, fewer bleed hallucinations, but parameter changes require re-ASR.
- **off:** baseline; used as the reference in benchmarks.

### 5.5 Dedup

A segment is removed as a duplicate only if **all** hold:
1. text similarity ≥ `dedup_similarity` and time distance ≤ `dedup_max_time_gap_s`;
2. energy/ownership evidence indicates one of them is bleed of the other (its frames are not owned, consistent with `C`);
3. there is no evidence of two independent speakers (both tracks owning the frames → never dedup).

Utterances shorter than `dedup_min_words` (e.g. two people shouting "Nem!") need strong energy evidence. A copy containing owned speech of its own is never removed. Uncertain cases are **kept** and flagged `possible_duplicate`; so are longer identical sentences that both tracks own (a third person whose own mic missed it). Removed items go to `suppressed` with `duplicate_of` and `suppression_reason="duplicate"`.

**Overlap marking:** segments of different speakers overlapping ≥ 0.3 s, both owning the shared frames and saying different words, get `overlap=true` (md: ⟂); both are kept.

Cross-correlation / spectral similarity between tracks may be explored in M3 as *additional* evidence only, never as a sole criterion.

### 5.6 Quality flags

Segments are flagged (not deleted) when: `possible_wrong_speaker` (another mic owns it, this is the only copy); low `avg_logprob`; high `no_speech_prob` together with a lowish `avg_logprob` (both are per 30 s decoding window); high compression ratio / repeated n-grams; text far too long for the voiced duration; segment outside VAD speech; possible duplicate; low ownership; overlaps a recording gap; `refilled` (recovered by re-transcription, §5.1). VAD speech that stays without words is listed as `possible_missed_speech`. Flags appear in JSON, in md as a marker, and in the GUI review list.

### 5.7 Cache

Layered, each layer with its own key; files written atomically (temp file + rename); incomplete results never count as done.

| Layer | Key includes | Invalidated by |
|---|---|---|
| `features` (energy envelopes) | file fingerprint, track-local chunk range, frame_ms, `processing_version` | audio change |
| `coupling` + `ownership` | all feature keys, offsets, coupling/ownership params | offsets, bleed params |
| `asr` | file fingerprint, track-local chunk + context range, model id + revision, compute type, ASR mode + params, VAD params, faster-whisper/ctranslate2 versions, `processing_version`; in `pre` mode also the ownership key | model/ASR/VAD change (and bleed params in pre mode) |
| merge / export | not cached (cheap) | – |

File fingerprint: size + hash of the first and last MB of each file a range touches (no mtime, so copying a session folder keeps its cache); full hash optional (`--full-hash`).

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
  "suppressed": [ /* same shape + "suppression_reason": "bleed"|"duplicate"|"chunk_boundary_duplicate", "duplicate_of" */ ],
  // also: "bleed_mode_applied", "chunks" (owned/context ranges, cache key, cached),
  // "vad_intervals" per participant, "processing_events" (worker crashes, retries, warnings);
  // "possible_missed_speech" (VAD speech without words, for review);
  // per segment "boundary_trimmed_words" and, after a manual reassignment, "reassigned_from"
  // (+ quality flag "reassigned"; the speaker's audio_start_sample stays on the original track).
  // audio_start_sample is track-local, start/end session time.
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

- **Segmented files:** each channel is written as consecutive, individually closed FLAC files (default 5 min) under `rec/chNN_<participant>/NNNN.flac`, flushed every second. WAV option uses RF64.
- Manifest: `[recording]` in `session.toml`, updated atomically after every closed segment, gap, alarm and state change: device, host API, channels, stream settings; per segment start sample, frame count, SHA-256, status (`open`/`closed`/`recovered`/`damaged`); gaps (`start_sample`, `length_samples`, `reason`, `padded`, `estimated`); alarms; metrics.
- Lost audio never shifts the timeline: ring-buffer drops (exact), ADC timestamp jumps and time without a device (estimated) are filled with silence and recorded as gaps; the gaps reach the transcript (`overlaps_gap`).
- `record verify` checks the files against the manifest; `record recover` makes `open` segments of a crashed run readable (original kept as `.partial`) and marks the run `interrupted`. Readers expose a channel's segments as one virtual track.
- Recording profiles: `rig.toml` (device + host API by name, channel → participant, segment length, format); explicit flags override it.
- While recording: idle-channel warning (an assigned channel at its noise floor for 10 min while others are active), reconnect after an unplug (PortAudio re-initialised to see the device again), sleep blocked (Windows `SetThreadExecutionState`, Linux `systemd-inhibit`).
- Verify the stream's actual sample format and rate; record them in the manifest (24-bit file ≠ guaranteed 24-bit end-to-end).
- Before recording, check free space for the planned duration (default 8 h) + 10 % reserve; warn during recording when remaining time drops below 30 min.

### 8.3 Recording from the GUI

- The GUI never records in its own process. It starts a **detached recorder process** (`python -m sessionscribe.core.audio.record_child`; Windows: new process group, no console, breaking away from a launcher's job; Linux: new session) that records exactly like `record start`.
- **No pipes.** The two sides share only files in `<session>/rec/control/`: `status.json` (atomic, ~4 Hz: state, heartbeat, clock, levels, buffer, disk, gaps, current mic-check prompt, command ack), `events.jsonl` (append-only alarms, prompts, mic-check result, end; rotated above 1 MB), `command.json` (GUI → recorder: `stop`), `recorder.lock` (pid, start, session) and `child.log`. The recorder never waits for a reader.
- Closing or crashing the GUI does not stop the recording; the next window reconnects (last followed session in the preferences, `recorder.lock` + heartbeat). A lock whose process is gone means a crash: the GUI offers Recover (`record recover` + `verify`).
- The GUI lists devices through a short-lived recorder process, so it never loads PortAudio/ASIO itself (an ASIO driver serves one client).

### 8.4 Host APIs

- Windows: if ASIO is requested, set `SD_ENABLE_ASIO` **before importing sounddevice** (decided in a startup config step). WASAPI is the fallback, but its multichannel behaviour is tested, not assumed.
- Linux: ALSA, PulseAudio/PipeWire, JACK – whatever PortAudio exposes.
- At startup and in `doctor`: list host APIs, device channel counts, supported rates, and actually open the requested N-channel stream briefly.

## 9. GUI (PySide6)

Tabs: **Session**, **Record** (device, rig, guided mic check with prompts and per-channel result, level meters, alarms with advice, start/stop, recover; §8.3), **Settings** (Basic / Advanced, mode-aware disabling; advanced settings show a translated name with the TOML key underneath), **Run** (progress, ETA, log, cancel/resume), **Result**.

Result view: colour-coded transcript; filter for flagged / overlap / suppressed items; clicking a line plays that time range from the original track (QtMultimedia); recording gaps shown inline. No text editing in v1.

Manual reassignment: any kept segment (typically `possible_wrong_speaker`, including `probable_speaker = "unknown"`) can be moved to another participant and back. It is stored in `session.toml` as `[[reassignments]]` (segment id → participant, no transcript text), applied in the core after bleed handling, and re-exported from the cache without ASR. The moved segment keeps its other flags and gains `reassigned` + `reassigned_from`. Unknown segment ids (e.g. after settings changed the segmentation) give a warning, never an error. Restoring a suppressed item is a later feature.

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
- **Triage reference** per clip (required from M3): each of our own per-track segments (bleed handling off) is labelled by ear as real speech, bleed (duplicate of another track), wrong speaker (only copy, on the wrong track), speech + bleed, no speech, or unsure. No text is corrected. Made with `bench ref triage` (keyboard labelling with playback; CSV fallback); see DECISIONS 2026-10-30.
- Verbatim, hand-corrected reference transcripts (for WER) are a later feature (§14). Until then WER/CER are measured against draft transcripts and are informational only.
- **Real recordings are personal data: stored locally, git-ignored, never committed.** Public tests use synthetic audio only.

### 11.2 Metrics (`sessionscribe bench`)

| Metric | Checks |
|---|---|
| WER / CER | recognition accuracy |
| Speech recall | reference utterances (triage: speech / wrong-speaker units) present in output |
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
| M3 | **Prerequisite:** `bench ref triage` and complete triage references (§11.1) for clip01 and the ≥ 5 min clip(s); M1 benchmark repeated on the ≥ 5 min clip. Then: coupling (auto + guided), ownership, post/pre modes, dedup with evidence rules, overlap marking | On the triage reference set vs `off`: speech recall does not drop beyond an agreed tolerance, no hand-marked overlap is lost silently, residual duplicates substantially reduced; tolerances fixed from first measurements and written to DECISIONS |
| M4 | GUI: Session, Settings, Run, Result (review + playback), i18n, themes; Record tab in a second step (detached recorder process, file-based status) | Full workflow A from GUI; cancel + resume; flagged items reviewable; a `possible_wrong_speaker` segment (incl. `probable_speaker = "unknown"`) can be reassigned to a participant by hand, without re-ASR |
| M5 | Recorder: device/channel browser with explicit device + host API selection saved in the manifest (§8.1), meters, guided mic check with alarms for assigned channels without signal or suspiciously quiet, segmented writer, manifest, recovery | 8 h, 8-ch recording on reference machine without loss; mic check alarms on a muted/unplugged and on a very low-gain assigned channel (tested); fault injection: device unplug, low disk, slowed disk writes, process kill → manifest correctly reports what exists and what is missing; restart recovery works |
| M6 | Release: README (EN + HU summary), supported-configuration matrix, licence notices, install tests on Windows and Linux, optional one-dir build | Fresh install on both OSes following README succeeds; on a clean environment **without system CUDA/cuDNN in PATH** the `cuda` extra alone is sufficient (`doctor` passes incl. real CUDA inference) |

## 14. Later features (do not design out)

Live transcription; campaign glossary + post-correction list; hybrid fast/quality re-run; session summary; diarization for mixed recordings; cross-correlation evidence; transcript editor; verbatim hand-corrected reference transcripts for WER benchmarking (the `ref labels`/`ref build --verified` round trip exists since M1).

Known risk for live transcription: short ASR windows cost accuracy. On clip01, 30 s chunks raised WER from 21.9 % to 31.9 % (29.9 % with re-transcription of word-less speech) against whole tracks, because Whisper re-windows and skips speech after window starts (DECISIONS 2026-10-30). A live mode must measure this and plan for it (longer windows, later re-runs of provisional segments).

## 15. Open questions

- Default ownership margin and attenuation – measured in M3.
- Default ASR mode – `quality` (decided by the WER ranking; M1 WER figures are relative to a draft transcript). Confirm the ranking on a ≥ 5 min clip (§11.1) before the M3 tolerances are fixed.
- Recording segment length (5 min default) – confirm in M5.
- Project name.
