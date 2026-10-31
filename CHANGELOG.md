# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org/)
(0.x: settings and output formats may still change; files carry a schema version and are
migrated when read).

## Unreleased – 0.1.0

First release, once the defaults are confirmed on a real session (see
[docs/RELEASING.md](docs/RELEASING.md)).

### Transcription
- Per-track transcription with faster-whisper: `quality` mode (default,
  `WhisperModel.transcribe()` with temperature fallback) and `fast` mode (batched
  pipeline); settings know which mode honours them.
- Presets (`conversation`, `fast`, `conservative`, `strong`), a full settings model in
  TOML with schema migrations, initial prompt and hotwords.
- Sessions of up to 8 tracks and 8 hours: integer-sample timeline, offsets, chunked
  processing with owned ranges, a layered cache (resume after interruption, re-export
  without ASR), crashing ASR worker processes detected and re-run once.
- Re-transcription of speech the first pass left without words.
- Bleed handling: coupling (automatic, guided by the mic check, or manual), ownership,
  `post` (default, attribution after ASR) and `pre` (attenuation before ASR) modes,
  silent-mic detection, evidence-based dedup, overlap marks; every removal recorded with
  a reason, uncertain cases kept and flagged.
- Exports: txt, srt, vtt, json, md; manual reassignment of a segment to a participant.

### Recording
- One multi-channel input stream (ASIO, WASAPI, ALSA/PipeWire), segmented 24-bit FLAC
  (or RF64), manifest with gaps, alarms (no signal, quiet mic, clipping, device lost,
  disk), guided mic check, recovery after a crash, `record verify`.
- 8-hour, 8-channel recording on the reference machine without loss.

### Desktop app
- Session, Record, Settings, Run and Result tabs; English and Hungarian; light and dark
  themes; System check; playback from the original track.
- The recording runs in a detached process steered through files: closing or crashing the
  window does not stop it, and reopening reconnects.
- About dialog with versions and licences.

### Tools and documentation
- `doctor` (versions, CUDA libraries, a real inference, audio devices), `record-probe`,
  `bench` (reference clips, triage labelling, metrics, matrices), `bench clip cut`.
- `sessionscribe demo`: a demo session from public-domain LibriVox voices.
- Install guide for Windows and Linux, supported configurations, Hungarian session-day
  checklist, third-party notices; fresh-install CI on Windows and Ubuntu; a check that
  the `cuda` extra alone runs CUDA inference without a system CUDA install.
