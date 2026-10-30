# SessionScribe

> Working title. Early development – nothing usable yet.

Local, multi-track speech-to-text built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Designed for tabletop RPG sessions recorded with one lavalier mic per participant
(up to 8 tracks, sessions up to 8 hours): each track is transcribed separately,
cross-track bleed is handled, and everything is merged into one speaker-labelled,
timestamped transcript (txt, srt, vtt, json, md).

- Runs fully offline after models are downloaded; nothing is sent to the cloud.
- Windows 10/11 and Linux; NVIDIA GPU (CUDA) recommended, CPU fallback.
- Hungarian and English are first-class; any Whisper language works.
- Guiding principle: losing real speech is worse than showing a duplicate. Filters
  keep and flag when uncertain, and every removal is recorded with a reason.

See [docs/SPEC.md](docs/SPEC.md) for the specification and
[docs/DECISIONS.md](docs/DECISIONS.md) for the decision log.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                      # add --extra cuda for pip-installed cuBLAS/cuDNN 9
uv run pre-commit install
uv run ruff check . && uv run ruff format .
uv run pytest -m "not gpu and not slow and not audio_hw"   # fast suite (= CI)
uv run pytest -m "gpu or slow or audio_hw"                 # local: GPU, real model, audio interface
```

Hardware checks:

```bash
uv run sessionscribe doctor                                   # versions, CUDA, real inference, audio
uv run sessionscribe doctor --host-api asio --channels 1-8    # also opens an 8-channel ASIO stream
uv run sessionscribe record-probe --host-api asio --channels 1-8 --minutes 30
```

`record-probe` writes one 24-bit FLAC per channel to `work/probe/<timestamp>/` (use
`--no-write` to only count) and reports frame counts, overflows, timestamp continuity,
clock deviation and per-channel peak/RMS levels.

Record a session (the device is always named; save it once in a rig file):

```bash
uv run sessionscribe record devices                                   # names, host APIs, channels
uv run sessionscribe record rig save rig.toml --device "UMC ASIO Driver" --host-api asio     --channels 1-8 --names gm,anna,bela,... --labels "Mesélő,Anna,Béla,..."
uv run sessionscribe record meters --rig rig.toml                      # identify channels
uv run sessionscribe record check --rig rig.toml                       # mic check with alarms
uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05   # Ctrl+C stops
uv run sessionscribe record verify --session recordings/2026-11-05
uv run sessionscribe transcribe --session recordings/2026-11-05 -l hu
```

`record start` begins with a guided mic check (each participant speaks a few seconds);
its turns are saved for guided coupling. After a crash: `record recover --session ...`.

Transcribe a session (one file or segment folder per participant):

```bash
uv run sessionscribe session init recordings/2026-10-29/      # writes session.toml; edit names/offsets
uv run sessionscribe transcribe --session recordings/2026-10-29 -l hu --formats json,txt,srt,vtt,md
uv run sessionscribe export --session recordings/2026-10-29   # after renaming / offsets: no ASR
uv run sessionscribe clean --session recordings/2026-10-29    # delete the work folder (cache)
```

Completed chunks are cached in `<session>/work/` (it contains transcript text), so an
interrupted run resumes and renaming or moving a participant only re-exports.

Transcribe one file and inspect settings:

```bash
uv run sessionscribe transcribe track.flac --language hu --out output/   # json + txt
uv run sessionscribe transcribe track.flac --language hu --mode fast     # batched pipeline
uv run sessionscribe config show --preset fast --set decoding.beam_size=3
uv run sessionscribe config init my-settings.toml
uv run sessionscribe config validate my-settings.toml
uv run sessionscribe presets list
```

`quality` mode (the default) uses `WhisperModel.transcribe()` with temperature fallback.
`fast` mode uses the batched pipeline, which ignores some options; the CLI warns when a
setting you chose has no effect in the selected mode.

Benchmarks run against local reference clips, which are personal data and git-ignored.
[docs/REFERENCE_GUIDE.md](docs/REFERENCE_GUIDE.md) describes how to prepare a reference.

```bash
uv run sessionscribe bench run --refs bench/refs --matrix m1   # writes work/bench/<run>/results.md
uv run sessionscribe bench score work/bench/<run>              # re-score without ASR
uv run sessionscribe bench ref triage bench/refs/clip01         # label segments by ear (M3)
uv run sessionscribe bench triage score bench/refs/clip01 out.json
uv run sessionscribe bench bleed bench/refs/clip01               # off vs post/pre on the triage
```

Bleed handling defaults to `post` (attribution after ASR, never re-transcribes); the
`strong` preset uses `pre` mode. A `[mic_check]` table in `session.toml` (participant →
time range where only they speak) enables `bleed.coupling = "guided"`.

CUDA: ctranslate2 needs CUDA 12 cuBLAS and cuDNN 9. Either install the optional extra
(`uv sync --extra cuda`) or a system CUDA/cuDNN; `doctor` shows where each library was found.

### Desktop app

The GUI is an optional extra (PySide6, LGPLv3, dynamically linked):

```bash
uv sync --extra gui          # combine with --extra cuda as needed
uv run sessionscribe-gui
```

Tabs: **Session** (create from a folder or open `session.toml`, edit participants),
**Settings** (preset, basic settings, generated advanced form; settings with no effect in
the chosen ASR mode are disabled), **Run** (progress, ETA, cancel; running again resumes
from the cache; export only), **Result** (filters, playback from the original track,
reassigning a segment to another participant without re-ASR). The interface is English
or Hungarian and follows the system light/dark theme unless set under *View*. Manual
reassignment also works from the CLI:

```bash
uv run sessionscribe reassign --session recordings/2026-10-29 --segment p2-c004-0003 --to p5
uv run sessionscribe reassign --session recordings/2026-10-29 --segment p2-c004-0003 --undo
```

Translations: after changing GUI strings run
`pyside6-lupdate <gui .py files> -ts src/sessionscribe/gui/i18n/sessionscribe_hu.ts`,
translate, then `pyside6-lrelease` to the `.qm` next to it (a test checks completeness).

## Privacy

Participants' voices are personal data. Whoever records a session is responsible
for informing participants and obtaining consent where required.

## Licence

MIT – see [LICENSE](LICENSE). Third-party components (e.g. PySide6, LGPLv3) keep
their own licences.
