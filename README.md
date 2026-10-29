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

Transcribe one file and inspect settings:

```bash
uv run sessionscribe transcribe track.flac --language hu --out output/   # json + txt
uv run sessionscribe transcribe track.flac --language hu --mode fast     # batched pipeline
uv run sessionscribe config show --preset fast-draft --set decoding.beam_size=3
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
```

CUDA: ctranslate2 needs CUDA 12 cuBLAS and cuDNN 9. Either install the optional extra
(`uv sync --extra cuda`) or a system CUDA/cuDNN; `doctor` shows where each library was found.

## Privacy

Participants' voices are personal data. Whoever records a session is responsible
for informing participants and obtaining consent where required.

## Licence

MIT – see [LICENSE](LICENSE). Third-party components (e.g. PySide6, LGPLv3) keep
their own licences.
