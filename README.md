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

## Privacy

Participants' voices are personal data. Whoever records a session is responsible
for informing participants and obtaining consent where required.

## Licence

MIT – see [LICENSE](LICENSE). Third-party components (e.g. PySide6, LGPLv3) keep
their own licences.
