# Try it without recording: the demo session

```bash
uv run ttrpg demo                 # builds ./demo (downloads ~6 MB once)
uv run ttrpg transcribe --session demo
```

or in the desktop app: *File → Open session…* → `demo/session.toml`, then **Run** and
**Result**.

## What is in it

Four fictional players – *Game Master*, *Bodor*, *Kira* and *Lin* – talk for about two
minutes, one lavalier track each (`demo/tracks/ch01_gm.flac` …). Their voices are short
excerpts of four LibriVox volunteers reading public-domain stories, so the "conversation"
is a patchwork of sentences, but the audio situation is the one Local TTRPG Recorder is built for:

- **Bleed:** every track also carries the other voices, quieter and duller, as a lavalier
  picks up the people next to it.
- **Mic check:** the session starts with each player speaking alone for a few seconds;
  their time ranges are in `[mic_check]` of `session.toml`, so
  `bleed.coupling = "guided"` can be tried as well.
- **True overlap:** two players speak at once once.
- **Offset:** Lin's mic was switched on 0.75 s late; `offset_s` in `session.toml` puts the
  track back in place.

## What to look at

- The transcript (`demo/output/Local TTRPG Recorder demo.txt`) has one speaker per line, with
  bleed copies removed. The JSON's `suppressed` list says which ones and why.
- In the **Result** tab, filter for flagged items: possible duplicates, low ownership,
  possible missed speech and Whisper's typical "Thank you." on near-silence are kept and
  marked for review rather than deleted. Play a segment to hear the original track.
- Right-click a segment to move it to another player; **Run → Export only** updates the
  outputs without transcribing again.
- Try the `strong` preset (pre-ASR attenuation) or `bleed.mode = "off"` and compare.

## Sources and licence

LibriVox recordings are in the public domain
([librivox.org/pages/public-domain](https://librivox.org/pages/public-domain/)); the
texts are by authors who died more than 70 years ago, so they are in the public domain in
the EU as well. Only the first 1.5 MB of each file is downloaded from the Internet Archive
and checked against a fixed SHA-256; `demo/SOURCES.md` repeats these credits.

| Player | Reading | Author | Read by |
|---|---|---|---|
| Game Master | Trusty John (The Blue Fairy Book) | Andrew Lang, ed. (1844–1912) | Clarica |
| Bodor | The Four-Fifteen Express | Amelia B. Edwards (1831–1892) | Paul Allerton |
| Kira | The History of England | Jane Austen (1775–1817) | Kiki Baessell |
| Lin | Ningyo-no-Haka | Lafcadio Hearn (1850–1904) | David Barnes |

From [LibriVox Short Story Collection Vol. 014](https://librivox.org/short-story-collection-vol-14/).
