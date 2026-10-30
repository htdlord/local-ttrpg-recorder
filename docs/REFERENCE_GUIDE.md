# Reference transcripts

A reference transcript says what each participant actually said: who, from when to
when, and what. The benchmark (`sessionscribe bench run`) scores ASR output against it.
References are made from real recordings, so **they are personal data**: they live in
the git-ignored `bench/refs/` folder and must never be committed, pasted into issues or
logged. The `bench` commands print counts, ids and numbers only.

> **Status (2026-10-30):** for M3, the *triage reference* below is used: every segment of
> our own output is labelled by ear, with no text correction. The verbatim workflow after
> it still works, but hand-correcting text is a later feature (SPEC §14).

## Triage reference (M3)

The triage reference does not correct any text. It records, segment by segment, what each piece of our own output **is**. That is enough to measure what bleed handling keeps and removes. Expect about 2–4 s per unit: a 109 s clip has about 80 units, so 5 minutes; a 5 min clip takes 15–20 minutes.

### Steps

1. The clip folder needs `clip.toml` and `tracks/` (see "Clip folder" below).
2. Open a terminal (not a redirected one) and run:
   ```bash
   uv run sessionscribe bench ref triage bench/refs/clip01
   ```
   The first run transcribes the clip with bleed handling **off**, using the default settings and the clip's language. It takes a GPU minute or two and is cached in the clip's `work/` folder. Every segment of that output, plus every "possible missed speech" stretch, becomes one *unit*.
3. Each unit is shown on one line and played from its own track with 0.3 s of context:
   ```
   12/81 · gm · 00:01:23.4 (2.1 s) · own -2 dB / seat_3 -14 dB · suggestion: s (Enter)
     <the ASR text, on the terminal only>
   ```
   The levels are relative to each track's normal speaking level: 0 dB is someone speaking into their own mic.
4. Press one key per unit; the label is saved immediately (`triage.json`). `q` quits; the next run continues at the first unlabelled unit.
5. When every unit is labelled, the status becomes `complete`. Check it with `--stats`.

### Keys

| Key | Meaning |
|---|---|
| `Enter` | accept the suggestion (only when one is shown) |
| `s` | own speech of this track's participant |
| `b` | bleed: someone else's voice that **their own track has too** (droppable) |
| `w` | wrong speaker: someone else said it, but **only this track has it** (must be reassigned, never dropped) |
| `v` | mixed: own speech **and** bleed in the same segment (counts as own speech) |
| `o` | true overlap: this participant speaks while someone else speaks too |
| `n` | no speech: noise, laughter without words, hallucinated text |
| `?` | unsure: not scored |
| `1`–`8` after `b`, `w`, `v` | the source (b, v) or the true speaker (w), by the number shown; `Enter` = unknown, `Esc` = cancel |
| `space` / `m` / `x` | replay own track / all tracks mixed / the loudest other track |
| `u` | back one unit (relabel it) |
| `h` | help (the same table and the example below) |
| `q` | save and quit |

### `b` or `w`? An example

Anna says "Nem!", and Béla's lavalier picks it up faintly.

- **On Béla's track:** if **Anna's track also has** "Nem!", the copy on Béla's track is bleed: **`b`**, source Anna. Dropping it loses nothing.
- **On Béla's track:** if **Anna's track missed it**, so that only Béla's track has the words, it is **`w`**, true speaker Anna. This is the only copy: it must be reassigned to Anna, never dropped.
- **On Anna's track:** the same "Nem!" is simply **`s`**.

Use `x` (the loudest other track) or `m` (all mixed) to check whether the other track has it.

### Other options

- `--rebuild`: regenerate the units, e.g. after new default settings or another model. Labels carry over by segment id, otherwise by the same speaker and ≥ 50 % time overlap; labels that no longer match are listed.
- `--only unlabelled|all|suggested|SPEAKER` and `--from N`: choose what to visit, e.g. `--only all --from 30` to review.
- `--export triage.csv` / `--import triage.csv`: label in a spreadsheet instead.
  - Columns: `id, speaker, kind, start, end, suggestion, label, source`. Fill `label` (and `source` for b/w/v with a participant id).
  - Listen in Audacity using the clip tracks; the `start`/`end` columns give the times.
- `--stats`: label counts, how many suggestions were shown and accepted, seconds per unit.
- `--no-play`: label without audio.

### Scoring

```bash
uv run sessionscribe bench triage score bench/refs/clip01 work/m3/clip01.json
```

This reads a session output JSON of the clip and prints numbers only:

- speech recall (s/w/v/o kept, also weighted by time);
- false suppression (s/w/v/o suppressed);
- residual duplicates (b kept);
- overlap kept and marked;
- no-speech kept;
- wrong-speaker attribution.

## Verbatim reference (later, SPEC §14)

## Clip folder

```
bench/refs/clip01/
  clip.toml          language, duration, track file -> speaker id (+ display name)
  tracks/            chNN_<speaker>.wav|flac, one per participant (never modified)
  source/            an existing transcript, if any (never modified)
  draft/             import.json, asr.json, asr/<speaker>.json, diff.md
  labels/            Audacity label files (export and re-import)
  reference.json     the validated reference (generated by 'bench ref build')
```

`clip.toml` is generated from the track file names:

```bash
uv run sessionscribe bench clip init bench/refs/clip01 --language hu \
    --names-from bench/refs/clip01/source/session.jsonl
```

## Workflow

```bash
C=bench/refs/clip01
uv run sessionscribe bench ref import $C   # existing transcript -> draft/import.json
uv run sessionscribe bench ref draft $C    # our ASR (large-v3, quality) -> draft/asr.json
uv run sessionscribe bench ref diff $C     # where the two disagree -> draft/diff.md, labels/_diff.txt
uv run sessionscribe bench ref labels $C   # draft -> labels/<speaker>.txt + _bleed.txt
# ... correct in Audacity, export labels to $C/labels/edited.txt ...
uv run sessionscribe bench ref build $C --labels $C/labels/edited.txt
uv run sessionscribe bench ref check $C
```

The import never drops anything. Segments the old tool marked as duplicates are
**bleed candidates**: they go to the `_bleed` label track (and to `excluded` in the
reference) instead of disappearing, so you can bring them back if they were real speech.

`ref diff` lists the passages where the imported transcript and our own ASR draft
disagree (different text, or only one of them has something). Those are the places to
listen to most carefully; `draft/diff.md` shows both texts side by side.

## Correcting in Audacity (3.x)

1. **File > Import > Audio**: all files in `tracks/` (one audio track each).
2. **File > Import > Labels**: every file in `labels/` (each becomes a label track).
   Move each speaker's label track under that speaker's audio track. `_bleed` and
   `_diff` can sit at the bottom.
3. Listen and correct:
   - Click a label to select its region; play the selection to hear it.
   - Drag a label's edges to fix the start and end times.
   - Click into the label text to fix the words.
   - **New utterance:** select the region on the speaker's audio track, press Ctrl+B,
     and type `seat_3 | what was said`. No id is needed; one is assigned at build time.
4. Label text format: `<speaker> | <id> [tags] text`.
   - **Wrong speaker:** change the prefix (`gm` -> `seat_3`). The label can stay on
     its track, because the prefix decides.
   - **Bleed / not real speech of that person:** prefix the speaker with `_` (`_seat_3`).
     Do this instead of deleting, so the decision is recorded. A bleed candidate on the
     `_bleed` track that *was* real speech: remove the `_` and fix the speaker.
   - Labels you delete anyway are kept in `excluded` as "deleted while editing".
5. **File > Export > Export Labels…** to `labels/edited.txt`. Audacity writes all label
   tracks into one UTF-8 file. If your version writes one file per track, pass each
   with its own `--labels`.
6. Run `bench ref build`. Fix what it reports and repeat from step 3.
7. Close Audacity **without exporting audio**. The WAV/FLAC files must stay untouched.
   Never use Edit > Delete/Cut on the audio.

To edit an existing reference again, run `bench ref labels $C --force`, which exports
from `reference.json` (it already contains your corrections), then repeat from step 1.

## Conventions

- **Verbatim:** write what was said, including fillers ("hát", "ööö", "um"),
  repetitions and false starts as heard. Do not correct grammar.
- **Names:** spell fantasy and character names the way the campaign spells them.
- **Numbers:** as spoken. Normalisation for scoring happens in the bench.
- **Punctuation and case** do not affect scores; use them for readability.
- **Length:** one label per utterance, at most about 15 s; split at sentence ends or pauses.
- **Times:** start where the voice starts and end where it ends. Leave out a breath before
  or after the voice.
- **One person cannot overlap themselves.** Two labels of the same speaker that overlap
  are rejected; merge them or fix the boundary.

### Tags

Put tags right after the id: `gm | r012 [overlap,unsure] text`.

| Tag | Meaning | Scored |
|---|---|---|
| `overlap` | genuinely talking at the same time as someone else | yes |
| `unsure` | words not certain even after listening | yes |
| `short` | short answer ("igen", "nem", "okay"); used by the M3 dedup tests | yes |
| `inaudible` | speech, but not intelligible; text may be empty | no |
| `laugh` | laughter or a non-speech sound worth marking | no |

Draft hints from the import end in `?`: `overlap?` (possible crosstalk) and `dup?`
(possible duplicate). Replace each with a final tag, or remove it once you have
listened. `--verified` refuses any hint that remains.

## Verified

Build with `--verified` only when every passage has been listened to. The benchmark
always runs, but reports based on an unverified or draft reference are marked as
provisional.
