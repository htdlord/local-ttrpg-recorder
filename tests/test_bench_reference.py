"""Reference tooling on synthetic clips: import, Audacity round-trip, build, validate, diff."""

import json

import numpy as np
import pytest
import soundfile as sf

from sessionscribe.core.bench.clip import Clip, init_clip
from sessionscribe.core.bench.diff import compare, diff_label_lines
from sessionscribe.core.bench.draft import import_jsonl
from sessionscribe.core.bench.labels import (
    build_reference,
    parse_labels,
    read_labels,
    write_label_files,
)
from sessionscribe.core.bench.reference import (
    Reference,
    Utterance,
    read_reference,
    validate_reference,
    write_reference,
)
from sessionscribe.core.bench.text import cer, normalize


@pytest.fixture
def clip(tmp_path):
    folder = tmp_path / "clip01"
    (folder / "tracks").mkdir(parents=True)
    for name in ("ch01_gm", "ch03_seat_1", "ch05_seat_3"):
        sf.write(folder / "tracks" / f"{name}.wav", np.zeros(48_000 * 20, np.float32), 48_000)
    init_clip(folder, "hu", {1: "Mesélő"})
    return Clip.load(folder)


def jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), "utf-8")
    return path


RECORDS = [
    {"id": 1, "start": 0.5, "end": 2.5, "channel": 1, "text": "Jó ember, ő bízik.", "flags": []},
    {
        "id": 2,
        "start": 0.6,
        "end": 2.6,
        "channel": 5,
        "text": "Jó ember ő bízik",
        "flags": ["possible_duplicate_overlap"],
        "duplicate_of": 1,
    },
    {
        "id": 3,
        "start": 3.0,
        "end": 5.0,
        "channel": 5,
        "text": "Árvíztűrő tükörfúrógép",
        "flags": ["possible_crosstalk", "possible_duplicate_overlap"],
        "duplicate_of": None,
    },
    {"id": 4, "start": 6.0, "end": 7.0, "channel": 3, "text": "Igen.", "flags": ["weird"]},
]


def test_clip_init(clip):
    cfg = clip.config
    assert cfg.speakers == ["gm", "seat_1", "seat_3"]
    assert [t.channel for t in cfg.tracks] == [1, 3, 5]
    assert cfg.tracks[0].name == "Mesélő" and cfg.language == "hu"
    assert cfg.duration_s == 20.0
    assert (clip.folder / "clip.toml").read_bytes()[:3] != b"\xef\xbb\xbf"


def test_import_keeps_everything(clip, tmp_path):
    ref, stats = import_jsonl(clip, jsonl(tmp_path / "s.jsonl", RECORDS))
    assert stats.records == 4 and stats.utterances == 3 and stats.excluded == 1
    assert len(ref.utterances) + len(ref.excluded) == len(RECORDS)
    (bleed,) = ref.excluded
    assert bleed.utterance.speaker == "seat_3" and "duplicate of #1" in bleed.reason
    by_id = {u.id: u for u in ref.utterances}
    assert by_id["r003"].tags == ["overlap?", "dup?"]
    assert by_id["r003"].origin == "import:s.jsonl#3"
    assert stats.unknown_flags["weird"] == 1


def test_import_rejects_unknown_channel(clip, tmp_path):
    with pytest.raises(ValueError, match="channel 9"):
        import_jsonl(clip, jsonl(tmp_path / "s.jsonl", [dict(RECORDS[0], channel=9)]))


def test_label_files_utf8_without_bom_and_lossless_round_trip(clip, tmp_path):
    ref, _ = import_jsonl(clip, jsonl(tmp_path / "s.jsonl", RECORDS))
    paths = write_label_files(clip, ref)
    names = sorted(p.name for p in paths)
    assert names == ["_bleed.txt", "gm.txt", "seat_1.txt", "seat_3.txt"]
    for p in paths:
        assert not p.read_bytes().startswith(b"\xef\xbb\xbf")
    raw = (clip.labels_dir / "seat_3.txt").read_bytes()
    assert "Árvíztűrő tükörfúrógép".encode() in raw  # á, ű, ő as UTF-8
    errors = []
    rebuilt, validation = build_reference(clip, read_labels(paths, errors), ref)
    assert errors == [] and validation.ok
    assert [u.model_dump() for u in rebuilt.utterances] == [u.model_dump() for u in ref.utterances]
    assert [e.utterance.id for e in rebuilt.excluded] == ["r002"]


def test_edited_labels(clip, tmp_path):
    ref, _ = import_jsonl(clip, jsonl(tmp_path / "s.jsonl", RECORDS))
    # As exported by Audacity: one file, spectral-selection lines, edits applied.
    edited = "\n".join(
        [
            "0.480000\t2.510000\tgm | r001 Jó ember, ő bízik szerintem.",
            "\\\t0.000000\t0.000000",  # spectral line: skipped
            "3.000000\t5.000000\tgm | r003 [overlap] Árvíztűrő tükörfúrógép",  # speaker changed
            "6.000000\t7.000000\t_seat_1 | r004 Igen.",  # marked as bleed
            "8.000000\t9.000000\tseat_3 | Új mondat, ű és ő.",  # new, no id
            "10.0\t10.5\t_diff | seat_3 differs CER 0.40 r003",  # hint only
        ]
    )
    errors = []
    labels = parse_labels(edited, errors)
    assert errors == []
    out, validation = build_reference(clip, labels, ref)
    assert validation.ok, validation.errors
    by_id = {u.id: u for u in out.utterances}
    assert set(by_id) == {"r001", "r003", "r005"}
    assert by_id["r001"].text.endswith("szerintem.") and by_id["r001"].start == 0.48
    assert by_id["r003"].speaker == "gm" and by_id["r003"].tags == ["overlap"]
    assert by_id["r005"].origin == "manual" and by_id["r005"].text == "Új mondat, ű és ő."
    reasons = {e.utterance.id: e.reason for e in out.excluded}
    assert reasons["r004"] == "marked as not this speaker's speech"
    assert reasons["r002"].endswith("deleted while editing")  # bleed line removed in editing
    assert out.excluded and all(e.reason for e in out.excluded)

    path = clip.reference_path
    write_reference(path, out)
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert read_reference(path) == out


def test_parse_errors():
    errors = []
    parse_labels("1.0\t2.0\tno speaker prefix\nabc\tdef\tgm | x\nonly-one-field\n", errors)
    assert len(errors) == 3


def test_build_validation(clip):
    labels = parse_labels(
        "\n".join(
            [
                "1.0\t2.0\tgm | r001 a",
                "1.5\t3.0\tgm | r002 b",  # overlaps r001
                "4.0\t3.0\tseat_1 | r003 c",  # end before start
                "5.0\t6.0\tnobody | r004 d",  # unknown speaker
                "7.0\t8.0\tgm | r005 [funny] e",  # unknown tag
                "9.0\t10.0\tgm | r005 f",  # duplicate id
                "11.0\t12.0\tgm | r006",  # empty text
                "13.0\t14.0\tgm | r007 [inaudible]",  # empty but unscored: fine
                "15.0\t16.0\tgm | r008 [dup?] g",  # hint: warning, error if verified
                "19.0\t25.0\tgm | r009 h",  # past the 20 s clip
            ]
        ),
        [],
    )
    _, v = build_reference(clip, labels, None)
    text = "\n".join(v.errors)
    for needle in (
        "overlap each other",
        "end must be after start",
        "unknown speaker",
        "unknown tag [funny]",
        "appears twice",
        "empty text",
        "ends after the clip",
    ):
        assert needle in text
    assert "r007" not in text
    assert any("dup?" in w for w in v.warnings)
    _, strict = build_reference(clip, labels, None, verified=True)
    assert any("dup?" in e for e in strict.errors)


def test_verified_status(clip):
    labels = parse_labels("1.0\t2.0\tgm | r001 a", [])
    ref, v = build_reference(clip, labels, None, verified=True)
    assert v.ok and ref.status == "verified"
    assert validate_reference(ref, clip.config.speakers, 20.0, verified=True).ok


def test_normalize_and_cer():
    assert normalize("Jó  ember, ŐT [laugh] láttad?!") == "jó ember őt láttad"
    assert normalize("Bloodwell-lel, isn't — it") == "bloodwell-lel isn't it"
    assert normalize("Árvíz") == "árvíz"  # NFC
    assert cer("Árvíztűrő", "árvíztűrő!") == 0.0
    assert cer("", "") == 0.0 and cer("", "x") == 1.0
    assert 0 < cer("tükörfúrógép", "tükörfurógép") < 0.2


def u(uid, speaker, start, end, text):
    return Utterance(id=uid, speaker=speaker, start=start, end=end, text=text)


def test_diff_pairs_by_speaker_and_time():
    imported = Reference(
        clip="c",
        language="hu",
        utterances=[u("r001", "gm", 0, 2, "jó ember bízik"), u("r002", "gm", 5, 6, "igen")],
    )
    asr = Reference(
        clip="c",
        language="hu",
        utterances=[
            u("r001", "gm", 0.1, 1.0, "jó ember"),
            u("r002", "gm", 1.0, 2.1, "bízik"),
            u("r003", "seat_1", 0.0, 2.0, "jó ember bízik"),  # bleed on another track
            u("r004", "gm", 8, 9, "hallucináció"),
        ],
    )
    items = compare(imported, asr, ["gm", "seat_1"])
    kinds = [(d.kind, d.speaker, d.import_id, d.asr_ids) for d in items]
    assert ("match", "gm", "r001", ["r001", "r002"]) in kinds
    assert ("only_import", "gm", "r002", []) in kinds
    assert ("only_asr", "gm", None, ["r004"]) in kinds
    assert ("only_asr", "seat_1", None, ["r003"]) in kinds
    lines = diff_label_lines(items)
    assert len(lines) == 3 and all("\t_diff | " in line for line in lines)
