"""``sessionscribe bench``: reference clips, reference transcripts and benchmark runs.

Reference material is personal data. These commands print counts, ids and numbers only;
transcript text goes to files inside the git-ignored clip or work folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

bench_app = typer.Typer(
    help="Benchmarks and reference transcripts (dev tooling).", no_args_is_help=True
)
clip_app = typer.Typer(help="Reference clip folders.", no_args_is_help=True)
ref_app = typer.Typer(
    help="Build a hand-corrected reference transcript (Audacity round-trip).",
    no_args_is_help=True,
)
bench_app.add_typer(clip_app, name="clip")
bench_app.add_typer(ref_app, name="ref")

ClipArg = Annotated[
    Path, typer.Argument(help="Clip folder, e.g. bench/refs/clip01.", file_okay=False)
]


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(typer.style("[FAIL]", fg=typer.colors.RED, bold=True) + f" {message}", err=True)
    return typer.Exit(code)


def _load_clip(folder: Path):  # -> Clip
    from sessionscribe.core.bench.clip import Clip

    try:
        return Clip.load(folder)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


def _show_validation(validation) -> None:
    for w in validation.warnings:
        typer.echo(typer.style("[WARN]", fg=typer.colors.YELLOW, bold=True) + f" {w}")
    for e in validation.errors:
        typer.echo(typer.style("[FAIL]", fg=typer.colors.RED, bold=True) + f" {e}")


@clip_app.command("init")
def clip_init(
    folder: ClipArg,
    language: Annotated[str, typer.Option(help="Language of the clip (hu, en, ...).")],
    names_from: Annotated[
        Path | None,
        typer.Option(help="JSONL transcript whose 'player' field names each channel."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing clip.toml.")] = False,
) -> None:
    """Create clip.toml from tracks/chNN_<speaker>.wav file names."""
    from sessionscribe.core.bench.clip import CLIP_FILE, init_clip

    if (folder / CLIP_FILE).exists() and not force:
        raise _fail(f"{folder / CLIP_FILE} exists; use --force to overwrite")
    names: dict[int, str] = {}
    if names_from is not None:
        for line in names_from.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("player"):
                    names.setdefault(int(rec["channel"]), str(rec["player"]))
    try:
        config = init_clip(folder, language, names)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(
        f"Wrote {folder / CLIP_FILE}: {len(config.tracks)} tracks "
        f"({', '.join(config.speakers)}), {config.duration_s:.2f} s, language {config.language}"
    )


@ref_app.command("import")
def ref_import(
    folder: ClipArg,
    jsonl: Annotated[
        Path | None, typer.Option(help="Existing transcript (default: source/session.jsonl).")
    ] = None,
) -> None:
    """Import an existing JSONL transcript as draft/import.json; nothing is dropped."""
    from sessionscribe.core.bench.draft import import_jsonl
    from sessionscribe.core.bench.reference import write_reference

    clip = _load_clip(folder)
    path = jsonl or folder / "source" / "session.jsonl"
    try:
        reference, stats = import_jsonl(clip, path)
    except (OSError, ValueError, KeyError) as exc:
        raise _fail(f"import failed: {exc}") from exc
    out = clip.draft_dir / "import.json"
    write_reference(out, reference)
    hints = ", ".join(f"{k} {v}" for k, v in sorted(stats.hints.items())) or "none"
    typer.echo(
        f"Imported {stats.records} segments: {stats.utterances} utterances, "
        f"{stats.excluded} bleed candidates (excluded, on the _bleed label track); hints: {hints}"
    )
    if stats.unknown_flags:
        typer.echo(f"Ignored unknown flags: {dict(stats.unknown_flags)}")
    typer.echo(str(out))


@ref_app.command("draft")
def ref_draft(
    folder: ClipArg,
    preset: Annotated[str, typer.Option(help="Settings preset.")] = "conversation",
    model: Annotated[str | None, typer.Option(help="Model (default: preset's).")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="section.field=value")] = None,
) -> None:
    """Transcribe every track with our ASR into draft/asr.json (for comparison)."""
    from sessionscribe.core.audio.reader import load_for_asr
    from sessionscribe.core.bench.draft import reference_from_asr
    from sessionscribe.core.bench.reference import write_reference
    from sessionscribe.core.config import load_settings
    from sessionscribe.core.exporters.single import Speaker, write_outputs
    from sessionscribe.core.pipeline.asr import AsrEngine, AsrError

    clip = _load_clip(folder)
    cli = {"decoding": {"language": clip.config.language}}
    if model:
        cli["model"] = {"id": model}
    try:
        settings = load_settings(preset=preset, overrides=set_ or (), cli=cli)
        engine = AsrEngine.load(settings)
    except (ValueError, AsrError) as exc:
        raise _fail(str(exc)) from exc
    results = {}
    for track in clip.config.tracks:
        audio, _ = load_for_asr(clip.track_path(track), max_duration_s=None)
        result = engine.transcribe(audio, source_id=track.speaker)
        results[track.speaker] = result
        who = Speaker(track.speaker, track.speaker, Path(track.file).name)
        write_outputs(result, settings, who, clip.draft_dir / "asr", track.speaker, ("json",))
        typer.echo(f"  {track.speaker:10} {len(result.segments):3d} segments, RTF {result.rtf:.3f}")
    out = clip.draft_dir / "asr.json"
    write_reference(out, reference_from_asr(clip, results))
    typer.echo(f"{out} ({settings.model.id}, {settings.asr.mode} mode)")


@ref_app.command("diff")
def ref_diff(folder: ClipArg) -> None:
    """Compare draft/import.json with draft/asr.json; writes draft/diff.md and labels/_diff.txt."""
    from sessionscribe.core.bench.diff import (
        compare,
        diff_ids,
        diff_label_lines,
        diff_markdown,
    )
    from sessionscribe.core.bench.labels import DIFF_TRACK
    from sessionscribe.core.bench.reference import read_reference
    from sessionscribe.core.fileio import atomic_write_text

    clip = _load_clip(folder)
    try:
        imported = read_reference(clip.draft_dir / "import.json")
        asr = read_reference(clip.draft_dir / "asr.json")
    except OSError as exc:
        raise _fail(f"{exc}; run 'bench ref import' and 'bench ref draft' first") from exc
    items = compare(imported, asr, clip.config.speakers)
    atomic_write_text(clip.draft_dir / "diff.md", diff_markdown(items))
    lines = diff_label_lines(items)
    atomic_write_text(clip.labels_dir / f"{DIFF_TRACK}.txt", "".join(f"{x}\n" for x in lines))
    counts: dict[str, int] = {}
    for d in items:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    typer.echo(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for d in items:
        if d.kind != "match":
            rate = "" if d.cer is None else f" CER {d.cer:.2f}"
            typer.echo(f"  {d.start:7.2f} {d.speaker:10} {d.kind:12}{rate:9} {diff_ids(d)}")
    typer.echo(f"{clip.draft_dir / 'diff.md'}\n{clip.labels_dir / (DIFF_TRACK + '.txt')}")


@ref_app.command("labels")
def ref_labels(
    folder: ClipArg,
    source: Annotated[
        Path | None,
        typer.Option("--from", help="Reference to export (default: reference.json, else draft)."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite existing label files.")] = False,
) -> None:
    """Write Audacity label files: one per speaker plus _bleed (UTF-8, no BOM)."""
    from sessionscribe.core.bench.labels import write_label_files
    from sessionscribe.core.bench.reference import read_reference

    clip = _load_clip(folder)
    path = source or (
        clip.reference_path if clip.reference_path.exists() else clip.draft_dir / "import.json"
    )
    existing = [p for p in clip.config.speakers if (clip.labels_dir / f"{p}.txt").exists()]
    if existing and not force:
        raise _fail(f"label files exist in {clip.labels_dir}; use --force to overwrite")
    try:
        reference = read_reference(path)
    except OSError as exc:
        raise _fail(str(exc)) from exc
    written = write_label_files(clip, reference)
    typer.echo(f"From {path}: {len(written)} label files in {clip.labels_dir}")
    typer.echo("Next: open the tracks and label files in Audacity (docs/REFERENCE_GUIDE.md).")


@ref_app.command("build")
def ref_build(
    folder: ClipArg,
    labels: Annotated[
        list[Path], typer.Option("--labels", help="Exported label file(s).", exists=True)
    ],
    base: Annotated[
        Path | None,
        typer.Option(help="Reference the labels came from (default: reference.json or draft)."),
    ] = None,
    verified: Annotated[
        bool, typer.Option(help="Mark as verified (all passages listened to).")
    ] = False,
) -> None:
    """Build reference.json from edited labels and validate it."""
    from sessionscribe.core.bench.labels import build_reference, read_labels
    from sessionscribe.core.bench.reference import read_reference, write_reference

    clip = _load_clip(folder)
    base_path = base or (
        clip.reference_path if clip.reference_path.exists() else clip.draft_dir / "import.json"
    )
    base_ref = read_reference(base_path) if base_path.exists() else None
    errors: list[str] = []
    parsed = read_labels(labels, errors)
    reference, validation = build_reference(
        clip,
        parsed,
        base_ref,
        verified=verified,
        source=", ".join(p.name for p in labels),
    )
    validation.errors = errors + validation.errors
    _show_validation(validation)
    if not validation.ok:
        raise _fail(f"{len(validation.errors)} error(s); reference.json not written")
    write_reference(clip.reference_path, reference)
    typer.echo(
        f"Wrote {clip.reference_path} ({reference.status}): {len(reference.utterances)} "
        f"utterances, {len(reference.excluded)} excluded"
    )


@ref_app.command("check")
def ref_check(
    folder: ClipArg,
    verified: Annotated[bool, typer.Option(help="Apply the verified rules.")] = False,
) -> None:
    """Validate reference.json."""
    from sessionscribe.core.bench.reference import read_reference, validate_reference

    clip = _load_clip(folder)
    try:
        reference = read_reference(clip.reference_path)
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    validation = validate_reference(
        reference,
        clip.config.speakers,
        clip.config.duration_s,
        verified=verified or reference.status == "verified",
    )
    _show_validation(validation)
    if not validation.ok:
        raise _fail(f"{len(validation.errors)} error(s)")
    typer.echo(
        f"[ OK ] {clip.reference_path} ({reference.status}): {len(reference.utterances)} "
        f"utterances, {len(reference.excluded)} excluded"
    )
