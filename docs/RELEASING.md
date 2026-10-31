# Releasing

Releases are git tags with a GitHub release; users install from the tag
([INSTALL.md](INSTALL.md)). Nothing is published to PyPI.

## Gate for v0.1.0

v0.1.0 is tagged only after the defaults are confirmed on real data (owner decision,
2026-10-31):

- [ ] First real session recorded with the Record tab or `record start`; `record verify`
      clean (or every gap explained).
- [ ] clip02 cut from it (`bench clip cut`, [SESSION_CHECKLIST_HU.md](SESSION_CHECKLIST_HU.md))
      and triaged (`bench ref triage`).
- [ ] M1 benchmark repeated on the ≥ 5 min clip; default ASR mode confirmed (SPEC §15).
- [ ] Bleed defaults (ownership margin, attenuation, dedup) and the M3 tolerances fixed
      from clip01 + clip02 and written to DECISIONS; `processing_version` bumped if
      outputs change.
- [ ] Open M5 owner checks done or listed as known limits in SUPPORTED.md (mic check with
      everyone speaking, unplug test, quiet-channel `record check`, WASAPI `audio_hw`).

## Steps

1. Fast suite green in CI; locally also `uv run pytest -m "gpu or slow or audio_hw"`.
2. `uv run python scripts/clean_env_check.py` passes (and the `--no-cuda-extra` control
   still fails as expected); note the result in DECISIONS if anything changed.
3. `uv sync --locked --extra gui --extra cuda`, then
   `uv run python scripts/third_party_notices.py`; commit `THIRD_PARTY_NOTICES.md`.
4. Update [SUPPORTED.md](SUPPORTED.md) and the README screenshots if the UI changed
   (demo session only, never real recordings).
5. Set `version` in `pyproject.toml` (`__version__` reads it from the package metadata);
   `uv lock`; move the "Unreleased" section of [CHANGELOG.md](../CHANGELOG.md) under the
   version and date.
6. PR, review, merge to `main`.
7. `git tag -a v0.1.0 -m "SessionScribe 0.1.0"` on the merge commit, `git push --tags`.
8. The `Install` workflow runs on the tag: green on Windows and Ubuntu.
9. GitHub release from the tag with the CHANGELOG section as notes.
10. Follow [INSTALL.md](INSTALL.md) once on a second machine or a fresh folder.

## Not in v0.1.0

- One-folder build (PyInstaller or similar): optional, moved to SPEC §14. Needs a child
  process dispatcher first, because the recorder, the device list and the inference check
  start `python -m ...` with `sys.executable`, which is the app itself in a frozen build;
  the ASR worker uses `multiprocessing` with `spawn` and would need `freeze_support()`.
