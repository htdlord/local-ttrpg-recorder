"""Command names from before the rename (SessionScribe), kept as deprecated aliases.

They work through v0.1.0 and are removed in the first release after it.
"""

from __future__ import annotations

import sys

ALIASES = {"sessionscribe": "ttrpg", "sessionscribe-gui": "ttrpg-gui"}
REMOVED_AFTER = "v0.1.0"


def warning(old: str) -> str:
    return (
        f"warning: '{old}' is deprecated and will be removed after {REMOVED_AFTER}; "
        f"use '{ALIASES[old]}' (the project is now Local TTRPG Recorder)"
    )


def old_cli() -> None:
    """``sessionscribe``: same as ``ttrpg``, with a warning on stderr."""
    print(warning("sessionscribe"), file=sys.stderr)
    from ttrpg_recorder.cli import app

    app()


def old_gui() -> int:
    """``sessionscribe-gui``: same as ``ttrpg-gui``, with a warning on stderr and in the
    window's status bar."""
    print(warning("sessionscribe-gui"), file=sys.stderr)
    from ttrpg_recorder.gui.app import main

    return main(deprecated_alias="sessionscribe-gui")
