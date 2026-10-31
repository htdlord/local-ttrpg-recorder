"""Window preferences (QSettings) carried over from the name before the rename."""

from __future__ import annotations

import logging

from PySide6.QtCore import QSettings

logger = logging.getLogger(__name__)

LEGACY_ORG = "SessionScribe"  # the project's name until 2026-10-31


def migrate_prefs(new: QSettings, old: QSettings) -> bool:
    """Copy every key of ``old`` into an empty ``new`` (theme, language, recent sessions,
    recording folder and rig). ``old`` is left as it is; returns whether anything was copied."""
    keys = old.allKeys()
    if new.allKeys() or not keys:
        return False
    for key in keys:
        new.setValue(key, old.value(key))
    new.sync()
    logger.info("copied %d window preference(s) from %s", len(keys), LEGACY_ORG)
    return True
