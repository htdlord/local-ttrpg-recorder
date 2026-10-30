"""Shared GUI state: the open session, its resolved settings and the latest result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from sessionscribe.core.config import DEFAULT_PRESET, Settings, load_settings
from sessionscribe.core.result import SessionResult
from sessionscribe.core.session import Session, SessionConfig, write_session


def preset_settings(preset: str | None) -> Settings:
    """The base the Advanced form diffs against: the preset alone."""
    return load_settings(preset=preset or DEFAULT_PRESET)


def session_settings(config: SessionConfig) -> Settings:
    return load_settings(
        preset=None if config.preset else DEFAULT_PRESET,
        session=config.settings,
        session_preset=config.preset,
    )


class AppState(QObject):
    session_changed = Signal()
    settings_changed = Signal()
    result_changed = Signal()

    def __init__(self, process_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.session: Session | None = None
        self.settings: Settings = load_settings()
        self.result: SessionResult | None = None
        # Tests inject fake engines here (isolation, factory, model_identity).
        self.process_kwargs: dict[str, Any] = dict(process_kwargs or {})

    def set_session(self, session: Session | None) -> None:
        self.session = session
        self.result = None
        self.settings = session_settings(session.config) if session else load_settings()
        self.session_changed.emit()
        self.settings_changed.emit()
        self.result_changed.emit()

    def set_settings(self, settings: Settings) -> None:
        self.settings = settings
        self.settings_changed.emit()

    def set_result(self, result: SessionResult | None) -> None:
        self.result = result
        self.result_changed.emit()

    def save_session(self) -> Path:
        assert self.session is not None
        path = self.session.folder / "session.toml"
        write_session(path, self.session.config)
        return path

    def output_dir(self) -> Path:
        assert self.session is not None
        configured = self.settings.output.output_dir
        if configured is None:
            return self.session.folder / "output"
        return configured if configured.is_absolute() else self.session.folder / configured
