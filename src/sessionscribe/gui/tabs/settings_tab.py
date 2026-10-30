"""Settings tab: preset, Basic fields and a generated, mode-aware Advanced form."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sessionscribe.core.config import list_presets, user_presets_dir, write_settings
from sessionscribe.gui.models import settings_form as sf
from sessionscribe.gui.models.colors import Tokens
from sessionscribe.gui.state import AppState, preset_settings
from sessionscribe.gui.widgets import button, label


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()


class FieldEditor(QWidget):
    """One setting: a control matching its kind, plus a muted mode note."""

    def __init__(self, spec: sf.FieldSpec, on_change: Any) -> None:
        super().__init__()
        self.spec = spec
        self.on_change = on_change
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.control: QWidget
        self.boxes: dict[str, QCheckBox] = {}
        if spec.kind == "bool":
            self.control = QCheckBox()
            self.control.toggled.connect(lambda v: self.on_change(spec, v))
        elif spec.kind == "choice":
            combo = QComboBox()
            combo.setEditable(spec.editable_choices)
            combo.addItems(list(spec.choices))
            if spec.editable_choices:
                combo.lineEdit().editingFinished.connect(  # type: ignore[union-attr]
                    lambda: self.on_change(spec, combo.currentText().strip())
                )
            combo.activated.connect(lambda _i: self.on_change(spec, combo.currentText()))
            self.control = combo
        elif spec.kind == "multichoice":
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            for choice in spec.choices:
                box = QCheckBox(choice)
                box.toggled.connect(lambda _v: self.on_change(spec, self.selected()))
                self.boxes[choice] = box
                rl.addWidget(box)
            rl.addStretch(1)
            self.control = row
        else:
            edit = QLineEdit()
            if spec.optional:
                edit.setPlaceholderText(self.tr("automatic"))
            edit.editingFinished.connect(lambda: self._text_changed(edit))
            self.control = edit
        if spec.description:
            self.control.setToolTip(spec.description)
        layout.addWidget(self.control, 1)
        self.note = label(muted=True)
        layout.addWidget(self.note)

    def _mode_notes(self) -> dict[str, str]:
        # Spelled out so the translation tools see each string.
        return {
            "no effect in this mode": self.tr("no effect in this mode"),
            "only the first value is used in this mode": self.tr(
                "only the first value is used in this mode"
            ),
            "reported in the output, does not filter in this mode": self.tr(
                "reported in the output, does not filter in this mode"
            ),
        }

    def selected(self) -> list[str]:
        return [c for c, box in self.boxes.items() if box.isChecked()]

    def _text_changed(self, edit: QLineEdit) -> None:
        try:
            value = sf.parse_text(self.spec, edit.text())
        except ValueError:
            self.show_error(self.tr("not a valid value"))
            return
        self.on_change(self.spec, value)

    def show_error(self, message: str) -> None:
        self.note.setProperty("error", True)
        self.note.setText(message)
        self.note.style().unpolish(self.note)
        self.note.style().polish(self.note)

    def set_value(self, value: Any, mode: str) -> None:
        self.control.blockSignals(True)
        for box in self.boxes.values():
            box.blockSignals(True)
        if isinstance(self.control, QCheckBox):
            self.control.setChecked(bool(value))
        elif isinstance(self.control, QComboBox):
            if self.control.findText(str(value)) < 0:
                self.control.addItem(str(value))
            self.control.setCurrentText(str(value))
        elif isinstance(self.control, QLineEdit):
            self.control.setText(sf.format_value(self.spec, value))
        else:
            for choice, box in self.boxes.items():
                box.setChecked(choice in (value or []))
        self.control.blockSignals(False)
        for box in self.boxes.values():
            box.blockSignals(False)
        note = sf.mode_note(self.spec, mode)
        self.note.setProperty("error", False)
        self.note.setText(self._mode_notes().get(note, note) if note else "")
        self.note.style().unpolish(self.note)
        self.note.style().polish(self.note)
        self.control.setEnabled(not sf.is_disabled(self.spec, mode))


class SettingsTab(QWidget):
    def __init__(self, state: AppState, tokens: Tokens) -> None:
        super().__init__()
        self.state, self.tokens = state, tokens
        self.editors: dict[str, FieldEditor] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("Preset")))
        self.preset = QComboBox()
        self.preset.addItems(list(list_presets()))
        top.addWidget(self.preset)
        top.addStretch(1)
        self.reset_btn = button(self.tr("Reset to preset"), tokens, "undo")
        self.save_preset_btn = button(self.tr("Save as preset…"), tokens, "save")
        self.save_btn = button(self.tr("Save to session"), tokens, "save", primary=True)
        for b in (self.reset_btn, self.save_preset_btn, self.save_btn):
            top.addWidget(b)
        outer.addLayout(top)
        self.status = label(muted=True)
        outer.addWidget(self.status)

        basic = QGroupBox(self.tr("Basic"))
        bf = QFormLayout(basic)
        bf.setVerticalSpacing(8)
        basic_titles = {
            "model.id": self.tr("Model"),
            "decoding.language": self.tr("Language"),
            "asr.mode": self.tr("ASR mode"),
            "bleed.mode": self.tr("Bleed handling"),
            "output.formats": self.tr("Output formats"),
            "output.timestamp_style": self.tr("Timestamps"),
        }
        for spec in sf.basic_specs():
            bf.addRow(basic_titles[spec.path], self._editor(spec, key=f"basic:{spec.path}"))
        outer.addWidget(basic)

        self.advanced_toggle = QCheckBox(self.tr("Show advanced settings"))
        outer.addWidget(self.advanced_toggle)
        self.advanced = QTabWidget()
        section_titles = {
            "model": self.tr("Model"),
            "asr": self.tr("ASR"),
            "decoding": self.tr("Decoding"),
            "vocabulary": self.tr("Vocabulary"),
            "vad": self.tr("VAD"),
            "bleed": self.tr("Bleed"),
            "dedup": self.tr("Dedup"),
            "quality": self.tr("Quality flags"),
            "output": self.tr("Output"),
        }
        by_section: dict[str, list[sf.FieldSpec]] = {}
        for spec in sf.field_specs():
            by_section.setdefault(spec.section, []).append(spec)
        for section, specs in by_section.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setVerticalSpacing(8)
            for spec in specs:
                form.addRow(_title(spec.name), self._editor(spec, key=spec.path))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.advanced.addTab(scroll, section_titles.get(section, _title(section)))
        self.advanced.setVisible(False)
        outer.addWidget(self.advanced, 1)
        outer.addStretch(0)

        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        self.preset.activated.connect(lambda _i: self.reset_to_preset())
        self.reset_btn.clicked.connect(self.reset_to_preset)
        self.save_btn.clicked.connect(self.save_to_session)
        self.save_preset_btn.clicked.connect(lambda: self.save_as_preset())  # not checked
        state.settings_changed.connect(self.refresh)
        state.session_changed.connect(self._session_changed)
        self._session_changed()

    def _editor(self, spec: sf.FieldSpec, key: str) -> FieldEditor:
        editor = FieldEditor(spec, self._changed)
        self.editors[key] = editor
        return editor

    def _changed(self, spec: sf.FieldSpec, value: Any) -> None:
        try:
            settings = sf.with_value(self.state.settings, spec.path, value)
        except ValidationError as exc:
            msg = exc.errors()[0]["msg"]
            for key, editor in self.editors.items():
                if key.endswith(spec.path):
                    editor.show_error(msg)
            return
        self.state.set_settings(settings)

    def _session_changed(self) -> None:
        session = self.state.session
        self.save_btn.setEnabled(session is not None)
        preset = (session.config.preset if session else None) or "conversation"
        self.preset.setCurrentText(preset)
        self.refresh()

    def refresh(self) -> None:
        settings = self.state.settings
        mode = settings.asr.mode
        for editor in self.editors.values():
            editor.set_value(sf.get_value(settings, editor.spec.path), mode)
        diff = sf.settings_diff(preset_settings(self.preset.currentText()), settings)
        changed = sf.changed_paths(diff)
        self.status.setText(
            self.tr("{0} setting(s) differ from the preset: {1}").format(
                len(changed), ", ".join(changed)
            )
            if changed
            else self.tr("Same as the preset.")
        )

    def reset_to_preset(self) -> None:
        self.state.set_settings(preset_settings(self.preset.currentText()))

    def save_to_session(self) -> None:
        session = self.state.session
        if session is None:
            return
        preset = self.preset.currentText()
        session.config.preset = None if preset == "conversation" else preset
        session.config.settings = sf.settings_diff(preset_settings(preset), self.state.settings)
        self.state.save_session()
        self.status.setText(self.tr("Saved to {0}").format(session.folder / "session.toml"))

    def save_as_preset(self, path: Path | None = None) -> None:
        if path is None:
            folder = user_presets_dir()
            folder.mkdir(parents=True, exist_ok=True)
            chosen, _ = QFileDialog.getSaveFileName(
                self, self.tr("Save preset"), str(folder / "my-preset.toml"), "*.toml"
            )
            if not chosen:
                return
            path = Path(chosen)
        write_settings(self.state.settings, path)
        self.status.setText(self.tr("Saved to {0}").format(path))
