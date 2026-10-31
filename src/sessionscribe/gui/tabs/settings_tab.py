"""Settings tab: preset, Basic fields and a generated, mode-aware Advanced form."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt
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


def _field_title(title: str, key: str) -> QWidget:
    """The translated name with its TOML key underneath (small, muted)."""
    w = QWidget()
    layout = QVBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(QLabel(title))
    caption = QLabel(key)
    caption.setProperty("caption", True)
    caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(caption)
    return w


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
            form.setVerticalSpacing(12)  # each title has its TOML key under it
            titles = self.field_titles()
            for spec in specs:
                form.addRow(
                    _field_title(titles.get(spec.path, _title(spec.name)), spec.path),
                    self._editor(spec, key=spec.path),
                )
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

    def field_titles(self) -> dict[str, str]:
        """Translated names of the settings (the TOML key is shown under each)."""
        return {
            "model.id": self.tr("Model"),
            "model.revision": self.tr("Model revision"),
            "model.device": self.tr("Device"),
            "model.compute_type": self.tr("Compute type"),
            "model.cpu_threads": self.tr("CPU threads"),
            "model.num_workers": self.tr("Parallel workers"),
            "model.model_dir": self.tr("Model folder"),
            "model.offline": self.tr("Offline only"),
            "asr.mode": self.tr("ASR mode"),
            "asr.batch_size": self.tr("Batch size"),
            "asr.oom_fallback": self.tr("Smaller batches on GPU out-of-memory"),
            "asr.chunk_owned_s": self.tr("Chunk length (s)"),
            "asr.chunk_context_s": self.tr("Chunk context (s)"),
            "asr.chunk_boundary_search_s": self.tr("Chunk boundary search (s)"),
            "asr.refill_holes": self.tr("Re-transcribe speech without words"),
            "asr.refill_min_s": self.tr("Shortest refill (s)"),
            "asr.refill_pad_s": self.tr("Refill padding (s)"),
            "decoding.language": self.tr("Language"),
            "decoding.task": self.tr("Task"),
            "decoding.beam_size": self.tr("Beam size"),
            "decoding.best_of": self.tr("Best of"),
            "decoding.temperature": self.tr("Temperatures"),
            "decoding.condition_on_previous_text": self.tr("Condition on previous text"),
            "decoding.no_speech_threshold": self.tr("No-speech threshold"),
            "decoding.log_prob_threshold": self.tr("Log probability threshold"),
            "decoding.compression_ratio_threshold": self.tr("Compression ratio threshold"),
            "decoding.hallucination_silence_threshold": self.tr(
                "Hallucination silence threshold (s)"
            ),
            "decoding.word_timestamps": self.tr("Word timestamps"),
            "vocabulary.initial_prompt": self.tr("Initial prompt"),
            "vocabulary.hotwords": self.tr("Hotwords"),
            "vad.enabled": self.tr("Voice activity detection"),
            "vad.threshold": self.tr("Speech threshold"),
            "vad.min_speech_duration_ms": self.tr("Shortest speech (ms)"),
            "vad.min_silence_duration_ms": self.tr("Shortest silence (ms)"),
            "vad.speech_pad_ms": self.tr("Speech padding (ms)"),
            "vad.max_speech_duration_s": self.tr("Longest speech (s)"),
            "bleed.mode": self.tr("Bleed handling"),
            "bleed.coupling": self.tr("Coupling"),
            "bleed.ownership_margin_db": self.tr("Ownership margin (dB)"),
            "bleed.pre_attenuation_db": self.tr("Pre-ASR attenuation (dB)"),
            "bleed.mute": self.tr("Mute instead of attenuate"),
            "bleed.frame_ms": self.tr("Frame length (ms)"),
            "bleed.hold_ms": self.tr("Ownership hold (ms)"),
            "bleed.release_ms": self.tr("Release (ms)"),
            "bleed.floor_above_noise_db": self.tr("Floor above noise (dB)"),
            "bleed.suppress_below": self.tr("Suppress below ownership"),
            "bleed.low_ownership_below": self.tr("Flag below ownership"),
            "bleed.text_similarity": self.tr("Text similarity (%)"),
            "bleed.silent_mic_range_db": self.tr("Silent mic range (dB)"),
            "bleed.coupling_min_frames": self.tr("Coupling: minimum frames"),
            "bleed.coupling_default_db": self.tr("Coupling: default (dB)"),
            "dedup.enabled": self.tr("Remove duplicates"),
            "dedup.similarity": self.tr("Text similarity (%)"),
            "dedup.max_time_gap_s": self.tr("Largest time gap (s)"),
            "dedup.min_words": self.tr("Fewest words"),
            "dedup.mark_overlaps": self.tr("Mark overlaps"),
            "quality.enabled": self.tr("Quality flags"),
            "quality.low_logprob": self.tr("Low log probability"),
            "quality.high_no_speech": self.tr("High no-speech probability"),
            "quality.high_no_speech_logprob": self.tr("… with log probability below"),
            "quality.high_compression": self.tr("High compression ratio"),
            "quality.repeated_ngram": self.tr("Repeated n-gram length"),
            "quality.repeated_ngram_times": self.tr("Repetitions that count as a loop"),
            "quality.max_chars_per_s": self.tr("Most characters per second"),
            "quality.min_vad_fraction": self.tr("Least VAD speech share"),
            "output.formats": self.tr("Output formats"),
            "output.timestamp_style": self.tr("Timestamps"),
            "output.merge_same_speaker_gap_s": self.tr("Merge same speaker within (s)"),
            "output.filename_template": self.tr("File name template"),
            "output.output_dir": self.tr("Output folder"),
            "output.work_dir": self.tr("Work folder"),
        }

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
