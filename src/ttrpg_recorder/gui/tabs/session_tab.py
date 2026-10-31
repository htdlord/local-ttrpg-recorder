"""Session tab: create a session from a folder or open one; edit participants."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ttrpg_recorder.core.session import SESSION_FILE, Session, SessionConfig, suggest_session
from ttrpg_recorder.gui.models import participants as pm
from ttrpg_recorder.gui.models.colors import Tokens
from ttrpg_recorder.gui.state import AppState
from ttrpg_recorder.gui.widgets import EmptyState, button, label


class SessionTab(QWidget):
    saved = Signal()

    def __init__(self, state: AppState, tokens: Tokens) -> None:
        super().__init__()
        self.state, self.tokens = state, tokens
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        bar = QHBoxLayout()
        self.new_btn = button(self.tr("New from folder…"), tokens, "add")
        self.open_btn = button(self.tr("Open session…"), tokens, "open")
        self.save_btn = button(self.tr("Save"), tokens, "save", primary=True)
        bar.addWidget(self.new_btn)
        bar.addWidget(self.open_btn)
        bar.addStretch(1)
        bar.addWidget(self.save_btn)
        outer.addLayout(bar)

        self.stack = QStackedWidget()
        self.empty = EmptyState(
            tokens,
            "session",
            self.tr("No session open"),
            self.tr(
                "Create one from a folder with one file (or segment folder) per "
                "participant, or open an existing session.toml."
            ),
        )
        self.stack.addWidget(self.empty)

        editor = QWidget()
        ed = QVBoxLayout(editor)
        ed.setContentsMargins(0, 0, 0, 0)
        ed.setSpacing(16)
        info = QGroupBox(self.tr("Session"))
        form = QFormLayout(info)
        self.name = QLineEdit()
        self.date = QLineEdit()
        self.date.setPlaceholderText("YYYY-MM-DD")
        self.folder = label(muted=True)
        form.addRow(self.tr("Name"), self.name)
        form.addRow(self.tr("Date"), self.date)
        form.addRow(self.tr("Folder"), self.folder)
        ed.addWidget(info)

        box = QGroupBox(self.tr("Participants"))
        bl = QVBoxLayout(box)
        self.table = QTableWidget(0, len(pm.COLUMNS))
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Id"),
                self.tr("Name"),
                self.tr("Source"),
                self.tr("Channel"),
                self.tr("Offset (s)"),
                self.tr("Colour"),
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        bl.addWidget(self.table)
        rows = QHBoxLayout()
        self.add_btn = button(self.tr("Add participant"), tokens, "add")
        self.remove_btn = button(self.tr("Remove selected"), tokens, "remove")
        rows.addWidget(self.add_btn)
        rows.addWidget(self.remove_btn)
        rows.addStretch(1)
        bl.addLayout(rows)
        self.errors = label()
        self.errors.setProperty("error", True)
        bl.addWidget(self.errors)
        ed.addWidget(box, 1)

        self.recording = QGroupBox(self.tr("Recording"))
        rl = QVBoxLayout(self.recording)
        self.recording_info = label(muted=True)
        rl.addWidget(self.recording_info)
        ed.addWidget(self.recording)
        self.stack.addWidget(editor)
        outer.addWidget(self.stack, 1)

        # clicked(bool) would fill the optional path argument with ``checked``
        self.new_btn.clicked.connect(lambda: self.new_from_folder())
        self.open_btn.clicked.connect(lambda: self.open_session())
        self.save_btn.clicked.connect(self.save)
        self.add_btn.clicked.connect(lambda: self.table.insertRow(self.table.rowCount()))
        self.remove_btn.clicked.connect(self._remove_selected)
        state.session_changed.connect(self.refresh)
        self.refresh()

    # --- actions ---

    def new_from_folder(self, folder: Path | None = None) -> None:
        if folder is None:
            chosen = QFileDialog.getExistingDirectory(self, self.tr("Session folder"))
            if not chosen:
                return
            folder = Path(chosen)
        try:
            config = suggest_session(folder)
        except (OSError, ValueError) as exc:
            self.errors.setText(str(exc))
            return
        self.state.set_session(Session(folder, config))

    def open_session(self, path: Path | None = None) -> None:
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, self.tr("Open session"), "", f"{SESSION_FILE} (*.toml)"
            )
            if not chosen:
                return
            path = Path(chosen)
        try:
            self.state.set_session(Session.load(path))
        except (OSError, ValueError) as exc:
            self.errors.setText(str(exc))

    def rows(self) -> list[pm.ParticipantRow]:
        out = []
        for r in range(self.table.rowCount()):
            values = []
            for c in range(len(pm.COLUMNS)):
                item = self.table.item(r, c)
                values.append(item.text() if item is not None else "")
            out.append(pm.ParticipantRow(*values))
        return out

    def save(self) -> bool:
        session = self.state.session
        if session is None:
            return False
        base = session.config.model_copy(
            update={"name": self.name.text().strip(), "date": self.date.text().strip() or None}
        )
        result = pm.validate(base, self.rows())
        self._show_errors(result)
        if not result.ok or result.config is None:
            return False
        session.config = result.config
        path = self.state.save_session()
        self.state.set_session(Session.load(path))
        self.saved.emit()
        return True

    # --- view ---

    def _remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def _show_errors(self, result: pm.Validation) -> None:
        lines = list(result.errors)
        for row, errs in sorted(result.row_errors.items()):
            lines += [self.tr("Row {0}: {1}").format(row + 1, e) for e in errs]
        self.errors.setText("\n".join(lines))

    def refresh(self) -> None:
        session = self.state.session
        self.save_btn.setEnabled(session is not None)
        if session is None:
            self.stack.setCurrentIndex(0)
            return
        self.stack.setCurrentIndex(1)
        c: SessionConfig = session.config
        self.name.setText(c.name or session.folder.name)
        self.date.setText(c.date or "")
        self.folder.setText(str(session.folder))
        self.table.setRowCount(0)
        for p in c.participants:
            row = pm.ParticipantRow.from_participant(p)
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, value in enumerate(
                (row.id, row.name, row.source, row.channel, row.offset_s, row.color)
            ):
                self.table.setItem(r, col, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.errors.setText("")
        rec = c.recording
        self.recording.setVisible(rec is not None)
        if rec is not None:
            minutes = rec.metrics.frames / rec.samplerate / 60
            self.recording_info.setText(
                self.tr(
                    "{0} ({1}), {2} channels, {3:.1f} min, state {4}; {5} gap(s), {6} alarm(s)"
                ).format(
                    rec.device_name,
                    rec.host_api,
                    len(rec.channels),
                    minutes,
                    rec.state,
                    len(rec.gaps),
                    len(rec.alarms),
                )
            )
