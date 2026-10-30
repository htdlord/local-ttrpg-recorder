"""Result tab: transcript with filters, details, playback and manual reassignment."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QSplitter,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from sessionscribe.core.timeline import format_clock
from sessionscribe.gui import icons
from sessionscribe.gui.audio_preview import PreviewPlayer
from sessionscribe.gui.models import results as rm
from sessionscribe.gui.models.colors import Tokens, speaker_color
from sessionscribe.gui.state import AppState
from sessionscribe.gui.theme import transcript_font
from sessionscribe.gui.widgets import EmptyState, button, label

COLUMNS = ("time", "speaker", "text", "flags")
FLAG_ICONS = {
    "possible_wrong_speaker": "review",
    "possible_duplicate": "duplicate",
    "low_ownership": "review",
    "possible_missed_speech": "missed",
    "reassigned": "reassigned",
    "refilled": "review",
}
KIND_ICONS = {"suppressed": "suppressed", "missed": "missed", "gap": "gap"}

AnyIndex = QModelIndex | QPersistentModelIndex


class ResultModel(QAbstractTableModel):
    def __init__(self, tokens: Tokens) -> None:
        super().__init__()
        self.tokens = tokens
        self.rows: list[rm.ResultRow] = []
        self.names: dict[str, str] = {}
        self.colors: dict[str, str] = {}
        self.font = transcript_font()
        self.bold = QFont(self.font)
        self.bold.setBold(True)
        self.headers = ["", "", "", ""]

    def set_rows(self, rows: list[rm.ResultRow], names: dict[str, str], colors: dict[str, str]):
        self.beginResetModel()
        self.rows, self.names, self.colors = rows, names, colors
        self.endResetModel()

    def rowCount(self, parent: AnyIndex = QModelIndex()) -> int:  # noqa: B008
        return len(self.rows)

    def columnCount(self, parent: AnyIndex = QModelIndex()) -> int:  # noqa: B008
        return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = 0) -> Any:
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def data(self, index: AnyIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        row = self.rows[index.row()]
        col = COLUMNS[index.column()]
        muted = row.kind != "segment"
        if role == Qt.ItemDataRole.DisplayRole:
            if col == "time":
                return format_clock(row.start)
            if col == "speaker":
                return self.names.get(row.speaker or "", row.speaker or "")
            if col == "text":
                if row.kind == "gap":
                    return f"— {row.reason} ({row.end_s - row.start_s:.1f} s) —"
                if row.kind == "missed":
                    return f"({row.end_s - row.start_s:.1f} s)"
                return row.text
            return ""
        if role == Qt.ItemDataRole.FontRole:
            return self.bold if col == "speaker" else self.font
        if role == Qt.ItemDataRole.ForegroundRole:
            if col == "speaker" and row.speaker in self.colors:
                return QBrush(QColor(self.colors[row.speaker]))
            if col == "time" or muted:
                return QBrush(QColor(self.tokens.text_muted))
            return None
        if role == Qt.ItemDataRole.DecorationRole and col == "flags":
            name = KIND_ICONS.get(row.kind) or next(
                (FLAG_ICONS[f] for f in row.flags if f in FLAG_ICONS), None
            )
            if row.overlap and name is None:
                name = "overlap"
            if name:
                color = self.tokens.warning if row.needs_review else self.tokens.text_muted
                return icons.icon(name, color, 16)
            return None
        if role == Qt.ItemDataRole.ToolTipRole and col == "flags":
            parts = list(row.flags)
            if row.reason:
                parts.append(row.reason)
            if row.overlap:
                parts.append("overlap")
            return ", ".join(parts) or None
        if role == Qt.ItemDataRole.TextAlignmentRole and col == "time":
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


class ResultTab(QWidget):
    reassigned = Signal()  # the session file changed; outputs need a re-export

    def __init__(self, state: AppState, tokens: Tokens, player: PreviewPlayer | None = None):
        super().__init__()
        self.state, self.tokens = state, tokens
        self.player = player or PreviewPlayer(parent=self)
        self.all_rows: list[rm.ResultRow] = []
        self.filter = rm.ResultFilter()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        self.only_review = QCheckBox(self.tr("Needs review"))
        self.only_overlap = QCheckBox(self.tr("Overlap"))
        self.show_suppressed = QCheckBox(self.tr("Suppressed"))
        self.show_missed = QCheckBox(self.tr("Possible missed speech"))
        self.show_missed.setChecked(True)
        self.speaker = QComboBox()
        self.search = QLineEdit()
        self.search.setPlaceholderText(self.tr("Search the text …"))
        self.search.setClearButtonEnabled(True)
        for w in (self.only_review, self.only_overlap, self.show_suppressed, self.show_missed):
            bar.addWidget(w)
        bar.addWidget(self.speaker)
        bar.addWidget(self.search, 1)
        outer.addLayout(bar)
        self.counts = label(muted=True)
        outer.addWidget(self.counts)

        self.stack = QStackedWidget()
        self.stack.addWidget(
            EmptyState(
                tokens,
                "result",
                self.tr("No result yet"),
                self.tr(
                    "Transcribe the session on the Run tab, or open a session that "
                    "was transcribed before."
                ),
            )
        )
        split = QSplitter()
        self.model = ResultModel(tokens)
        self.model.headers = [self.tr("Time"), self.tr("Speaker"), self.tr("Text"), ""]
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        split.addWidget(self.table)

        details = QGroupBox(self.tr("Details"))
        dl = QVBoxLayout(details)
        form = QFormLayout()
        self.d_id = label(muted=True)
        self.d_kind = label()
        self.d_flags = label()
        self.d_other = label()
        self.d_ownership = label()
        form.addRow(self.tr("Segment"), self.d_id)
        form.addRow(self.tr("Kind"), self.d_kind)
        form.addRow(self.tr("Flags"), self.d_flags)
        form.addRow(self.tr("Other speaker"), self.d_other)
        form.addRow(self.tr("Ownership"), self.d_ownership)
        dl.addLayout(form)
        play = QHBoxLayout()
        self.play_btn = button(self.tr("Play"), tokens, "play")
        self.play_other_btn = button(self.tr("Play other track"), tokens, "play")
        self.stop_btn = button(self.tr("Stop"), tokens, "stop")
        for b in (self.play_btn, self.play_other_btn, self.stop_btn):
            play.addWidget(b)
        dl.addLayout(play)
        reassign = QGroupBox(self.tr("Reassign"))
        rl = QVBoxLayout(reassign)
        self.target = QComboBox()
        rl.addWidget(self.target)
        rb = QHBoxLayout()
        self.reassign_btn = button(
            self.tr("Move to participant"), tokens, "reassigned", primary=True
        )
        self.undo_btn = button(self.tr("Undo"), tokens, "undo")
        rb.addWidget(self.reassign_btn)
        rb.addWidget(self.undo_btn)
        rl.addLayout(rb)
        self.reassign_note = label(muted=True)
        rl.addWidget(self.reassign_note)
        dl.addWidget(reassign)
        dl.addStretch(1)
        split.addWidget(details)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        self.stack.addWidget(split)
        outer.addWidget(self.stack, 1)

        for box in (self.only_review, self.only_overlap, self.show_suppressed, self.show_missed):
            box.toggled.connect(self.apply_filter)
        self.speaker.activated.connect(lambda _i: self.apply_filter())
        self.search.textChanged.connect(lambda _t: self.apply_filter())
        self.table.selectionModel().currentRowChanged.connect(lambda *_: self._show_details())
        self.play_btn.clicked.connect(lambda: self.play(other=False))
        self.play_other_btn.clicked.connect(lambda: self.play(other=True))
        self.stop_btn.clicked.connect(self.player.stop)
        self.reassign_btn.clicked.connect(self.reassign)
        self.undo_btn.clicked.connect(self.undo)
        state.result_changed.connect(self.refresh)
        self.refresh()

    # --- data ---

    def refresh(self) -> None:
        result = self.state.result
        if result is None:
            self.stack.setCurrentIndex(0)
            self.counts.setText("")
            return
        self.stack.setCurrentIndex(1)
        self.all_rows = rm.rows_from_result(result)
        names = {p.id: p.display_name for p in result.participants}
        colors = {
            p.id: speaker_color(self.tokens, i, p.color) for i, p in enumerate(result.participants)
        }
        self.model.names, self.model.colors = names, colors
        current = self.speaker.currentData()
        self.speaker.clear()
        self.speaker.addItem(self.tr("All speakers"), None)
        for pid, name in names.items():
            self.speaker.addItem(name, pid)
        if current:
            self.speaker.setCurrentIndex(max(0, self.speaker.findData(current)))
        self.target.clear()
        for pid, name in names.items():
            self.target.addItem(name, pid)
        c = rm.counts(self.all_rows)
        self.counts.setText(
            self.tr("{0} segments · {1} to review · {2} suppressed · {3} recording gap(s)").format(
                c["segments"], c["review"], c["suppressed"], c["gaps"]
            )
        )
        self.apply_filter()

    def apply_filter(self) -> None:
        f = self.filter
        f.only_review = self.only_review.isChecked()
        f.only_overlap = self.only_overlap.isChecked()
        f.show_suppressed = self.show_suppressed.isChecked()
        f.show_missed = self.show_missed.isChecked()
        pid = self.speaker.currentData()
        f.speakers = {pid} if pid else set()
        f.text = self.search.text().strip()
        self.model.set_rows(rm.filter_rows(self.all_rows, f), self.model.names, self.model.colors)
        self.table.resizeRowsToContents()
        self._show_details()

    def selected(self) -> rm.ResultRow | None:
        index = self.table.currentIndex()
        if not index.isValid() or index.row() >= len(self.model.rows):
            return None
        return self.model.rows[index.row()]

    def select_id(self, segment_id: str) -> bool:
        for i, row in enumerate(self.model.rows):
            if row.id == segment_id:
                self.table.selectRow(i)
                self._show_details()
                return True
        return False

    def _show_details(self) -> None:
        row = self.selected()
        has = row is not None and row.kind != "gap"
        for w in (self.play_btn, self.stop_btn):
            w.setEnabled(has)
        self.play_other_btn.setEnabled(
            bool(row and row.other_speaker and row.other_speaker != "unknown")
        )
        self.reassign_btn.setEnabled(bool(row and row.can_reassign))
        self.undo_btn.setEnabled(bool(row and row.reassigned_from))
        if row is None:
            for field in (self.d_id, self.d_kind, self.d_flags, self.d_other, self.d_ownership):
                field.setText("")
            return
        self.d_id.setText(row.id)
        self.d_kind.setText(row.kind + (f" ({row.reason})" if row.reason else ""))
        self.d_flags.setText(", ".join(row.flags) or "-")
        other = row.other_speaker or "-"
        if row.reassigned_from:
            other = self.tr("moved from {0}").format(
                self.model.names.get(row.reassigned_from, row.reassigned_from)
            )
        self.d_other.setText(self.model.names.get(other, other))
        self.d_ownership.setText("-" if row.ownership is None else f"{row.ownership:.2f}")
        if row.speaker:
            self.target.setCurrentIndex(max(0, self.target.findData(row.speaker)))

    # --- actions ---

    def play(self, *, other: bool) -> None:
        row = self.selected()
        session = self.state.session
        if row is None or session is None:
            return
        who = row.other_speaker if other else row.speaker
        if not who or who == "unknown":
            return
        self.player.play(session, who, row.start, row.end)

    def reassign(self) -> None:
        row = self.selected()
        session = self.state.session
        if row is None or session is None or not row.can_reassign:
            return
        to = self.target.currentData()
        original = row.reassigned_from or row.speaker
        session.config.reassign(row.id, None if to == original else to)
        self.state.save_session()
        self.reassign_note.setText(self.tr("Saved. Re-exporting …"))
        self.reassigned.emit()

    def undo(self) -> None:
        row = self.selected()
        session = self.state.session
        if row is None or session is None:
            return
        session.config.reassign(row.id, None)
        self.state.save_session()
        self.reassign_note.setText(self.tr("Saved. Re-exporting …"))
        self.reassigned.emit()
