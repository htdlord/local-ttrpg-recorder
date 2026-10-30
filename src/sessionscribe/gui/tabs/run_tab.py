"""Run tab: process the session in the background; progress, ETA, log, cancel/resume."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from sessionscribe.gui.models.colors import Tokens
from sessionscribe.gui.models.results import eta_seconds, format_duration
from sessionscribe.gui.state import AppState
from sessionscribe.gui.widgets import button, label
from sessionscribe.gui.workers import ProcessWorker, start


class RunTab(QWidget):
    finished = Signal()  # a result is available (the window switches to Result)

    def __init__(self, state: AppState, tokens: Tokens) -> None:
        super().__init__()
        self.state, self.tokens = state, tokens
        self.worker: ProcessWorker | None = None
        self.worker_thread: QThread | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        box = QGroupBox(self.tr("Processing"))
        bl = QVBoxLayout(box)
        bl.setSpacing(8)
        self.summary = label(muted=True)
        bl.addWidget(self.summary)
        buttons = QHBoxLayout()
        self.run_btn = button(self.tr("Transcribe"), tokens, "play", primary=True)
        self.cancel_btn = button(self.tr("Cancel"), tokens, "stop")
        self.export_btn = button(self.tr("Export only (no ASR)"), tokens, "export")
        for b in (self.run_btn, self.cancel_btn, self.export_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        bl.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        bl.addWidget(self.progress)
        self.eta = label(muted=True)
        bl.addWidget(self.eta)
        outer.addWidget(box)

        log_box = QGroupBox(self.tr("Log"))
        ll = QVBoxLayout(log_box)
        self.log = QListWidget()
        ll.addWidget(self.log)
        outer.addWidget(log_box, 1)

        self.run_btn.clicked.connect(lambda: self.start(export_only=False))
        self.export_btn.clicked.connect(lambda: self.start(export_only=True))
        self.cancel_btn.clicked.connect(self.cancel)
        state.session_changed.connect(self.refresh)
        state.settings_changed.connect(self.refresh)
        self.refresh()

    @property
    def running(self) -> bool:
        return self.worker is not None

    def refresh(self) -> None:
        s = self.state.session
        idle = not self.running
        self.run_btn.setEnabled(s is not None and idle)
        self.export_btn.setEnabled(s is not None and idle)
        self.cancel_btn.setEnabled(not idle)
        if s is None:
            self.summary.setText(self.tr("Open or create a session first."))
            return
        st = self.state.settings
        self.summary.setText(
            self.tr(
                "{0}: {1} participant(s); model {2}, {3} mode, bleed handling {4}, "
                "language {5}. Completed chunks are cached: running again resumes."
            ).format(
                s.name,
                len(s.participants),
                st.model.id,
                st.asr.mode,
                st.bleed.mode,
                st.decoding.language,
            )
        )

    def add_log(self, text: str) -> None:
        self.log.addItem(text)
        self.log.scrollToBottom()

    def start(self, *, export_only: bool) -> None:
        session = self.state.session
        if session is None or self.running:
            return
        self.progress.setValue(0)
        self.eta.setText("")
        self.add_log(self.tr("Export from the cache …") if export_only else self.tr("Started …"))
        worker = ProcessWorker(
            session,
            self.state.settings,
            self.state.output_dir(),
            export_only=export_only,
            process_kwargs=self.state.process_kwargs,
        )
        worker.progress.connect(self._progress)
        worker.message.connect(self.add_log)
        worker.finished.connect(self._finished)
        worker.failed.connect(self._failed)
        worker.cancelled.connect(self._cancelled)
        self.worker = worker
        self.worker_thread = start(worker, self)
        self.refresh()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.add_log(self.tr("Cancelling … (the current chunk finishes or stops)"))

    def _progress(self, done: int, total: int, elapsed: float) -> None:
        self.progress.setValue(int(1000 * done / max(total, 1)))
        eta = eta_seconds(done, total, elapsed)
        self.eta.setText(
            self.tr("{0:.0f} % · about {1} left").format(
                100 * done / max(total, 1), format_duration(eta)
            )
            if eta is not None
            else self.tr("{0:.0f} %").format(100 * done / max(total, 1))
        )

    def _done(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.refresh()

    def _finished(self, result: Any, stats: Any, written: list[Any]) -> None:
        self._done()
        self.progress.setValue(1000)
        self.eta.setText("")
        self.add_log(
            self.tr(
                "Done: {0} chunk(s), {1} from cache, {2} transcribed; {3} file(s) written to {4}"
            ).format(
                stats.chunks, stats.cached, stats.transcribed, len(written), self.state.output_dir()
            )
        )
        self.state.set_result(result)
        self.finished.emit()

    def _failed(self, message: str) -> None:
        self._done()
        self.add_log(self.tr("Failed: {0}").format(message))

    def _cancelled(self) -> None:
        self._done()
        self.add_log(self.tr("Cancelled. Completed chunks are kept; Transcribe resumes."))
