"""Background work for the GUI: processing, export and System check off the GUI thread.

Each worker is a QObject moved to a QThread; core events become Qt signals. Nothing
here touches widgets. Cancelling sets a flag the core polls; completed chunks stay in
the cache, so running again resumes.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from sessionscribe.core.config import Settings
from sessionscribe.core.events import Event, ProgressEvent, WarningEvent
from sessionscribe.core.session import Session


class ProcessWorker(QObject):
    progress = Signal(int, int, float)  # done, total, elapsed seconds
    message = Signal(str)  # warnings: never transcript text
    finished = Signal(object, object, list)  # SessionResult, RunStats, written paths
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        session: Session,
        settings: Settings,
        out_dir: Any,
        *,
        export_only: bool = False,
        process_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.session, self.settings, self.out_dir = session, settings, out_dir
        self.export_only = export_only
        self.process_kwargs = dict(process_kwargs or {})
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _sink(self, t0: float) -> Any:
        def sink(event: Event) -> None:
            if isinstance(event, ProgressEvent) and event.stage == "asr" and event.total:
                self.progress.emit(int(event.done), int(event.total), time.monotonic() - t0)
            elif isinstance(event, WarningEvent):
                self.message.emit(event.message)

        return sink

    @Slot()
    def run(self) -> None:
        from sessionscribe.core.exporters.formats import write_session_outputs
        from sessionscribe.core.pipeline.asr import Cancelled
        from sessionscribe.core.pipeline.run import export_session, process_session

        t0 = time.monotonic()
        try:
            if self.export_only:
                kw = {k: v for k, v in self.process_kwargs.items() if k == "model_identity"}
                result, stats = export_session(
                    self.session, self.settings, sink=self._sink(t0), **kw
                )
            else:
                result, stats = process_session(
                    self.session,
                    self.settings,
                    sink=self._sink(t0),
                    cancel=lambda: self._cancel,
                    **self.process_kwargs,
                )
            written = write_session_outputs(result, self.out_dir)
        except Cancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surfaced in the GUI; the cache stays consistent
            self.failed.emit(str(exc))
            return
        self.finished.emit(result, stats, written)


class DoctorWorker(QObject):
    result = Signal(object)  # CheckResult
    finished = Signal(object)  # DoctorReport
    failed = Signal(str)

    def __init__(self, inference: bool = True, audio: bool = True) -> None:
        super().__init__()
        self.inference, self.audio = inference, audio

    @Slot()
    def run(self) -> None:
        from sessionscribe.core.doctor import DoctorOptions, run_doctor

        try:
            report = run_doctor(
                DoctorOptions(inference=self.inference, audio=self.audio), self.result.emit
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(report)


def start(worker: QObject, parent: QObject) -> QThread:
    """Run ``worker.run`` in a new thread that ends (and cleans up) with the worker."""
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)  # type: ignore[attr-defined]
    for name in ("finished", "failed", "cancelled"):
        signal = getattr(worker, name, None)
        if signal is not None:
            signal.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
