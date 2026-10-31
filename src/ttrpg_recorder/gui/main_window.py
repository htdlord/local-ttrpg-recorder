"""Main window: tabs, menus, preferences (language, theme), recent sessions, System check."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from PySide6.QtCore import QSettings, QThread, QUrl
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from ttrpg_recorder.core.session import Session
from ttrpg_recorder.gui import i18n, icons, theme, workers
from ttrpg_recorder.gui.audio_preview import PreviewPlayer
from ttrpg_recorder.gui.state import AppState
from ttrpg_recorder.gui.tabs.record_tab import RecordTab
from ttrpg_recorder.gui.tabs.result_tab import ResultTab
from ttrpg_recorder.gui.tabs.run_tab import RunTab
from ttrpg_recorder.gui.tabs.session_tab import SessionTab
from ttrpg_recorder.gui.tabs.settings_tab import SettingsTab
from ttrpg_recorder.gui.widgets import button, label
from ttrpg_recorder.gui.workers import DoctorWorker, ProcessWorker, start

ORG = "SessionScribe"
MAX_RECENT = 8
NOTICES = "THIRD_PARTY_NOTICES.md"
NOTICES_URL = "https://github.com/htdlord/sessionscribe/blob/main/" + NOTICES


def notices_url() -> QUrl:
    """The notices next to a source checkout, else on GitHub."""
    path = Path(__file__).resolve().parents[3] / NOTICES
    return QUrl.fromLocalFile(str(path)) if path.is_file() else QUrl(NOTICES_URL)


class AboutDialog(QDialog):
    """Version, licences (own MIT, third-party notices, Qt) and the privacy note."""

    def __init__(self, parent: Any, tokens: Any) -> None:
        from importlib.metadata import version

        import PySide6

        from ttrpg_recorder import __version__

        super().__init__(parent)
        self.setWindowTitle(self.tr("About SessionScribe"))
        self.open_url: Callable[[QUrl], bool] = QDesktopServices.openUrl
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 16)
        layout.setSpacing(8)
        layout.addWidget(label(f"SessionScribe {__version__}", heading=True))
        libs = [f"{name} {version(name)}" for name in ("faster-whisper", "ctranslate2")]
        libs.append(f"PySide6 {PySide6.__version__}")
        self.versions = label(", ".join(libs), muted=True)
        layout.addWidget(self.versions)
        for text in (
            self.tr("Local multi-track speech-to-text for tabletop RPG sessions."),
            self.tr(
                "Own code: MIT licence. Third-party packages keep their own licences; "
                "PySide6/Qt is LGPLv3 and installed as replaceable libraries."
            ),
            self.tr(
                "Participants' voices are personal data: whoever records is responsible "
                "for informing participants and obtaining consent where required."
            ),
        ):
            note = label(text)
            note.setWordWrap(True)
            layout.addWidget(note)
        row = QHBoxLayout()
        self.notices_btn = button(self.tr("Third-party notices"), tokens)
        self.notices_btn.clicked.connect(lambda: self.open_url(notices_url()))
        about_qt = button(self.tr("About Qt"), tokens)
        about_qt.clicked.connect(lambda: QMessageBox.aboutQt(self))
        close = button(self.tr("Close"), tokens, primary=True)
        close.clicked.connect(self.accept)
        row.addWidget(self.notices_btn)
        row.addWidget(about_qt)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        self.resize(520, 0)


class SystemCheckDialog(QDialog):
    STATUS: ClassVar[dict[int, str]] = {0: "SKIP", 1: "INFO", 2: "OK", 3: "WARN", 4: "FAIL"}

    def __init__(self, parent: Any, tokens: Any, run: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("System check"))
        self.resize(760, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.heading = label(self.tr("Checking the environment …"), heading=True)
        layout.addWidget(self.heading)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            [self.tr("Status"), self.tr("Check"), self.tr("Result")]
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        self.close_btn = button(self.tr("Close"), tokens)
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)
        self.worker_thread: QThread | None = None
        if run:
            worker = DoctorWorker()
            worker.result.connect(self.add_result)
            worker.finished.connect(self.done_report)
            worker.failed.connect(self.heading.setText)  # bound slot: runs in the GUI thread
            self.worker_thread = start(worker)

    def add_result(self, result: Any) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        for col, text in enumerate(
            (self.STATUS.get(int(result.status), "?"), result.name, result.summary)
        ):
            self.table.setItem(r, col, QTableWidgetItem(text))

    def done_report(self, report: Any) -> None:
        verdict = self.STATUS.get(int(report.worst), "?")
        self.heading.setText(self.tr("System check finished: {0}").format(verdict))


class MainWindow(QMainWindow):
    def __init__(
        self,
        state: AppState | None = None,
        settings: QSettings | None = None,
        player: PreviewPlayer | None = None,
    ) -> None:
        super().__init__()
        self.state = state or AppState()
        self.prefs = settings or QSettings(ORG, ORG)
        self.player = player
        self.setWindowTitle("SessionScribe")
        self.resize(1200, 800)
        self._current_export: ProcessWorker | None = None
        self.confirm_close: Callable[[], bool] = self._ask_close
        self.build()
        if self._reconnected:
            self.tabs.setCurrentWidget(self.record_tab)
        self.state.session_changed.connect(self._session_opened)

    # --- building (again after a theme or language change) ---

    def build(self) -> None:
        app = QApplication.instance()
        theme_name = str(self.prefs.value("theme", "system"))
        assert isinstance(app, QApplication)
        self.tokens = theme.apply(app, theme_name)
        t = self.tokens
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.session_tab = SessionTab(self.state, t)
        self.record_tab = RecordTab(self.state, t, self.prefs)
        self.settings_tab = SettingsTab(self.state, t)
        self.run_tab = RunTab(self.state, t)
        self.result_tab = ResultTab(self.state, t, self.player)
        for widget, name, title in (
            (self.session_tab, "session", self.tr("Session")),
            (self.record_tab, "record", self.tr("Record")),
            (self.settings_tab, "settings", self.tr("Settings")),
            (self.run_tab, "run", self.tr("Run")),
            (self.result_tab, "result", self.tr("Result")),
        ):
            self.tabs.addTab(widget, icons.icon(name, t.text_muted, 20), title)
        self.setCentralWidget(self.tabs)
        self.run_tab.finished.connect(lambda: self.tabs.setCurrentWidget(self.result_tab))
        self.result_tab.reassigned.connect(self._reexport)
        self.session_tab.saved.connect(self._remember)
        self.record_tab.finished.connect(self._recorded)
        # a recording that runs (or crashed) since the last window: follow it again
        self._reconnected = self.record_tab.reconnect()
        self._build_menus()
        self._update_title()

    def _build_menus(self) -> None:
        bar = self.menuBar()
        bar.clear()
        file_menu = bar.addMenu(self.tr("&File"))
        new = file_menu.addAction(self.tr("New session from folder…"))
        new.triggered.connect(lambda: self.session_tab.new_from_folder())
        opn = file_menu.addAction(self.tr("Open session…"))
        opn.triggered.connect(lambda: self.session_tab.open_session())
        self.recent_menu = file_menu.addMenu(self.tr("Recent sessions"))
        self._fill_recent()
        file_menu.addSeparator()
        quit_action = file_menu.addAction(self.tr("Quit"))
        quit_action.triggered.connect(self.close)

        tools = bar.addMenu(self.tr("&Tools"))
        check = tools.addAction(self.tr("System check…"))
        check.triggered.connect(self.system_check)

        view = bar.addMenu(self.tr("&View"))
        theme_menu = view.addMenu(self.tr("Theme"))
        self._choice_menu(
            theme_menu,
            "theme",
            (
                ("system", self.tr("System")),
                ("light", self.tr("Light")),
                ("dark", self.tr("Dark")),
            ),
        )
        lang_menu = view.addMenu(self.tr("Language"))
        self._choice_menu(
            lang_menu,
            "language",
            (
                ("system", self.tr("System")),
                ("en", "English"),
                ("hu", "Magyar"),
            ),
        )

        help_menu = bar.addMenu(self.tr("&Help"))
        about = help_menu.addAction(self.tr("About SessionScribe…"))
        about.triggered.connect(self.about)

    def about(self) -> AboutDialog:
        dialog = AboutDialog(self, self.tokens)
        dialog.open()
        return dialog

    def _choice_menu(self, menu: Any, key: str, choices: tuple[tuple[str, str], ...]) -> None:
        group = QActionGroup(menu)
        current = str(self.prefs.value(key, "system"))
        for value, text in choices:
            action = QAction(text, menu, checkable=True)
            action.setChecked(value == current)
            action.triggered.connect(lambda _c=False, v=value: self.set_preference(key, v))
            group.addAction(action)
            menu.addAction(action)

    def set_preference(self, key: str, value: str) -> None:
        self.prefs.setValue(key, value)
        if key == "language":
            app = QApplication.instance()
            assert isinstance(app, QApplication)
            i18n.install(app, value)
        current = self.tabs.currentIndex()
        self.build()
        self.tabs.setCurrentIndex(current)

    # --- sessions ---

    def recent(self) -> list[str]:
        value = self.prefs.value("recent", [])
        if isinstance(value, str):
            return [value]
        if isinstance(value, list | tuple):
            return [str(v) for v in value]
        return []

    def _remember(self) -> None:
        if self.state.session is None:
            return
        path = str(self.state.session.folder / "session.toml")
        items = [path] + [p for p in self.recent() if p != path]
        self.prefs.setValue("recent", items[:MAX_RECENT])
        self._fill_recent()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for path in self.recent():
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _c=False, p=path: self.open_path(Path(p)))
        self.recent_menu.setEnabled(bool(self.recent()))

    def open_path(self, path: Path) -> None:
        try:
            self.state.set_session(Session.load(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.tr("Open session"), str(exc))

    def _session_opened(self) -> None:
        self._update_title()
        session = self.state.session
        if session is None:
            return
        if (session.folder / "session.toml").is_file():
            self._remember()
        # A session transcribed before: show its result straight from the cache.
        worker = ProcessWorker(
            session,
            self.state.settings,
            self.state.output_dir(),
            export_only=True,
            process_kwargs=self.state.process_kwargs,
        )
        # Bound slot (not a lambda), so it runs in the GUI thread; a newer session's
        # export supersedes this one.
        worker.finished.connect(self._export_finished)
        self._current_export = worker
        start(worker)

    def _export_finished(self, result: Any, *_: Any) -> None:
        if self.sender() is self._current_export:
            self._current_export = None
            self.state.set_result(result)

    def _reexport(self) -> None:
        self.run_tab.start(export_only=True)

    def _update_title(self) -> None:
        s = self.state.session
        self.setWindowTitle(f"SessionScribe - {s.name}" if s else "SessionScribe")
        self.statusBar().showMessage(str(s.folder) if s else self.tr("No session open"))

    def _recorded(self, folder: Path) -> None:
        self.open_path(folder / "session.toml")
        self.tabs.setCurrentWidget(self.session_tab)

    def _ask_close(self) -> bool:
        answer = QMessageBox.question(
            self,
            self.tr("Recording in progress"),
            self.tr(
                "The recording continues in the background after the window closes. "
                "Open SessionScribe again to follow or stop it. Close the window?"
            ),
        )
        return answer == QMessageBox.StandardButton.Yes

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.record_tab.recording and not self.confirm_close():
            event.ignore()
            return
        # Completed chunks stay cached; a cancelled run resumes next time.
        workers.stop_all()
        super().closeEvent(event)

    def system_check(self) -> None:
        SystemCheckDialog(self, self.tokens).exec()
