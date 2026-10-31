"""Record tab: rig, guided mic check, live meters and alarms of a detached recorder.

The recording runs in its own process (``core.audio.record_child``); this tab only
starts it and then follows ``rec/control/`` files, so closing or crashing the GUI does
not stop the recording and a new window reconnects to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ttrpg_recorder.core.audio import control
from ttrpg_recorder.core.audio.manifest import RigConfig, read_rig, write_rig
from ttrpg_recorder.gui import icons
from ttrpg_recorder.gui.models import recording as rm
from ttrpg_recorder.gui.models.colors import Tokens
from ttrpg_recorder.gui.state import AppState
from ttrpg_recorder.gui.widgets import button, label
from ttrpg_recorder.gui.workers import DevicesWorker, start

POLL_MS = 250
PREF_SESSION = "recording/session"
PREF_RIG = "recording/rig"
RIG_IN_SESSION = "rig.toml"  # copy of the rig used, in rec/control/


class LevelMeter(QWidget):
    """Horizontal meter: RMS bar, peak tick, red when the peak clipped."""

    def __init__(self, tokens: Tokens) -> None:
        super().__init__()
        self.tokens = tokens
        self.rms = 0.0
        self.peak = 0.0
        self.clipped = False
        self.setMinimumHeight(16)
        self.setMinimumWidth(160)

    def set_level(self, meter: rm.Meter) -> None:
        self.rms = meter.fraction(meter.rms_db)
        self.peak = meter.fraction(meter.peak_db)
        self.clipped = meter.clipped
        self.update()

    def paintEvent(self, event: Any) -> None:
        t = self.tokens
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 3.5, -0.5, -3.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t.surface_alt))
        p.drawRoundedRect(r, 4, 4)
        colour = QColor(t.error if self.clipped else t.accent)
        p.setBrush(colour)
        p.drawRoundedRect(QRectF(r.x(), r.y(), r.width() * self.rms, r.height()), 4, 4)
        x = r.x() + r.width() * self.peak
        p.setPen(colour)
        p.drawLine(int(x), int(r.top()), int(x), int(r.bottom()))
        p.end()


class RecordTab(QWidget):
    finished = Signal(Path)  # the recorded session folder, after a clean stop

    def __init__(self, state: AppState, tokens: Tokens, prefs: QSettings) -> None:
        super().__init__()
        self.state, self.tokens, self.prefs = state, tokens, prefs
        self.session_dir: Path | None = None
        self.reader: control.EventReader | None = None
        self.proc: Any = None
        self.link_state: control.LinkState = "none"
        self.base_rig: RigConfig | None = None
        self.devices: list[dict[str, Any]] = []
        self.meter_widgets: list[tuple[QLabel, LevelMeter, QLabel]] = []
        self.names: dict[str, str] = {}  # participant id -> display name
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.poll)
        self._build()
        self._load_rig_pref()
        self._refresh_buttons()

    # --- layout ---

    def _build(self) -> None:
        t = self.tokens
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(16)
        setup = QGroupBox(self.tr("Interface"))
        form = QFormLayout(setup)
        dev_row = QHBoxLayout()
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.setMinimumWidth(240)
        self.host_api = QLineEdit()
        self.host_api.setPlaceholderText("asio")
        self.host_api.setMaximumWidth(96)
        self.device.activated.connect(lambda _i: self._device_chosen())
        self.refresh_btn = button(self.tr("Refresh"), t, "undo")
        self.refresh_btn.clicked.connect(lambda: self.refresh_devices())
        dev_row.addWidget(self.device, 1)
        dev_row.addWidget(self.host_api)
        dev_row.addWidget(self.refresh_btn)
        form.addRow(self.tr("Device"), dev_row)
        self.device_note = label(muted=True)
        form.addRow("", self.device_note)
        left.addWidget(setup)

        rig_box = QGroupBox(self.tr("Channels and participants"))
        rl = QVBoxLayout(rig_box)
        self.rig_table = QTableWidget(0, 3)
        self.rig_table.setHorizontalHeaderLabels(
            [self.tr("Channel"), self.tr("Id"), self.tr("Name")]
        )
        self.rig_table.horizontalHeader().setStretchLastSection(True)
        self.rig_table.verticalHeader().setVisible(False)
        self.rig_table.setAlternatingRowColors(True)
        rl.addWidget(self.rig_table, 1)
        rows = QHBoxLayout()
        self.add_btn = button(self.tr("Add channel"), t, "add")
        self.add_btn.clicked.connect(lambda: self.rig_table.insertRow(self.rig_table.rowCount()))
        self.remove_btn = button(self.tr("Remove selected"), t, "remove")
        self.remove_btn.clicked.connect(self._remove_rows)
        self.load_rig_btn = button(self.tr("Load rig…"), t, "open")
        self.load_rig_btn.clicked.connect(lambda: self.load_rig())
        self.save_rig_btn = button(self.tr("Save rig…"), t, "save")
        self.save_rig_btn.clicked.connect(lambda: self.save_rig())
        for b in (self.add_btn, self.remove_btn):
            rows.addWidget(b)
        rows.addStretch(1)
        for b in (self.load_rig_btn, self.save_rig_btn):
            rows.addWidget(b)
        rl.addLayout(rows)
        self.rig_errors = label()
        self.rig_errors.setProperty("error", True)
        rl.addWidget(self.rig_errors)
        left.addWidget(rig_box, 1)

        session_box = QGroupBox(self.tr("Recording"))
        sf = QFormLayout(session_box)
        folder_row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText(self.tr("New session folder, e.g. recordings/2026-11-05"))
        self.browse_btn = button(self.tr("Browse…"), t, "open")
        self.browse_btn.clicked.connect(lambda: self.browse_folder())
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.browse_btn)
        sf.addRow(self.tr("Session folder"), folder_row)
        self.hours = QDoubleSpinBox()
        self.hours.setRange(0.5, 24.0)
        self.hours.setSingleStep(0.5)
        self.hours.setValue(8.0)
        self.hours.setSuffix(" h")
        sf.addRow(self.tr("Planned length"), self.hours)
        self.check_box = QCheckBox(self.tr("Guided mic check at the start"))
        self.check_box.setChecked(True)
        self.force_box = QCheckBox(self.tr("Keep recording if the mic check fails"))
        sf.addRow("", self.check_box)
        sf.addRow("", self.force_box)
        buttons = QHBoxLayout()
        self.start_btn = button(self.tr("Start recording"), t, "record", primary=True)
        self.start_btn.clicked.connect(lambda: self.start_recording())
        self.stop_btn = button(self.tr("Stop"), t, "stop")
        self.stop_btn.clicked.connect(self.stop_recording)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch(1)
        sf.addRow("", buttons)
        left.addWidget(session_box)
        outer.addLayout(left, 5)

        right = QVBoxLayout()
        right.setSpacing(16)
        live = QGroupBox(self.tr("Live"))
        ll = QVBoxLayout(live)
        self.state_label = label(self.tr("Not recording"), heading=True)
        ll.addWidget(self.state_label)
        self.stats = label(muted=True)
        ll.addWidget(self.stats)
        self.prompt = QLabel()
        self.prompt.setWordWrap(True)
        self.prompt.setProperty("heading", True)
        self.prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt.setStyleSheet(
            f"background: {t.accent}; color: {t.accent_text}; border-radius: 8px; padding: 16px;"
        )
        self.prompt.hide()
        ll.addWidget(self.prompt)
        self.meter_grid = QGridLayout()
        self.meter_grid.setHorizontalSpacing(8)
        self.meter_grid.setVerticalSpacing(4)
        self.meter_grid.setColumnStretch(1, 1)
        ll.addLayout(self.meter_grid)
        self.banner = label()
        self.banner.setProperty("error", True)
        self.banner.hide()
        ll.addWidget(self.banner)
        actions = QHBoxLayout()
        self.recover_btn = button(self.tr("Recover"), t, "undo")
        self.recover_btn.clicked.connect(self.recover)
        self.open_btn = button(self.tr("Open in Session tab"), t, "session", primary=True)
        self.open_btn.clicked.connect(self._open_session)
        actions.addWidget(self.recover_btn)
        actions.addWidget(self.open_btn)
        actions.addStretch(1)
        ll.addLayout(actions)
        right.addWidget(live)

        self.check_box_group = QGroupBox(self.tr("Mic check"))
        cl = QVBoxLayout(self.check_box_group)
        self.check_table = QTableWidget(0, 3)
        self.check_table.setHorizontalHeaderLabels(
            [self.tr("Participant"), self.tr("Result"), self.tr("Alarms")]
        )
        self.check_table.horizontalHeader().setStretchLastSection(True)
        self.check_table.verticalHeader().setVisible(False)
        cl.addWidget(self.check_table)
        self.check_box_group.hide()
        right.addWidget(self.check_box_group)

        alarms = QGroupBox(self.tr("Alarms"))
        al = QVBoxLayout(alarms)
        self.alarm_list = QListWidget()
        al.addWidget(self.alarm_list)
        right.addWidget(alarms, 1)
        outer.addLayout(right, 6)

    # --- rig ---

    def rig_rows(self) -> list[rm.RigRow]:
        out = []
        for r in range(self.rig_table.rowCount()):
            cells = [self.rig_table.item(r, c) for c in range(3)]
            out.append(rm.RigRow(*[(i.text() if i else "") for i in cells]))
        return out

    def set_rig(self, rig: RigConfig) -> None:
        self.base_rig = rig
        self.device.setEditText(rig.device)
        self.host_api.setText(rig.host_api)
        self.rig_table.setRowCount(0)
        for row in rm.rows_from_rig(rig):
            r = self.rig_table.rowCount()
            self.rig_table.insertRow(r)
            for c, text in enumerate((row.channel, row.id, row.name)):
                self.rig_table.setItem(r, c, QTableWidgetItem(text))

    def current_rig(self) -> RigConfig | None:
        result = rm.build_rig(
            self.device.currentText().strip(),
            self.host_api.text().strip(),
            self.rig_rows(),
            self.base_rig,
        )
        self.rig_errors.setText("\n".join(result.errors))
        return result.rig

    def load_rig(self, path: Path | None = None) -> bool:
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, self.tr("Load rig"), str(self.prefs.value(PREF_RIG, "")), "rig (*.toml)"
            )
            if not chosen:
                return False
            path = Path(chosen)
        try:
            self.set_rig(read_rig(path))
        except (OSError, ValueError) as exc:
            self.rig_errors.setText(str(exc))
            return False
        self.prefs.setValue(PREF_RIG, str(path))
        self.rig_errors.setText("")
        return True

    def save_rig(self, path: Path | None = None) -> bool:
        rig = self.current_rig()
        if rig is None:
            return False
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, self.tr("Save rig"), str(self.prefs.value(PREF_RIG, "rig.toml")), "*.toml"
            )
            if not chosen:
                return False
            path = Path(chosen)
        write_rig(path, rig)
        self.prefs.setValue(PREF_RIG, str(path))
        return True

    def _load_rig_pref(self) -> None:
        path = self.prefs.value(PREF_RIG, "")
        if path and Path(str(path)).is_file():
            self.load_rig(Path(str(path)))

    def _remove_rows(self) -> None:
        for r in sorted({i.row() for i in self.rig_table.selectedIndexes()}, reverse=True):
            self.rig_table.removeRow(r)

    # --- devices ---

    def refresh_devices(self) -> None:
        self.refresh_btn.setEnabled(False)
        self.device_note.setText(self.tr("Looking for devices …"))
        worker = DevicesWorker()
        worker.finished.connect(self.set_devices)
        worker.failed.connect(self._devices_failed)
        self._devices_worker = worker
        start(worker)

    def set_devices(self, devices: list[dict[str, Any]]) -> None:
        self.devices = devices
        current = self.device.currentText()
        self.device.clear()
        for d in devices:
            self.device.addItem(f"{d['name']}  ·  {d['host_api']}  ·  {d['channels']} ch", d)
        self.device.setEditText(current)
        self.refresh_btn.setEnabled(True)
        self.device_note.setText(self.tr("{0} input device(s)").format(len(devices)))

    def _devices_failed(self, message: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.device_note.setText(message)

    def _device_chosen(self) -> None:
        d = self.device.currentData()
        if isinstance(d, dict):
            self.device.setEditText(str(d["name"]))
            api = str(d["host_api"]).split()[0].lower()
            self.host_api.setText("asio" if "asio" in api else api)

    # --- recording ---

    def browse_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, self.tr("Session folder"))
        if chosen:
            self.folder.setText(chosen)

    @property
    def recording(self) -> bool:
        return self.link_state in ("running", "unresponsive")

    def start_recording(self) -> bool:
        rig = self.current_rig()
        text = self.folder.text().strip()
        if rig is None:
            return False
        if not text:
            self.rig_errors.setText(self.tr("Choose a session folder."))
            return False
        session = Path(text).resolve()
        link = control.probe(session)
        if link.state in ("running", "unresponsive"):
            self.attach(session)
            return True
        folder = control.control_dir(session)
        folder.mkdir(parents=True, exist_ok=True)
        rig_path = folder / RIG_IN_SESSION
        write_rig(rig_path, rig)
        for name in (control.STATUS_FILE, control.COMMAND_FILE):
            (folder / name).unlink(missing_ok=True)  # a resumed session starts fresh
        self.proc = control.spawn_recorder(
            session,
            rm.child_args(
                rig_path,
                hours=self.hours.value(),
                check=self.check_box.isChecked(),
                force=self.force_box.isChecked(),
            ),
        )
        self.alarm_list.clear()
        self.check_table.setRowCount(0)
        self.check_box_group.hide()
        self.attach(session, from_start=True)
        return True

    def attach(self, session: Path, *, from_start: bool = False) -> None:
        """Follow the recorder of ``session`` (just started or found by a new window)."""
        self.session_dir = session
        self.prefs.setValue(PREF_SESSION, str(session))
        self.folder.setText(str(session))
        folder = control.control_dir(session)
        self.reader = control.EventReader(folder)
        if from_start:
            self.link_state = "running"  # the process is starting up
        self.timer.start()
        self.poll()

    def reconnect(self) -> bool:
        """At start-up: attach to the recording the last window followed, if any."""
        saved = self.prefs.value(PREF_SESSION, "")
        if not saved:
            return False
        session = Path(str(saved))
        if control.probe(session).state not in ("running", "unresponsive", "crashed"):
            return False
        self.attach(session)
        return True

    def stop_recording(self) -> None:
        if self.session_dir is not None:
            control.send_command(control.control_dir(self.session_dir), "stop")
            self.state_label.setText(self.tr("Stopping …"))

    def recover(self) -> None:
        from ttrpg_recorder.core.audio.recovery import recover, verify

        if self.session_dir is None:
            return
        try:
            recover(self.session_dir)
            report = verify(self.session_dir)
        except (OSError, ValueError) as exc:
            self.banner.setText(str(exc))
            return
        problems = [s for s in report.segments if not s.ok]
        control.mark_recovered(self.session_dir)
        self.poll()  # ended now: Open in Session tab / Start continues into this folder
        self.banner.setText(
            self.tr("Recovered: {0} segment file(s), {1} with a problem").format(
                len(report.segments), len(problems)
            )
        )
        self.banner.show()

    def _open_session(self) -> None:
        if self.session_dir is not None:
            self.finished.emit(self.session_dir)

    # --- following the files ---

    def poll(self) -> None:
        if self.session_dir is None:
            return
        if self.proc is not None and self.proc.poll() is not None:
            self.proc = None  # reaped (no zombie on Linux)
        if self.reader is not None:
            for event in self.reader.poll():
                self._event(event)
        link = control.probe(self.session_dir)
        status = link.status
        if self.link_state == "running" and link.state == "none":
            link = control.Link("running", status, None)  # still starting up
        self.link_state = link.state
        self._show_status(link)
        if link.state in ("ended", "crashed") and self.timer.isActive():
            self.timer.stop()
            if link.state == "ended" and status.get("state") == "stopped":
                self.finished.emit(self.session_dir)
        self._refresh_buttons()

    def _show_status(self, link: control.Link) -> None:
        s = link.status
        self.names.update(zip(s.get("participants") or [], s.get("names") or [], strict=False))
        state = s.get("state", "starting")
        texts = {
            "starting": self.tr("Starting …"),
            "mic_check": self.tr("Mic check"),
            "recording": self.tr("Recording"),
            "stopping": self.tr("Stopping …"),
            "stopped": self.tr("Stopped"),
            "failed": self.tr("Failed"),
        }
        title = texts.get(state, state)
        if link.state == "unresponsive":
            title = self.tr("Recorder not responding")
        elif link.state == "crashed":
            title = self.tr("Recording interrupted")
        self.state_label.setText(title)
        parts = [self.tr("Time {0}").format(rm.format_clock(s.get("time_s")))]
        if s.get("buffer_fill") is not None:
            parts.append(self.tr("buffer {0:.0%}").format(float(s["buffer_fill"])))
        if s.get("disk_left_s") is not None:
            parts.append(self.tr("disk {0:.1f} h").format(float(s["disk_left_s"]) / 3600))
        parts.append(self.tr("{0} gap(s)").format(int(s.get("gaps", 0))))
        if s.get("device"):
            parts.append(f"{s['device']} ({s.get('host_api', '')})")
        self.stats.setText("  ·  ".join(parts))
        self._show_prompt(s.get("prompt") if state == "mic_check" else None)
        self._show_meters(rm.meters(s))
        if link.state == "crashed":
            self.banner.setText(
                self.tr(
                    "The recorder stopped without a clean stop (crash or power loss). "
                    "Recover makes the last files readable; then you can continue into the "
                    "same folder."
                )
            )
            self.banner.show()
        elif state == "failed" and s.get("error"):
            self.banner.setText(str(s["error"]))
            self.banner.show()
        elif state == "stopped":
            summary = s.get("summary") or {}
            self.banner.setText(
                self.tr("Stopped: {0} min, {1} segment file(s), {2} gap(s)").format(
                    f"{float(summary.get('seconds', 0)) / 60:.1f}",
                    summary.get("segments", 0),
                    summary.get("gaps", 0),
                )
            )
            self.banner.setProperty("error", bool(summary.get("gaps")))
            self.banner.show()
        else:
            self.banner.hide()

    def _show_prompt(self, prompt: dict[str, Any] | None) -> None:
        if not prompt:
            self.prompt.hide()
            return
        if prompt.get("kind") == "noise":
            text = self.tr("Everyone quiet, please ({0:.0f} s)").format(prompt.get("seconds", 0))
        else:
            text = self.tr("{0}: please speak now ({1:.0f} s)").format(
                prompt.get("label", ""), prompt.get("seconds", 0)
            )
        self.prompt.setText(text)
        self.prompt.show()

    def _show_meters(self, meters: list[rm.Meter]) -> None:
        if len(meters) != len(self.meter_widgets):
            while self.meter_grid.count():
                item = self.meter_grid.takeAt(0)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.deleteLater()
            self.meter_widgets = []
            for row, m in enumerate(meters):
                name = label(m.name)
                meter = LevelMeter(self.tokens)
                db = label(muted=True)
                db.setMinimumWidth(56)
                db.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.meter_grid.addWidget(name, row, 0)
                self.meter_grid.addWidget(meter, row, 1)
                self.meter_grid.addWidget(db, row, 2)
                self.meter_widgets.append((name, meter, db))
        for (name, meter, db), m in zip(self.meter_widgets, meters, strict=True):
            name.setText(m.name)
            meter.set_level(m)
            db.setText(f"{m.peak_db:.0f} dB")

    def _alarm_text(self, event: dict[str, Any]) -> str:
        advice = {
            "device_lost": self.tr("Check the cable and the interface; it reconnects by itself."),
            "reconnected": self.tr("Audio is back; the missing time is marked as a gap."),
            "idle_channel": self.tr(
                "Unplugged or muted mic? Nothing to do if they are just quiet."
            ),
            "disk_low": self.tr("Free some disk space now."),
            "disk_full": self.tr("The disk is full: the recording stopped."),
            "fatal": self.tr("The recorder cannot go on: restart it into the same folder."),
        }
        clock = rm.format_clock(event.get("time_s"))
        text = f"{clock}  {event.get('message', '')}"
        extra = advice.get(str(event.get("kind", "")))
        return f"{text}\n{extra}" if extra else text

    def _event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "alarm":
            severity = rm.alarm_severity(event)
            colour = {
                "error": self.tokens.error,
                "warning": self.tokens.warning,
                "info": self.tokens.success,
            }[severity]
            item = QListWidgetItem(icons.icon("alarm", colour, 20), self._alarm_text(event))
            self.alarm_list.insertItem(0, item)
        elif kind == "mic_check":
            self._show_check(event)
        elif kind == "failed":
            item = QListWidgetItem(
                icons.icon("alarm", self.tokens.error, 20), str(event.get("message", ""))
            )
            self.alarm_list.insertItem(0, item)

    def _show_check(self, event: dict[str, Any]) -> None:
        verdicts = {"PASS": self.tr("OK"), "WARN": self.tr("Check"), "FAIL": self.tr("Failed")}
        kinds = {
            "no_signal": self.tr("no signal"),
            "wrong_channel": self.tr("wrong channel"),
            "quiet": self.tr("too quiet"),
            "clipping": self.tr("clipping"),
        }
        colours = {
            "PASS": self.tokens.success,
            "WARN": self.tokens.warning,
            "FAIL": self.tokens.error,
        }
        self.check_table.setRowCount(0)
        for c in event.get("checks", []):
            r = self.check_table.rowCount()
            self.check_table.insertRow(r)
            pid = str(c.get("participant", ""))
            self.check_table.setItem(r, 0, QTableWidgetItem(self.names.get(pid, pid)))
            verdict = QTableWidgetItem(verdicts.get(c.get("verdict", ""), c.get("verdict", "")))
            verdict.setForeground(QColor(colours.get(c.get("verdict", ""), self.tokens.text)))
            self.check_table.setItem(r, 1, verdict)
            alarms = "; ".join(
                f"{kinds.get(a['kind'], a['kind'])}: {a['message']}" for a in c.get("alarms", [])
            )
            self.check_table.setItem(r, 2, QTableWidgetItem(alarms))
        if event.get("error"):
            self.alarm_list.insertItem(
                0,
                QListWidgetItem(
                    icons.icon("alarm", self.tokens.error, 20),
                    self.tr("Mic check failed: {0}").format(event["error"]),
                ),
            )
        if event.get("stopping"):
            self.alarm_list.insertItem(
                0,
                QListWidgetItem(
                    icons.icon("alarm", self.tokens.error, 20),
                    self.tr(
                        "The mic check failed, so the recording stopped. Fix the channels and "
                        "start again into the same folder, or tick 'Keep recording if the mic "
                        "check fails'."
                    ),
                ),
            )
        self.check_box_group.show()

    def _refresh_buttons(self) -> None:
        active = self.recording
        editable = not active
        for w in (
            self.device,
            self.host_api,
            self.refresh_btn,
            self.rig_table,
            self.add_btn,
            self.remove_btn,
            self.load_rig_btn,
            self.folder,
            self.browse_btn,
            self.hours,
            self.check_box,
            self.force_box,
        ):
            w.setEnabled(editable)
        self.start_btn.setEnabled(editable)
        self.stop_btn.setEnabled(active)
        self.recover_btn.setVisible(self.link_state == "crashed")
        self.open_btn.setVisible(self.link_state == "ended")
