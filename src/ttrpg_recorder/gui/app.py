"""``ttrpg-gui``: start the desktop application."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "The GUI needs the optional 'gui' extra (PySide6): run 'uv sync --extra gui'.",
            file=sys.stderr,
        )
        return 1
    from ttrpg_recorder.gui import i18n
    from ttrpg_recorder.gui.main_window import ORG, MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("SessionScribe")
    app.setOrganizationName(ORG)
    prefs = QSettings(ORG, ORG)
    i18n.install(app, str(prefs.value("language", "system")))
    window = MainWindow(settings=prefs)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
