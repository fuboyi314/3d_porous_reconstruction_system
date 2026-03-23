"""Application entry point for the porous media reconstruction desktop app."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.core.config import DEFAULT_RUNTIME_CONFIG, PROJECT_NAME
from src.ui.main_window import MainWindow


def main() -> int:
    """Start the Qt application and display the main window."""
    runtime_config = DEFAULT_RUNTIME_CONFIG
    runtime_config.ensure_directories()

    app = QApplication(sys.argv)
    app.setApplicationName(PROJECT_NAME)

    window = MainWindow(runtime_config=runtime_config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
