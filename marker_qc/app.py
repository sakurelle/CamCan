from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from marker_qc.config import AppSettings, SettingsStore
from marker_qc.gui.main_window import MainWindow
from marker_qc.paths import AppPaths
from marker_qc.storage import Storage


def configure_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
    )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Marker QC")
    app.setOrganizationName("MarkerQC")
    paths = AppPaths.resolve()
    configure_logging(paths.logs / "marker_qc.log")
    store = SettingsStore(paths.settings)
    try:
        settings = store.load()
    except RuntimeError as exc:
        settings = AppSettings()
        QMessageBox.warning(None, "Настройки восстановлены", str(exc))
    try:
        storage = Storage(paths.database, paths.images)
    except Exception as exc:
        QMessageBox.critical(None, "Ошибка запуска", f"Не удалось открыть базу данных:\n{exc}")
        return 2
    window = MainWindow(paths, store, settings, storage)
    window.show()
    if os.environ.get("MARKER_QC_SMOKE_TEST"):
        (paths.root / "smoke-ready.txt").write_text("MarkerQC GUI initialized", encoding="utf-8")
        QTimer.singleShot(1000, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

