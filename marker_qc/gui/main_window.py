from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QFontDatabase, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from marker_qc.camera import CameraDescriptor, CameraLoop, enumerate_cameras
from marker_qc.config import AppSettings, CameraProfile, SettingsStore, full_frame_roi
from marker_qc.gui.dialogs import CalibrationDialog, PointEditorDialog, RoiDialog
from marker_qc.gui.widgets import array_to_qimage
from marker_qc.measurement import draw_measurement, measure_points
from marker_qc.models import Calibration, Decision, InspectionDraft, InspectionStage, MeasurementResult
from marker_qc.paths import AppPaths
from marker_qc.processing import ProcessingResult, process_frame
from marker_qc.realtime import AnalysisPacket, LatestFrameMailbox
from marker_qc.storage import Storage, StorageError


class CameraWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, descriptor: CameraDescriptor, resolution: tuple[int, int]):
        super().__init__()
        self.descriptor = descriptor
        self.resolution = resolution
        self.loop = CameraLoop()

    @Slot()
    def run(self) -> None:
        self.loop.run(self.descriptor, self.resolution, self.frame_ready.emit, self.error.emit)
        self.finished.emit()

    def stop(self) -> None:
        self.loop.stop()


class ScanWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(enumerate_cameras(cancelled=self._stop.is_set))
        except Exception as exc:
            self.error.emit(str(exc))


class RealtimeAnalysisWorker(QObject):
    result_ready = Signal(object)
    finished = Signal()

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.mailbox = LatestFrameMailbox()

    def submit(self, sequence: int, frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        self.mailbox.put(sequence, frame, roi)

    def stop(self) -> None:
        self.mailbox.close()

    @Slot()
    def run(self) -> None:
        while True:
            item = self.mailbox.take()
            if item is None:
                break
            sequence, frame, roi = item
            try:
                packet = AnalysisPacket(sequence, frame, process_frame(frame, roi, self.settings))
            except Exception as exc:
                packet = AnalysisPacket(sequence, frame, None, str(exc))
            self.result_ready.emit(packet)
        self.finished.emit()


@dataclass(frozen=True)
class MeasurementSnapshot:
    sequence: int
    frame: np.ndarray
    processing: ProcessingResult
    measurement: MeasurementResult
    overlay: np.ndarray


class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths, settings_store: SettingsStore, settings: AppSettings, storage: Storage):
        super().__init__()
        self.paths, self.settings_store, self.settings, self.storage = paths, settings_store, settings, storage
        self.setWindowTitle("Marker QC — контроль качества маркеров")
        self.resize(1320, 820)
        self.current_frame: np.ndarray | None = None
        self.snapshot: MeasurementSnapshot | None = None
        self._display_image: np.ndarray | None = None
        self._manual_frozen = False
        self._saving = False
        self._frame_sequence = 0
        self._last_saved_sequence = -1
        self._connect_started: float | None = None
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.analysis_thread: QThread | None = None
        self.analysis_worker: RealtimeAnalysisWorker | None = None
        self._build_ui()
        self._restore_last_camera()
        self._refresh_history()
        self._update_calibration_status()
        if not os.environ.get("MARKER_QC_SKIP_CAMERA_SCAN") and self.settings.last_camera is None:
            self.refresh_cameras()

    def _build_ui(self) -> None:
        menu = self.menuBar().addMenu("Настройки")
        data_action = QAction("Каталог данных…", self)
        data_action.triggered.connect(lambda: QMessageBox.information(self, "Каталог данных", str(self.paths.root)))
        menu.addAction(data_action)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        control = QWidget()
        root = QVBoxLayout(control)
        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.resolution_combo = QComboBox()
        for resolution in ((640, 480), (1280, 720), (1920, 1080)):
            self.resolution_combo.addItem(f"{resolution[0]}×{resolution[1]}", resolution)
        self.refresh_button = QPushButton("Обновить камеры")
        self.connect_button = QPushButton("Подключить")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.connect_button.clicked.connect(self.toggle_camera)
        camera_row.addWidget(QLabel("Камера:"))
        camera_row.addWidget(self.camera_combo, 1)
        camera_row.addWidget(QLabel("Разрешение:"))
        camera_row.addWidget(self.resolution_combo)
        camera_row.addWidget(self.refresh_button)
        camera_row.addWidget(self.connect_button)
        root.addLayout(camera_row)
        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.preview = QLabel("Камера не подключена")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(720, 540)
        self.preview.setStyleSheet("background:#202124;color:#ddd;border:1px solid #555")
        left_layout.addWidget(self.preview, 1)
        secondary_actions = QHBoxLayout()
        secondary_actions.setSpacing(8)
        self.roi_button = QPushButton("Область поиска")
        self.calibrate_button = QPushButton("Калибровка")
        self.edit_points_button = QPushButton("Коррекция точек")
        self.roi_button.clicked.connect(self.select_roi)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.edit_points_button.clicked.connect(self.edit_points)
        for button in (self.roi_button, self.calibrate_button, self.edit_points_button):
            button.setFixedHeight(36)
            secondary_actions.addWidget(button)
        left_layout.addLayout(secondary_actions)
        self.measure_button = QPushButton("ИЗМЕРИТЬ")
        self.measure_button.setMinimumHeight(62)
        self.measure_button.setStyleSheet(
            "QPushButton{font-size:20px;font-weight:bold;background:#1769aa;color:white;border-radius:5px;}"
            "QPushButton:disabled{background:#aeb8c2;color:#eef1f3;}"
            "QPushButton:hover:enabled{background:#0d5d9f;}"
        )
        self.measure_button.clicked.connect(self.save_measurement)
        left_layout.addWidget(self.measure_button)
        splitter.addWidget(left)
        right = QWidget()
        self.info_panel = right
        right.setFixedWidth(390)
        right_layout = QVBoxLayout(right)
        self.calibration_label = QLabel()
        self.calibration_label.setWordWrap(True)
        right_layout.addWidget(self.calibration_label)
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(10)
        self.length_label = self._metric_label("—")
        self.width_label = self._metric_label("—")
        self.decision_label = self._metric_label("—")
        for row, (caption, value) in enumerate((("Длина:", self.length_label), ("Ширина:", self.width_label), ("Итог:", self.decision_label))):
            caption_label = QLabel(caption)
            caption_label.setStyleSheet("font-size:20px;font-weight:600;color:#333")
            metrics.addWidget(caption_label, row, 0)
            metrics.addWidget(value, row, 1)
        metrics.setColumnStretch(1, 1)
        right_layout.addLayout(metrics)
        self.state_label = QLabel("Маркер не обнаружен")
        self.state_label.setWordWrap(True)
        self.state_label.setStyleSheet("color:#666;font-size:16px")
        right_layout.addWidget(self.state_label)
        self.diagnostic_button = QPushButton("Показать диагностику")
        self.diagnostic_button.setCheckable(True)
        self.diagnostic_text = QTextEdit()
        self.diagnostic_text.setReadOnly(True)
        self.diagnostic_text.hide()
        self.diagnostic_button.toggled.connect(self._toggle_diagnostics)
        right_layout.addWidget(self.diagnostic_button)
        right_layout.addWidget(self.diagnostic_text)
        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)
        self.tabs.addTab(control, "Измерение")
        history = QWidget()
        history_layout = QVBoxLayout(history)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Дата", "Длина", "Ширина", "Итог", "Точки", "Фотография"])
        history_layout.addWidget(self.history_table)
        buttons = QHBoxLayout()
        refresh_history = QPushButton("Обновить")
        export = QPushButton("Экспорт CSV")
        export_excel = QPushButton("Экспорт Excel")
        self.clear_history_button = QPushButton("Очистить историю")
        self.clear_history_button.setStyleSheet("color:#b00020")
        refresh_history.clicked.connect(self._refresh_history)
        export.clicked.connect(self.export_csv)
        export_excel.clicked.connect(self.export_excel)
        self.clear_history_button.clicked.connect(self.clear_history)
        buttons.addWidget(refresh_history)
        buttons.addWidget(export)
        buttons.addWidget(export_excel)
        buttons.addStretch()
        buttons.addWidget(self.clear_history_button)
        history_layout.addLayout(buttons)
        self.tabs.addTab(history, "История")
        self.measure_button.setEnabled(False)
        self.edit_points_button.setEnabled(False)
        self.statusBar().showMessage("Готово")

    @staticmethod
    def _metric_label(text: str) -> QLabel:
        label = QLabel(text)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(20)
        font.setBold(True)
        label.setFont(font)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setMinimumWidth(225)
        label.setStyleSheet("color:#666")
        return label

    def _toggle_diagnostics(self, visible: bool) -> None:
        self.diagnostic_text.setVisible(visible)
        self.diagnostic_button.setText("Скрыть диагностику" if visible else "Показать диагностику")

    def _error(self, message: str) -> None:
        logging.error(message)
        QMessageBox.critical(self, "Ошибка", message)
        self.statusBar().showMessage(message, 10000)

    def _restore_last_camera(self) -> None:
        profile = self.settings.last_camera
        if not profile:
            return
        descriptor = CameraDescriptor(
            profile.index, profile.backend, profile.resolution[0], profile.resolution[1], profile.name
        )
        self.camera_combo.clear()
        self.camera_combo.addItem(f"{descriptor.label} — сохранённая", descriptor)
        index = self.resolution_combo.findData(profile.resolution)
        if index < 0:
            self.resolution_combo.addItem(f"{profile.resolution[0]}×{profile.resolution[1]}", profile.resolution)
            index = self.resolution_combo.count() - 1
        self.resolution_combo.setCurrentIndex(index)
        self.statusBar().showMessage("Загружена последняя работающая камера. Нажмите «Подключить».")

    def refresh_cameras(self) -> None:
        if self.scan_thread and self.scan_thread.isRunning():
            return
        self.camera_combo.clear()
        self.camera_combo.addItem("Поиск камер…", None)
        self.refresh_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.scan_thread = QThread(self)
        self.scan_worker = ScanWorker()
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.finished.connect(self._cameras_found)
        self.scan_worker.error.connect(self._error)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.error.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self._scan_finished)
        self.scan_thread.start()

    def _scan_finished(self) -> None:
        self.scan_worker = None
        self.scan_thread = None
        self.refresh_button.setEnabled(True)
        self.connect_button.setEnabled(True)

    def _cameras_found(self, cameras) -> None:
        self.camera_combo.clear()
        for camera in cameras:
            self.camera_combo.addItem(camera.label, camera)
        if not cameras:
            self.camera_combo.addItem("Камеры не найдены", None)

    def toggle_camera(self) -> None:
        if self.camera_thread and self.camera_thread.isRunning():
            self.disconnect_camera()
            return
        if self.scan_thread and self.scan_thread.isRunning():
            self.statusBar().showMessage("Дождитесь завершения поиска камер.", 4000)
            return
        descriptor = self.camera_combo.currentData()
        if not descriptor:
            self._error("Выберите доступную камеру.")
            return
        self._start_analysis()
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(descriptor, tuple(self.resolution_combo.currentData()))
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.error.connect(self._camera_error)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_thread.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.start()
        self._connect_started = time.perf_counter()
        logging.info("Camera connection requested: index=%s backend=%s resolution=%s", descriptor.index, descriptor.backend, self.resolution_combo.currentData())
        self.connect_button.setText("Отключить")
        self.statusBar().showMessage("Подключение к камере…")

    def _start_analysis(self) -> None:
        self.analysis_thread = QThread(self)
        self.analysis_worker = RealtimeAnalysisWorker(self.settings.processing)
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.result_ready.connect(self._analysis_ready)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.start()

    def disconnect_camera(self) -> bool:
        if self.camera_worker:
            self.camera_worker.stop()
        if self.camera_thread:
            self.camera_thread.quit()
            if not self.camera_thread.wait(5000):
                logging.warning("Camera thread is still blocked inside the driver after 5 seconds")
                self.statusBar().showMessage("Драйвер камеры ещё завершает операцию. Повторите отключение через несколько секунд.", 8000)
                return False
        if self.analysis_worker:
            self.analysis_worker.stop()
        if self.analysis_thread:
            if not self.analysis_thread.wait(5000):
                logging.warning("Analysis thread did not finish within 5 seconds")
                return False
        self.camera_worker = self.camera_thread = None
        self.analysis_worker = self.analysis_thread = None
        self.snapshot = None
        self._manual_frozen = False
        self.measure_button.setEnabled(False)
        self.edit_points_button.setEnabled(False)
        self.connect_button.setText("Подключить")
        self._show_unavailable("Маркер не обнаружен")
        self.statusBar().showMessage("Камера отключена")
        return True

    def _camera_error(self, message: str) -> None:
        self.connect_button.setText("Подключить")
        if self.analysis_worker:
            self.analysis_worker.stop()
        logging.error("Camera connection failed: %s", message)
        self._error(message + "\nНажмите «Обновить камеры», если сохранённое устройство недоступно.")

    def _on_frame(self, frame: np.ndarray) -> None:
        self.current_frame = frame.copy()
        self._frame_sequence += 1
        selected = self._selected_camera()
        if selected:
            descriptor, _ = selected
            actual = (frame.shape[1], frame.shape[0])
            profile = CameraProfile(descriptor.index, descriptor.backend, actual, descriptor.name)
            if self.settings.last_camera != profile:
                self.settings.last_camera = profile
                self.settings_store.save(self.settings)
            if self._connect_started is not None:
                logging.info("First camera frame reached UI in %.3fs", time.perf_counter() - self._connect_started)
                self._connect_started = None
        self._update_calibration_status()
        if self.analysis_worker:
            self.analysis_worker.submit(self._frame_sequence, frame, self._roi_for_frame(frame))
        if self.snapshot is None and not self._manual_frozen:
            self._show_image(frame)
        self.statusBar().showMessage(f"Камера подключена: {frame.shape[1]}×{frame.shape[0]}; анализ в реальном времени")

    def _analysis_ready(self, packet: AnalysisPacket) -> None:
        if self._manual_frozen:
            return
        calibration = self._valid_calibration(packet.frame)
        if packet.error or not packet.processing or not calibration:
            self.snapshot = None
            self._show_image(packet.frame)
            message = packet.error or "Нужна калибровка для текущей камеры и разрешения"
            self._show_unavailable("Маркер не обнаружен" if "не найден" in message.lower() else "Не удалось измерить")
            self.diagnostic_text.setPlainText(message)
            return
        processing = packet.processing
        if processing.roi_rect != (0, 0, 0, 0) and processing.roi_rect != self._roi_for_frame(packet.frame):
            return
        measurement = measure_points(processing.points, calibration.scale_mm_per_px, self.settings.norms,
                                     automatic=True, confidence=processing.confidence,
                                     diagnostics=processing.diagnostics)
        overlay = draw_measurement(packet.frame, measurement, self.settings.norms)
        self.snapshot = MeasurementSnapshot(packet.sequence, packet.frame, processing, measurement, overlay)
        self._show_image(overlay)
        self._show_result(measurement)
        self.diagnostic_text.setPlainText("\n".join(processing.diagnostics))
        self.measure_button.setEnabled(True)
        self.edit_points_button.setEnabled(True)

    def _show_unavailable(self, message: str) -> None:
        self.length_label.setText("—")
        self.width_label.setText("—")
        self.decision_label.setText("—")
        for label in (self.length_label, self.width_label, self.decision_label):
            label.setStyleSheet("color:#777")
        self.state_label.setText(message)
        self.measure_button.setEnabled(False)
        self.edit_points_button.setEnabled(False)

    def _show_result(self, result: MeasurementResult) -> None:
        self.length_label.setText(f"{result.length_mm:.4f} мм")
        self.width_label.setText(f"{result.width_mm:.4f} мм")
        color = "#087f23" if result.dimensions_ok else "#b00020"
        self.length_label.setStyleSheet(f"color:{'#087f23' if result.length_ok else '#b00020'}")
        self.width_label.setStyleSheet(f"color:{'#087f23' if result.width_ok else '#b00020'}")
        self.decision_label.setText("ДА" if result.dimensions_ok else "НЕТ")
        self.decision_label.setStyleSheet(f"color:{color}")
        self.state_label.setText("Точки исправлены вручную; нажмите «Измерить» для сохранения." if not result.automatic else "Автоматическое измерение")

    def _show_image(self, image: np.ndarray) -> None:
        self._display_image = image
        displayed = image.copy()
        if displayed.ndim == 3:
            roi = self._roi_for_frame(displayed)
            if roi != full_frame_roi((displayed.shape[1], displayed.shape[0])):
                x, y, width, height = roi
                cv2.rectangle(displayed, (x, y), (x + width - 1, y + height - 1), (0, 210, 255), 2)
        pixmap = QPixmap.fromImage(array_to_qimage(displayed))
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._display_image is not None:
            self._show_image(self._display_image)

    def _selected_camera(self) -> tuple[CameraDescriptor, tuple[int, int]] | None:
        descriptor, resolution = self.camera_combo.currentData(), self.resolution_combo.currentData()
        return (descriptor, tuple(resolution)) if descriptor and resolution else None

    def _roi_for_frame(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        selected = self._selected_camera()
        resolution = (frame.shape[1], frame.shape[0])
        if not selected:
            return full_frame_roi(resolution)
        descriptor, _ = selected
        return self.settings.roi_for(descriptor.index, descriptor.backend, resolution)

    def select_roi(self) -> None:
        selected = self._selected_camera()
        if self.current_frame is None or not selected:
            self._error("Сначала подключите камеру и дождитесь изображения.")
            return
        frame = self.current_frame.copy()
        descriptor, _ = selected
        resolution = (frame.shape[1], frame.shape[0])
        dialog = RoiDialog(frame, self.settings.roi_for(descriptor.index, descriptor.backend, resolution), self)
        if not dialog.exec():
            return
        roi = dialog.roi
        if roi is None:
            return
        try:
            self.settings.set_roi(descriptor.index, descriptor.backend, resolution, roi)
        except ValueError as exc:
            self._error(str(exc))
            return
        self.settings_store.save(self.settings)
        self.snapshot = None
        self._show_unavailable("Область поиска изменена; ожидается новое измерение")
        self._show_image(frame)
        self.statusBar().showMessage(f"Область поиска: {roi}. Калибровка сохранена.", 6000)

    def calibrate(self) -> None:
        selected = self._selected_camera()
        if self.current_frame is None or not selected:
            self._error("Сначала подключите камеру и дождитесь изображения.")
            return
        descriptor, _ = selected
        frame = self.current_frame.copy()
        dialog = CalibrationDialog(frame, self)
        if not dialog.exec():
            return
        p1, p2, distance, scale = dialog.calibration_data
        height, width = frame.shape[:2]
        self.settings.calibration = Calibration.create(descriptor.index, descriptor.backend, descriptor.name,
            (width, height), (0, 0, width, height), scale, distance, (p1, p2))
        self.settings_store.save(self.settings)
        self.storage.audit("", "CALIBRATION_CHANGED", self.settings.calibration.__dict__)
        self._update_calibration_status()

    def _valid_calibration(self, frame: np.ndarray | None = None) -> Calibration | None:
        selected, calibration = self._selected_camera(), self.settings.calibration
        if not selected or not calibration:
            return None
        descriptor, requested = selected
        source = frame if frame is not None else self.current_frame
        actual = (source.shape[1], source.shape[0]) if source is not None else requested
        return calibration if calibration.matches(descriptor.index, descriptor.backend, actual) else None

    def _update_calibration_status(self) -> None:
        calibration = self._valid_calibration()
        if calibration:
            self.calibration_label.setText(f"Калибровка: {calibration.scale_mm_per_px:.8f} мм/px · {calibration.resolution[0]}×{calibration.resolution[1]}")
            self.calibration_label.setStyleSheet("color:#087f23")
        else:
            self.calibration_label.setText("Нужна калибровка для выбранной камеры и разрешения")
            self.calibration_label.setStyleSheet("color:#b00020")

    def edit_points(self) -> None:
        snapshot = self.snapshot
        if not snapshot:
            self._error("Нет корректного измерения для редактирования.")
            return
        self._manual_frozen = True
        dialog = PointEditorDialog(snapshot.frame, snapshot.measurement.points,
                                   snapshot.measurement.scale_mm_per_px, self)
        if not dialog.exec():
            self._manual_frozen = False
            return
        measurement = measure_points(dialog.points, snapshot.measurement.scale_mm_per_px,
            self.settings.norms, automatic=False, confidence=1.0,
            diagnostics=snapshot.measurement.diagnostics + ("Измерительные точки скорректированы вручную.",))
        overlay = draw_measurement(snapshot.frame, measurement, self.settings.norms)
        self.snapshot = MeasurementSnapshot(snapshot.sequence, snapshot.frame, snapshot.processing, measurement, overlay)
        self._show_image(overlay)
        self._show_result(measurement)
        self.measure_button.setEnabled(True)

    def save_measurement(self) -> None:
        snapshot = self.snapshot
        if self._saving or not snapshot:
            return
        if snapshot.sequence == self._last_saved_sequence:
            self.statusBar().showMessage("Этот кадр уже сохранён. Дождитесь следующего измерения.", 4000)
            return
        self._saving = True
        self.measure_button.setEnabled(False)
        result = snapshot.measurement
        decision = Decision.APPROVED_STAGE1 if result.dimensions_ok else Decision.REJECTED
        draft = InspectionDraft("__REALTIME__",
            f"AUTO-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}", "",
            InspectionStage.FIRST, result, True, True, decision,
            "Ручная корректировка точек" if not result.automatic else "")
        try:
            inspection_id = self.storage.save_inspection(draft, self.settings.norms,
                {"raw": snapshot.frame, "roi": snapshot.processing.roi,
                 "mask": snapshot.processing.mask, "overlay": snapshot.overlay})
            self._last_saved_sequence = snapshot.sequence
            self.statusBar().showMessage(f"Результат сохранён · запись {inspection_id[:8]}", 6000)
            self._refresh_history()
            self._manual_frozen = False
        except (StorageError, OSError) as exc:
            self._error(str(exc))
        finally:
            self._saving = False
            QTimer.singleShot(700, lambda: self.measure_button.setEnabled(self.snapshot is not None))

    def _refresh_history(self) -> None:
        rows = self.storage.history()
        self.history_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [row["created_at"], f"{row['length_mm']:.4f}", f"{row['width_mm']:.4f}",
                "ДА" if row["dimensions_ok"] else "НЕТ",
                "Авто" if row["automatic"] else "Вручную", row["overlay_path"]]
            for column, value in enumerate(values):
                self.history_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.history_table.resizeColumnsToContents()

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт результатов", "marker_qc_export.csv", "CSV (*.csv)")
        if path:
            try:
                self.storage.export_csv(Path(path))
                self.statusBar().showMessage(f"CSV сохранён: {path}", 6000)
            except OSError as exc:
                self._error(f"Не удалось экспортировать CSV: {exc}")

    def export_excel(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт Excel", "marker_qc_export.xlsx", "Excel (*.xlsx)")
        if path:
            try:
                self.storage.export_xlsx(Path(path))
                self.statusBar().showMessage(f"Excel сохранён: {path}", 6000)
            except (OSError, StorageError) as exc:
                self._error(f"Не удалось экспортировать Excel: {exc}")

    def clear_history(self) -> None:
        if not self._confirm_clear_history():
            return
        try:
            result = self.storage.clear_history()
            self._refresh_history()
            message = f"Удалено записей: {result.deleted_records}; файлов: {result.deleted_files}."
            if result.warnings:
                message += "\n" + "\n".join(result.warnings)
                logging.warning("History cleanup warnings: %s", result.warnings)
            QMessageBox.information(self, "История очищена", message)
        except StorageError as exc:
            logging.exception("History cleanup failed")
            self._error(str(exc))

    def _confirm_clear_history(self) -> bool:
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Warning)
        confirmation.setWindowTitle("Очистить историю")
        confirmation.setText("Удалить всю историю измерений и связанные фотографии? Это действие невозможно отменить.")
        confirmation.addButton("Отмена", QMessageBox.RejectRole)
        delete_button = confirmation.addButton("Удалить", QMessageBox.DestructiveRole)
        confirmation.exec()
        return confirmation.clickedButton() is delete_button

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.disconnect_camera():
            event.ignore()
            return
        if self.scan_worker:
            self.scan_worker.stop()
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.quit()
            if not self.scan_thread.wait(5000):
                logging.warning("Camera scan is still blocked inside the driver")
                self.statusBar().showMessage("Поиск камер ещё завершается. Повторите закрытие через несколько секунд.", 8000)
                event.ignore()
                return
        event.accept()
