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
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QDesktopServices, QFont, QFontDatabase, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHeaderView, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from marker_qc.camera import CameraDescriptor, CameraLoop, enumerate_cameras
from marker_qc.config import AppSettings, CameraProfile, SettingsStore, full_frame_roi
from marker_qc.gui.dialogs import CalibrationDialog, PointEditorDialog, RoiDialog
from marker_qc.gui.widgets import array_to_qimage
from marker_qc.measurement import draw_measurement_geometry, draw_measurement_text, measure_points
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
        self.setFont(QFont("Segoe UI", 10))
        self.setMinimumSize(1100, 700)
        self.resize(1320, 820)
        self.current_frame: np.ndarray | None = None
        self.snapshot: MeasurementSnapshot | None = None
        self._display_image: np.ndarray | None = None
        self._display_measurement: MeasurementResult | None = None
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
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        camera_panel = QFrame()
        camera_panel.setObjectName("cameraPanel")
        camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        camera_grid = QGridLayout(camera_panel)
        camera_grid.setContentsMargins(16, 12, 16, 12)
        camera_grid.setHorizontalSpacing(10)
        camera_grid.setVerticalSpacing(5)
        camera_title = QLabel("Источник изображения")
        camera_title.setObjectName("sectionTitle")
        camera_grid.addWidget(camera_title, 0, 0, 1, 6)
        camera_label = QLabel("Камера")
        camera_label.setObjectName("fieldLabel")
        resolution_label = QLabel("Разрешение")
        resolution_label.setObjectName("fieldLabel")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(280)
        self.resolution_combo = QComboBox()
        self.resolution_combo.setMinimumWidth(132)
        for resolution in ((640, 480), (1280, 720), (1920, 1080)):
            self.resolution_combo.addItem(f"{resolution[0]}×{resolution[1]}", resolution)
        self.refresh_button = self._button("Обновить камеры", "utility")
        self.connect_button = self._button("Подключить", "secondary")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.connect_button.clicked.connect(self.toggle_camera)
        camera_fields = QHBoxLayout()
        camera_fields.setSpacing(10)
        camera_fields.addWidget(camera_label)
        camera_fields.addWidget(self.camera_combo, 1)
        camera_fields.addSpacing(6)
        camera_fields.addWidget(resolution_label)
        camera_fields.addWidget(self.resolution_combo)
        camera_fields.addWidget(self.refresh_button)
        camera_fields.addWidget(self.connect_button)
        camera_grid.addLayout(camera_fields, 1, 0, 1, 6)
        root.addWidget(camera_panel)

        splitter = QSplitter()
        splitter.setHandleWidth(10)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        self.preview = QLabel("Камера не подключена")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(620, 420)
        self.preview.setObjectName("cameraPreview")
        left_layout.addWidget(self.preview, 1)

        tools_panel = QFrame()
        tools_panel.setObjectName("toolsPanel")
        tools_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tools_layout = QVBoxLayout(tools_panel)
        tools_layout.setContentsMargins(12, 10, 12, 10)
        tools_layout.setSpacing(8)
        secondary_actions = QHBoxLayout()
        secondary_actions.setContentsMargins(0, 0, 0, 0)
        secondary_actions.setSpacing(8)
        self.roi_button = self._button("Область поиска", "secondary")
        self.calibrate_button = self._button("Калибровка", "secondary")
        self.edit_points_button = self._button("Коррекция точек", "secondary")
        self.show_roi_checkbox = QCheckBox("Показывать рамку области поиска")
        self.show_roi_checkbox.setChecked(True)
        self.show_roi_checkbox.toggled.connect(self._roi_visibility_changed)
        self.roi_button.clicked.connect(self.select_roi)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.edit_points_button.clicked.connect(self.edit_points)
        for button in (self.roi_button, self.calibrate_button, self.edit_points_button):
            secondary_actions.addWidget(button)
        secondary_actions.addStretch()
        tools_layout.addLayout(secondary_actions)
        roi_visibility_row = QHBoxLayout()
        roi_visibility_row.setContentsMargins(2, 0, 0, 0)
        roi_visibility_row.addWidget(self.show_roi_checkbox)
        roi_visibility_row.addStretch()
        tools_layout.addLayout(roi_visibility_row)
        left_layout.addWidget(tools_panel)

        self.measure_button = self._button("Измерить", "primary")
        self.measure_button.clicked.connect(self.save_measurement)
        splitter.addWidget(left)

        right = QFrame()
        right.setObjectName("infoPanel")
        self.info_panel = right
        right.setFixedWidth(390)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 18, 20, 18)
        right_layout.setSpacing(12)

        technical_title = QLabel("Камера и калибровка")
        technical_title.setObjectName("sectionTitle")
        right_layout.addWidget(technical_title)
        self.resolution_status_label = QLabel("Разрешение: —")
        self.resolution_status_label.setObjectName("secondaryText")
        right_layout.addWidget(self.resolution_status_label)
        self.calibration_label = QLabel()
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setObjectName("technicalStatus")
        right_layout.addWidget(self.calibration_label)

        divider_one = QFrame()
        divider_one.setFrameShape(QFrame.HLine)
        divider_one.setObjectName("divider")
        right_layout.addWidget(divider_one)
        metrics_title = QLabel("Измеренные значения")
        metrics_title.setObjectName("sectionTitle")
        right_layout.addWidget(metrics_title)
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        self.length_label = self._metric_label("—")
        self.width_label = self._metric_label("—")
        for row, (caption, value) in enumerate((("Длина", self.length_label), ("Ширина", self.width_label))):
            caption_label = QLabel(caption)
            caption_label.setObjectName("metricCaption")
            metrics.addWidget(caption_label, row, 0)
            metrics.addWidget(value, row, 1)
        metrics.setColumnStretch(1, 1)
        right_layout.addLayout(metrics)

        result_title = QLabel("Результат контроля")
        result_title.setObjectName("sectionTitle")
        right_layout.addWidget(result_title)
        self.decision_label = self._metric_label("—")
        self.decision_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.decision_label.setMinimumWidth(0)
        right_layout.addWidget(self.decision_label)
        self.state_label = QLabel("Маркер не обнаружен")
        self.state_label.setWordWrap(True)
        self.state_label.setObjectName("secondaryText")
        right_layout.addWidget(self.state_label)

        divider_two = QFrame()
        divider_two.setFrameShape(QFrame.HLine)
        divider_two.setObjectName("divider")
        right_layout.addWidget(divider_two)
        actions_title = QLabel("Действия")
        actions_title.setObjectName("sectionTitle")
        right_layout.addWidget(actions_title)
        right_layout.addWidget(self.measure_button)
        self.diagnostic_button = self._button("Показать диагностику", "secondary")
        self.diagnostic_button.setCheckable(True)
        self.diagnostic_text = QTextEdit()
        self.diagnostic_text.setReadOnly(True)
        self.diagnostic_text.setObjectName("diagnosticText")
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
        history_layout.setContentsMargins(18, 18, 18, 18)
        history_layout.setSpacing(12)
        history_title = QLabel("История измерений")
        history_title.setObjectName("pageTitle")
        history_layout.addWidget(history_title)
        self.history_table = QTableWidget()
        self.history_table.setObjectName("historyTable")
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Дата и время", "Длина, мм", "Ширина, мм", "Итог", "Точки", "Снимок"])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setShowGrid(False)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.verticalHeader().setDefaultSectionSize(38)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        self.history_table.itemDoubleClicked.connect(self._open_history_image)
        history_layout.addWidget(self.history_table)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.refresh_history_button = self._button("Обновить", "utility")
        self.export_csv_button = self._button("Экспорт CSV", "utility")
        self.export_excel_button = self._button("Экспорт Excel", "utility")
        self.clear_history_button = self._button("Очистить историю", "destructive")
        self.refresh_history_button.clicked.connect(self._refresh_history)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_excel_button.clicked.connect(self.export_excel)
        self.clear_history_button.clicked.connect(self.clear_history)
        buttons.addWidget(self.refresh_history_button)
        buttons.addWidget(self.export_csv_button)
        buttons.addWidget(self.export_excel_button)
        buttons.addStretch()
        buttons.addWidget(self.clear_history_button)
        history_layout.addLayout(buttons)
        self.tabs.addTab(history, "История")

        self.measure_button.setEnabled(False)
        self.edit_points_button.setEnabled(False)
        self.statusBar().showMessage("Готово")
        self.camera_combo.currentIndexChanged.connect(self._update_calibration_status)
        self.resolution_combo.currentIndexChanged.connect(self._update_calibration_status)
        self._apply_styles()

    @staticmethod
    def _button(text: str, role: str) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("buttonRole", role)
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedHeight(44 if role == "primary" else 36)
        return button

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QTabWidget::pane { background: #f3f5f7; }
            QMenuBar { background: #ffffff; border-bottom: 1px solid #e4e8ec; padding: 3px; }
            QMenuBar::item:selected { background: #eef4fa; border-radius: 4px; }
            QTabBar::tab { background: transparent; color: #53606d; padding: 10px 20px; margin-right: 2px; }
            QTabBar::tab:selected { color: #235f92; font-weight: 600; border-bottom: 2px solid #3478b8; }
            QFrame#cameraPanel, QFrame#toolsPanel, QFrame#infoPanel {
                background: #ffffff; border: 1px solid #e1e6ea; border-radius: 8px;
            }
            QLabel#cameraPreview {
                background: #171a1d; color: #aeb7c0; border: 1px solid #d8dee4; border-radius: 8px;
            }
            QLabel#sectionTitle, QLabel#pageTitle { color: #20272e; font-weight: 650; font-size: 15px; }
            QLabel#pageTitle { font-size: 18px; }
            QLabel#fieldLabel { color: #5d6873; font-size: 13px; }
            QLabel#metricCaption { color: #69747f; font-size: 14px; font-weight: 550; }
            QLabel#secondaryText { color: #66727e; font-size: 13px; }
            QLabel#technicalStatus { font-size: 13px; }
            QFrame#divider { color: #e7ebef; background: #e7ebef; max-height: 1px; border: none; }
            QComboBox {
                background: #ffffff; color: #26313a; border: 1px solid #cfd7de; border-radius: 6px;
                min-height: 34px; padding: 0 10px;
            }
            QComboBox:hover { border-color: #8faac2; }
            QComboBox:focus { border: 1px solid #3478b8; }
            QPushButton {
                border-radius: 7px; padding: 0 14px; font-size: 13px; font-weight: 550;
            }
            QPushButton[buttonRole="primary"] { background: #2f73ad; color: white; border: 1px solid #2f73ad; }
            QPushButton[buttonRole="primary"]:hover { background: #28679c; border-color: #28679c; }
            QPushButton[buttonRole="primary"]:pressed { background: #225985; border-color: #225985; }
            QPushButton[buttonRole="primary"]:disabled { background: #c7d1d9; color: #f4f6f7; border-color: #c7d1d9; }
            QPushButton[buttonRole="secondary"] { background: #ffffff; color: #2d5f87; border: 1px solid #b9c9d7; }
            QPushButton[buttonRole="secondary"]:hover { background: #f0f6fb; border-color: #7fa2bf; }
            QPushButton[buttonRole="secondary"]:pressed, QPushButton[buttonRole="secondary"]:checked { background: #e2eef7; border-color: #5e8eb5; }
            QPushButton[buttonRole="secondary"]:disabled { background: #f5f6f7; color: #a6afb7; border-color: #e1e5e8; }
            QPushButton[buttonRole="utility"] { background: #f7f9fa; color: #394650; border: 1px solid #d3dae0; }
            QPushButton[buttonRole="utility"]:hover { background: #edf2f5; border-color: #aebbc5; }
            QPushButton[buttonRole="utility"]:pressed { background: #e2e8ec; }
            QPushButton[buttonRole="utility"]:disabled { background: #f5f6f7; color: #a6afb7; border-color: #e4e7e9; }
            QPushButton[buttonRole="destructive"] { background: #ffffff; color: #b33a3a; border: 1px solid #dfaaaa; }
            QPushButton[buttonRole="destructive"]:hover { background: #fff2f1; border-color: #cc7474; }
            QPushButton[buttonRole="destructive"]:pressed { background: #f8dfdd; }
            QPushButton[buttonRole="destructive"]:disabled { color: #b9a6a6; border-color: #e6dada; }
            QCheckBox { color: #46525c; spacing: 8px; font-size: 13px; }
            QTextEdit#diagnosticText { background: #f7f9fa; color: #44515c; border: 1px solid #dce2e7; border-radius: 6px; padding: 8px; }
            QTableWidget#historyTable { background: #ffffff; alternate-background-color: #f8fafb; border: 1px solid #e0e5e9; border-radius: 8px; color: #2b343b; }
            QTableWidget#historyTable::item { padding: 7px; border-bottom: 1px solid #edf0f2; }
            QTableWidget#historyTable::item:selected { background: #dcebf7; color: #20303d; }
            QHeaderView::section { background: #eef2f5; color: #4c5964; border: none; border-right: 1px solid #dde3e8; border-bottom: 1px solid #d8dfe5; padding: 9px; font-weight: 600; }
            QStatusBar { background: #ffffff; color: #5f6b75; border-top: 1px solid #e1e6ea; }
        """)

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

    def _roi_visibility_changed(self, _visible: bool) -> None:
        if self._display_image is not None:
            self._show_image(self._display_image, self._display_measurement)

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
        index = next(
            (item_index for item_index in range(self.resolution_combo.count())
             if tuple(self.resolution_combo.itemData(item_index)) == tuple(profile.resolution)),
            -1,
        )
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
        geometry_overlay = draw_measurement_geometry(packet.frame, measurement)
        persisted_overlay = draw_measurement_text(geometry_overlay, measurement, self.settings.norms)
        self.snapshot = MeasurementSnapshot(packet.sequence, packet.frame, processing, measurement, persisted_overlay)
        self._show_image(geometry_overlay, measurement)
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
        color = "#18864b" if result.dimensions_ok else "#c13d3d"
        self.length_label.setStyleSheet(f"color:{'#18864b' if result.length_ok else '#c13d3d'}")
        self.width_label.setStyleSheet(f"color:{'#18864b' if result.width_ok else '#c13d3d'}")
        self.decision_label.setText("Соответствует" if result.dimensions_ok else "Вне допуска")
        self.decision_label.setStyleSheet(f"color:{color}")
        self.state_label.setText("Точки исправлены вручную; нажмите «Измерить» для сохранения." if not result.automatic else "Автоматическое измерение")

    def _show_image(self, image: np.ndarray, measurement: MeasurementResult | None = None) -> None:
        self._display_image = image
        self._display_measurement = measurement
        displayed = image.copy()
        if displayed.ndim == 3 and self.show_roi_checkbox.isChecked():
            roi = self._roi_for_frame(displayed)
            if roi != full_frame_roi((displayed.shape[1], displayed.shape[0])):
                x, y, width, height = roi
                cv2.rectangle(displayed, (x, y), (x + width - 1, y + height - 1), (0, 0, 220), 2)
        # Текст рисуется последним: тёмная подложка закрывает рамку ROI под ним.
        if measurement is not None:
            displayed = draw_measurement_text(displayed, measurement, self.settings.norms)
        pixmap = QPixmap.fromImage(array_to_qimage(displayed))
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._display_image is not None:
            self._show_image(self._display_image, self._display_measurement)

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
        selected = self._selected_camera()
        if selected:
            descriptor, resolution = selected
            actual = ((self.current_frame.shape[1], self.current_frame.shape[0])
                      if self.current_frame is not None else resolution)
            self.resolution_status_label.setText(
                f"{descriptor.name} · {descriptor.backend} · {actual[0]}×{actual[1]}"
            )
        else:
            self.resolution_status_label.setText("Камера и разрешение не выбраны")
        calibration = self._valid_calibration()
        if calibration:
            self.calibration_label.setText(f"Калибровка активна · {calibration.scale_mm_per_px:.8f} мм/px")
            self.calibration_label.setStyleSheet("color:#18864b")
        else:
            self.calibration_label.setText("Нужна калибровка для выбранной камеры и разрешения")
            self.calibration_label.setStyleSheet("color:#c13d3d")

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
        geometry_overlay = draw_measurement_geometry(snapshot.frame, measurement)
        persisted_overlay = draw_measurement_text(geometry_overlay, measurement, self.settings.norms)
        self.snapshot = MeasurementSnapshot(
            snapshot.sequence, snapshot.frame, snapshot.processing, measurement, persisted_overlay
        )
        self._show_image(geometry_overlay, measurement)
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
                "Авто" if row["automatic"] else "Вручную", Path(row["overlay_path"]).name]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in (1, 2):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                elif column in (3, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 3:
                    item.setForeground(QBrush(QColor("#18864b" if row["dimensions_ok"] else "#c13d3d")))
                if column == 5:
                    item.setData(Qt.UserRole, row["overlay_path"])
                    item.setToolTip(f"Двойной щелчок — открыть снимок\n{row['overlay_path']}")
                self.history_table.setItem(row_index, column, item)

    def _open_history_image(self, item: QTableWidgetItem) -> None:
        if item.column() != 5:
            return
        path_text = item.data(Qt.UserRole)
        if not path_text:
            return
        path = Path(path_text)
        if not path.is_file():
            self.statusBar().showMessage(f"Файл снимка не найден: {path}", 6000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

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
