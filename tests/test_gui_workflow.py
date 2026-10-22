import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["MARKER_QC_SKIP_CAMERA_SCAN"] = "1"

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication

from marker_qc.camera import CameraDescriptor
from marker_qc.config import AppSettings, CameraProfile, SettingsStore
from marker_qc.gui.main_window import MainWindow
from marker_qc.models import Calibration, MeasurementPoints
from marker_qc.paths import AppPaths
from marker_qc.processing import ProcessingResult
from marker_qc.realtime import AnalysisPacket
from marker_qc.storage import Storage
from marker_qc.measurement import draw_measurement_geometry, measure_points
from marker_qc.config import Norms
from marker_qc.gui.widgets import RoiCanvas


def test_live_result_is_not_saved_until_measure_and_double_save_is_blocked(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    frame = np.full((400, 640, 3), 240, dtype=np.uint8)
    points = MeasurementPoints((100, 100), (450, 100), (100, 100), (450, 100), (275, 158))
    processing = ProcessingResult(
        frame.copy(), np.zeros((400, 640), np.uint8), np.ones((400, 640), np.uint8) * 255,
        frame.copy(), points, 0.9, ("test",),
    )
    descriptor = CameraDescriptor(0, "DirectShow", 640, 400, "Тестовая камера")
    settings = AppSettings(calibration=Calibration.create(
        0, "DirectShow", descriptor.name, (640, 400), (10, 10, 20, 20),
        0.02, 10.0, ((0, 0), (500, 0)),
    ))
    storage = Storage(paths.database, paths.images)
    window = MainWindow(paths, SettingsStore(paths.settings), settings, storage)
    window.camera_combo.clear()
    window.camera_combo.addItem(descriptor.label, descriptor)

    window._analysis_ready(AnalysisPacket(7, frame, processing))
    assert storage.history() == []
    assert window._display_image is not None
    assert np.all(window._display_image[:80] == 240)
    assert tuple(window._display_image[100, 200]) != (240, 240, 240)
    window.save_measurement()
    assert len(storage.history()) == 1
    saved_overlay = cv2.imread(storage.history()[0]["overlay_path"])
    assert saved_overlay is not None
    assert np.all(saved_overlay[:80] == 240)
    assert tuple(saved_overlay[100, 200]) != (240, 240, 240)
    window.save_measurement()
    assert len(storage.history()) == 1
    window.close()
    app.processEvents()


def test_information_panel_width_and_metric_fonts_are_stable(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    window = MainWindow(paths, SettingsStore(paths.settings), AppSettings(), Storage(paths.database, paths.images))
    width = window.info_panel.width()
    application_font = app.font()
    metric_fonts = [
        label.font()
        for label in (window.length_label, window.width_label, window.bend_label, window.decision_label)
    ]
    assert window.styleSheet() == ""
    assert window.camera_combo.styleSheet() == ""
    assert window.measure_button.styleSheet() == ""
    assert window.show_roi_checkbox.styleSheet() == ""
    assert window.tabs.styleSheet() == ""
    window._show_unavailable("Маркер не обнаружен")
    assert window.info_panel.width() == width == 360
    points = MeasurementPoints((0, 0), (350, 0), (0, 0), (350, 0), (175, 58))
    window._show_result(measure_points(points, 0.02, Norms(), automatic=True, confidence=1.0))
    assert window.info_panel.width() == width
    assert window.decision_label.text() == "Соответствует"
    assert window.bend_label.text() == "Да · t=0.500"
    window._show_result(measure_points(points, 0.03, Norms(), automatic=True, confidence=1.0))
    assert window.decision_label.text() == "Вне допуска"
    assert window.info_panel.width() == width
    assert all(font.family() == application_font.family() for font in metric_fonts)
    assert all(font.pointSize() == application_font.pointSize() for font in metric_fonts)
    assert all(font.weight() == application_font.weight() for font in metric_fonts)
    window.close()
    app.processEvents()


def test_off_center_bend_fails_measurement_and_is_reported_in_ui(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    window = MainWindow(paths, SettingsStore(paths.settings), AppSettings(), Storage(paths.database, paths.images))
    points = MeasurementPoints((0, 0), (350, 0), (0, 0), (350, 0), (100, 58))
    result = measure_points(points, 0.02, Norms(), automatic=True, confidence=1.0)
    assert result.length_ok and result.width_ok
    assert np.isclose(result.bend_position, 100 / 350)
    assert not result.bend_centered
    assert not result.dimensions_ok
    window._show_result(result)
    assert window.bend_label.text().startswith("Нет · t=")
    assert window.decision_label.text() == "Вне допуска"
    window.close()
    app.processEvents()


def test_measurement_geometry_points_are_hollow_with_visible_centers() -> None:
    image = np.full((110, 110, 3), 180, dtype=np.uint8)
    points = MeasurementPoints((20, 20), (80, 20), (20, 50), (80, 50), (50, 80))
    result = measure_points(points, 0.02, Norms(), automatic=True, confidence=1.0)
    overlay = draw_measurement_geometry(image, result)
    assert tuple(overlay[20, 20]) != (180, 180, 180)
    assert tuple(overlay[20, 26]) != (180, 180, 180)
    assert np.max(np.abs(overlay[17, 18].astype(np.int16) - 180)) < 20


def test_saved_camera_profile_is_restored_without_scan(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    profile = CameraProfile(3, "DirectShow", (1280, 720), "YX-AK23B")
    window = MainWindow(paths, SettingsStore(paths.settings), AppSettings(last_camera=profile), Storage(paths.database, paths.images))
    descriptor = window.camera_combo.currentData()
    assert descriptor.index == 3
    assert descriptor.backend == "DirectShow"
    assert tuple(window.resolution_combo.currentData()) == (1280, 720)
    assert window.resolution_combo.count() == 3
    window.close()
    app.processEvents()


def test_cancelled_history_cleanup_keeps_records(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    storage = Storage(paths.database, paths.images)
    window = MainWindow(paths, SettingsStore(paths.settings), AppSettings(), storage)
    called = False

    def should_not_run():
        nonlocal called
        called = True

    monkeypatch.setattr(window, "_confirm_clear_history", lambda: False)
    monkeypatch.setattr(storage, "clear_history", should_not_run)
    window.clear_history()
    assert not called
    window.close()
    app.processEvents()


def test_roi_canvas_can_select_and_reset_full_frame() -> None:
    app = QApplication.instance() or QApplication([])
    canvas = RoiCanvas(np.zeros((480, 640, 3), dtype=np.uint8))
    canvas.set_selection((100, 50, 400, 300))
    assert canvas.selection == (100, 50, 400, 300)
    canvas.set_selection((0, 0, 640, 480))
    assert canvas.selection == (0, 0, 640, 480)
    canvas.deleteLater()
    app.processEvents()


def test_button_roles_and_roi_visibility_control_are_consistent(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    window = MainWindow(paths, SettingsStore(paths.settings), AppSettings(), Storage(paths.database, paths.images))
    assert window.measure_button.text() == "Измерить"
    assert window.measure_button.property("buttonRole") == "primary"
    assert all(button.property("buttonRole") == "secondary" for button in (
        window.roi_button, window.calibrate_button, window.edit_points_button, window.diagnostic_button,
    ))
    assert all(button.property("buttonRole") == "utility" for button in (
        window.refresh_history_button, window.export_csv_button, window.export_excel_button,
    ))
    assert window.clear_history_button.property("buttonRole") == "destructive"
    assert window.show_roi_checkbox.isChecked()
    window.close()
    app.processEvents()


def test_roi_frame_can_be_hidden_without_changing_camera_image(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    paths = AppPaths.resolve(tmp_path)
    settings = AppSettings()
    settings.set_roi(0, "DirectShow", (640, 480), (200, 150, 200, 200))
    window = MainWindow(paths, SettingsStore(paths.settings), settings, Storage(paths.database, paths.images))
    descriptor = CameraDescriptor(0, "DirectShow", 640, 480, "Test")
    window.camera_combo.clear()
    window.camera_combo.addItem(descriptor.label, descriptor)
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    captured = []

    from marker_qc.gui import main_window as main_window_module
    original = main_window_module.array_to_qimage
    monkeypatch.setattr(main_window_module, "array_to_qimage", lambda image: (captured.append(image.copy()), original(image))[1])
    window._show_image(frame)
    with_frame = captured[-1]
    assert tuple(with_frame[150, 200]) == (0, 0, 220)
    assert np.all(with_frame[:110] == 180)
    assert tuple(with_frame[100, 100]) == (180, 180, 180)
    assert tuple(with_frame[470, 20]) == (180, 180, 180)

    window.show_roi_checkbox.setChecked(False)
    without_frame = captured[-1]
    assert tuple(without_frame[150, 200]) != (0, 0, 220)
    assert np.array_equal(without_frame, frame)
    window.close()
    app.processEvents()
