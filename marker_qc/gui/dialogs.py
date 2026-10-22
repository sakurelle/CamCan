from __future__ import annotations

from decimal import Decimal

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout,
)

from marker_qc.config import Norms
from marker_qc.geometry import GeometryError, calibration_scale, dimensions_from_points
from marker_qc.models import MeasurementPoints
from marker_qc.gui.widgets import ImageCanvas, RoiCanvas


class RoiDialog(QDialog):
    def __init__(self, image: np.ndarray, initial=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка рабочей области")
        self.canvas = RoiCanvas(image, initial)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Reset).setText("Весь кадр")
        buttons.button(QDialogButtonBox.Reset).clicked.connect(
            lambda: self.canvas.set_selection((0, 0, image.shape[1], image.shape[0]))
        )
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Выделите область поиска маркера. Калибровка и видеопоток используют полный кадр."))
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.resize(min(image.shape[1] + 40, 1200), min(image.shape[0] + 100, 850))

    @property
    def roi(self):
        return self.canvas.selection

    def _accept(self) -> None:
        roi = self.roi
        if not roi or roi[2] < 50 or roi[3] < 50:
            QMessageBox.warning(self, "Некорректная область", "Выберите рабочую область размером не менее 50×50 пикселей.")
            return
        self.accept()


class CalibrationDialog(QDialog):
    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Калибровка масштаба")
        self.canvas = ImageCanvas(image)
        self.canvas.set_max_points(2)
        self.canvas.points_changed.connect(self._update)
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.001, 1000.0)
        self.distance.setDecimals(4)
        self.distance.setValue(10.0)
        self.distance.valueChanged.connect(self._update)
        self.result_label = QLabel("Щёлкните две точки шкалы.")
        self.zoom_label = QLabel("100 %")
        minus, plus = QPushButton("−"), QPushButton("+")
        minus.clicked.connect(lambda: self._zoom(-0.25))
        plus.clicked.connect(lambda: self._zoom(0.25))
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Реальное расстояние, мм:"))
        controls.addWidget(self.distance)
        controls.addStretch()
        controls.addWidget(minus)
        controls.addWidget(self.zoom_label)
        controls.addWidget(plus)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Укажите центры двух отметок проверенной шкалы. Для точности используйте увеличение."))
        layout.addLayout(controls)
        layout.addWidget(scroll)
        layout.addWidget(self.result_label)
        layout.addWidget(buttons)
        self._zoom_value = 1.0
        self.resize(900, 700)

    def _zoom(self, delta: float) -> None:
        self._zoom_value = min(max(self._zoom_value + delta, 0.5), 5.0)
        self.canvas.set_zoom(self._zoom_value)
        self.zoom_label.setText(f"{self._zoom_value * 100:.0f} %")

    def _update(self, *_args) -> None:
        if len(self.canvas.points) != 2:
            return
        points = self.canvas.points
        try:
            scale = calibration_scale(points["P1"], points["P2"], self.distance.value())
            pixels = self.distance.value() / scale
            orientation = " (ориентир старой схемы: 0,0164–0,0184)" if not 0.0164 <= scale <= 0.0184 else ""
            self.result_label.setText(f"Расстояние: {pixels:.3f} px; масштаб: {scale:.8f} мм/px{orientation}")
        except GeometryError as exc:
            self.result_label.setText(str(exc))

    @property
    def calibration_data(self):
        points = self.canvas.points
        if len(points) != 2:
            return None
        return points["P1"], points["P2"], self.distance.value(), calibration_scale(points["P1"], points["P2"], self.distance.value())

    def _accept(self) -> None:
        if self.calibration_data is None:
            QMessageBox.warning(self, "Недостаточно данных", "Выберите две различные точки шкалы.")
            return
        self.accept()


class PointEditorDialog(QDialog):
    def __init__(self, image: np.ndarray, points: MeasurementPoints, scale: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ручная корректировка измерительных точек")
        source = {"L1": points.l1, "L2": points.l2, "W1": points.w1, "W2": points.w2, "WC": points.wc}
        self.canvas = ImageCanvas(image, source)
        self.canvas.points_changed.connect(self._update)
        self.scale = scale
        self.info = QLabel()
        self._zoom_value = 1.0
        minus, plus = QPushButton("−"), QPushButton("+")
        minus.clicked.connect(lambda: self._zoom(-0.25))
        plus.clicked.connect(lambda: self._zoom(0.25))
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Перетащите ближайшую точку мышью. Красные: длина; синие: ширина."))
        zoom_row.addStretch()
        zoom_row.addWidget(minus)
        zoom_row.addWidget(plus)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(zoom_row)
        layout.addWidget(scroll)
        layout.addWidget(self.info)
        layout.addWidget(buttons)
        self._update()
        self.resize(950, 720)

    def _zoom(self, delta: float) -> None:
        self._zoom_value = min(max(self._zoom_value + delta, 0.5), 5.0)
        self.canvas.set_zoom(self._zoom_value)

    @property
    def points(self) -> MeasurementPoints:
        p = self.canvas.points
        return MeasurementPoints(l1=p["L1"], l2=p["L2"], w1=p["W1"], w2=p["W2"], wc=p["WC"])

    def _update(self, *_args) -> None:
        try:
            length, width = dimensions_from_points(self.points, self.scale)
            self.info.setText(f"Длина: {length:.4f} мм; ширина: {width:.4f} мм")
        except GeometryError as exc:
            self.info.setText(str(exc))

    def _accept(self) -> None:
        try:
            dimensions_from_points(self.points, self.scale)
        except GeometryError as exc:
            QMessageBox.warning(self, "Некорректные точки", str(exc))
            return
        self.accept()


class NormsDialog(QDialog):
    def __init__(self, norms: Norms, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Нормативные размеры")
        self.fields: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout()
        labels = {
            "length_nominal_mm": "Номинал длины, мм",
            "length_tolerance_mm": "Допуск длины ±, мм",
            "width_nominal_mm": "Номинал ширины, мм",
            "width_tolerance_mm": "Допуск ширины ±, мм",
        }
        for name, label in labels.items():
            field = QDoubleSpinBox()
            field.setDecimals(4)
            field.setRange(0.0, 1000.0)
            field.setValue(float(getattr(norms, name)))
            self.fields[name] = field
            form.addRow(label, field)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Изменение нормативов будет записано в журнале аудита."))
        layout.addLayout(form)
        layout.addWidget(buttons)

    @property
    def norms(self) -> Norms:
        return Norms(**{name: format(Decimal(str(field.value())), "f") for name, field in self.fields.items()})

