from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


def array_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        contiguous = np.ascontiguousarray(image)
        return QImage(contiguous.data, contiguous.shape[1], contiguous.shape[0], contiguous.strides[0], QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()


class ImageCanvas(QWidget):
    points_changed = Signal(object)

    COLORS = {
        "L1": Qt.red, "L2": Qt.red,
        "W1": Qt.blue, "W2": Qt.blue, "WC": Qt.blue,
        "P1": Qt.red, "P2": Qt.red,
    }

    def __init__(self, image: np.ndarray, points: dict[str, tuple[float, float]] | None = None, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap.fromImage(array_to_qimage(image))
        self._image_size = (image.shape[1], image.shape[0])
        self._zoom = 1.0
        self._points = dict(points or {})
        self._dragging: str | None = None
        self._max_points = 0
        self.setMouseTracking(True)
        self._resize_to_zoom()

    @property
    def points(self) -> dict[str, tuple[float, float]]:
        return dict(self._points)

    def set_max_points(self, count: int) -> None:
        self._max_points = count

    def set_zoom(self, zoom: float) -> None:
        self._zoom = min(max(float(zoom), 0.5), 5.0)
        self._resize_to_zoom()
        self.update()

    def _resize_to_zoom(self) -> None:
        self.setFixedSize(int(self._image_size[0] * self._zoom), int(self._image_size[1] * self._zoom))

    def _image_point(self, event: QMouseEvent) -> tuple[float, float]:
        return (
            min(max(event.position().x() / self._zoom, 0.0), self._image_size[0] - 1.0),
            min(max(event.position().y() / self._zoom, 0.0), self._image_size[1] - 1.0),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        point = self._image_point(event)
        if self._points:
            nearest, distance = min(
                ((name, (value[0] - point[0]) ** 2 + (value[1] - point[1]) ** 2) for name, value in self._points.items()),
                key=lambda item: item[1],
            )
            if distance <= (14.0 / self._zoom) ** 2:
                self._dragging = nearest
                return
        if self._max_points and len(self._points) < self._max_points:
            name = f"P{len(self._points) + 1}"
            self._points[name] = point
            self.points_changed.emit(self.points)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._points[self._dragging] = self._image_point(event)
            self.points_changed.emit(self.points)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._points[self._dragging] = self._image_point(event)
            self._dragging = None
            self.points_changed.emit(self.points)
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(self.rect(), self._pixmap)
        def screen(name: str) -> QPoint:
            p = self._points[name]
            return QPoint(int(p[0] * self._zoom), int(p[1] * self._zoom))
        if "L1" in self._points and "L2" in self._points:
            painter.setPen(QPen(Qt.red, 2))
            painter.drawLine(screen("L1"), screen("L2"))
        if "W1" in self._points and "W2" in self._points:
            painter.setPen(QPen(Qt.blue, 2))
            painter.drawLine(screen("W1"), screen("W2"))
            if "WC" in self._points:
                a = np.array(self._points["W1"], dtype=float)
                b = np.array(self._points["W2"], dtype=float)
                c = np.array(self._points["WC"], dtype=float)
                vector = b - a
                denominator = float(vector @ vector)
                if denominator > 0:
                    projection = a + vector * float((c - a) @ vector) / denominator
                    target = QPoint(int(projection[0] * self._zoom), int(projection[1] * self._zoom))
                    painter.drawLine(screen("WC"), target)
        if "P1" in self._points and "P2" in self._points:
            painter.setPen(QPen(Qt.red, 2))
            painter.drawLine(screen("P1"), screen("P2"))
        for name, value in self._points.items():
            position = QPoint(int(value[0] * self._zoom), int(value[1] * self._zoom))
            color = self.COLORS.get(name, Qt.magenta)
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(position, 6, 6)
            painter.drawLine(position + QPoint(-3, 0), position + QPoint(3, 0))
            painter.drawLine(position + QPoint(0, -3), position + QPoint(0, 3))
            painter.drawText(position + QPoint(7, -7), name)


class RoiCanvas(QWidget):
    selection_changed = Signal(object)

    def __init__(self, image: np.ndarray, initial: tuple[int, int, int, int] | None = None, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap.fromImage(array_to_qimage(image))
        self._image_size = (image.shape[1], image.shape[0])
        self._start: QPoint | None = None
        self._end: QPoint | None = None
        if initial:
            x, y, w, h = initial
            self._start, self._end = QPoint(x, y), QPoint(x + w, y + h)
        self.setFixedSize(*self._image_size)

    @property
    def selection(self) -> tuple[int, int, int, int] | None:
        if self._start is None or self._end is None:
            return None
        frame_width, frame_height = self._image_size
        x1 = min(max(min(self._start.x(), self._end.x()), 0), frame_width)
        x2 = min(max(max(self._start.x(), self._end.x()), 0), frame_width)
        y1 = min(max(min(self._start.y(), self._end.y()), 0), frame_height)
        y2 = min(max(max(self._start.y(), self._end.y()), 0), frame_height)
        return x1, y1, x2 - x1, y2 - y1

    def set_selection(self, roi: tuple[int, int, int, int]) -> None:
        x, y, width, height = roi
        self._start = QPoint(int(x), int(y))
        self._end = QPoint(int(x + width), int(y + height))
        self.selection_changed.emit(self.selection)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._start is not None:
            self._end = event.position().toPoint()
            self.selection_changed.emit(self.selection)
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self._pixmap)
        if self.selection:
            x, y, w, h = self.selection
            painter.setPen(QPen(Qt.red, 2))
            painter.drawRect(x, y, max(w - 1, 0), max(h - 1, 0))

