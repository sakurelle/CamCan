from __future__ import annotations

import math

from marker_qc.models import MeasurementPoints, Point


class GeometryError(ValueError):
    """Некорректная или вырожденная геометрия измерительных точек."""


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def point_to_line_distance(point: Point, line_a: Point, line_b: Point) -> float:
    ax, ay = map(float, line_a)
    bx, by = map(float, line_b)
    px, py = map(float, point)
    dx, dy = bx - ax, by - ay
    denominator = math.hypot(dx, dy)
    if denominator == 0.0:
        raise GeometryError("Точки W1 и W2 совпадают: ширину вычислить невозможно.")
    return abs(dx * (ay - py) - (ax - px) * dy) / denominator


def calibration_scale(point_a: Point, point_b: Point, real_distance_mm: float) -> float:
    if not math.isfinite(real_distance_mm) or real_distance_mm <= 0:
        raise GeometryError("Реальное расстояние калибровки должно быть больше нуля.")
    pixels = euclidean_distance(point_a, point_b)
    if pixels == 0.0:
        raise GeometryError("Калибровочные точки должны различаться.")
    return float(real_distance_mm) / pixels


def dimensions_from_points(points: MeasurementPoints, scale_mm_per_px: float) -> tuple[float, float]:
    if not math.isfinite(scale_mm_per_px) or scale_mm_per_px <= 0:
        raise GeometryError("Масштаб должен быть положительным конечным числом.")
    length_px = euclidean_distance(points.l1, points.l2)
    if length_px == 0.0:
        raise GeometryError("Точки L1 и L2 совпадают: длину вычислить невозможно.")
    width_px = point_to_line_distance(points.wc, points.w1, points.w2)
    if width_px == 0.0:
        raise GeometryError("Точка WC лежит на линии W1–W2: геометрия ширины вырожденна.")
    return length_px * scale_mm_per_px, width_px * scale_mm_per_px


def widget_to_image(
    widget_point: Point,
    image_size: tuple[int, int],
    widget_size: tuple[int, int],
) -> Point:
    """Преобразовать координаты QLabel с KeepAspectRatio в координаты изображения."""
    iw, ih = image_size
    ww, wh = widget_size
    if min(iw, ih, ww, wh) <= 0:
        raise GeometryError("Размеры изображения и виджета должны быть положительными.")
    scale = min(ww / iw, wh / ih)
    shown_w, shown_h = iw * scale, ih * scale
    offset_x, offset_y = (ww - shown_w) / 2.0, (wh - shown_h) / 2.0
    x = (widget_point[0] - offset_x) / scale
    y = (widget_point[1] - offset_y) / scale
    return min(max(x, 0.0), iw - 1.0), min(max(y, 0.0), ih - 1.0)

