from __future__ import annotations

import cv2
import numpy as np

from marker_qc.config import Norms
from marker_qc.geometry import dimensions_from_points, project_point_to_line, projection_parameter
from marker_qc.models import MeasurementPoints, MeasurementResult


def measure_points(
    points: MeasurementPoints,
    scale_mm_per_px: float,
    norms: Norms,
    *,
    automatic: bool,
    confidence: float,
    diagnostics: tuple[str, ...] = (),
) -> MeasurementResult:
    length, width = dimensions_from_points(points, scale_mm_per_px)
    length_ok = norms.check_length(length)
    width_ok = norms.check_width(width)
    bend_position = projection_parameter(points.wc, points.w1, points.w2)
    bend_centered = norms.check_bend_position(bend_position)
    return MeasurementResult(
        length_mm=length,
        width_mm=width,
        length_ok=length_ok,
        width_ok=width_ok,
        dimensions_ok=length_ok and width_ok and bend_centered,
        points=points,
        scale_mm_per_px=scale_mm_per_px,
        automatic=automatic,
        confidence=confidence,
        diagnostics=diagnostics + (f"Положение перегиба на W1–W2: t={bend_position:.4f}",),
        bend_position=bend_position,
        bend_centered=bend_centered,
    )


def draw_measurement_geometry(image: np.ndarray, result: MeasurementResult) -> np.ndarray:
    """Draw measurement points and guide lines."""
    image = image.copy()
    to_int = lambda p: (int(round(p[0])), int(round(p[1])))
    l1, l2 = to_int(result.points.l1), to_int(result.points.l2)
    w1, w2, wc = to_int(result.points.w1), to_int(result.points.w2), to_int(result.points.wc)
    red, blue = (0, 0, 255), (255, 0, 0)
    cv2.line(image, l1, l2, red, 2)
    cv2.line(image, w1, w2, blue, 2)
    projection, _ = project_point_to_line(wc, w1, w2)
    cv2.line(image, wc, tuple(np.rint(projection).astype(int)), blue, 2)
    for point in (l1, l2):
        _draw_hollow_point(image, point, red)
    for point in (w1, w2, wc):
        _draw_hollow_point(image, point, blue)
    return image


def _draw_hollow_point(image: np.ndarray, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.circle(image, point, 6, color, 2, cv2.LINE_AA)
    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)

