from __future__ import annotations

import cv2
import numpy as np

from marker_qc.config import Norms
from marker_qc.geometry import dimensions_from_points
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
    return MeasurementResult(
        length_mm=length,
        width_mm=width,
        length_ok=length_ok,
        width_ok=width_ok,
        dimensions_ok=length_ok and width_ok,
        points=points,
        scale_mm_per_px=scale_mm_per_px,
        automatic=automatic,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def draw_measurement_geometry(image: np.ndarray, result: MeasurementResult) -> np.ndarray:
    """Draw measurement points and guide lines without the text panel."""
    image = image.copy()
    to_int = lambda p: (int(round(p[0])), int(round(p[1])))
    l1, l2 = to_int(result.points.l1), to_int(result.points.l2)
    w1, w2, wc = to_int(result.points.w1), to_int(result.points.w2), to_int(result.points.wc)
    red, blue = (0, 0, 255), (255, 0, 0)
    cv2.line(image, l1, l2, red, 2)
    cv2.line(image, w1, w2, blue, 2)
    vector = np.array(w2, dtype=float) - np.array(w1, dtype=float)
    denominator = float(vector @ vector)
    if denominator > 0:
        projection = np.array(w1, dtype=float) + vector * float(
            (np.array(wc, dtype=float) - np.array(w1, dtype=float)) @ vector
        ) / denominator
        cv2.line(image, wc, tuple(np.rint(projection).astype(int)), blue, 2)
    for point in (l1, l2):
        cv2.circle(image, point, 5, red, -1)
    for point in (w1, w2, wc):
        cv2.circle(image, point, 5, blue, -1)
    return image


def draw_measurement(image: np.ndarray, result: MeasurementResult, norms: Norms) -> np.ndarray:
    """Draw the complete persisted overlay: geometry plus live-reading panel."""
    image = draw_measurement_geometry(image, result)
    return draw_measurement_text(image, result, norms)


def draw_measurement_text(image: np.ndarray, result: MeasurementResult, norms: Norms) -> np.ndarray:
    """Draw the compact live-reading panel in the upper-left corner."""
    image = image.copy()
    # Числа рисуются латиницей: OpenCV не гарантирует кириллические глифы.
    lines = (
        f"L={result.length_mm:.4f} mm [{'OK' if result.length_ok else 'FAIL'}]",
        f"W={result.width_mm:.4f} mm [{'OK' if result.width_ok else 'FAIL'}]",
        f"L: {norms.length_bounds[0]}..{norms.length_bounds[1]}  W: {norms.width_bounds[0]}..{norms.width_bounds[1]}",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    padding_x, padding_y = 12, 10
    line_height = 25
    text_width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text in lines)
    panel_x, panel_y = 10, 10
    panel_width = min(text_width + padding_x * 2, max(image.shape[1] - panel_x, 1))
    panel_height = min(line_height * len(lines) + padding_y * 2, max(image.shape[0] - panel_y, 1))
    if panel_width > 0 and panel_height > 0:
        panel = image[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width]
        dark = np.zeros_like(panel)
        cv2.addWeighted(panel, 0.35, dark, 0.65, 0, panel)
    green = (80, 220, 105)
    for index, text in enumerate(lines):
        y = panel_y + padding_y + 17 + index * line_height
        if y < image.shape[0]:
            cv2.putText(image, text, (panel_x + padding_x, y), font, font_scale, green, thickness, cv2.LINE_AA)
    return image

