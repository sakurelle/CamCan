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


def draw_measurement(roi: np.ndarray, result: MeasurementResult, norms: Norms) -> np.ndarray:
    image = roi.copy()
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
    # Числа рисуются латиницей: OpenCV не гарантирует кириллические глифы.
    lines = (
        f"L={result.length_mm:.4f} mm [{'OK' if result.length_ok else 'FAIL'}]",
        f"W={result.width_mm:.4f} mm [{'OK' if result.width_ok else 'FAIL'}]",
        f"L: {norms.length_bounds[0]}..{norms.length_bounds[1]}  W: {norms.width_bounds[0]}..{norms.width_bounds[1]}",
    )
    for index, text in enumerate(lines):
        y = 25 + index * 25
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return image

