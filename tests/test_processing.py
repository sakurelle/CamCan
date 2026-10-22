import cv2
import numpy as np
import pytest
from pathlib import Path

from marker_qc.config import ProcessingSettings
from marker_qc.processing import ProcessingError, process_frame
from marker_qc.geometry import dimensions_from_points


def synthetic_marker(*, offset=(0, 0), touch_border=False, second_object=False, noise=False, uneven=False):
    height, width = 400, 640
    if uneven:
        gradient = np.tile(np.linspace(225, 255, width, dtype=np.uint8), (height, 1))
        image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    else:
        image = np.full((height, width, 3), 245, dtype=np.uint8)
    t = np.linspace(0, 1, 160)
    if touch_border:
        p0, p1, p2 = np.array([0, 40]), np.array([260, 300]), np.array([540, 220])
    else:
        p0, p1, p2 = np.array([70, 70]), np.array([290, 310]), np.array([550, 215])
    curve = ((1 - t)[:, None] ** 2 * p0 + 2 * (1 - t)[:, None] * t[:, None] * p1 + t[:, None] ** 2 * p2)
    curve += np.array(offset)
    cv2.polylines(image, [curve.astype(np.int32)], False, (30, 30, 30), 34, cv2.LINE_AA)
    if second_object:
        cv2.rectangle(image, (110, 300), (260, 355), (25, 25, 25), -1)
    if noise:
        rng = np.random.default_rng(42)
        coords = rng.integers([0, 0], [width, height], size=(300, 2))
        for x, y in coords:
            image[y, x] = 0
    return image


@pytest.mark.parametrize("kwargs", [{}, {"noise": True}, {"uneven": True}, {"offset": (15, 10)}])
def test_valid_synthetic_marker(kwargs) -> None:
    image = synthetic_marker(**kwargs)
    result = process_frame(image, (0, 0, image.shape[1], image.shape[0]), ProcessingSettings())
    assert np.count_nonzero(result.mask) > 1000
    assert result.confidence >= 0.5


def test_marker_touching_border_is_rejected() -> None:
    image = synthetic_marker(touch_border=True)
    with pytest.raises(ProcessingError, match="границ"):
        process_frame(image, (0, 0, image.shape[1], image.shape[0]), ProcessingSettings())


def test_multiple_objects_are_rejected() -> None:
    image = synthetic_marker(second_object=True)
    with pytest.raises(ProcessingError, match="несколько"):
        process_frame(image, (0, 0, image.shape[1], image.shape[0]), ProcessingSettings())


def test_empty_roi_never_returns_measurement() -> None:
    image = np.full((400, 640, 3), 245, dtype=np.uint8)
    with pytest.raises(ProcessingError):
        process_frame(image, (0, 0, 640, 400), ProcessingSettings())


def _on_square_canvas(image, size=800):
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y = (size - image.shape[0]) // 2
    x = (size - image.shape[1]) // 2
    canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    return canvas


@pytest.mark.parametrize("angle", [0, 30, 45, 90, 135, 180, 270])
def test_measurement_is_rotation_invariant(angle) -> None:
    image = _on_square_canvas(synthetic_marker())
    center = (image.shape[1] / 2, image.shape[0] / 2)
    rotated = cv2.warpAffine(
        image, cv2.getRotationMatrix2D(center, angle, 1.0),
        (image.shape[1], image.shape[0]), borderValue=(245, 245, 245),
    )
    baseline = process_frame(image, None, ProcessingSettings())
    measured = process_frame(rotated, None, ProcessingSettings())
    expected = dimensions_from_points(baseline.points, 1.0)
    actual = dimensions_from_points(measured.points, 1.0)
    assert actual[0] == pytest.approx(expected[0], rel=0.002)
    assert actual[1] == pytest.approx(expected[1], rel=0.005)


@pytest.mark.parametrize("offset", [(-40, 0), (40, 0), (0, -40), (0, 40)])
def test_measurement_is_translation_invariant(offset) -> None:
    image = _on_square_canvas(synthetic_marker())
    transform = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    moved = cv2.warpAffine(image, transform, (800, 800), borderValue=(245, 245, 245))
    baseline = dimensions_from_points(process_frame(image, None, ProcessingSettings()).points, 1.0)
    actual = dimensions_from_points(process_frame(moved, None, ProcessingSettings()).points, 1.0)
    assert actual == pytest.approx(baseline, abs=0.01)


def test_roi_limits_processing_area() -> None:
    image = synthetic_marker()
    result = process_frame(image, (40, 40, 560, 320), ProcessingSettings())
    assert result.roi.shape == (320, 560, 3)
    assert result.roi_rect == (40, 40, 560, 320)


def test_full_frame_is_default_roi() -> None:
    image = synthetic_marker()
    result = process_frame(image, None, ProcessingSettings())
    assert result.roi.shape == image.shape
    assert result.roi_rect == (0, 0, 640, 400)


def test_points_are_translated_from_roi_to_full_frame() -> None:
    image = synthetic_marker()
    roi = (40, 40, 560, 320)
    cropped = image[40:360, 40:600]
    local = process_frame(cropped, None, ProcessingSettings())
    full = process_frame(image, roi, ProcessingSettings())
    for local_point, full_point in zip(local.points.__dict__.values(), full.points.__dict__.values()):
        assert full_point == pytest.approx((local_point[0] + 40, local_point[1] + 40))
    assert dimensions_from_points(full.points, 1.0) == pytest.approx(dimensions_from_points(local.points, 1.0))


def test_object_outside_roi_does_not_interfere() -> None:
    image = synthetic_marker()
    cv2.rectangle(image, (80, 365), (560, 395), (20, 20, 20), -1)
    result = process_frame(image, (40, 35, 560, 325), ProcessingSettings())
    assert result.points.l1[1] < 365


@pytest.mark.parametrize("roi", [(-1, 0, 100, 100), (0, 0, 0, 100), (600, 0, 100, 100)])
def test_invalid_roi_is_rejected(roi) -> None:
    with pytest.raises(ProcessingError, match="границ|область"):
        process_frame(synthetic_marker(), roi, ProcessingSettings())



