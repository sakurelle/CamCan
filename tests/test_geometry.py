import math

import pytest

from marker_qc.geometry import (
    GeometryError, calibration_scale, dimensions_from_points, euclidean_distance,
    point_to_line_distance, widget_to_image,
)
from marker_qc.models import MeasurementPoints


def test_euclidean_distance_and_scale() -> None:
    assert euclidean_distance((0, 0), (3, 4)) == 5
    assert calibration_scale((0, 0), (300, 400), 10) == pytest.approx(0.02)


@pytest.mark.parametrize("angle", [0, 15, 45, 89, 135])
def test_point_to_line_distance_is_rotation_invariant(angle: float) -> None:
    radians = math.radians(angle)
    direction = (math.cos(radians), math.sin(radians))
    normal = (-direction[1], direction[0])
    a = (10.0, 10.0)
    b = (a[0] + direction[0] * 100, a[1] + direction[1] * 100)
    point = (a[0] + direction[0] * 35 + normal[0] * 12, a[1] + direction[1] * 35 + normal[1] * 12)
    assert point_to_line_distance(point, a, b) == pytest.approx(12.0)


def test_degenerate_geometry_is_rejected() -> None:
    with pytest.raises(GeometryError):
        point_to_line_distance((1, 1), (0, 0), (0, 0))
    with pytest.raises(GeometryError):
        calibration_scale((2, 2), (2, 2), 10)
    points = MeasurementPoints((1, 1), (1, 1), (0, 0), (1, 0), (0, 1))
    with pytest.raises(GeometryError):
        dimensions_from_points(points, 0.02)


def test_widget_to_image_with_letterboxing() -> None:
    # 640×480 inside a square 800×800: 100 px top/bottom letterbox.
    assert widget_to_image((400, 400), (640, 480), (800, 800)) == pytest.approx((320, 240))
    assert widget_to_image((0, 100), (640, 480), (800, 800)) == pytest.approx((0, 0))

