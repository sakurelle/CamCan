import pytest

from marker_qc.config import Norms
from marker_qc.geometry import dimensions_from_points
from marker_qc.models import MeasurementPoints


SCALE = 0.0176981171364162
ROWS = [
    # archived length, archived width, coordinates
    (6.8831, 1.1953, (125, 110, 149, 96, 481, 270, 497, 236, 298, 229)),
    (6.9281, 1.1960, (113, 84, 135, 71, 475, 233, 488, 199, 287, 198)),
    (7.1362, 1.1980, (101, 129, 124, 112, 482, 261, 480, 228, 283, 235)),
]


def _points(c):
    x1, y1, x2, y2, x3, y3, x4, y4, cx, cy = c
    return MeasurementPoints((x1, y1), (x3, y3), (x2, y2), (x4, y4), (cx, cy))


def test_saved_coordinates_reproduce_two_latest_archive_results() -> None:
    for archived_length, archived_width, coordinates in ROWS[1:]:
        length, width = dimensions_from_points(_points(coordinates), SCALE)
        assert length == pytest.approx(archived_length, abs=0.0002)
        assert width == pytest.approx(archived_width, abs=0.0002)


def test_first_archive_row_contains_scale_inconsistency() -> None:
    length, width = dimensions_from_points(_points(ROWS[0][2]), SCALE)
    assert abs(length - ROWS[0][0]) > 0.02
    assert width == pytest.approx(ROWS[0][1], abs=0.005)


@pytest.mark.parametrize("length,width,expected", [
    (6.8831, 1.1953, False),
    (6.9281, 1.1960, True),
    (7.1362, 1.1980, False),
])
def test_archive_values_use_new_strict_norms(length, width, expected) -> None:
    norms = Norms()
    assert (norms.check_length(length) and norms.check_width(width)) is expected

