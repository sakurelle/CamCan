from decimal import Decimal

import pytest

from marker_qc.config import Norms


@pytest.mark.parametrize("value,expected", [
    ("6.90", True), ("7.00", True), ("7.10", True), ("6.89", False), ("7.11", False),
    (Decimal("6.899999999999999"), False), (Decimal("7.100000000000001"), False),
])
def test_length_boundaries(value, expected) -> None:
    assert Norms().check_length(value) is expected


@pytest.mark.parametrize("value,expected", [
    ("1.10", True), ("1.15", True), ("1.20", True), ("1.09", False), ("1.21", False),
    (Decimal("1.099999999999999"), False), (Decimal("1.200000000000001"), False),
])
def test_width_boundaries(value, expected) -> None:
    assert Norms().check_width(value) is expected


@pytest.mark.parametrize("value,expected", [
    ("0.40", True), ("0.50", True), ("0.60", True), ("0.3999", False), ("0.6001", False),
])
def test_bend_position_boundaries(value, expected) -> None:
    assert Norms().check_bend_position(value) is expected


def test_invalid_bend_position_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 ≤ минимум"):
        Norms(bend_position_min="0.7", bend_position_max="0.3").bend_position_bounds

