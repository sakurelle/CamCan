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

