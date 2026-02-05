# tests/test_validate.py
"""
Deterministic tests for validate_rect_inside_safe. See: docs/ALGORITHM.md A4.
"""

from __future__ import annotations

import pytest

from app.core.geometry import oriented_rectangle
from app.core.validate import validate_rect_inside_safe
from shapely.geometry import Polygon


def test_validate_rect_inside_safe_contained() -> None:
    safe = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    rect = oriented_rectangle(10, 10, 4, 2, 0)
    ok, min_cl = validate_rect_inside_safe(safe, rect)
    assert ok is True
    assert min_cl >= 0


def test_validate_rect_inside_safe_not_contained() -> None:
    safe = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    rect = oriented_rectangle(15, 5, 4, 2, 0)
    ok, _ = validate_rect_inside_safe(safe, rect)
    assert ok is False


def test_validate_rect_inside_safe_partial() -> None:
    safe = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    rect = oriented_rectangle(8, 5, 6, 2, 0)
    ok, _ = validate_rect_inside_safe(safe, rect)
    assert ok is False


def test_validate_rect_inside_safe_returns_min_clearance() -> None:
    safe = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    rect = oriented_rectangle(10, 10, 2, 2, 0)
    ok, min_cl = validate_rect_inside_safe(safe, rect)
    assert ok is True
    assert min_cl >= 0
    assert min_cl <= 10
