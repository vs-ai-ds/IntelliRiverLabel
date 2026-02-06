# tests/test_multiparts.py
"""Tests for app.core.multiparts (merge_nearby_components, describe_components)."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from app.core.multiparts import describe_components, merge_nearby_components


def test_describe_components_single() -> None:
    """Single polygon: count 1, area > 0."""
    poly = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    d = describe_components(poly)
    assert d["component_count"] == 1
    assert d["total_area"] == 50.0


def test_describe_components_two() -> None:
    """Two polygons: count 2, total area sum."""
    a = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    b = Polygon([(10, 0), (15, 0), (15, 5), (10, 5)])
    from shapely.geometry import MultiPolygon
    multi = MultiPolygon([a, b])
    d = describe_components(multi)
    assert d["component_count"] == 2
    assert d["total_area"] == 25.0 + 25.0


def test_merge_nearby_two_close_rectangles() -> None:
    """Two close rectangles merge into one (or fewer) when distance is large enough."""
    # Two 4x4 squares, 2 units apart (gap between 4 and 6 on x).
    a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    b = Polygon([(6, 0), (10, 0), (10, 4), (6, 4)])
    from shapely.geometry import MultiPolygon
    multi = MultiPolygon([a, b])
    assert describe_components(multi)["component_count"] == 2
    # Merge with distance 1.5: gap is 2, so buffer(1.5) makes them overlap -> one polygon after negative buffer
    merged = merge_nearby_components(multi, distance_pt=1.5)
    assert merged is not None and not merged.is_empty
    desc = describe_components(merged)
    assert desc["component_count"] == 1
    assert desc["total_area"] > 0
