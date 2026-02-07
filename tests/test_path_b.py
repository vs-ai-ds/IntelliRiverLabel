# tests/test_path_b.py
"""
Tests for Phase B path: centerline (cross-section midpoints) and clearance.
See: docs/ALGORITHM.md Phase B.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Polygon

from app.core.config import PADDING_PT
from app.core.path_b import (
    build_internal_path_polyline,
    min_clearance_along_path,
    path_length,
)
from app.core.phase_b import try_phase_b_curved
from app.core.types import LabelSpec


def test_centerline_returns_linestring_for_buffered_polyline() -> None:
    """Cross-section midpoint builder returns a LineString for a simple buffered polyline."""
    # Simple wiggly line buffered to a polygon (river-like)
    line = LineString([(0, 10), (20, 12), (40, 8), (60, 15), (80, 10)])
    poly = line.buffer(8.0, resolution=4)
    if poly.geom_type == "MultiPolygon" and poly.geoms:
        poly = poly.geoms[0]
    safe = poly.buffer(-2.0)
    if safe.is_empty:
        safe = poly
    path = build_internal_path_polyline(safe, seed=42)
    assert path is not None
    assert isinstance(path, LineString)
    assert not path.is_empty
    assert path_length(path) > 0
    coords = list(path.coords)
    assert len(coords) >= 10


def test_clearance_reasonable_for_controlled_polygon() -> None:
    """Clearance along path returns small, reasonable values (not thousands)."""
    # Rectangle 100 x 20; path along horizontal midline
    poly = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
    path = LineString([(10, 10), (50, 10), (90, 10)])
    clearance = min_clearance_along_path(path, poly)
    # Distance from (x,10) to boundary (y=0 or y=20) is 10 pt
    assert 0 < clearance <= 15.0
    assert clearance < 100.0


def test_try_phase_b_curved_returns_result_or_reason() -> None:
    """try_phase_b_curved returns either (PlacementResult with mode phase_b_curved, "") or (None, reason)."""
    line = LineString([(0, 10), (50, 12), (100, 10), (150, 8), (200, 10)])
    poly = line.buffer(12.0, resolution=4)
    if poly.geom_type == "MultiPolygon" and poly.geoms:
        poly = poly.geoms[0]
    safe = poly.buffer(-3.0)
    if safe.is_empty:
        safe = poly
    label = LabelSpec(text="River", font_family="sans-serif", font_size_pt=10.0)
    result, reason = try_phase_b_curved(poly, safe, label, PADDING_PT, seed=42, geometry_source="test")
    if result is not None:
        assert reason == ""
        assert result.mode == "phase_b_curved"
    else:
        assert isinstance(reason, str) and len(reason) > 0
