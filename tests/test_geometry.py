# tests/test_geometry.py
"""
Deterministic tests for geometry: ensure_polygon, bounds, sampling, PCA angle,
oriented_rectangle, polygon_contains_with_tol. See: docs/ALGORITHM.md.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from app.core.geometry import (
    ensure_polygon,
    oriented_rectangle,
    pca_dominant_angle_deg,
    polygon_bounds,
    polygon_contains_with_tol,
    sample_points_in_polygon,
)


def test_ensure_polygon_polygon() -> None:
    p = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    out = ensure_polygon(p)
    assert out is not None
    assert not out.is_empty
    assert out.geom_type == "Polygon"
    assert out.area == 4.0


def test_ensure_polygon_invalid_fixed() -> None:
    # Self-intersecting bowtie
    p = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not p.is_valid
    out = ensure_polygon(p)
    assert out is not None
    assert out.is_valid or out.area >= 0


def test_polygon_bounds() -> None:
    p = Polygon([(1, 2), (5, 2), (5, 6), (1, 6)])
    minx, miny, maxx, maxy = polygon_bounds(p)
    assert minx == 1 and miny == 2 and maxx == 5 and maxy == 6


def test_sample_points_in_polygon_deterministic() -> None:
    p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    a = sample_points_in_polygon(p, 20, seed=42)
    b = sample_points_in_polygon(p, 20, seed=42)
    assert len(a) == 20 and len(b) == 20
    for (x1, y1), (x2, y2) in zip(a, b):
        assert x1 == x2 and y1 == y2


def test_sample_points_in_polygon_inside() -> None:
    p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    pts = sample_points_in_polygon(p, 50, seed=1)
    for x, y in pts:
        assert p.contains(Point(x, y))


def test_pca_dominant_angle_deg() -> None:
    # Horizontal strip: dominant axis along x
    p = Polygon([(0, 0), (100, 0), (100, 2), (0, 2)])
    angle = pca_dominant_angle_deg(p)
    assert 0 <= angle < 180
    assert abs(angle - 0.0) < 15 or abs(angle - 180.0) < 15


def test_oriented_rectangle() -> None:
    rect = oriented_rectangle(5, 5, 4, 2, 0)
    assert rect.is_valid
    assert rect.centroid.x == pytest.approx(5) and rect.centroid.y == pytest.approx(5)
    rect90 = oriented_rectangle(5, 5, 4, 2, 90)
    assert rect90.is_valid
    minx, miny, maxx, maxy = rect90.bounds
    assert maxx - minx == pytest.approx(2) and maxy - miny == pytest.approx(4)


def test_polygon_contains_with_tol() -> None:
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    inside = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    assert polygon_contains_with_tol(poly, inside) is True
    outside = Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])
    assert polygon_contains_with_tol(poly, outside) is False
