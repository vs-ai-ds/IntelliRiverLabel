# app/core/geometry.py
"""
Geometry helpers: polygon normalization, bounds, sampling, PCA angle,
oriented rectangle, containment. See: docs/ALGORITHM.md A2–A4.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from app.core.config import CONTAINMENT_TOLERANCE_PT


def ensure_polygon(geom: BaseGeometry) -> Polygon | MultiPolygon:
    """
    Return geom as Polygon or MultiPolygon; fix invalid with buffer(0).
    See: docs/ARCHITECTURE.md.
    """
    if geom is None or geom.is_empty:
        return Polygon()
    if isinstance(geom, (Polygon, MultiPolygon)):
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom  # type: ignore[return-value]
    if hasattr(geom, "geoms"):
        polys = [ensure_polygon(g) for g in geom.geoms]
        polys = [p for p in polys if p is not None and not p.is_empty]
        if not polys:
            return Polygon()
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)
    return Polygon()


def polygon_bounds(geom: BaseGeometry) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy)."""
    if geom is None or geom.is_empty:
        return (0.0, 0.0, 0.0, 0.0)
    b = geom.bounds
    return (b[0], b[1], b[2], b[3])


def sample_points_in_polygon(
    geom: BaseGeometry,
    n: int,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """
    Deterministic rejection sampling: sample n points inside polygon.
    Uses bbox + contains check. See: docs/ALGORITHM.md A2.
    """
    if geom is None or geom.is_empty or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon_bounds(geom)
    out: list[tuple[float, float]] = []
    max_attempts = n * 50
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        p = Point(x, y)
        if geom.contains(p):
            out.append((x, y))
        attempts += 1
    return out[:n]


def _boundary_coords(geom: BaseGeometry) -> np.ndarray:
    """Collect boundary coordinates as (N, 2) array."""
    if geom is None or geom.is_empty:
        return np.zeros((0, 2))
    if isinstance(geom, Polygon):
        xy = np.array(geom.exterior.coords)
        return xy
    if isinstance(geom, MultiPolygon):
        parts = [_boundary_coords(p) for p in geom.geoms]
        return np.vstack([p for p in parts if len(p) > 0]) if parts else np.zeros((0, 2))
    return np.zeros((0, 2))


def pca_dominant_angle_deg(geom: BaseGeometry) -> float:
    """
    Dominant direction via PCA of polygon boundary points (numpy only, no sklearn).
    Returns angle in degrees [0, 180). See: docs/ALGORITHM.md A3.
    """
    xy = _boundary_coords(geom)
    if xy.shape[0] < 2:
        return 0.0
    xy = xy - np.mean(xy, axis=0)
    cov = np.cov(xy.T)
    if cov.size == 0:
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argmax(eigvals)
    v = eigvecs[:, idx]
    angle_rad = math.atan2(v[1], v[0])
    angle_deg = math.degrees(angle_rad)
    angle_deg = angle_deg % 180.0
    if angle_deg < 0:
        angle_deg += 180.0
    return angle_deg


def oriented_rectangle(
    cx: float, cy: float, width_pt: float, height_pt: float, angle_deg: float
) -> Polygon:
    """
    Axis-aligned rectangle centered at (cx, cy) with given width/height (pt),
    then rotated by angle_deg around center. See: docs/ALGORITHM.md A4.
    """
    hw = width_pt / 2.0
    hh = height_pt / 2.0
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    corners = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh),
    ]
    rotated = [
        (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
        for x, y in corners
    ]
    return Polygon(rotated)


def bbox_pt_to_polygon(
    bbox_pt: list[tuple[float, float]],
    buffer_pt: float = 0.0,
) -> Polygon:
    """Build a polygon from 4-corner bbox_pt; optionally buffer. For collision geometry."""
    if not bbox_pt or len(bbox_pt) < 3:
        return Polygon()
    poly = Polygon(bbox_pt)
    if buffer_pt > 0 and not poly.is_empty:
        poly = poly.buffer(buffer_pt)
    return poly if not poly.is_empty else Polygon()


def polygon_contains_with_tol(
    poly: BaseGeometry,
    rect: BaseGeometry,
    tolerance_pt: float = CONTAINMENT_TOLERANCE_PT,
) -> bool:
    """
    True if rect is fully inside poly (with tolerance). See: docs/ALGORITHM.md A4.
    """
    if poly is None or rect is None or poly.is_empty or rect.is_empty:
        return False
    if tolerance_pt > 0:
        buffered = poly.buffer(-tolerance_pt)
        if buffered.is_empty:
            return False
        return buffered.contains(rect) or buffered.covers(rect)
    return poly.contains(rect) or poly.covers(rect)
