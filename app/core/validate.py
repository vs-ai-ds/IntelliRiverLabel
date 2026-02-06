# app/core/validate.py
"""
Validate that label rectangle is inside safe polygon. Return (ok, min_clearance_pt).
See: docs/ALGORITHM.md, docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

from app.core.config import CONTAINMENT_TOLERANCE_PT
from app.core.geometry import polygon_contains_with_tol


def _min_distance_to_boundary(poly: BaseGeometry, geom: BaseGeometry) -> float:
    """
    Minimum distance between geom and poly boundary.
    For contained geometries this equals the "clearance" to the boundary.
    """
    if poly is None or poly.is_empty or geom is None or geom.is_empty:
        return 0.0
    boundary = poly.boundary
    if boundary is None or boundary.is_empty:
        return 0.0
    try:
        return float(boundary.distance(geom))
    except Exception:
        return 0.0


def validate_rect_inside_safe(
    safe_poly: BaseGeometry,
    rect_poly: BaseGeometry,
    tolerance_pt: float = CONTAINMENT_TOLERANCE_PT,
) -> tuple[bool, float]:
    """
    True if rect_poly is fully inside safe_poly (with tolerance).
    Also returns min clearance from rect to safe boundary (min_clearance_pt).
    See: docs/ALGORITHM.md.
    """
    if safe_poly is None or safe_poly.is_empty or rect_poly is None or rect_poly.is_empty:
        return False, 0.0

    ok = polygon_contains_with_tol(safe_poly, rect_poly, tolerance_pt=tolerance_pt)
    min_clearance = _min_distance_to_boundary(safe_poly, rect_poly)
    return ok, min_clearance