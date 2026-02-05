# app/core/preprocess.py
"""
Preprocess river polygon: simplify, normalize orientation, build safe polygon
via inward buffer. See: docs/ARCHITECTURE.md, docs/ALGORITHM.md (A1).
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from app.core.config import MIN_BUFFER_PT, PADDING_PT


def _normalize_polygon(poly: Polygon) -> Polygon:
    """Ensure exterior ring is CCW and holes are CW (Shapely convention)."""
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_empty and isinstance(poly, Polygon):
        return poly
    return poly


def simplify_geometry(geom: BaseGeometry, tolerance: float = 0.0) -> BaseGeometry:
    """
    Simplify polygon(s); tolerance 0 returns geometry unchanged.
    See: docs/ARCHITECTURE.md (preprocess).
    """
    if tolerance <= 0:
        return geom
    if isinstance(geom, Polygon):
        return geom.simplify(tolerance, preserve_topology=True)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon(
            _normalize_polygon(p.simplify(tolerance, preserve_topology=True))
            for p in geom.geoms
        )
    return geom


def safe_polygon(
    geom: BaseGeometry,
    padding_pt: float = PADDING_PT,
    min_buffer_pt: float = MIN_BUFFER_PT,
) -> BaseGeometry:
    """
    Compute safe polygon as inward buffer: safe = P.buffer(-padding_pt).
    If safe is empty, reduce padding down to min_buffer_pt; if still empty,
    returns empty polygon (caller should treat as internal infeasible).
    See: docs/ALGORITHM.md A1.
    """
    if geom is None or geom.is_empty:
        return Polygon()

    def buffer_inward(g: BaseGeometry, d: float) -> BaseGeometry:
        if isinstance(g, Polygon):
            return g.buffer(-d)
        if isinstance(g, MultiPolygon):
            buffered = [p.buffer(-d) for p in g.geoms]
            non_empty = [b for b in buffered if b is not None and not b.is_empty]
            if not non_empty:
                return Polygon()
            if len(non_empty) == 1:
                return non_empty[0]
            return MultiPolygon(non_empty)
        return Polygon()

    d = padding_pt
    safe = buffer_inward(geom, d)
    while (safe is None or safe.is_empty) and d > min_buffer_pt:
        d = max(min_buffer_pt, d / 2.0)
        safe = buffer_inward(geom, d)
    if safe is None or safe.is_empty:
        return Polygon()
    return safe


def preprocess_river(
    geom: BaseGeometry,
    padding_pt: float = PADDING_PT,
    min_buffer_pt: float = MIN_BUFFER_PT,
    simplify_tolerance: float = 0.0,
) -> tuple[BaseGeometry, BaseGeometry]:
    """
    Full preprocessing: simplify (optional), then compute safe polygon.
    Returns (original_or_simplified_geom, safe_polygon).
    See: docs/ARCHITECTURE.md, docs/ALGORITHM.md.
    """
    simplified = simplify_geometry(geom, simplify_tolerance)
    safe = safe_polygon(simplified, padding_pt=padding_pt, min_buffer_pt=min_buffer_pt)
    return simplified, safe
