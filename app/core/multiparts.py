# app/core/multiparts.py
"""
Helpers for multi-part geometry: merge nearby components (braided rivers), describe components.
Does not change placement algorithm; used for preprocessing only.
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from app.core.io import extract_polygon_components


def describe_components(geom: BaseGeometry) -> dict:
    """
    Return counts and total area for polygon components.
    Keys: component_count, total_area.
    """
    polys = extract_polygon_components(geom)
    n = len(polys)
    area = sum(float(p.area) for p in polys) if polys else 0.0
    return {"component_count": n, "total_area": area}


def merge_nearby_components(geom: BaseGeometry, distance_pt: float) -> BaseGeometry:
    """
    Merge polygon components that are within distance_pt of each other.
    Approach: buffer each by distance_pt, union, then negative buffer by same amount.
    Returns Polygon or MultiPolygon; validates with buffer(0).
    """
    if geom is None or geom.is_empty or distance_pt <= 0:
        return geom
    polys = extract_polygon_components(geom)
    if not polys:
        return geom
    if len(polys) == 1:
        out = polys[0].buffer(0) if not polys[0].is_valid else polys[0]
        return out if not out.is_empty else Polygon()
    try:
        buffered = [p.buffer(distance_pt) for p in polys]
        merged = unary_union(buffered)
        if merged is None or merged.is_empty:
            return geom
        unmerged = merged.buffer(-distance_pt)
        if unmerged is None or unmerged.is_empty:
            return geom
        fixed = unmerged.buffer(0)
        if fixed is None or fixed.is_empty:
            return geom
        polygons = extract_polygon_components(fixed)
        if not polygons:
            return geom
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)
    except Exception:
        return geom
