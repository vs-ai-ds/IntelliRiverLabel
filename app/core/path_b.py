# app/core/path_b.py
"""
Internal path approximation for Phase B: cross-section midpoints centerline.
No rendering. See: docs/ALGORITHM.md Phase B.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from app.core.config import (
    PATH_MIN_POINTS,
    PATH_SLICES,
    PATH_SMOOTHING_WINDOW,
)
from app.core.geometry import polygon_bounds


def _boundary_coords(geom: BaseGeometry) -> np.ndarray:
    """Boundary coordinates as (N, 2)."""
    if geom is None or geom.is_empty:
        return np.zeros((0, 2))
    from shapely.geometry import Polygon, MultiPolygon
    if isinstance(geom, Polygon):
        return np.array(geom.exterior.coords)
    if isinstance(geom, MultiPolygon):
        parts = [_boundary_coords(p) for p in geom.geoms]
        return np.vstack([p for p in parts if len(p) > 0]) if parts else np.zeros((0, 2))
    return np.zeros((0, 2))


def _pca_axis_xy(geom: BaseGeometry) -> tuple[float, float]:
    """Unit vector along dominant PCA axis (x, y)."""
    if geom is None or geom.is_empty:
        return (1.0, 0.0)
    xy = _boundary_coords(geom)
    if xy.shape[0] < 2:
        return (1.0, 0.0)
    xy = xy - np.mean(xy, axis=0)
    cov = np.cov(xy.T)
    if cov.size == 0:
        return (1.0, 0.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argmax(eigvals)
    v = eigvecs[:, idx]
    n = np.linalg.norm(v)
    if n <= 0:
        return (1.0, 0.0)
    return (float(v[0] / n), float(v[1] / n))


def _smooth_polyline(xy: np.ndarray, window: int) -> np.ndarray:
    """Moving average along the sequence; window must be odd."""
    if xy.shape[0] < window or window < 1:
        return xy
    kernel = np.ones(window) / window
    out = np.zeros_like(xy)
    for c in range(xy.shape[1]):
        out[:, c] = np.convolve(xy[:, c], kernel, mode="same")
    return out


def _snap_coords_inside(poly: BaseGeometry, coords: list[tuple[float, float]], inset: float = 0.75) -> list[tuple[float, float]]:
    """
    Ensure coords are inside poly by snapping any outliers to the nearest point
    on an *inner* buffered polygon (to preserve clearance).
    - If inner buffer becomes empty, falls back to poly.
    """
    if poly is None or poly.is_empty or not coords:
        return coords

    try:
        inner = poly.buffer(-float(inset))
        if inner.is_empty:
            inner = poly
    except Exception:
        inner = poly

    out: list[tuple[float, float]] = []
    for x, y in coords:
        p = Point(float(x), float(y))
        try:
            if inner.covers(p) or inner.contains(p):
                out.append((float(x), float(y)))
                continue
            # Snap to nearest point on inner polygon
            q, _ = nearest_points(inner, p)
            out.append((float(q.x), float(q.y)))
        except Exception:
            out.append((float(x), float(y)))
    return out


def _longest_segment(geom: BaseGeometry) -> LineString | None:
    """If MultiLineString, return longest LineString; if LineString, return it; else None."""
    if geom is None or geom.is_empty:
        return None
    from shapely.geometry import MultiLineString
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        best = None
        best_len = -1.0
        for g in geom.geoms:
            if isinstance(g, LineString) and not g.is_empty:
                L = g.length
                if L > best_len:
                    best_len = L
                    best = g
        return best
    return None


def build_internal_path_polyline(
    safe_poly: BaseGeometry,
    seed: int | None,
) -> LineString | None:
    """
    Build centerline by cross-section midpoints: slice along dominant PCA axis,
    intersect each perpendicular with safe polygon, take longest segment midpoint, smooth.
    Deterministic (slice count and geometry only; seed unused).
    """
    if safe_poly is None or safe_poly.is_empty:
        return None
    simplified = safe_poly.simplify(0.5, preserve_topology=True)
    if simplified.is_empty:
        simplified = safe_poly
    ax, ay = _pca_axis_xy(simplified)
    minx, miny, maxx, maxy = polygon_bounds(safe_poly)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    diag = math.hypot(maxx - minx, maxy - miny)
    half_len = diag
    t_min = (minx - cx) * ax + (miny - cy) * ay
    t_max = (maxx - cx) * ax + (maxy - cy) * ay
    for x, y in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
        t = (x - cx) * ax + (y - cy) * ay
        t_min = min(t_min, t)
        t_max = max(t_max, t)
    n_slices = max(80, min(200, PATH_SLICES))
    t_vals = np.linspace(float(t_min), float(t_max), n_slices, endpoint=True)
    midpoints: list[tuple[float, float]] = []
    for t in t_vals:
        ox = cx + t * ax
        oy = cy + t * ay
        px1 = ox - ay * half_len
        py1 = oy + ax * half_len
        px2 = ox + ay * half_len
        py2 = oy - ax * half_len
        line = LineString([(px1, py1), (px2, py2)])
        try:
            inter = line.intersection(safe_poly)
        except Exception:
            continue
        seg = _longest_segment(inter)
        if seg is not None and not seg.is_empty:
            c = seg.centroid
            midpoints.append((float(c.x), float(c.y)))
    if len(midpoints) < PATH_MIN_POINTS:
        return None
    xy = np.array(midpoints)
    window = PATH_SMOOTHING_WINDOW if PATH_SMOOTHING_WINDOW % 2 == 1 else PATH_SMOOTHING_WINDOW + 1
    xy = _smooth_polyline(xy, max(3, window))
    coords = [(float(xy[i, 0]), float(xy[i, 1])) for i in range(len(xy))]
    # Smoothing can push points outside; snap back inside (prefer inner buffer to keep clearance)
    coords = _snap_coords_inside(safe_poly, coords, inset=0.75)
    if len(coords) < PATH_MIN_POINTS:
        return None
    return LineString(coords)


def path_length(path: LineString) -> float:
    """Total length of the path in pt."""
    if path is None or path.is_empty:
        return 0.0
    return float(path.length)


def sample_along_path(path: LineString, step_pt: float) -> list[Point]:
    """Sample points along path at step_pt intervals."""
    if path is None or path.is_empty or step_pt <= 0:
        return []
    length = path_length(path)
    if length <= 0:
        return []
    n = max(1, int(length / step_pt) + 1)
    dists = np.linspace(0, length, n, endpoint=True)
    points = []
    for d in dists:
        pt = path.interpolate(d)
        if pt is not None and not pt.is_empty:
            points.append(pt)
    return points


def min_clearance_along_path(path: LineString, boundary_poly: BaseGeometry) -> float:
    """Minimum distance from sampled points along path to boundary of boundary_poly (original polygon)."""
    if path is None or boundary_poly is None or path.is_empty or boundary_poly.is_empty:
        return 0.0
    from app.core.config import PATH_SAMPLE_STEP_PT
    pts = sample_along_path(path, PATH_SAMPLE_STEP_PT)
    if not pts:
        return 0.0
    boundary = boundary_poly.boundary
    if boundary is None or boundary.is_empty:
        return 0.0
    min_d = float("inf")
    for p in pts:
        d = boundary.distance(p)
        if d < min_d:
            min_d = float(d)
    return min_d if min_d != float("inf") else 0.0
