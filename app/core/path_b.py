# app/core/path_b.py
"""
Internal path approximation for Phase B: center points, PCA projection, smoothing.
No rendering. See: docs/ALGORITHM.md Phase B.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from app.core.config import K_TOP_CLEARANCE, N_SAMPLE_POINTS, PATH_SMOOTHING_WINDOW
from app.core.geometry import sample_points_in_polygon


def _clearance_pt(geom: BaseGeometry, x: float, y: float) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.boundary.distance(Point(x, y)))


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
    pad = window // 2
    kernel = np.ones(window) / window
    out = np.zeros_like(xy)
    for c in range(xy.shape[1]):
        out[:, c] = np.convolve(xy[:, c], kernel, mode="same")
    return out


def build_internal_path_polyline(
    safe_poly: BaseGeometry,
    seed: int | None,
) -> LineString | None:
    """
    Build internal path: top-K sample points by clearance, project onto PCA axis,
    sort by projected coordinate, smooth, return LineString. Deterministic with seed.
    """
    if safe_poly is None or safe_poly.is_empty:
        return None
    pts = sample_points_in_polygon(safe_poly, N_SAMPLE_POINTS, seed=seed)
    if len(pts) < 2:
        return None
    with_clearance = [(x, y, _clearance_pt(safe_poly, x, y)) for x, y in pts]
    with_clearance.sort(key=lambda t: t[2], reverse=True)
    top = with_clearance[: min(K_TOP_CLEARANCE * 2, len(with_clearance))]
    xy = np.array([[t[0], t[1]] for t in top])
    ax, ay = _pca_axis_xy(safe_poly)
    proj = xy[:, 0] * ax + xy[:, 1] * ay
    order = np.argsort(proj)
    xy = xy[order]
    window = PATH_SMOOTHING_WINDOW if PATH_SMOOTHING_WINDOW % 2 == 1 else PATH_SMOOTHING_WINDOW + 1
    xy = _smooth_polyline(xy, max(3, window))
    coords = [(float(xy[i, 0]), float(xy[i, 1])) for i in range(len(xy))]
    if len(coords) < 2:
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
    """Minimum distance from path to boundary of boundary_poly (sample along path)."""
    if path is None or boundary_poly is None or path.is_empty or boundary_poly.is_empty:
        return 0.0
    from app.core.config import PATH_SAMPLE_STEP_PT
    pts = sample_along_path(path, PATH_SAMPLE_STEP_PT)
    if not pts:
        return 0.0
    boundary = boundary_poly.boundary
    min_d = float("inf")
    for p in pts:
        d = boundary.distance(p)
        if d < min_d:
            min_d = float(d)
    return min_d if min_d != float("inf") else 0.0
