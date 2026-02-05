# app/core/candidates_b.py
"""
Phase B: propose a path window of required length with best clearance proxy.
See: docs/ALGORITHM.md B2.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from app.core.path_b import min_clearance_along_path, path_length
from app.core.config import PATH_SAMPLE_STEP_PT


def _curvature_proxy(path: LineString) -> float:
    """Sum of absolute angle changes between segments (degrees)."""
    if path is None or path.is_empty:
        return 0.0
    coords = list(path.coords)
    if len(coords) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(coords) - 1):
        ax = coords[i][0] - coords[i - 1][0]
        ay = coords[i][1] - coords[i - 1][1]
        bx = coords[i + 1][0] - coords[i][0]
        by = coords[i + 1][1] - coords[i][1]
        a_rad = math.atan2(ay, ax)
        b_rad = math.atan2(by, bx)
        delta = math.degrees(abs((b_rad - a_rad + math.pi) % (2 * math.pi) - math.pi))
        total += delta
    return total


def propose_path_window(
    path: LineString,
    required_len: float,
    boundary_poly: BaseGeometry,
) -> LineString | None:
    """
    Return a sub-LineString window of length >= required_len with best clearance proxy.
    Scans windows by arclength; prefers high min clearance and low curvature.
    """
    if path is None or path.is_empty or required_len <= 0 or boundary_poly is None:
        return None
    total_len = path_length(path)
    if total_len < required_len:
        return None

    best_window: LineString | None = None
    best_score = float("-inf")
    step = max(1.0, required_len * 0.2)
    start_dist = 0.0
    while start_dist + required_len <= total_len:
        end_dist = start_dist + required_len
        seg = _segment_by_length(path, start_dist, end_dist)
        if seg is not None and path_length(seg) >= required_len * 0.95:
            clearance = min_clearance_along_path(seg, boundary_poly)
            curv = _curvature_proxy(seg)
            score = clearance - 0.01 * curv
            if score > best_score:
                best_score = score
                best_window = seg
        start_dist += step
        if start_dist + required_len > total_len:
            break

    return best_window


def _segment_by_length(path: LineString, start_dist: float, end_dist: float) -> LineString | None:
    """Extract sub-path from start_dist to end_dist along path."""
    if path is None or path.is_empty:
        return None
    length = path_length(path)
    if start_dist >= length or end_dist <= start_dist:
        return None
    end_dist = min(end_dist, length)
    pts = []
    for d in [start_dist, end_dist]:
        pt = path.interpolate(d)
        if pt is not None:
            pts.append((pt.x, pt.y))
    if len(pts) < 2:
        return None
    n = max(10, int((end_dist - start_dist) / PATH_SAMPLE_STEP_PT) + 1)
    dists = np.linspace(start_dist, end_dist, n, endpoint=True)
    coords = []
    for d in dists:
        pt = path.interpolate(d)
        coords.append((pt.x, pt.y))
    return LineString(coords)
