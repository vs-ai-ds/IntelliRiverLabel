# app/core/candidates_b.py
"""
Phase B: propose a path window of required length with best clearance proxy.
See: docs/ALGORITHM.md.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from app.core.path_b import min_clearance_along_path, path_length
from app.core.config import PATH_SAMPLE_STEP_PT, PATH_WINDOW_STEP_PT


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
    return float(total)


def propose_path_window(
    path: LineString,
    required_len: float,
    boundary_poly: BaseGeometry,
) -> LineString | None:
    """
    Return a sub-LineString window of length >= required_len with best clearance proxy.
    Scans windows by arclength; prefers high min clearance and low curvature.
    boundary_poly should be SAFE polygon (recommended).
    """
    if path is None or path.is_empty or required_len <= 0 or boundary_poly is None or boundary_poly.is_empty:
        return None

    total_len = path_length(path)
    if total_len < required_len:
        return None

    best_window: LineString | None = None
    best_score = float("-inf")
    step = max(1.0, float(PATH_WINDOW_STEP_PT))

    start_dist = 0.0
    while start_dist + required_len <= total_len + 1e-9:
        end_dist = start_dist + required_len
        seg = _segment_by_length(path, start_dist, end_dist)
        if seg is None or seg.is_empty:
            start_dist += step
            continue

        seg_len = path_length(seg)
        if seg_len < required_len * 0.98:
            start_dist += step
            continue

        # CRITICAL: Verify segment is inside boundary_poly before scoring
        # Clip segment to ensure it's fully inside
        try:
            seg_clipped = seg.intersection(boundary_poly)
            if seg_clipped.is_empty:
                start_dist += step
                continue
            # Use clipped segment (or longest part if MultiLineString)
            from shapely.geometry import MultiLineString
            if isinstance(seg_clipped, LineString):
                seg = seg_clipped
            elif isinstance(seg_clipped, MultiLineString) and seg_clipped.geoms:
                seg = max(seg_clipped.geoms, key=lambda x: x.length)
            else:
                start_dist += step
                continue
            # Re-check length after clipping
            seg_len = path_length(seg)
            if seg_len < required_len * 0.95:
                start_dist += step
                continue
        except Exception:
            # If clipping fails, skip this segment
            start_dist += step
            continue

        # Clearance proxy against SAFE polygon boundary
        clearance = float(min_clearance_along_path(seg, boundary_poly))
        curv = _curvature_proxy(seg)

        # Prefer high clearance, low curvature
        score = clearance - 0.01 * curv

        if score > best_score:
            best_score = score
            best_window = seg

        start_dist += step

    return best_window


def _segment_by_length(path: LineString, start_dist: float, end_dist: float) -> LineString | None:
    """Extract sub-path from start_dist to end_dist along path."""
    if path is None or path.is_empty:
        return None
    length = path_length(path)
    if length <= 0 or start_dist >= length or end_dist <= start_dist:
        return None

    start_dist = max(0.0, float(start_dist))
    end_dist = min(float(end_dist), float(length))
    if end_dist <= start_dist:
        return None

    n = max(10, int((end_dist - start_dist) / max(1.0, float(PATH_SAMPLE_STEP_PT))) + 1)
    dists = np.linspace(start_dist, end_dist, n, endpoint=True)
    coords = []
    for d in dists:
        pt = path.interpolate(float(d))
        if pt is not None and not pt.is_empty:
            coords.append((float(pt.x), float(pt.y)))
    if len(coords) < 2:
        return None
    return LineString(coords)
