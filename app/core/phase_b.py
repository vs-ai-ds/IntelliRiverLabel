# app/core/phase_b.py
"""
Phase B curved placement: internal path, window selection, validation.
Returns PlacementResult or None; caller falls back to Phase A. See: docs/ALGORITHM.md.
"""

from __future__ import annotations

import math

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

PHASE_B_WARNING = "Phase B curved not available; used Phase A."

from app.core.candidates_b import propose_path_window
from app.core.config import (
    CURVE_EXTRA_CLEARANCE_PT,
    CURVE_FIT_MARGIN,
    CURVE_MAX_POINTS,
)
from app.core.path_b import (
    build_internal_path_polyline,
    min_clearance_along_path,
    path_length,
)
from app.core.text_metrics import measure_text_pt
from app.core.types import LabelSpec, PlacementResult


def _curvature_total_deg(path: LineString) -> float:
    """Sum of absolute angle changes along path (degrees)."""
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


def _straightness_ratio(path: LineString) -> float:
    """Chord length / path length; 1 = straight."""
    if path is None or path.is_empty:
        return 1.0
    length = path_length(path)
    if length <= 0:
        return 1.0
    coords = list(path.coords)
    if len(coords) < 2:
        return 1.0
    chord = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    if chord <= 0:
        return 1.0
    return min(1.0, chord / length)


def _downsample_path(coords: list[tuple[float, float]], max_pts: int) -> list[tuple[float, float]]:
    """Downsample to at most max_pts points."""
    if len(coords) <= max_pts:
        return list(coords)
    step = (len(coords) - 1) / (max_pts - 1)
    return [coords[int(i * step)] for i in range(max_pts)]


def try_phase_b_curved(
    polygon: BaseGeometry,
    safe_poly: BaseGeometry,
    label_spec: LabelSpec,
    padding_pt: float,
    seed: int | None,
    geometry_source: str = "",
) -> PlacementResult | None:
    """
    Attempt Phase B curved placement. Returns PlacementResult or None.
    On any failure (path build, window, clearance), returns None for Phase A fallback.
    """
    try:
        path = build_internal_path_polyline(safe_poly, seed)
        if path is None or path.is_empty:
            return None

        w_pt, _ = measure_text_pt(label_spec.text, label_spec.font_family, label_spec.font_size_pt)
        if w_pt <= 0:
            return None
        required_len = w_pt * CURVE_FIT_MARGIN

        window = propose_path_window(path, required_len, polygon)
        if window is None or path_length(window) < required_len * 0.95:
            return None

        min_clearance = min_clearance_along_path(window, polygon)
        required_clearance = padding_pt + CURVE_EXTRA_CLEARANCE_PT
        if min_clearance < required_clearance:
            return None

        coords = list(window.coords)
        path_pt = _downsample_path(coords, CURVE_MAX_POINTS)
        centroid = window.centroid
        anchor_pt = (float(centroid.x), float(centroid.y))

        curvature_deg = _curvature_total_deg(window)
        straightness = _straightness_ratio(window)
        fit_margin = min_clearance / padding_pt if padding_pt > 0 else 1.0

        b = window.bounds
        bbox_pt = [
            (b[0], b[1]),
            (b[2], b[1]),
            (b[2], b[3]),
            (b[0], b[3]),
        ]

        return PlacementResult(
            label_text=label_spec.text,
            font_size_pt=label_spec.font_size_pt,
            font_family=label_spec.font_family,
            geometry_source=geometry_source or "unknown",
            mode="phase_b_curved",
            confidence=min(1.0, 0.5 + fit_margin * 0.2),
            anchor_pt=anchor_pt,
            angle_deg=0.0,
            bbox_pt=bbox_pt,
            path_pt=path_pt,
            min_clearance_pt=min_clearance,
            fit_margin_ratio=fit_margin,
            curvature_total_deg=curvature_deg,
            straightness_ratio=straightness,
            warnings=[],
        )
    except Exception:
        return None
