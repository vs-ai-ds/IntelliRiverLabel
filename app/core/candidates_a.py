# app/core/candidates_a.py
"""
Phase A candidates: sample points in safe polygon, clearance, top K, angle candidates
from PCA + config deltas, upright orientation. See: docs/ALGORITHM.md A2–A3.
"""

from __future__ import annotations

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from app.core.config import (
    ANGLE_OFFSETS_DEG,
    K_TOP_CLEARANCE,
    N_SAMPLE_POINTS,
    SEED,
)
from app.core.geometry import pca_dominant_angle_deg, sample_points_in_polygon
from app.core.types import CandidatePoint


def _clearance_pt(geom: BaseGeometry, x: float, y: float) -> float:
    """Distance from (x, y) to boundary of geom."""
    if geom is None or geom.is_empty:
        return 0.0
    p = Point(x, y)
    return float(geom.boundary.distance(p))


def generate_candidate_points(
    safe_polygon: BaseGeometry,
    n_sample: int = N_SAMPLE_POINTS,
    k_top: int = K_TOP_CLEARANCE,
    seed: int | None = SEED,
) -> list[CandidatePoint]:
    """
    Sample n points inside safe polygon, compute clearance, keep top K.
    See: docs/ALGORITHM.md A2.
    """
    if safe_polygon is None or safe_polygon.is_empty:
        return []
    pts = sample_points_in_polygon(safe_polygon, n_sample, seed=seed)
    with_clearance = [
        (x, y, _clearance_pt(safe_polygon, x, y)) for x, y in pts
    ]
    with_clearance.sort(key=lambda t: t[2], reverse=True)
    top = with_clearance[:k_top]
    return [
        CandidatePoint(x=x, y=y, clearance=cl, base_score=cl, features={"clearance": cl})
        for x, y, cl in top
    ]


def angle_candidates_deg(
    geom: BaseGeometry,
    offsets_deg: tuple[float, ...] = ANGLE_OFFSETS_DEG,
) -> list[float]:
    """
    Base angle from PCA of boundary, then base ± each offset.
    Enforce upright: return angles in [0, 180) so text is not upside-down.
    See: docs/ALGORITHM.md A3.
    """
    base = pca_dominant_angle_deg(geom)
    angles: list[float] = []
    for off in offsets_deg:
        a = base + off
        a = a % 180.0
        if a < 0:
            a += 180.0
        if a not in angles:
            angles.append(a)
    for off in offsets_deg:
        a = base - off
        a = a % 180.0
        if a < 0:
            a += 180.0
        if a not in angles:
            angles.append(a)
    return sorted(set(angles))
