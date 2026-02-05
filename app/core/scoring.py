# app/core/scoring.py
"""
Heuristic Phase A scoring: clearance, fit_margin_ratio, centering_score, angle_penalty.
All weights from config. See: docs/ALGORITHM.md A5.
"""

from __future__ import annotations

import math

from app.core.config import (
    SCORE_WEIGHT_ANGLE_PENALTY,
    SCORE_WEIGHT_CENTERING,
    SCORE_WEIGHT_CLEARANCE,
    SCORE_WEIGHT_FIT_MARGIN,
)


def fit_margin_ratio(min_clearance: float, padding_pt: float) -> float:
    """How comfortably the label fits; 0 if no room, >0 if margin exists."""
    if padding_pt <= 0:
        return 1.0
    return min_clearance / padding_pt


def centering_score(
    cx: float, cy: float,
    bounds: tuple[float, float, float, float],
) -> float:
    """
    Prefer center of bbox; avoid ends if elongated.
    Returns 0 at edges, 1 at center (linear falloff).
    """
    minx, miny, maxx, maxy = bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 and h <= 0:
        return 1.0
    dx = abs((cx - (minx + maxx) / 2) / w) if w > 0 else 0.0
    dy = abs((cy - (miny + maxy) / 2) / h) if h > 0 else 0.0
    d = max(dx, dy)
    return max(0.0, 1.0 - d)


def angle_penalty_deg(angle_deg: float) -> float:
    """Optional readability: penalize steep angles. 0 = no penalty, 90 = max penalty."""
    a = abs(angle_deg % 180.0 - 90.0)
    return a / 90.0


def score_phase_a(
    clearance: float,
    fit_margin: float,
    centering: float,
    angle_deg: float,
) -> float:
    """
    Combined heuristic score. Higher is better.
    Weights from config. See: docs/ALGORITHM.md A5.
    """
    s = (
        SCORE_WEIGHT_CLEARANCE * clearance
        + SCORE_WEIGHT_FIT_MARGIN * fit_margin
        + SCORE_WEIGHT_CENTERING * centering
        - SCORE_WEIGHT_ANGLE_PENALTY * angle_penalty_deg(angle_deg)
    )
    return s
