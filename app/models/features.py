# app/models/features.py
"""
Convert a candidate (point, angle, rect validation stats) into a numeric feature vector.
Stable key ordering for model input. See: docs/AI_MODEL.md.
"""

from __future__ import annotations

from app.core.scoring import angle_penalty_deg, centering_score, fit_margin_ratio


FEATURE_ORDER: tuple[str, ...] = (
    "clearance_pt",
    "fit_margin_ratio",
    "centering_score",
    "angle_penalty",
    "bbox_min_clearance_pt",
)


def candidate_to_features(
    cx: float,
    cy: float,
    clearance_pt: float,
    angle_deg: float,
    bbox_min_clearance_pt: float,
    bounds: tuple[float, float, float, float],
    padding_pt: float,
) -> dict[str, float]:
    """
    Build a feature dict for one candidate (point + angle + rect validation stats).
    Includes all FEATURE_ORDER keys.
    """
    fit = bbox_min_clearance_pt / padding_pt if padding_pt > 0 else 1.0
    return {
        "clearance_pt": float(clearance_pt),
        "fit_margin_ratio": float(fit),
        "centering_score": float(centering_score(cx, cy, bounds)),
        "angle_penalty": float(angle_penalty_deg(angle_deg)),
        "bbox_min_clearance_pt": float(bbox_min_clearance_pt),
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Stable ordering for model input. Uses FEATURE_ORDER."""
    return [features[k] for k in FEATURE_ORDER if k in features]
