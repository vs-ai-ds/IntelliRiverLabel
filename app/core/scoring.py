# app/core/scoring.py
"""
Heuristic Phase A scoring: clearance, fit_margin_ratio, centering_score, angle_penalty.
Optional learned model blend. See: docs/ALGORITHM.md A5, docs/AI_MODEL.md.
"""

from __future__ import annotations

import math

from app.core.config import (
    ENABLE_LEARNED_RANKING,
    LEARNED_BLEND_ALPHA,
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


_model_cache: tuple[object, list[str]] | None | bool = False


def _get_model() -> tuple[object, list[str]] | None:
    """Load model once; cache result. None or False means not loaded yet."""
    global _model_cache
    if _model_cache is False:
        try:
            from app.models.registry import load_model
            _model_cache = load_model()
        except Exception:
            _model_cache = None
    return _model_cache if isinstance(_model_cache, tuple) else None


def score_with_model(features_dict: dict[str, float]) -> float:
    """
    Optional model score from feature dict. Returns 0.0 if model missing.
    Used only when ENABLE_LEARNED_RANKING and model exists.
    """
    model_tuple = _get_model()
    if model_tuple is None:
        return 0.0
    model, feature_names = model_tuple
    try:
        from app.models.features import FEATURE_ORDER, features_to_vector
        vec = features_to_vector(features_dict)
        if not vec:
            return 0.0
        import numpy as np
        X = np.array([vec], dtype=np.float64)
        pred = model.predict(X)
        return float(pred[0])
    except Exception:
        return 0.0


def score_blended(
    heuristic_score: float,
    features_dict: dict[str, float],
    alpha: float | None = None,
    use_model: bool | None = None,
) -> float:
    """
    final = alpha * heuristic + (1 - alpha) * model_score.
    use_model True = try model; False = heuristic only; None = use ENABLE_LEARNED_RANKING.
    If model not available, returns heuristic_score.
    """
    if use_model is None:
        use_model = ENABLE_LEARNED_RANKING
    if not use_model:
        return heuristic_score
    model_tuple = _get_model()
    if model_tuple is None:
        return heuristic_score
    a = LEARNED_BLEND_ALPHA if alpha is None else alpha
    model_s = score_with_model(features_dict)
    return a * heuristic_score + (1.0 - a) * model_s
