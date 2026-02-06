# app/models/train.py
"""
Generate synthetic river-like polygons and train a regressor for candidate ranking.
Saves model.joblib and metadata.json to app/models/artifacts/. See: docs/AI_MODEL.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

from app.core.config import N_SAMPLE_POINTS, PADDING_PT, SEED
from app.core.geometry import polygon_bounds, sample_points_in_polygon
from app.core.preprocess import preprocess_river
from app.models.features import FEATURE_ORDER, candidate_to_features, features_to_vector


def _wiggly_polyline(n_pts: int = 30, width: float = 80.0, seed: int | None = 42) -> LineString:
    """Generate a wiggly centerline and return as LineString."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_pts)
    x = width * (t + 0.1 * rng.standard_normal(n_pts))
    y = width * (t * 0.5 + 0.15 * np.sin(t * 8) + 0.1 * rng.standard_normal(n_pts))
    coords = list(zip(x.tolist(), y.tolist()))
    return LineString(coords)


def _synthetic_polygon(seed: int | None = None, buffer_width: float = 15.0) -> Polygon | None:
    """Create a river-like polygon by buffering a wiggly polyline. Vary width/noise."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 40))
    w = float(rng.uniform(50, 120))
    line = _wiggly_polyline(n, w, seed=seed)
    try:
        poly = line.buffer(buffer_width + rng.uniform(-3, 5), resolution=4)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if hasattr(poly, "geoms"):
            poly = poly.geoms[0] if poly.geoms else None
        return poly if isinstance(poly, Polygon) else None
    except Exception:
        return None


def generate_training_data(
    n_polygons: int = 100,
    seed: int | None = SEED,
) -> tuple[list[list[float]], list[float]]:
    """
    Generate synthetic polygons, run candidate generation, compute features and oracle y.
    Returns (X_list of feature vectors, y_list). Oracle: y = clearance + fit_margin - angle_penalty.
    """
    from app.core.candidates_a import generate_candidate_points, angle_candidates_deg
    from app.core.geometry import oriented_rectangle, pca_dominant_angle_deg
    from app.core.text_metrics import measure_text_pt
    from app.core.validate import validate_rect_inside_safe
    from app.core.types import LabelSpec
    from app.core.config import DEFAULT_FONT_FAMILY

    label = LabelSpec(text="RIVER", font_family=DEFAULT_FONT_FAMILY, font_size_pt=12.0)
    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0 or h_pt <= 0:
        w_pt, h_pt = 40.0, 12.0

    X_all: list[list[float]] = []
    y_all: list[float] = []
    rng = np.random.default_rng(seed)

    for i in range(n_polygons):
        poly = _synthetic_polygon(seed=seed + i if seed is not None else i)
        if poly is None or poly.is_empty:
            continue
        try:
            _, safe_poly = preprocess_river(poly, padding_pt=PADDING_PT)
            if safe_poly.is_empty:
                continue
            candidates = generate_candidate_points(safe_poly, n_sample=min(200, N_SAMPLE_POINTS), k_top=30, seed=seed + i * 7)
            bounds = polygon_bounds(safe_poly)
            angles = angle_candidates_deg(poly)
            if not angles:
                angles = [0.0]
            for cp in candidates:
                for angle_deg in angles:
                    rect = oriented_rectangle(cp.x, cp.y, w_pt, h_pt, angle_deg)
                    ok, bbox_min_clearance_pt = validate_rect_inside_safe(safe_poly, rect)
                    if not ok:
                        continue
                    fit = bbox_min_clearance_pt / PADDING_PT if PADDING_PT > 0 else 1.0
                    from app.core.scoring import centering_score, angle_penalty_deg
                    centering = centering_score(cp.x, cp.y, bounds)
                    angle_pen = angle_penalty_deg(angle_deg)
                    y = cp.clearance + fit - angle_pen
                    features = candidate_to_features(
                        cp.x, cp.y, cp.clearance, angle_deg,
                        bbox_min_clearance_pt, bounds, PADDING_PT,
                    )
                    X_all.append(features_to_vector(features))
                    y_all.append(y)
        except Exception:
            continue

    return X_all, y_all


def train_and_save(
    n_polygons: int = 100,
    seed: int | None = SEED,
) -> Path:
    """
    Generate data, train GradientBoostingRegressor (or RandomForestRegressor fallback), save artifact.
    Returns path to artifacts dir.
    """
    X_list, y_list = generate_training_data(n_polygons=n_polygons, seed=seed)
    if len(X_list) < 10:
        raise ValueError("Insufficient training data")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=seed, min_samples_leaf=5)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=seed, min_samples_leaf=5)

    import numpy as np
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    model.fit(X, y)

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, artifacts_dir / "model.joblib")
    meta = {
        "feature_names": list(FEATURE_ORDER),
        "version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X_list),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return artifacts_dir


def train_model(n_polygons: int = 100, seed: int | None = SEED) -> Path:
    """
    Train the learned ranking model on synthetic data and save to app/models/artifacts/.
    Returns path to the artifacts directory. Use small n_polygons for fast runs.
    """
    return train_and_save(n_polygons=n_polygons, seed=seed)
