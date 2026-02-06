# app/models/registry.py
"""
Load trained model artifact if present. Returns (model, feature_names) or None.
No hard failure if missing. See: docs/AI_MODEL.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.models.features import FEATURE_ORDER


def get_model_metadata() -> dict | None:
    """
    Read metadata for UI display (artifact path, timestamp, feature list).
    Does not load the model. Returns None if artifact dir or metadata.json missing.
    """
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "metadata.json"
    if not model_path.exists():
        return None
    out: dict = {
        "artifact_path": str(model_path),
        "feature_names": list(FEATURE_ORDER),
    }
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out["feature_names"] = meta.get("feature_names", out["feature_names"])
            if "trained_timestamp_utc" in meta:
                out["trained_timestamp_utc"] = meta["trained_timestamp_utc"]
            if "timestamp_utc" in meta:
                out["trained_timestamp_utc"] = meta["timestamp_utc"]
        except Exception:
            pass
    return out


def load_model() -> tuple[object, list[str]] | None:
    """
    Load model and feature names from app/models/artifacts/ (path relative to this file).
    Returns (model, feature_names) or None if artifact missing or load fails.
    """
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "metadata.json"
    if not model_path.exists():
        import logging
        logging.getLogger(__name__).warning(
            "Learned model not found at %s; using heuristic-only ranking.",
            str(model_path),
        )
        return None
    try:
        import joblib
        model = joblib.load(model_path)
        feature_names = list(FEATURE_ORDER)
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feature_names = meta.get("feature_names", feature_names)
        return (model, feature_names)
    except Exception:
        return None
