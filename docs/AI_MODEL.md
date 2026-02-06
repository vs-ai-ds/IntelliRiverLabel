# Learned Ranking Model (Optional)

This project includes an optional learned ranking stage to improve candidate selection beyond fixed heuristics.

## What the model does

Given features computed for each candidate placement, the model predicts a suitability score. Final selection uses:

- **blended_score** = alpha × heuristic_score + (1 − alpha) × model_score

Config: `ENABLE_LEARNED_RANKING` (default False), `LEARNED_BLEND_ALPHA` (default 0.6). If the model artifact is missing, placement falls back to heuristic-only (no failure).

## Training approach (synthetic data)

Training uses **synthetic river-like polygons** only (no external datasets):

1. **Synthetic polygon generation** (`app/models/train.py`):
   - Build a wiggly centerline polyline (random walk with controlled noise).
   - Buffer the polyline to a polygon with varying width and noise; resolution and width are randomized for diversity.
   - Ensure validity (e.g. `buffer(0)` when needed). Only valid polygons are used.

2. **Candidate generation and oracle target**:
   - For each synthetic polygon, run Phase A–style candidate generation (sample points, angles).
   - For each feasible (point, angle) candidate, compute the feature vector and a **oracle goodness** target:
     - **y** = clearance + fit_margin − angle_penalty (weighted combination).

3. **Regressor**:
   - **GradientBoostingRegressor** (default) or **RandomForestRegressor** (fallback if sklearn version lacks GBR).
   - Trained on (X = feature vectors, y = oracle y). Artifact path: `app/models/artifacts/model.joblib`; metadata (feature names, version, timestamp) in `app/models/artifacts/metadata.json`.

## Candidate features (Phase A)

Stable ordering for model input is defined in `app/models/features.py` (`FEATURE_ORDER`). Features include:

- **clearance_pt** — clearance at candidate point
- **fit_margin_ratio** — available margin / required padding
- **centering_score** — how well the point is centered in the safe region
- **angle_penalty** — penalty for near-vertical angles (readability)
- **bbox_min_clearance_pt** — minimum clearance from the label bbox to the safe boundary (from validation)

## Registry and scoring

- **`app/models/registry.py`**: `load_model()` returns `(model, feature_names)` if the artifact exists, else `None`. No hard failure if missing.
- **`app/core/scoring.py`**: `score_with_model(features_dict)` returns the model score or 0.0 if no model; `score_blended(heuristic_score, features_dict, use_model=...)` combines heuristic and model when the caller enables learned ranking.

## Synthetic families

Training and evaluation can tag synthetic polygons by **family** (e.g. straight_wide, curved_narrow, braided). The model is trained on the same synthetic pipeline (wiggly polyline + buffer); it does not see real river data. Evaluation reports success rate and metrics per method and per family.

## What the model can and cannot do

- **Can**: Improve candidate ranking on river-like shapes; blend with heuristics so placement stays robust when the model is wrong.
- **Cannot**: Replace Phase A/B geometry (it only scores candidates); guarantee placement on arbitrary polygons; handle multi-label collision (that is layout logic).

## Where artifacts are stored

- **Model**: `app/models/artifacts/model.joblib`
- **Metadata** (feature names, version, trained timestamp): `app/models/artifacts/metadata.json`

Do not commit large artifacts; keep `app/models/artifacts/` in `.gitignore` if desired.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for baselines (centroid, heuristic-only, heuristic+model) and metrics produced by `app/core/evaluate.py`.
