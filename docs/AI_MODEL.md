# Learned Ranking Model (Optional)

This project includes an optional learned ranking stage to improve candidate selection beyond fixed heuristics.

## What the model does
Given features computed for each candidate placement, predict a "suitability" score.
Final selection uses:
- blended_score = alpha * heuristic_score + (1 - alpha) * model_score

## Why this qualifies as learned intelligence
- The model is trained on examples (synthetic + small curated)
- It improves rankings measurably over centroid/random baselines
- It remains explainable via feature importance and ablation

## Model choice (scikit-learn)
Recommended:
- GradientBoostingRegressor (fast, strong for tabular)
Alternative:
- RandomForestRegressor (robust, less tuning)

## Candidate features (Phase A)
- clearance_pt
- fit_margin_ratio
- bbox_inside_margin (min distance from bbox to safe boundary)
- centering_score (distance from safe centroid / major axis)
- angle_penalty (near-vertical penalty)
- local_clearance_mean / local_clearance_std (stability)

Phase B features (if available):
- segment_clearance_min
- curvature_total_deg
- straightness_ratio

## Training data
A) Synthetic generator:
- Generate a wiggly centerline polyline
- Buffer to polygon with varying width
- Create "good" labels using an oracle rule:
  - high clearance + good fit + low curvature
- This yields thousands of labeled examples fast.

B) Curated real examples:
- A small set where we manually approve placements to sanity-check the oracle.

## Evaluation
Compare:
- centroid baseline
- heuristic-only
- blended heuristic + model

Report:
- success rate
- average clearance
- average fit margin
- visual quality on 8–10 showcase rivers