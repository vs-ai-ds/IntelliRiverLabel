# Evaluation

Evaluation compares placement methods on a provided river WKT and a small batch of synthetic polygons. Implemented in **`app/core/evaluate.py`** (CLI) and **`app/core/plots.py`** (figures).

## What evaluation measures

- **Success** — Whether placement is feasible (label inside safe region).
- **min_clearance_pt**, **fit_margin_ratio** — Quality of fit.
- **family** — Synthetic case family (e.g. straight_wide, curved_narrow, braided) for breakdown.
- **collisions**, **duration_ms** — For multi-label / batch runs (when applicable).

## Families

Synthetic cases are tagged with a **family** (e.g. straight_wide, straight_narrow, curved_wide, curved_narrow, braided) so results can be aggregated by shape type. Families are assigned from a fixed list in the evaluator.

## How to run

From repo root (example):

```powershell
.\.venv\Scripts\python -m app.core.evaluate --run-name eval_01
```

Optional: `--geometry <path>`, `--n-synthetic 20`, `--seed 42`, `--repo-root <path>`.

## Inputs

- **Provided geometry**: One river WKT file (default: `docs/assets/problem/Problem_1_river.wkt` if present).
- **Synthetic batch**: A small number of generated river-like polygons (default 20) from the same synthetic pipeline used for training (wiggly polyline + buffer). Each case is assigned a family for leaderboard breakdown.

## Baselines and modes

For each test case (river or synthetic polygon), the evaluator runs:

1. **baseline_centroid** — Place label at polygon centroid with zero rotation; no Phase A/B.
2. **heuristic_only** — Phase A candidate generation and selection using heuristic score only (learned ranking off).
3. **heuristic_plus_model** — Phase A with learned ranking enabled (blended score). This row is produced only when the model artifact exists (`app/models/artifacts/model.joblib`).

## Metrics

Per run (per source × method), the following are recorded:

- **success** — Whether placement is feasible (label inside safe region).
- **min_clearance_pt** — Minimum distance from label bbox to river boundary.
- **fit_margin_ratio** — Available margin relative to required padding.
- **mode_used** — Result mode (e.g. `phase_a_straight`, `external_fallback`, `baseline_centroid`).
- **family** — Case family (for synthetic cases).
- **n_labels**, **collisions**, **duration_ms** — When multi-label or timing is recorded.

## Leaderboard

The evaluator writes:

- **leaderboard.json** — `by_method`: success_rate, avg_clearance, avg_fit_margin, collision_rate, avg_duration_ms per method; `by_family`: success_count, n, success_rate per family.
- **leaderboard.csv** — Same by_method data in CSV form for spreadsheets.

Meaning: **success_rate** = fraction of cases where placement succeeded; **collision_rate** = average collisions per case (for multi-label); **avg_clearance** / **avg_fit_margin** = mean quality metrics.

## Outputs (from evaluate.py)

All written under **`reports/<run_name>/`** (e.g. `reports/eval_01/`):

- **evaluation_results.csv** — One row per (source, method) with columns: source, method, success, min_clearance_pt, fit_margin_ratio, mode_used, family, n_labels, collisions, duration_ms (when applicable).
- **evaluation_summary.json** — Aggregates: run_name, timestamp_utc, n_cases, success_rate (by method), mean_min_clearance_pt (by method), mean_fit_margin_ratio (by method).
- **leaderboard.json** / **leaderboard.csv** — Per-method and per-family aggregates (see Leaderboard above).
- **plots/** (from `app/core/plots.py`):
  - **success_rate.png** — Bar chart of success rate by method.
  - **min_clearance_pt.png** — Boxplot of min_clearance_pt by method.
  - **success_by_family.png** — Success rate by family.
  - **collision_rate_by_method.png** — Collision rate by method.

Plots are generated automatically at the end of the same evaluation run. Do not commit generated report artifacts under `reports/`.
