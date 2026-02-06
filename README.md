# IntelliRiverLabel

Places a river name label **inside** the river polygon (Phase A) with safe padding and readable placement. Optional learned ranking improves candidate selection. Supports **multi-label** placement with **collision avoidance**, **batch mode** (many rivers), **zoom buckets** (consistent scale by zoom), and **evaluation families** with **leaderboard** export. Outputs: `before.png`, `after.png`, `debug.png`, and `placement.json` (or `placements.json` for multi-label) to `reports/<run_name>/`.

## Demo

🌐 **Live Demo:** https://intelliriverlabel.streamlit.app/

## Why this is hard

River labels must sit **inside** the polygon with padding, avoid clipping, and stay **readable** (orientation and clearance). Polygons are irregular, so centroid placement often fails or looks poor; we need many candidates and a way to pick the best.

## Our solution

- **Phase A (default):** Internal straight placement with padding; label fully inside the polygon with optional rotation. Candidates are scored by clearance, fit margin, centering, and angle; optionally blended with a **learned ranking** model (scikit-learn regressor trained on synthetic river-like polygons).
- **Phase B (optional):** Curved label along the river centerline when straight placement is insufficient.
- **Learned ranking (optional):** A small regressor trained on synthetic buffered polylines improves candidate selection; final score = alpha × heuristic + (1 − alpha) × model. The system works without it (heuristic-only).

## What it does

- Input geometry: `docs/assets/problem/Problem_1_river.wkt` (repo-relative).
- Outputs: `before.png`, `after.png`, `debug.png`, `placement.json` under `reports/<run_name>/`.

## Setup (Windows)

From the repo root:

1. Create the virtual environment: `py -m venv .venv` or `python -m venv .venv`
2. Install dependencies: `pip install -r requirements.txt` (or `.\.venv\Scripts\pip install -r requirements.txt` if not activated)

## Quickstart

### Streamlit (UI)

Run from the **repo root** (e.g. `D:\HackArena3.0\IntelliRiverLabel`) so the `app` package is found:

```powershell
cd D:\HackArena3.0\IntelliRiverLabel
set PYTHONPATH=%CD%
.\.venv\Scripts\python -m streamlit run app/ui/app.py
```

### Single CLI run

```powershell
.\.venv\Scripts\python -m app.core.runner --text "ELBE" --font-size-pt 12 --run-name demo_01
```

### Batch CLI run (directory of .wkt files)

```powershell
.\.venv\Scripts\python -m app.core.runner --batch-dir path/to/wkt_folder --labels "ELBE,MAIN" --run-name demo_01 --batch-output-dir reports
```
Output: `reports/batch_<run_name>/index.csv` and `cases/<case_id>/`.

### Evaluation run

```powershell
.\.venv\Scripts\python -m app.core.evaluate --run-name eval_01
```
Or run evaluation from the Streamlit sidebar (AI & Evaluation). Outputs: `evaluation_results.csv`, `leaderboard.csv`, `leaderboard.json`, and plots under `reports/<run_name>/`.

## Validation (baselines and metrics)

We compare:

- **Centroid baseline** — label at centroid, no rotation
- **Heuristic-only** — Phase A with heuristic score only
- **Heuristic + learned model** — Phase A with blended score (when model artifact exists)

Metrics: success (feasible placement), min_clearance_pt, fit_margin_ratio. Evaluation runner and plots are in `app/core/evaluate.py` and `app/core/plots.py`; see [docs/EVALUATION.md](docs/EVALUATION.md).

### How to run evaluation

From repo root:

```powershell
.\.venv\Scripts\python -m app.core.evaluate --run-name eval_01
```

### Expected outputs under `reports/eval_01/`

- `evaluation_results.csv` — per-case, per-method metrics (includes family, zoom_bucket, n_labels, collisions, duration_ms)
- `evaluation_summary.json` — aggregated success rate and mean metrics by method
- `leaderboard.csv` / `leaderboard.json` — aggregate per method and per family (success_rate, avg_clearance, collision_rate, etc.)
- `plots/success_rate.png` — bar chart of success rate by method
- `plots/min_clearance_pt.png` — distribution of min_clearance_pt by method
- `plots/success_by_family.png` — success rate by family
- `plots/collision_rate_by_method.png` — collision rate by method

Do not commit generated artifacts under `reports/`.

## Outputs (placement runs)

All run artifacts go to **`reports/<run_name>/`** (e.g. `reports/demo_01/`):

- `before.png`, `after.png`, `debug.png`
- `placement.json`
- `run_metadata.json`

## Documentation

- [Project spec](docs/PROJECT_SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Algorithm](docs/ALGORITHM.md)
- [Placement schema](docs/PLACEMENT_SCHEMA.md)
- [Evaluation](docs/EVALUATION.md)
- [AI / learned model](docs/AI_MODEL.md)
- [UI spec](docs/UI_SPEC.md)

## License

MIT — see [LICENSE](LICENSE).
