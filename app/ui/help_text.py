# app/ui/help_text.py
"""
Reusable help strings for UI tooltips and glossary.
Align with docs/UI_SPEC.md and docs/AI_MODEL.md.
"""

# Short one-liners for widget help=
TOOLTIP_WKT = "Well-Known Text: POLYGON, MULTIPOLYGON, or GEOMETRYCOLLECTION."
TOOLTIP_CURVED = "Place label along river centerline; falls back to straight if not feasible."
TOOLTIP_LEARNED = "Blend heuristic score with trained model (when model is trained)."
TOOLTIP_EVAL = "Run baselines vs heuristic vs heuristic+model on default + synthetic polygons."
TOOLTIP_RENDER_SCALE = "1x=800×600, 2x=1600×1200, 4x=3200×2400. Applies to PNG output; curved label appears in both PNG and SVG."
TOOLTIP_RUN_AGAIN = "Re-run with current geometry and options; uses a new folder name automatically (same styling as Run demo)."
TOOLTIP_OVERRIDE_NK = "More candidates (N) or keep more by clearance (K). Does not change seed or heuristic formula; use when placement fails or you want denser search."
TOOLTIP_TRAIN_SYNTHETIC = "Synthetic polygons only (real WKT training not implemented). Improves ranking on similar river-like shapes."
EVAL_SHORT = "Compare baseline, heuristic, and learned methods on default + synthetic geometry. Outputs leaderboard and plots; see Help for detail."

# Example WKT for "Merge nearby components (braided rivers)": two nearby polygon parts
WKT_BRAIDED_EXAMPLE = "MULTIPOLYGON (((10 10, 90 15, 85 80, 15 75, 10 10)), ((12 25, 88 30, 92 65, 18 60, 12 25)))"
TOOLTIP_MERGE_NEARBY = "Merge multiple polygon parts (e.g. braided rivers) within a distance; only affects multi-part geometry. Example WKT in Help & glossary."

# Shorter inline help; detail lives in GLOSSARY_MD
WKT_HELP = "Standard text format for geometry. Use POLYGON, MULTIPOLYGON, or GEOMETRYCOLLECTION."
CURVED_LABEL_HELP = "Place label along a centerline path; falls back to straight if not feasible."
LEARNED_RANKING_HELP = "Blend heuristic with a trained model to pick the best candidate."
SYNTHETIC_HELP = "Training uses generated river-like polygons only (no external datasets)."

# Full glossary for Help & glossary expander
GLOSSARY_MD = """
### Safe polygon
The river polygon shrunk inward by the **padding** distance. The label must fit fully inside this region.

### Clearance
Distance from a point (or the label box) to the boundary of the safe polygon. Higher clearance = more room.

### Fit margin
Ratio of actual clearance to required padding (e.g. 1.0 = exactly fits; >1 = extra room).

### Centroid baseline
Evaluation baseline that places the label at the polygon center with no rotation. Used to compare against heuristic and learned placement.

### WKT
Well-Known Text: a standard text format for vector geometry (points, lines, polygons). You can paste a WKT string or upload a .wkt/.txt file.

### Supported geometry types
**POLYGON**, **MULTIPOLYGON**, or **GEOMETRYCOLLECTION** containing polygons. For multi-part geometry, choose a component (Auto = best by safe area, or 0, 1, 2…).

### Example WKT: Merge nearby components (braided rivers)
Use **Paste WKT** and enable **Merge nearby components** to test merging two nearby polygon parts (e.g. braided channels):
```
MULTIPOLYGON (((10 10, 90 15, 85 80, 15 75, 10 10)), ((12 25, 88 30, 92 65, 18 60, 12 25)))
```

### Seed
Random seed for sampling candidate points (0 or positive). When **Deterministic** is on, the same seed gives the same result. Preset does not change the seed value; it only sets curved/learned/scale options.

### Padding / safe polygon
The app builds a *safe polygon* by shrinking the river inward by the padding distance (in points). The label is placed inside this safe region. Larger padding = smaller safe area; if the safe polygon becomes empty, reduce padding or font size.

### Straight vs curved
- **Straight**: Places the label at a single point and angle. Fast and robust.
- **Curved**: Tries to place the label along a centerline path inside the river. If the path is too short or the geometry unsuitable, placement falls back to straight.

### Learned ranking
An optional trained model blends with heuristic scores to pick the best placement. Improves results on river-like shapes. Requires training first (see AI & Evaluation).

### Synthetic data
Training uses **synthetic** river-like polygons only: the app generates wiggly polylines, buffers them to polygons, then runs the same placement pipeline to train a regressor. No external datasets.

### Evaluation
Runs baseline (centroid), heuristic-only, and heuristic+model on the default geometry plus synthetic polygons. Results and plots appear in the **Evaluate** tab. Supports **families** (e.g. straight_wide, curved_narrow) and outputs **leaderboard** (by method and family).

### Zoom buckets
Consistent placement across zoom levels: same geometry is run at several zoom buckets (e.g. 10, 12, 14) with scaled font and padding. Lets you pick the best zoom for export.

### All results tab
View **all** processed runs (Demo/CLI, Batch, Eval) in one place. Open any run in the Demo tab, set **Compare A** / **Compare B** for the Compare tab, or get the geometry path to **re-run with learned ranking** after training.

### Batch mode
Run placement on many rivers at once: provide a directory of .wkt files or a manifest (CSV with path, labels). Output is **reports/batch_<name>/index.csv** and **cases/<case_id>/** with placement(s) and images.

### Collision avoidance
When placing **multiple labels** on one river, each new label avoids already-placed labels. Longer labels are placed first; candidates overlapping occupied areas are penalized or rejected.

### WKT for multi-label and collision
**Same as single-label:** use **POLYGON**, **MULTIPOLYGON**, or **GEOMETRYCOLLECTION** (polygons). One geometry = one river. For **multiple labels on the same river**, give one WKT and a comma-separated label list (e.g. "ELBE, MAIN"). The app places each label in order with collision avoidance. For **batch** with multiple labels per file, use the same WKT per file and set "Labels" or the manifest `labels` column (e.g. "ELBE, MAIN"). Multi-part (MULTIPOLYGON) is supported; "Auto" picks the best part by safe area.

### Re-running after training (better results)
After you **train the model**, the AI can produce better placements for the same rivers. Re-run placement on the **same geometry** with **Use learned ranking** enabled. Use the **same seed** for reproducibility. In **All results**, open a past run and use "Re-run with learned ranking" to see the geometry path; load that path in the sidebar, enable Use learned ranking, then Run demo with a new name. The new run is comparable to the old (same seed, different ranking).

### Docs
- [ALGORITHM.md](docs/ALGORITHM.md) — Placement steps (Phase A/B, scoring).
- [EVALUATION.md](docs/EVALUATION.md) — Baselines, metrics, leaderboard.
- [AI_MODEL.md](docs/AI_MODEL.md) — Learned ranking and training.
- [BATCH.md](docs/BATCH.md) — Batch mode (UI and CLI); [BATCH_MANIFEST.md](docs/BATCH_MANIFEST.md) for manifest CSV schema.
"""

# Common failures and fixes
QUICK_TROUBLESHOOT_MD = """
### Geometry failed to load
- **Default not found**: Ensure you're running from the repo root and `docs/assets/problem/Problem_1_river.wkt` exists.
- **Upload / paste**: Use valid WKT (starts with POLYGON, MULTIPOLYGON, or GEOMETRYCOLLECTION). Remove extra text after the geometry.

### Safe polygon became empty
The river is too narrow for the current padding. **Fix**: Reduce padding (pt) or font size, then run again.

### Phase B fallback to Phase A
Curved placement wasn't feasible (path too short, window outside polygon, etc.). The reason is shown in the Demo tab. You can still use the straight label result.

### Run failed / no output
- Check that label text is not empty and length ≤ 50.
- Output folder name is sanitized to letters, numbers, underscore, hyphen (max 40 chars). If empty, a timestamped name is used.

### Placement always at centroid
Internal placement found no feasible candidate (safe polygon too small or label too large). **Fix**: Reduce padding or font size; check that the geometry is valid and has interior area.

### No curved label
Curved placement requires a long enough centerline and sufficient clearance. **Fix**: Try a longer/wider river segment, or reduce font size; see Debug tab for the fallback reason.

### Model not available
Use **Train model (synthetic)** in the AI & Evaluation section. "Use learned ranking" is disabled until the model artifact exists at `app/models/artifacts/model.joblib`.
"""
