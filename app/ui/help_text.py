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

# Shorter inline help; detail lives in GLOSSARY_MD
WKT_HELP = "Standard text format for geometry. Use POLYGON, MULTIPOLYGON, or GEOMETRYCOLLECTION."
CURVED_LABEL_HELP = "Place label along a centerline path; falls back to straight if not feasible."
LEARNED_RANKING_HELP = "Blend heuristic with a trained model to pick the best candidate."
SYNTHETIC_HELP = "Training uses generated river-like polygons only (no external datasets)."

# Full glossary for Help & glossary expander
GLOSSARY_MD = """
### WKT
Well-Known Text: a standard text format for vector geometry (points, lines, polygons). You can paste a WKT string or upload a .wkt/.txt file.

### Supported geometry types
**POLYGON**, **MULTIPOLYGON**, or **GEOMETRYCOLLECTION** containing polygons. For multi-part geometry, choose a component (Auto = best by safe area, or 0, 1, 2…).

### Seed
Random seed for sampling candidate points. When "Deterministic (use seed)" is on, the same inputs produce the same result. Turn off for non-deterministic runs.

### Padding / safe polygon
The app builds a *safe polygon* by shrinking the river inward by the padding distance (in points). The label is placed inside this safe region. Larger padding = smaller safe area; if the safe polygon becomes empty, reduce padding or font size.

### Phase A vs Phase B
- **Phase A (straight)**: Places a straight label at a single point and angle. Fast and robust.
- **Phase B (curved)**: Tries to place the label along a centerline path inside the river. If the path is too short or the geometry unsuitable, placement falls back to Phase A.

### Learned ranking
An optional trained model blends with heuristic scores to pick the best placement. Improves results on river-like shapes. Requires training first (see AI & Evaluation).

### Synthetic data
Training uses **synthetic** river-like polygons only: the app generates wiggly polylines, buffers them to polygons, then runs the same placement pipeline to train a regressor. No external datasets.

### Evaluation
Runs baseline (centroid), heuristic-only, and heuristic+model on the default geometry plus synthetic polygons. Results and plots appear in the **Evaluate** tab. Supports **families** (e.g. straight_wide, curved_narrow) and outputs **leaderboard** (by method and family).

### Zoom buckets
Consistent placement across zoom levels: same geometry is run at several zoom buckets (e.g. 10, 12, 14) with scaled font and padding. Lets you pick the best zoom for export.

### Batch mode
Run placement on many rivers at once: provide a directory of .wkt files or a manifest (CSV with path, labels). Output is **reports/batch_<name>/index.csv** and **cases/<case_id>/** with placement(s) and images.

### Collision avoidance
When placing **multiple labels** on one river, each new label avoids already-placed labels. Longer labels are placed first; candidates overlapping occupied areas are penalized or rejected.
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

### Model not trained
Use **Train model (synthetic)** in the AI & Evaluation section. "Use learned ranking" is disabled until the model artifact exists.
"""
