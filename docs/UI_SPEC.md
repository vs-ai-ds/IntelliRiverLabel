# UI Spec (Streamlit)

## Goals
- Make judging easy: strong visuals + explainability + exports
- Keep UI thin; core logic in app/core

## State model: active selection (single source of truth)

- **active_ref** (session state) is the single source of truth for "current run":
  - `{ "kind": "single" | "batch_root" | "batch_case" | "eval", "path": str, "case_id": str | None }`
- **Who sets active_ref:** All Results (Open in Demo, Use as A/B), Batch (Open case in Demo), Sidebar (Run demo / Run Again, Switch to run). Compare tab only sets compare_path_a / compare_path_b.
- **Who reads active_ref:** Demo, Debug, Export only. They render from `resolve_display_path(active_ref)`; no run picker inside these tabs. Eval runs: resolve_display_path returns None; use **Evaluate** tab (selected_eval_dir / last_eval_dir).
- **Run types:** **single** = one folder with placement. **batch_root** = reports/batch_*/. **batch_case** = reports/batch_*/cases/&lt;id&gt;/. **eval** = reports/eval_*/ (leaderboards/plots only; not openable in Demo/Debug/Export).
- **Run registry:** One scan at load (scan_reports); no tab rescans independently. All Results and Compare use the same records; Compare only placement runs (no eval).
- **Backward compatibility:** active_report_dir and last_report_dir are kept in sync when setting the current run.

## Tabs
1) **Demo** (default)
- Load `docs/assets/problem/Problem_1_river.wkt`
- Inputs: label text (default "ELBE"), font size pt (default 12), padding pt (from config), seed
- Toggle: Curved label (Phase B), default OFF
- Run button: runs placement (Phase B if curved ON, else Phase A) + render + reporting into `reports/<run_name>/`
- Show "Mode used": phase_a_straight / phase_b_curved / external_fallback
- Show before.png and after.png side by side; if Phase B used, show after.svg preview

2) **Debug**
- Show debug.png
- Show key metrics from placement.json in a readable way

3) **Export**
- Download placement.json
- Download before.png, after.png, debug.png if present
- Download after.svg if curved mode was used

## Visual style
- Clear before/after panels
- Halo/stroke around text for readability