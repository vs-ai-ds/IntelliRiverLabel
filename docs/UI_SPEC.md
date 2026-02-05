# UI Spec (Streamlit)

## Goals
- Make judging easy: strong visuals + explainability + exports
- Keep UI thin; core logic in app/core

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