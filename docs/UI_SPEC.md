# UI Spec (Streamlit)

## Goals
- Make judging easy: strong visuals + explainability + exports
- Keep UI thin; core logic in app/core

## Pages / Tabs
1) Demo (default)
- Load `assets/problem/Problem_1_river.wkt`
- Input: label text, font size pt, padding pt
- Toggle: Phase B curved (on/off)
- Buttons: Run, Export

2) Debug
- Show safe polygon
- Show candidate points colored by score (Phase A)
- Show chosen label rectangle
- If Phase B: show extracted path segment

3) Export
- Download placement.json
- Download before/after/debug PNG
- Download SVG (if Phase B produced curved placement)

## Visual style
- Clear before/after panels
- Halo/stroke around text for readability
- Optional dark background for contrast