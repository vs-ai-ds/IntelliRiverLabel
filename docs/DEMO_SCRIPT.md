# Demo Script (2 minutes)

0:00–0:10
"This tool places a river name inside a river polygon with padding, prioritizing the widest readable area."

0:10–0:30
Show the raw polygon and a baseline (centroid) result that clips or sits in narrow regions.

0:30–0:55
Enable Phase A:
- safe region inset
- candidate points in widest areas
- rectangle fit validation
Show debug overlay and metrics.

0:55–1:25
Enable rotation alignment (dominant direction) and show improved readability.

1:25–1:45
Enable Phase B (curved):
- internal path extraction
- select best segment by clearance/curvature
- export SVG showing text-on-path
If it fails: show automatic fallback to Phase A (robustness).

1:45–2:00
Export JSON + images. Highlight:
- padding guarantee
- explainable scoring
- extensible design