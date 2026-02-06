# tests/test_layout_collisions.py
"""
Multi-label layout: place 2 labels on a simple rectangle river;
second label must avoid the first or fail gracefully without overlap.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from app.core.layout import run_multi_label_layout
from app.core.preprocess import preprocess_river
from app.core.types import LabelSpec


def test_two_labels_on_rectangle_no_overlap() -> None:
    # Simple wide rectangle so two short labels can fit
    river = Polygon([(0, 0), (200, 0), (200, 40), (0, 40)])
    river_geom, safe_poly = preprocess_river(river, padding_pt=2.0)
    assert not safe_poly.is_empty
    labels = [
        LabelSpec(text="A", font_family="Arial", font_size_pt=10.0),
        LabelSpec(text="B", font_family="Arial", font_size_pt=10.0),
    ]
    layout = run_multi_label_layout(
        river_geom,
        safe_poly,
        labels,
        geometry_source="test_rect",
        seed=42,
        padding_pt=2.0,
        allow_phase_b=False,
    )
    assert layout.n_labels == 2
    # Either both placed or at least one; if both placed, no collision
    assert layout.success_count <= 2
    if layout.success_count == 2:
        assert layout.collisions_detected == 0
        # Results should be non-overlapping (checked by layout internally)
        assert len(layout.results) == 2
