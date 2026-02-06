# app/core/layout.py
"""
Multi-label placement orchestration with collision avoidance.
Orders labels by width (longer first), maintains occupied geometry, places one per label.
See: docs/ALGORITHM.md (collision avoidance).
"""

from __future__ import annotations

from dataclasses import dataclass
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from app.core.config import (
    COLLISION_MAX_AREA,
    COLLISION_WEIGHT,
    LABEL_BUFFER_EXTRA_PT,
    PADDING_PT,
    SEED,
)
from app.core.geometry import bbox_pt_to_polygon
from app.core.placement import run_placement, _placement_to_collision_geom
from app.core.text_metrics import measure_text_pt
from app.core.types import LabelSpec, PlacementResult


@dataclass
class LayoutSummary:
    """Summary of multi-label layout run."""
    success_count: int
    n_labels: int
    collisions_detected: int
    results: list[PlacementResult]


def _order_labels_by_width(labels: list[LabelSpec], font_family: str) -> list[LabelSpec]:
    """Place longer (wider) labels first; tie-break by text."""
    def key(lab: LabelSpec) -> tuple[float, str]:
        w, _ = measure_text_pt(lab.text, lab.font_family or font_family, lab.font_size_pt)
        return (-max(0, w), lab.text)
    return sorted(labels, key=key)


def run_multi_label_layout(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    labels: list[LabelSpec],
    geometry_source: str,
    seed: int | None = SEED,
    padding_pt: float = PADDING_PT,
    allow_phase_b: bool = False,
    use_learned_ranking: bool = False,
    existing_obstacles: list[BaseGeometry] | None = None,
    collision_weight: float = COLLISION_WEIGHT,
    collision_max_area: float = COLLISION_MAX_AREA,
) -> LayoutSummary:
    """
    Place multiple labels with collision avoidance. Longer labels first.
    Returns LayoutSummary with results and success/collision counts.
    """
    from app.core.config import DEFAULT_FONT_FAMILY

    if not labels:
        return LayoutSummary(success_count=0, n_labels=0, collisions_detected=0, results=[])

    ordered = _order_labels_by_width(labels, DEFAULT_FONT_FAMILY)
    obstacles = list(existing_obstacles) if existing_obstacles else []
    results: list[PlacementResult] = []
    collisions_detected = 0

    for label in ordered:
        occupied: BaseGeometry | None = None
        if obstacles:
            occupied = unary_union(obstacles)
        result = run_placement(
            river_geom,
            safe_poly,
            label,
            geometry_source,
            seed=seed,
            allow_phase_b=allow_phase_b,
            padding_pt=padding_pt,
            use_learned_ranking=use_learned_ranking,
            occupied=occupied,
            collision_weight=collision_weight,
            collision_max_area=collision_max_area,
        )
        results.append(result)
        if result.mode != "external_fallback":
            coll_geom = _placement_to_collision_geom(result, buffer_pt=LABEL_BUFFER_EXTRA_PT)
            if not coll_geom.is_empty:
                obstacles.append(coll_geom)
            if occupied is not None and not occupied.is_empty:
                inter = coll_geom.intersection(occupied)
                if not inter.is_empty and inter.area > collision_max_area:
                    collisions_detected += 1

    success_count = sum(1 for r in results if r.mode != "external_fallback")
    return LayoutSummary(
        success_count=success_count,
        n_labels=len(labels),
        collisions_detected=collisions_detected,
        results=results,
    )
