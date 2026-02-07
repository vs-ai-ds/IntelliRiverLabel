# tests/test_smoke_contract.py
"""
Validate PlacementResult serializes to placement schema shape; required keys exist.
Smoke test: run full placement on a small polygon and assert output shape and mode.
Deterministic, no dependency on real river.wkt. See: docs/PLACEMENT_SCHEMA.md.
"""

from __future__ import annotations

import json

import pytest
from shapely.geometry import Polygon

from app.core.preprocess import preprocess_river
from app.core.placement import run_placement
from app.core.reporting import placement_to_dict
from app.core.types import LabelSpec, PlacementResult


def _minimal_placement_result() -> PlacementResult:
    """Minimal valid result for schema contract tests."""
    return PlacementResult(
        label_text="X",
        font_size_pt=12.0,
        font_family="DejaVu Sans",
        geometry_source="docs/assets/problem/Problem_1_river.wkt",
        units="pt",
        mode="phase_a_straight",
        confidence=0.8,
        anchor_pt=(10.0, 20.0),
        angle_deg=0.0,
        bbox_pt=[(0, 0), (5, 0), (5, 3), (0, 3)],
        path_pt=None,
        min_clearance_pt=1.0,
        fit_margin_ratio=0.5,
        curvature_total_deg=0.0,
        straightness_ratio=1.0,
        warnings=[],
    )


REQUIRED_KEYS = [
    "schema_version",
    ("label", "text"),
    ("label", "font_size_pt"),
    ("label", "font_family"),
    ("input", "geometry_source"),
    ("input", "units"),
    ("result", "mode"),
    ("result", "confidence"),
    ("result", "anchor_pt"),
    ("result", "angle_deg"),
    ("result", "bbox_pt"),
    ("metrics", "min_clearance_pt"),
    ("metrics", "fit_margin_ratio"),
    ("metrics", "curvature_total_deg"),
    ("metrics", "straightness_ratio"),
    "warnings",
]


def test_placement_result_serializes_to_schema_shape() -> None:
    result = _minimal_placement_result()
    data = placement_to_dict(result)
    assert isinstance(data, dict)
    assert "label" in data and isinstance(data["label"], dict)
    assert "input" in data and isinstance(data["input"], dict)
    assert "result" in data and isinstance(data["result"], dict)
    assert "metrics" in data and isinstance(data["metrics"], dict)
    assert "warnings" in data and isinstance(data["warnings"], list)


def test_placement_schema_required_keys_exist() -> None:
    result = _minimal_placement_result()
    data = placement_to_dict(result)
    for key in REQUIRED_KEYS:
        if isinstance(key, tuple):
            obj = data
            for k in key:
                assert k in obj, f"Missing key: {key}"
                obj = obj[k]
        else:
            assert key in data, f"Missing key: {key}"


def test_placement_json_roundtrip() -> None:
    result = _minimal_placement_result()
    data = placement_to_dict(result)
    s = json.dumps(data)
    loaded = json.loads(s)
    assert loaded["label"]["text"] == result.label_text
    assert loaded["result"]["mode"] == result.mode
    assert loaded["metrics"]["min_clearance_pt"] == result.min_clearance_pt


ALLOWED_MODES = ("phase_a_straight", "phase_b_curved", "external_fallback")


def test_placement_smoke_run_produces_valid_result() -> None:
    """Run full placement on a small polygon; assert result has valid mode and schema."""
    # Simple elongated polygon (river-like): 80pt x 25pt
    coords = [(0, 0), (80, 0), (80, 25), (0, 25)]
    poly = Polygon(coords)
    river_geom, safe_poly = preprocess_river(poly, padding_pt=3.0)
    assert not safe_poly.is_empty
    label = LabelSpec(text="X", font_family="DejaVu Sans", font_size_pt=10.0)
    result = run_placement(
        river_geom,
        safe_poly,
        label,
        geometry_source="smoke_test",
        seed=42,
        allow_phase_b=False,
        padding_pt=3.0,
    )
    assert result is not None
    assert result.mode in ALLOWED_MODES
    data = placement_to_dict(result)
    assert data.get("schema_version") == "1.0"
    assert data["result"]["mode"] == result.mode
