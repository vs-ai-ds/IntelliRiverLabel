# tests/test_smoke_contract.py
"""
Validate PlacementResult serializes to placement schema shape; required keys exist.
Deterministic, no dependency on real river.wkt. See: docs/PLACEMENT_SCHEMA.md.
"""

from __future__ import annotations

import json

import pytest

from app.core.reporting import placement_to_dict
from app.core.types import PlacementResult


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
