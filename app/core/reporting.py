# app/core/reporting.py
"""
Create reports/<run_name>/ and write placement.json (exact schema), run_metadata.json.
See: docs/PLACEMENT_SCHEMA.md, docs/PROJECT_SPEC.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry.base import BaseGeometry

from app.core.config import (
    ANGLE_OFFSETS_DEG,
    CONTAINMENT_TOLERANCE_PT,
    DEFAULT_FONT_FAMILY,
    K_TOP_CLEARANCE,
    N_SAMPLE_POINTS,
    PADDING_PT,
    REPORTS_DIR,
    SCORE_WEIGHT_ANGLE_PENALTY,
    SCORE_WEIGHT_CENTERING,
    SCORE_WEIGHT_CLEARANCE,
    SCORE_WEIGHT_FIT_MARGIN,
    SEED,
)
from app.core.types import LabelSpec, PlacementResult


def placement_to_dict(result: PlacementResult) -> dict:
    """Exact structure for placement.json. See: docs/PLACEMENT_SCHEMA.md."""
    bbox = [{"x": float(x), "y": float(y)} for x, y in result.bbox_pt]
    out = {
        "label": {
            "text": result.label_text,
            "font_size_pt": result.font_size_pt,
            "font_family": result.font_family,
        },
        "input": {
            "geometry_source": result.geometry_source,
            "units": result.units,
        },
        "result": {
            "mode": result.mode,
            "confidence": result.confidence,
            "anchor_pt": {"x": result.anchor_pt[0], "y": result.anchor_pt[1]},
            "angle_deg": result.angle_deg,
            "bbox_pt": bbox,
            "path_pt": [{"x": float(x), "y": float(y)} for x, y in (result.path_pt or [])],
        },
        "metrics": {
            "min_clearance_pt": result.min_clearance_pt,
            "fit_margin_ratio": result.fit_margin_ratio,
            "curvature_total_deg": result.curvature_total_deg,
            "straightness_ratio": result.straightness_ratio,
        },
        "warnings": result.warnings,
    }
    if result.mode != "phase_b_curved":
        out["result"].pop("path_pt", None)
    return out


def run_metadata_dict(
    run_name: str,
    geometry_path: str,
    label_text: str,
    font_size_pt: float,
    padding_pt: float,
    seed: int | None,
) -> dict:
    """Timestamp and config snapshot for run_metadata.json."""
    return {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "geometry_path": geometry_path,
        "label_text": label_text,
        "font_size_pt": font_size_pt,
        "padding_pt": padding_pt,
        "seed": seed,
        "config": {
            "PADDING_PT": PADDING_PT,
            "N_SAMPLE_POINTS": N_SAMPLE_POINTS,
            "K_TOP_CLEARANCE": K_TOP_CLEARANCE,
            "ANGLE_OFFSETS_DEG": list(ANGLE_OFFSETS_DEG),
            "CONTAINMENT_TOLERANCE_PT": CONTAINMENT_TOLERANCE_PT,
            "SCORE_WEIGHT_CLEARANCE": SCORE_WEIGHT_CLEARANCE,
            "SCORE_WEIGHT_FIT_MARGIN": SCORE_WEIGHT_FIT_MARGIN,
            "SCORE_WEIGHT_CENTERING": SCORE_WEIGHT_CENTERING,
            "SCORE_WEIGHT_ANGLE_PENALTY": SCORE_WEIGHT_ANGLE_PENALTY,
            "DEFAULT_FONT_FAMILY": DEFAULT_FONT_FAMILY,
            "SEED": SEED,
        },
    }


def ensure_report_dir(
    repo_root: Path,
    run_name: str,
    output_dir: str | None = None,
) -> Path:
    """Create output_dir/<run_name>/ under repo_root; return path. Default output_dir from config."""
    base = output_dir if output_dir is not None else REPORTS_DIR
    out = (repo_root / base).resolve() / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_placement_json(report_dir: Path, result: PlacementResult) -> Path:
    """Write placement.json to report_dir. Returns path to file."""
    path = report_dir / "placement.json"
    data = placement_to_dict(result)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_curved_svg_if_applicable(
    report_dir: Path,
    result: PlacementResult,
    polygon: BaseGeometry,
    label_spec: LabelSpec,
) -> Path | None:
    """
    If result is phase_b_curved with path_pt, write after.svg and return path; else return None.
    """
    if result.mode != "phase_b_curved" or not result.path_pt:
        return None
    try:
        from app.core.render_svg import export_curved_svg
        out = report_dir / "after.svg"
        export_curved_svg(polygon, label_spec, result.path_pt, out)
        return out
    except Exception:
        return None


def write_run_metadata_json(
    report_dir: Path,
    run_name: str,
    geometry_path: str,
    label_text: str,
    font_size_pt: float,
    padding_pt: float,
    seed: int | None,
) -> Path:
    """Write run_metadata.json to report_dir."""
    path = report_dir / "run_metadata.json"
    data = run_metadata_dict(run_name, geometry_path, label_text, font_size_pt, padding_pt, seed)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_placements_json(
    report_dir: Path,
    results: list[PlacementResult],
    summary: dict,
    inputs: dict,
) -> Path:
    """Write placements.json for multi-label runs. Schema: labels, results (placement dicts), summary, input."""
    path = report_dir / "placements.json"
    data = {
        "labels": [{"text": r.label_text, "font_size_pt": r.font_size_pt, "font_family": r.font_family} for r in results],
        "results": [placement_to_dict(r) for r in results],
        "summary": summary,
        "input": inputs,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
