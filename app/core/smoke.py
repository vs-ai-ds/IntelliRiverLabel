# app/core/smoke.py
"""
Single entrypoint to verify Phase A end-to-end: placement + rendering for default inputs.
Does not run on import. See: docs/PROJECT_SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import (
    DEFAULT_FONT_FAMILY,
    DEFAULT_GEOMETRY_PATH,
    PADDING_PT,
    SEED,
)
from app.core.candidates_a import generate_candidate_points
from app.core.io import load_and_validate_river
from app.core.placement import run_placement_phase_a
from app.core.preprocess import preprocess_river
from app.core.render import render_before, render_after, render_debug
from app.core.reporting import (
    ensure_report_dir,
    write_placement_json,
    write_run_metadata_json,
)
from app.core.types import LabelSpec


def main() -> None:
    """Run Phase A placement and rendering with default inputs and run_name='smoke'."""
    repo_root = Path.cwd().resolve()
    geom_path = (repo_root / DEFAULT_GEOMETRY_PATH).resolve()
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")

    river_geom = load_and_validate_river(geom_path, repo_root=None)
    river_geom, safe_poly = preprocess_river(
        river_geom,
        simplify_tol_pt=0.0,
        padding_pt=PADDING_PT,
    )
    if safe_poly.is_empty:
        raise ValueError("Safe polygon is empty.")

    label = LabelSpec(text="ELBE", font_family=DEFAULT_FONT_FAMILY, font_size_pt=12.0)
    result = run_placement_phase_a(
        river_geom,
        safe_poly,
        label,
        geometry_source=DEFAULT_GEOMETRY_PATH,
        seed=SEED,
    )

    report_dir = ensure_report_dir(repo_root, "smoke")
    write_placement_json(report_dir, result)
    write_run_metadata_json(
        report_dir,
        "smoke",
        DEFAULT_GEOMETRY_PATH,
        label.text,
        label.font_size_pt,
        PADDING_PT,
        SEED,
    )
    render_before(river_geom, report_dir / "before.png")
    render_after(river_geom, result, report_dir / "after.png")
    candidates = generate_candidate_points(safe_poly, seed=SEED)
    render_debug(river_geom, safe_poly, candidates, result, report_dir / "debug.png")


if __name__ == "__main__":
    main()
