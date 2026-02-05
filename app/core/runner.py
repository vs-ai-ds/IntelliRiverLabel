# app/core/runner.py
"""
CLI entrypoint: load WKT, preprocess, run Phase A placement, render, export.
Default geometry: docs/assets/problem/Problem_1_river.wkt (config).
See: docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import (
    DEFAULT_GEOMETRY_PATH,
    DEFAULT_FONT_FAMILY,
    PADDING_PT,
    REPORTS_DIR,
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A river label placement.")
    p.add_argument("--text", type=str, default="ELBE", help="Label text")
    p.add_argument("--font-size-pt", type=float, default=12.0, dest="font_size_pt", help="Font size (pt)")
    p.add_argument("--padding-pt", type=float, default=PADDING_PT, dest="padding_pt", help="Padding (pt)")
    p.add_argument("--run-name", type=str, default="run", dest="run_name", help="Reports subdir name")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed (None for non-deterministic)")
    p.add_argument("--geometry", type=str, default=DEFAULT_GEOMETRY_PATH, help="Path to river WKT (repo-relative)")
    p.add_argument("--repo-root", type=str, default=None, dest="repo_root", help="Repo root (default: cwd)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()

    geom_path = repo_root / args.geometry
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")

    river_geom = load_and_validate_river(geom_path, repo_root=None)
    river_geom, safe_poly = preprocess_river(
        river_geom,
        simplify_tol_pt=0.0,
        padding_pt=args.padding_pt,
    )

    if safe_poly.is_empty:
        raise ValueError("Safe polygon is empty; cannot place label internally.")

    label = LabelSpec(text=args.text, font_family=DEFAULT_FONT_FAMILY, font_size_pt=args.font_size_pt)
    result = run_placement_phase_a(
        river_geom,
        safe_poly,
        label,
        geometry_source=args.geometry,
        seed=args.seed,
    )

    report_dir = ensure_report_dir(repo_root, args.run_name)
    write_placement_json(report_dir, result)
    write_run_metadata_json(
        report_dir,
        args.run_name,
        args.geometry,
        args.text,
        args.font_size_pt,
        args.padding_pt,
        args.seed,
    )

    render_before(river_geom, report_dir / "before.png")
    render_after(river_geom, result, report_dir / "after.png")
    candidates = generate_candidate_points(safe_poly, seed=args.seed)
    render_debug(river_geom, safe_poly, candidates, result, report_dir / "debug.png")


if __name__ == "__main__":
    main()
