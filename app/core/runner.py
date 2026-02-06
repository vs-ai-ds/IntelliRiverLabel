# app/core/runner.py
"""
CLI entrypoint: load WKT, preprocess, run Phase A placement, render, export.
Default geometry: docs/assets/problem/Problem_1_river.wkt (repo-relative).
See: docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import (
    DEFAULT_GEOMETRY_PATH,
    DEFAULT_FONT_FAMILY,
    PADDING_PT,
    SEED,
)
from app.core.candidates_a import generate_candidate_points
from app.core.io import load_and_validate_river
from app.core.placement import run_placement
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
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")
    p.add_argument("--geometry", type=str, default=DEFAULT_GEOMETRY_PATH, help="River WKT path (repo-relative)")
    p.add_argument("--output-dir", type=str, default="reports", dest="output_dir", help="Output directory (repo-relative)")
    p.add_argument("--repo-root", type=str, default=None, dest="repo_root", help="Repo root (default: cwd)")
    p.add_argument("--curved", action="store_true", dest="curved", help="Enable Phase B curved label placement")
    p.add_argument("--batch-dir", type=str, default=None, dest="batch_dir", help="Batch mode: directory of .wkt files")
    p.add_argument("--batch-manifest", type=str, default=None, dest="batch_manifest", help="Batch mode: CSV/JSON manifest path")
    p.add_argument("--labels", type=str, default="ELBE", help="Label(s): 'ELBE' or 'ELBE,MAIN'")
    p.add_argument("--batch-limit", type=int, default=None, dest="batch_limit", help="Max cases in batch")
    p.add_argument("--batch-output-dir", type=str, default="reports", dest="batch_output_dir", help="Batch output parent dir")
    p.add_argument("--zoom-buckets", type=str, default="", dest="zoom_buckets", help="Zoom buckets e.g. '10,12,14'")
    return p.parse_args()


def _resolve_geometry_path(repo_root: Path, path_arg: str) -> Path:
    """Resolve path: if relative, from repo root; else as-is then resolve."""
    p = Path(path_arg)
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()

    if args.batch_dir or args.batch_manifest:
        from app.core.batch import run_batch
        batch_dir = Path(args.batch_dir) if args.batch_dir else None
        manifest = Path(args.batch_manifest).resolve() if args.batch_manifest else None
        if batch_dir is not None and not batch_dir.is_absolute():
            batch_dir = repo_root / batch_dir
        out = run_batch(
            run_name=args.run_name,
            batch_dir=batch_dir,
            manifest_path=manifest,
            labels_text=args.labels,
            font_size_pt=args.font_size_pt,
            limit=args.batch_limit,
            repo_root=repo_root,
            padding_pt=args.padding_pt,
            seed=args.seed,
            allow_phase_b=args.curved,
        )
        print(out / "index.csv")
        return

    geom_path = _resolve_geometry_path(repo_root, args.geometry)
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
    result = run_placement(
        river_geom,
        safe_poly,
        label,
        geometry_source=args.geometry,
        seed=args.seed,
        allow_phase_b=args.curved,
        padding_pt=args.padding_pt,
    )

    report_dir = ensure_report_dir(repo_root, args.run_name, output_dir=args.output_dir)
    placement_path = write_placement_json(report_dir, result)
    svg_path = None
    if result.mode == "phase_b_curved" and result.path_pt:
        try:
            from app.core.render_svg import export_curved_svg
            svg_path = report_dir / "after.svg"
            export_curved_svg(river_geom, label, result.path_pt, svg_path)
        except Exception:
            svg_path = None
    write_run_metadata_json(
        report_dir,
        args.run_name,
        args.geometry,
        args.text,
        args.font_size_pt,
        args.padding_pt,
        args.seed,
    )

    before_path = report_dir / "before.png"
    after_path = report_dir / "after.png"
    debug_path = report_dir / "debug.png"
    render_before(river_geom, before_path)
    render_after(river_geom, result, after_path)
    candidates = generate_candidate_points(safe_poly, seed=args.seed)
    render_debug(river_geom, safe_poly, candidates, result, debug_path)

    for p in (placement_path, before_path, after_path, debug_path):
        print(p)
    if svg_path is not None:
        print(svg_path)
    print("Mode used:", result.mode)


if __name__ == "__main__":
    main()
