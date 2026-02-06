# app/core/evaluate.py
"""
Evaluation runner: compare centroid baseline, heuristic-only, heuristic+model.
Saves evaluation_results.csv and evaluation_summary.json under reports/<run_name>/.
See: docs/EVALUATION.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry.base import BaseGeometry

from app.core.config import (
    DEFAULT_GEOMETRY_PATH,
    DEFAULT_FONT_FAMILY,
    PADDING_PT,
    SEED,
)
from app.core.io import load_and_validate_river
from app.core.placement import run_placement_phase_a
from app.core.preprocess import preprocess_river
from app.core.types import LabelSpec
from app.core.reporting import ensure_report_dir


def _centroid_placement(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
) -> dict:
    """Baseline: place at centroid, no rotation. Returns metrics dict."""
    from app.core.geometry import oriented_rectangle
    from app.core.text_metrics import measure_text_pt
    from app.core.validate import validate_rect_inside_safe

    w_pt, h_pt = measure_text_pt(label.text, label.font_family, label.font_size_pt)
    if w_pt <= 0:
        w_pt = 40.0
    if h_pt <= 0:
        h_pt = 12.0
    c = river_geom.centroid
    cx, cy = float(c.x), float(c.y)
    rect = oriented_rectangle(cx, cy, w_pt, h_pt, 0.0)
    ok, min_cl = validate_rect_inside_safe(safe_poly, rect)
    fit = min_cl / PADDING_PT if PADDING_PT > 0 else 0.0
    return {
        "success": ok,
        "min_clearance_pt": min_cl,
        "fit_margin_ratio": fit,
        "mode_used": "baseline_centroid",
    }


def _run_one(
    river_geom: BaseGeometry,
    safe_poly: BaseGeometry,
    label: LabelSpec,
    geometry_source: str,
    seed: int | None,
    use_learned: bool,
) -> dict:
    """Run Phase A and return metrics dict."""
    try:
        result = run_placement_phase_a(
            river_geom,
            safe_poly,
            label,
            geometry_source,
            seed=seed,
            use_learned_ranking=use_learned,
        )
        return {
            "success": result.mode != "external_fallback",
            "min_clearance_pt": result.min_clearance_pt,
            "fit_margin_ratio": result.fit_margin_ratio,
            "mode_used": result.mode,
        }
    except Exception:
        return {
            "success": False,
            "min_clearance_pt": 0.0,
            "fit_margin_ratio": 0.0,
            "mode_used": "error",
        }


EVAL_FAMILIES = ("straight_wide", "straight_narrow", "curved_wide", "curved_narrow", "braided")


def _synthetic_batch(n: int, seed: int | None) -> list[tuple[BaseGeometry, str, str]]:
    """Generate n synthetic river polygons. Returns list of (geom, source_name, family)."""
    from app.models.train import _synthetic_polygon
    out = []
    for i in range(n):
        poly = _synthetic_polygon(seed=seed + i if seed is not None else i)
        if poly is not None and not poly.is_empty:
            family = EVAL_FAMILIES[i % len(EVAL_FAMILIES)]
            out.append((poly, f"synthetic_{i}", family))
    return out


def run_evaluation(
    run_name: str = "eval_01",
    geometry_path: str | None = None,
    n_synthetic: int = 20,
    seed: int | None = SEED,
    repo_root: Path | None = None,
) -> Path:
    """
    Run baselines and our modes on provided WKT + synthetic batch.
    Writes evaluation_results.csv and evaluation_summary.json to reports/<run_name>/.
    Returns report_dir.
    """
    root = repo_root or Path.cwd().resolve()
    report_dir = ensure_report_dir(root, run_name)
    label = LabelSpec(text="ELBE", font_family=DEFAULT_FONT_FAMILY, font_size_pt=12.0)

    rows: list[dict] = []
    cases: list[tuple[BaseGeometry, str]] = []

    if geometry_path:
        geom_path = root / geometry_path
        if geom_path.exists():
            try:
                river = load_and_validate_river(geom_path, repo_root=None)
                cases.append((river, geometry_path, "default"))
            except Exception:
                pass

    for poly, name, family in _synthetic_batch(n_synthetic, seed):
        cases.append((poly, name, family))

    for river_geom, source, family in cases:
        try:
            _, safe_poly = preprocess_river(river_geom, padding_pt=PADDING_PT)
        except Exception:
            continue
        if safe_poly.is_empty:
            continue

        def _row(method: str, metrics: dict) -> dict:
            return {"source": source, "method": method, "family": family, "n_labels": 1, "collisions": 0, "duration_ms": 0, **metrics}

        base_metrics = _centroid_placement(river_geom, safe_poly, label)
        rows.append(_row("baseline_centroid", base_metrics))

        h_metrics = _run_one(river_geom, safe_poly, label, source, seed, use_learned=False)
        rows.append(_row("heuristic_only", h_metrics))

        try:
            from app.models.registry import load_model
            if load_model() is not None:
                m_metrics = _run_one(river_geom, safe_poly, label, source, seed, use_learned=True)
                rows.append(_row("heuristic_plus_model", m_metrics))
        except Exception:
            pass

    # Summary
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        m = r["method"]
        if m not in by_method:
            by_method[m] = []
        by_method[m].append(r)

    summary = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "success_rate": {},
        "mean_min_clearance_pt": {},
        "mean_fit_margin_ratio": {},
    }
    for method, list_r in by_method.items():
        n = len(list_r)
        summary["success_rate"][method] = sum(1 for r in list_r if r["success"]) / n if n else 0.0
        summary["mean_min_clearance_pt"][method] = sum(r["min_clearance_pt"] for r in list_r) / n if n else 0.0
        summary["mean_fit_margin_ratio"][method] = sum(r["fit_margin_ratio"] for r in list_r) / n if n else 0.0

    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    import csv
    if rows:
        keys = ["source", "method", "family", "n_labels", "collisions", "duration_ms", "success", "min_clearance_pt", "fit_margin_ratio", "mode_used"]
        with open(report_dir / "evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    # Leaderboard: aggregate per method and per family
    leaderboard: dict = {"by_method": {}, "by_family": {}}
    for method, list_r in by_method.items():
        n = len(list_r)
        leaderboard["by_method"][method] = {
            "success_rate": sum(1 for r in list_r if r.get("success")) / n if n else 0,
            "avg_clearance": sum(r.get("min_clearance_pt", 0) for r in list_r) / n if n else 0,
            "avg_fit_margin": sum(r.get("fit_margin_ratio", 0) for r in list_r) / n if n else 0,
            "collision_rate": 0,
            "avg_duration_ms": 0,
        }
    for r in rows:
        fam = r.get("family", "default")
        if fam not in leaderboard["by_family"]:
            leaderboard["by_family"][fam] = {"success_count": 0, "n": 0}
        leaderboard["by_family"][fam]["n"] += 1
        if r.get("success"):
            leaderboard["by_family"][fam]["success_count"] += 1
    for fam, v in leaderboard["by_family"].items():
        v["success_rate"] = v["success_count"] / v["n"] if v["n"] else 0
    (report_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    with open(report_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "success_rate", "avg_clearance", "avg_fit_margin", "collision_rate", "avg_duration_ms"])
        for method, v in leaderboard["by_method"].items():
            w.writerow([method, v["success_rate"], v["avg_clearance"], v["avg_fit_margin"], v["collision_rate"], v["avg_duration_ms"]])

    try:
        from app.core.plots import generate_all_plots
        generate_all_plots(report_dir)
    except Exception:
        pass

    return report_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Run evaluation: baselines vs heuristic vs heuristic+model.")
    p.add_argument("--run-name", default="eval_01", dest="run_name", help="Reports subdir name")
    p.add_argument("--geometry", default=DEFAULT_GEOMETRY_PATH, help="River WKT path (repo-relative)")
    p.add_argument("--n-synthetic", type=int, default=20, dest="n_synthetic", help="Number of synthetic polygons")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--repo-root", default=None, dest="repo_root")
    args = p.parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    report_dir = run_evaluation(
        run_name=args.run_name,
        geometry_path=args.geometry,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        repo_root=root,
    )
    print(report_dir)


if __name__ == "__main__":
    main()
