# app/core/batch.py
"""
Multi-river batch mode: run placement on a directory of .wkt files or a manifest.
Output: reports/batch_<run_name>/index.csv and cases/<case_id>/ with placement(s), images.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from app.core.config import DEFAULT_FONT_FAMILY, PADDING_PT, REPORTS_DIR, SEED
from app.core.io import load_and_validate_river, parse_wkt, validate_geometry
from app.core.layout import run_multi_label_layout
from app.core.preprocess import preprocess_river
from app.core.render import render_before, render_after, render_debug
from app.core.reporting import ensure_report_dir, write_placement_json, write_placements_json, write_run_metadata_json
from app.core.types import LabelSpec
from app.core.candidates_a import generate_candidate_points


def _load_labels(text: str, font_size_pt: float = 12.0) -> list[LabelSpec]:
    """Parse labels from 'ELBE' or 'ELBE,MAIN' or JSON list."""
    text = (text or "").strip()
    if not text:
        return [LabelSpec(text="Label", font_family=DEFAULT_FONT_FAMILY, font_size_pt=font_size_pt)]
    if text.startswith("["):
        try:
            arr = json.loads(text)
            return [LabelSpec(text=str(t), font_family=DEFAULT_FONT_FAMILY, font_size_pt=font_size_pt) for t in arr]
        except Exception:
            pass
    return [
        LabelSpec(text=t.strip(), font_family=DEFAULT_FONT_FAMILY, font_size_pt=font_size_pt)
        for t in text.split(",") if t.strip()
    ]


def _batch_from_dir(
    wkt_dir: Path,
    labels: list[LabelSpec],
    run_name: str,
    repo_root: Path,
    limit: int | None = None,
    padding_pt: float = PADDING_PT,
    seed: int | None = SEED,
    allow_phase_b: bool = False,
) -> Path:
    """Run batch on all .wkt files in wkt_dir. Returns path to batch report dir."""
    batch_dir = ensure_report_dir(repo_root, f"batch_{run_name}", output_dir=REPORTS_DIR)
    cases_dir = batch_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    wkt_files = sorted(wkt_dir.glob("*.wkt"))[: (limit or 999)]
    rows: list[dict] = []
    for i, wkt_path in enumerate(wkt_files):
        case_id = f"case_{i:04d}_{wkt_path.stem}"
        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        geom_source = str(wkt_path.relative_to(repo_root)) if repo_root in wkt_path.parents else str(wkt_path)
        t0 = time.perf_counter()
        try:
            geom = load_and_validate_river(wkt_path, repo_root=None)
        except Exception:
            rows.append({
                "case_id": case_id, "geometry_source": geom_source, "labels": ",".join(l.text for l in labels),
                "mode_used": "error", "n_labels": len(labels), "success_count": 0,
                "mean_min_clearance": "", "mean_fit_margin": "", "collisions_detected": 0,
                "duration_ms": int((time.perf_counter() - t0) * 1000), "warnings_count": 0,
            })
            continue
        river_geom, safe_poly = preprocess_river(geom, padding_pt=padding_pt)
        if safe_poly.is_empty:
            rows.append({
                "case_id": case_id, "geometry_source": geom_source, "labels": ",".join(l.text for l in labels),
                "mode_used": "empty_safe", "n_labels": len(labels), "success_count": 0,
                "mean_min_clearance": "", "mean_fit_margin": "", "collisions_detected": 0,
                "duration_ms": int((time.perf_counter() - t0) * 1000), "warnings_count": 0,
            })
            continue
        layout = run_multi_label_layout(
            river_geom, safe_poly, labels, geom_source,
            seed=seed, padding_pt=padding_pt, allow_phase_b=allow_phase_b,
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if len(layout.results) == 1:
            write_placement_json(case_dir, layout.results[0])
            mode_used = layout.results[0].mode
        else:
            write_placements_json(
                case_dir,
                layout.results,
                {"success_count": layout.success_count, "collisions_detected": layout.collisions_detected},
                {"geometry_source": geom_source, "n_labels": len(labels)},
            )
            mode_used = layout.results[0].mode if layout.results else "error"
        mean_cl = sum(r.min_clearance_pt for r in layout.results) / len(layout.results) if layout.results else 0
        mean_fit = sum(r.fit_margin_ratio for r in layout.results) / len(layout.results) if layout.results else 0
        render_before(river_geom, case_dir / "before.png")
        for r in layout.results:
            render_after(river_geom, r, case_dir / "after.png")
            break
        candidates = generate_candidate_points(safe_poly, seed=seed)
        render_debug(river_geom, safe_poly, candidates, layout.results[0], case_dir / "debug.png")
        write_run_metadata_json(case_dir, run_name, geom_source, ",".join(l.text for l in labels), labels[0].font_size_pt, padding_pt, seed)
        rows.append({
            "case_id": case_id, "geometry_source": geom_source, "labels": ",".join(l.text for l in labels),
            "mode_used": mode_used, "n_labels": len(labels), "success_count": layout.success_count,
            "mean_min_clearance": round(mean_cl, 2), "mean_fit_margin": round(mean_fit, 2),
            "collisions_detected": layout.collisions_detected,
            "duration_ms": duration_ms, "warnings_count": sum(len(r.warnings) for r in layout.results),
        })
    index_path = batch_dir / "index.csv"
    if rows:
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return batch_dir


def run_batch(
    run_name: str,
    batch_dir: Path | None = None,
    manifest_path: Path | None = None,
    labels_text: str = "ELBE",
    font_size_pt: float = 12.0,
    limit: int | None = None,
    repo_root: Path | None = None,
    padding_pt: float = PADDING_PT,
    seed: int | None = SEED,
    allow_phase_b: bool = False,
) -> Path:
    """
    Run batch: from batch_dir (directory of .wkt) or manifest (CSV/JSON with paths).
    Returns report directory containing index.csv and cases/<case_id>/.
    """
    root = repo_root or Path.cwd().resolve()
    labels = _load_labels(labels_text, font_size_pt)
    if batch_dir is not None and batch_dir.is_dir():
        return _batch_from_dir(batch_dir, labels, run_name, root, limit=limit, padding_pt=padding_pt, seed=seed, allow_phase_b=allow_phase_b)
    if manifest_path is not None and manifest_path.exists():
        # Minimal manifest: CSV with path,label or path,labels
        batch_dir = (root / REPORTS_DIR / f"batch_{run_name}").resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)
        cases_dir = batch_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                if limit and i >= limit:
                    break
                path_key = "path" if "path" in row else "geometry"
                path_val = row.get(path_key, row.get("file", ""))
                if not path_val:
                    continue
                p = root / path_val.strip() if not Path(path_val).is_absolute() else Path(path_val)
                if not p.exists():
                    p = batch_dir.parent / path_val.strip()
                lab_text = row.get("labels", row.get("label", labels_text))
                case_labels = _load_labels(lab_text, font_size_pt)
                case_id = f"case_{i:04d}"
                case_dir = cases_dir / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                try:
                    geom = load_and_validate_river(p, repo_root=None)
                except Exception:
                    rows.append({"case_id": case_id, "geometry_source": path_val, "labels": lab_text, "mode_used": "error", "n_labels": len(case_labels), "success_count": 0, "mean_min_clearance": "", "mean_fit_margin": "", "collisions_detected": 0, "duration_ms": int((time.perf_counter() - t0) * 1000), "warnings_count": 0})
                    continue
                river_geom, safe_poly = preprocess_river(geom, padding_pt=padding_pt)
                if safe_poly.is_empty:
                    rows.append({"case_id": case_id, "geometry_source": path_val, "labels": lab_text, "mode_used": "empty_safe", "n_labels": len(case_labels), "success_count": 0, "mean_min_clearance": "", "mean_fit_margin": "", "collisions_detected": 0, "duration_ms": int((time.perf_counter() - t0) * 1000), "warnings_count": 0})
                    continue
                layout = run_multi_label_layout(river_geom, safe_poly, case_labels, path_val, seed=seed, padding_pt=padding_pt, allow_phase_b=allow_phase_b)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                if len(layout.results) == 1:
                    write_placement_json(case_dir, layout.results[0])
                else:
                    write_placements_json(case_dir, layout.results, {"success_count": layout.success_count, "collisions_detected": layout.collisions_detected}, {"geometry_source": path_val, "n_labels": len(case_labels)})
                mean_cl = sum(r.min_clearance_pt for r in layout.results) / len(layout.results) if layout.results else 0
                mean_fit = sum(r.fit_margin_ratio for r in layout.results) / len(layout.results) if layout.results else 0
                render_before(river_geom, case_dir / "before.png")
                if layout.results:
                    render_after(river_geom, layout.results[0], case_dir / "after.png")
                    candidates = generate_candidate_points(safe_poly, seed=seed)
                    render_debug(river_geom, safe_poly, candidates, layout.results[0], case_dir / "debug.png")
                write_run_metadata_json(case_dir, run_name, path_val, lab_text, font_size_pt, padding_pt, seed)
                rows.append({"case_id": case_id, "geometry_source": path_val, "labels": lab_text, "mode_used": layout.results[0].mode if layout.results else "error", "n_labels": len(case_labels), "success_count": layout.success_count, "mean_min_clearance": round(mean_cl, 2), "mean_fit_margin": round(mean_fit, 2), "collisions_detected": layout.collisions_detected, "duration_ms": duration_ms, "warnings_count": sum(len(r.warnings) for r in layout.results)})
        index_path = batch_dir / "index.csv"
        if rows:
            with open(index_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        return batch_dir
    raise ValueError("Provide batch_dir or manifest_path")
