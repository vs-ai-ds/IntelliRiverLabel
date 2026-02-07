# app/ui/run_registry.py
"""
Run registry: scan reports/ and normalize run folders into RunRecords.
Single source of truth for "what runs exist". No tab rescans reports/ independently.
See: docs/UI_SPEC.md, docs/ARCHITECTURE.md.

UI + State contract
------------------
Run types (run_type / active_ref.kind):
  single       — one folder with placement.json (demo/CLI run).
  batch_root   — reports/batch_<name>/; display resolves to first case with placement.
  batch_case   — reports/batch_<name>/cases/<id>/; one case with placement.
  eval         — reports/eval_<name>/; leaderboards/plots only; NOT openable in Demo/Debug/Export.

active_ref (single source of truth for "current run"):
  Format: { "kind": "single"|"batch_root"|"batch_case"|"eval", "path": str, "case_id": str|None }
  Who SETS active_ref: All Results (Open in Demo), Sidebar (Run demo). Run Again and Switch to run removed from sidebar.
  Who READS active_ref: Demo, Debug, Export — they render ONLY from resolve_display_path(active_ref).
  Eval runs: do not set active_ref for viewing; use last_eval_dir / selected_eval_dir for Evaluate tab.

What opens where:
  Placement runs (single, batch_case): Demo, Debug, Export show them. All Results "Open in Demo" sets active_ref.
  Eval runs: Evaluate tab only. All Results "Open in Evaluate tab" sets last_eval_dir (and selected_eval_dir).
  Compare: only placement runs (single + batch_case); no eval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RunType = Literal["single", "batch_root", "batch_case", "eval"]
ActiveRefKind = Literal["single", "batch_root", "batch_case", "eval"]


@dataclass
class RunRecord:
    """Normalized view of a run folder (single, batch root, batch case, or eval)."""
    run_id: str
    run_type: RunType
    display_name: str
    path: Path
    has_placement: bool
    has_images: bool
    thumbnail_path: Path | None
    metadata: dict
    metrics_summary: dict
    mtime: float


def _has_placement_at(path: Path) -> bool:
    return (path / "placement.json").exists() or (path / "placements.json").exists()


def _read_metrics(report_dir: Path) -> dict:
    """Read metrics from placement.json or placements.json. Backward-compatible."""
    pp = report_dir / "placements.json"
    if pp.exists():
        try:
            data = json.loads(pp.read_text(encoding="utf-8"))
            results = data.get("results", [])
            summary = data.get("summary", {})
            n = len(results)
            if not n:
                return {"success_count": 0, "n_labels": summary.get("n_labels", 0), "collisions_detected": summary.get("collisions_detected", 0)}
            mean_cl = sum(r.get("min_clearance_pt", 0) for r in results) / n
            mean_fit = sum(r.get("fit_margin_ratio", 0) for r in results) / n
            return {
                "min_clearance_pt": round(mean_cl, 2),
                "fit_margin_ratio": round(mean_fit, 2),
                "mode": results[0].get("mode") if results else "",
                "success_count": summary.get("success_count", n),
                "n_labels": summary.get("n_labels", n),
                "collisions_detected": summary.get("collisions_detected", 0),
            }
        except Exception:
            pass
    p = report_dir / "placement.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        r = data.get("result", {})
        m = data.get("metrics", {})
        return {
            "min_clearance_pt": m.get("min_clearance_pt"),
            "fit_margin_ratio": m.get("fit_margin_ratio"),
            "mode": r.get("mode"),
            "confidence": r.get("confidence"),
            "curvature_total_deg": m.get("curvature_total_deg"),
            "straightness_ratio": m.get("straightness_ratio"),
        }
    except Exception:
        return {}


def _read_metadata(report_dir: Path) -> dict:
    """Read run_metadata.json if present."""
    p = report_dir / "run_metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _thumbnail_for(report_dir: Path, run_type: RunType) -> Path | None:
    p = report_dir / "after.png"
    if p.exists():
        return p
    p = report_dir / "collage.png"
    if p.exists():
        return p
    cases = report_dir / "cases"
    if cases.is_dir():
        for sub in sorted(cases.iterdir()):
            if sub.is_dir():
                q = sub / "after.png"
                if q.exists():
                    return q
                break
    plots = report_dir / "plots" / "success_rate.png"
    if plots.exists():
        return plots
    return None


def _run_type_from_name(name: str) -> RunType:
    if name.startswith("batch_"):
        return "batch_root"
    if name.startswith("eval_"):
        return "eval"
    return "single"


def get_record_by_path(path: Path, repo_root: Path, reports_dir_name: str) -> RunRecord | None:
    """Build one RunRecord for the given path (single, batch case, batch root, or eval)."""
    if not path.exists() or not path.is_dir():
        return None
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    # Detect if this is inside a batch cases/ folder
    parts = path.parts
    if "cases" in parts:
        idx = parts.index("cases")
        if idx > 0 and idx < len(parts) - 1:
            batch_name = parts[idx - 1]
            case_id = parts[idx + 1]
            if batch_name.startswith("batch_") and case_id:
                has_pl = _has_placement_at(path)
                return RunRecord(
                    run_id=str(path),
                    run_type="batch_case",
                    display_name=f"{batch_name} / {case_id}",
                    path=path,
                    has_placement=has_pl,
                    has_images=(path / "after.png").exists() or (path / "before.png").exists(),
                    thumbnail_path=path / "after.png" if (path / "after.png").exists() else None,
                    metadata=_read_metadata(path),
                    metrics_summary=_read_metrics(path) if has_pl else {},
                    mtime=mtime,
                )
    name = path.name
    run_type = _run_type_from_name(name)
    has_pl = _has_placement_at(path)
    has_im = (path / "after.png").exists() or (path / "before.png").exists()
    thumb = _thumbnail_for(path, run_type)
    meta = _read_metadata(path)
    if not meta and run_type == "batch_root" and (path / "cases").is_dir():
        for sub in sorted((path / "cases").iterdir()):
            if sub.is_dir():
                meta = _read_metadata(sub)
                break
    return RunRecord(
        run_id=str(path),
        run_type=run_type,
        display_name=name,
        path=path,
        has_placement=has_pl,
        has_images=has_im,
        thumbnail_path=thumb,
        metadata=meta,
        metrics_summary=_read_metrics(path) if has_pl else {},
        mtime=mtime,
    )


def scan_reports(repo_root: Path, reports_dir_name: str) -> list[RunRecord]:
    """
    Scan reports/ and return normalized RunRecords (top-level single/batch_root/eval,
    plus one record per batch case with placement). Sorted by mtime newest first.
    """
    root = repo_root / reports_dir_name
    if not root.exists():
        return []
    out: list[RunRecord] = []
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        name = d.name
        run_type = _run_type_from_name(name)
        if run_type == "batch_root":
            rec = get_record_by_path(d, repo_root, reports_dir_name)
            if rec:
                out.append(rec)
            cases_dir = d / "cases"
            if cases_dir.is_dir():
                for sub in sorted(cases_dir.iterdir()):
                    if sub.is_dir() and _has_placement_at(sub):
                        out.append(get_record_by_path(sub, repo_root, reports_dir_name))
        elif run_type == "eval":
            rec = get_record_by_path(d, repo_root, reports_dir_name)
            if rec:
                out.append(rec)
        else:
            rec = get_record_by_path(d, repo_root, reports_dir_name)
            if rec:
                out.append(rec)
    out.sort(key=lambda r: r.mtime, reverse=True)
    return out


def records_with_placement(records: list[RunRecord]) -> list[RunRecord]:
    """Filter to records that have placement data (for Demo/Debug/Export picker). Order preserved (newest first if records are mtime-desc)."""
    return [r for r in records if r.has_placement]


def resolve_display_path(
    active_ref: dict | None,
    repo_root: Path,
    reports_dir_name: str,
) -> Path | None:
    """
    Resolve the folder path to show in Demo/Debug/Export.
    - single / batch_case: return path if it has placement, else None.
    - batch_root: return first case path that has placement.
    - eval: return None (eval tab owns eval content).
    """
    if not active_ref or not active_ref.get("path"):
        return None
    path = Path(active_ref["path"])
    if not path.exists():
        return None
    kind = active_ref.get("kind", "single")
    if kind == "eval":
        return None
    if kind in ("single", "batch_case"):
        return path if _has_placement_at(path) else None
    if kind == "batch_root":
        cases_dir = path / "cases"
        if not cases_dir.is_dir():
            return None
        for sub in sorted(cases_dir.iterdir()):
            if sub.is_dir() and _has_placement_at(sub):
                return sub
    return None


def active_ref_from_path(path_str: str | None, repo_root: Path, reports_dir_name: str) -> dict | None:
    """Build active_ref from a path string (e.g. from session_state)."""
    if not path_str or not path_str.strip():
        return None
    path = Path(path_str.strip()).resolve()
    rec = get_record_by_path(path, repo_root, reports_dir_name)
    if not rec:
        return {"kind": "single", "path": path_str, "case_id": None}
    case_id = path.name if rec.run_type == "batch_case" else None
    return {"kind": rec.run_type, "path": path_str, "case_id": case_id}
