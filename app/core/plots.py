# app/core/plots.py
"""
Evaluation plots: success_rate bar chart, min_clearance_pt distribution.
Saves under reports/<run_name>/plots/. See: docs/EVALUATION.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_results(report_dir: Path) -> tuple[list[dict], dict]:
    """Load evaluation_results.csv and evaluation_summary.json from report_dir."""
    import csv
    rows: list[dict] = []
    csv_path = report_dir / "evaluation_results.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["success"] = r.get("success", "").strip().lower() in ("true", "1", "yes")
                try:
                    r["min_clearance_pt"] = float(r.get("min_clearance_pt", 0))
                except (TypeError, ValueError):
                    r["min_clearance_pt"] = 0.0
                try:
                    r["fit_margin_ratio"] = float(r.get("fit_margin_ratio", 0))
                except (TypeError, ValueError):
                    r["fit_margin_ratio"] = 0.0
                rows.append(r)
    summary: dict = {}
    json_path = report_dir / "evaluation_summary.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text(encoding="utf-8"))
    return rows, summary


def plot_success_rate(report_dir: Path) -> Path:
    """
    Bar chart of success_rate by method. Saves to report_dir/plots/success_rate.png.
    Returns path to saved figure.
    """
    rows, summary = _load_results(report_dir)
    if not rows and not summary.get("success_rate"):
        return report_dir / "plots" / "success_rate.png"

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if summary.get("success_rate"):
        methods = list(summary["success_rate"].keys())
        rates = [summary["success_rate"][m] for m in methods]
    else:
        by_method: dict[str, list[bool]] = {}
        for r in rows:
            m = r.get("method", "unknown")
            if m not in by_method:
                by_method[m] = []
            by_method[m].append(r.get("success", False))
        methods = list(by_method.keys())
        rates = [
            sum(by_method[m]) / len(by_method[m]) if by_method[m] else 0.0
            for m in methods
        ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, rates, color=["#2ecc71", "#3498db", "#9b59b6"][: len(methods)])
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by method")
    plt.tight_layout()
    out = plots_dir / "success_rate.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_min_clearance_pt(report_dir: Path) -> Path:
    """
    Distribution of min_clearance_pt (boxplot by method). Saves to report_dir/plots/min_clearance_pt.png.
    Returns path to saved figure.
    """
    rows, _ = _load_results(report_dir)
    if not rows:
        return report_dir / "plots" / "min_clearance_pt.png"

    by_method: dict[str, list[float]] = {}
    for r in rows:
        m = r.get("method", "unknown")
        if m not in by_method:
            by_method[m] = []
        by_method[m].append(r.get("min_clearance_pt", 0.0))

    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = list(by_method.keys())
    data = [by_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=methods)
    ax.set_ylabel("min_clearance_pt")
    ax.set_xlabel("Method")
    ax.set_title("min_clearance_pt distribution by method")
    plt.tight_layout()
    out = plots_dir / "min_clearance_pt.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_success_by_family(report_dir: Path) -> Path:
    """Bar chart of success rate by family. Saves to report_dir/plots/success_by_family.png."""
    rows, _ = _load_results(report_dir)
    if not rows:
        return report_dir / "plots" / "success_by_family.png"
    by_family: dict[str, list[bool]] = {}
    for r in rows:
        fam = r.get("family", "default")
        if fam not in by_family:
            by_family[fam] = []
        by_family[fam].append(r.get("success", False))
    families = list(by_family.keys())
    rates = [sum(by_family[f]) / len(by_family[f]) if by_family[f] else 0 for f in families]
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(families, rates, color="steelblue", alpha=0.8)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Family")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by family")
    plt.tight_layout()
    out = plots_dir / "success_by_family.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_collision_rate_by_method(report_dir: Path) -> Path:
    """Bar chart of collision rate by method. Saves to report_dir/plots/collision_rate_by_method.png."""
    rows, _ = _load_results(report_dir)
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        out = plots_dir / "collision_rate_by_method.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Collision rate by method (no data)")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    by_method: dict[str, list[float]] = {}
    for r in rows:
        m = r.get("method", "unknown")
        if m not in by_method:
            by_method[m] = []
        by_method[m].append(float(r.get("collisions", 0)))
    methods = list(by_method.keys())
    rates = [sum(by_method[m]) / len(by_method[m]) if by_method[m] else 0 for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, rates, color="coral", alpha=0.8)
    ax.set_ylabel("Collision rate")
    ax.set_xlabel("Method")
    ax.set_title("Collision rate by method")
    plt.tight_layout()
    out = plots_dir / "collision_rate_by_method.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def generate_all_plots(report_dir: Path) -> list[Path]:
    """Generate all evaluation plots."""
    paths = []
    paths.append(plot_success_rate(report_dir))
    paths.append(plot_min_clearance_pt(report_dir))
    paths.append(plot_success_by_family(report_dir))
    paths.append(plot_collision_rate_by_method(report_dir))
    return paths
