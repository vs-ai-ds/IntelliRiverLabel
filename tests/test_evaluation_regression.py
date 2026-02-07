"""
Evaluation regression: run evaluation with fixed seed and assert heuristic_only success rate.
See: docs/EVALUATION.md, IMPROVEMENTS_AND_SUGGESTIONS.md §9.3.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.core.evaluate import run_evaluation
from app.core.config import DEFAULT_GEOMETRY_PATH, SEED


def test_evaluation_heuristic_only_success_rate_above_threshold() -> None:
    """Run evaluation with fixed seed; assert heuristic_only success_rate >= 0.5."""
    repo_root = Path(__file__).resolve().parent.parent
    geom_path = repo_root / DEFAULT_GEOMETRY_PATH
    if not geom_path.exists():
        pytest.skip(f"Default geometry not found: {DEFAULT_GEOMETRY_PATH}")
    run_name = "pytest_eval_regression"
    report_dir = run_evaluation(
        run_name=run_name,
        geometry_path=DEFAULT_GEOMETRY_PATH,
        n_synthetic=15,
        seed=42 if SEED is None else SEED,
        repo_root=repo_root,
        current_geometry=None,
    )
    try:
        lb_path = report_dir / "leaderboard.json"
        assert lb_path.exists(), "leaderboard.json not written"
        lb = json.loads(lb_path.read_text(encoding="utf-8"))
        by_method = lb.get("by_method", {})
        assert "heuristic_only" in by_method, "heuristic_only method missing"
        rate = by_method["heuristic_only"].get("success_rate", 0)
        assert rate >= 0.5, f"heuristic_only success_rate {rate} below 0.5 (regression?)"
    finally:
        if report_dir.exists():
            shutil.rmtree(report_dir, ignore_errors=True)
