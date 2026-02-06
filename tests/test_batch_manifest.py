# tests/test_batch_manifest.py
"""
Batch mode: temp directory with a few synthetic WKT files; run batch and assert index.csv exists with rows.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.batch import run_batch


def test_batch_from_dir_produces_index_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wkt_dir = root / "wkt_input"
        wkt_dir.mkdir()
        # Two simple polygons
        (wkt_dir / "r1.wkt").write_text(
            "POLYGON((0 0, 100 0, 100 30, 0 30, 0 0))"
        )
        (wkt_dir / "r2.wkt").write_text(
            "POLYGON((0 0, 80 0, 80 25, 0 25, 0 0))"
        )
        report_dir = run_batch(
            run_name="test_batch",
            batch_dir=wkt_dir,
            labels_text="ELBE",
            limit=5,
            repo_root=root,
        )
        index_csv = report_dir / "index.csv"
        assert index_csv.exists()
        rows = list(index_csv.read_text(encoding="utf-8").strip().split("\n"))
        assert len(rows) >= 2  # header + 2 cases
        assert "case_id" in rows[0] or "geometry_source" in rows[0]
        cases_dir = report_dir / "cases"
        assert cases_dir.is_dir()
