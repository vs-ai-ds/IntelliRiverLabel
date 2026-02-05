# app/ui/app.py
"""
Streamlit UI: thin wrapper over app.core. Tabs Demo, Debug, Export.
See: docs/UI_SPEC.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on path when Streamlit loads this file
# __file__ is app/ui/app.py, so parent.parent.parent is repo root
_repo_root = Path(__file__).resolve().parent.parent.parent

# Verify it contains app package, fallback to cwd
if not (_repo_root / "app" / "__init__.py").exists():
    _repo_root = Path.cwd().resolve()
    if not (_repo_root / "app" / "__init__.py").exists():
        raise RuntimeError(
            f"Cannot find repo root. Run from repo root directory.\n"
            f"Expected 'app/__init__.py' in: {_repo_root}\n"
            f"Current working directory: {Path.cwd()}"
        )

# Add repo root to sys.path (must be first to avoid conflicts)
_repo_root_str = str(_repo_root.resolve())
# Remove it if already present to avoid duplicates
sys.path = [p for p in sys.path if Path(p).resolve() != _repo_root.resolve()]
# Insert at the beginning so it's checked first
sys.path.insert(0, _repo_root_str)

import streamlit as st

from app.core.config import DEFAULT_GEOMETRY_PATH, DEFAULT_FONT_FAMILY, PADDING_PT, SEED
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


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _run_placement(
    geom_path: Path,
    label_text: str,
    font_size_pt: float,
    padding_pt: float,
    seed: int | None,
    run_name: str,
    curved_mode: bool,
) -> Path | None:
    """Run placement (Phase A or Phase B if curved_mode) and write to reports/<run_name>. Returns report_dir or None on error."""
    repo_root = _repo_root()
    if not geom_path.exists():
        return None
    try:
        river_geom = load_and_validate_river(geom_path, repo_root=None)
        river_geom, safe_poly = preprocess_river(
            river_geom,
            simplify_tol_pt=0.0,
            padding_pt=padding_pt,
        )
        if safe_poly.is_empty:
            return None
        label = LabelSpec(
            text=label_text,
            font_family=DEFAULT_FONT_FAMILY,
            font_size_pt=font_size_pt,
        )
        result = run_placement(
            river_geom,
            safe_poly,
            label,
            geometry_source=DEFAULT_GEOMETRY_PATH,
            seed=seed,
            allow_phase_b=curved_mode,
            padding_pt=padding_pt,
        )
        report_dir = ensure_report_dir(repo_root, run_name)
        write_placement_json(report_dir, result)
        if result.mode == "phase_b_curved" and result.path_pt:
            try:
                from app.core.render_svg import export_curved_svg
                export_curved_svg(river_geom, label, result.path_pt, report_dir / "after.svg")
            except Exception:
                pass
        write_run_metadata_json(
            report_dir,
            run_name,
            DEFAULT_GEOMETRY_PATH,
            label_text,
            font_size_pt,
            padding_pt,
            seed,
        )
        render_before(river_geom, report_dir / "before.png")
        render_after(river_geom, result, report_dir / "after.png")
        candidates = generate_candidate_points(safe_poly, seed=seed)
        render_debug(river_geom, safe_poly, candidates, result, report_dir / "debug.png")
        return report_dir
    except Exception:
        return None


st.set_page_config(page_title="IntelliRiverLabel", layout="wide")
repo_root = _repo_root()
geom_path = repo_root / DEFAULT_GEOMETRY_PATH

tab_demo, tab_debug, tab_export = st.tabs(["Demo", "Debug", "Export"])

with tab_demo:
    st.header("Demo")
    col1, col2 = st.columns(2)
    with col1:
        label_text = st.text_input("Label text", value="ELBE")
        font_size_pt = st.number_input("Font size (pt)", min_value=1.0, value=12.0, step=0.5)
        padding_pt = st.number_input("Padding (pt)", min_value=0.0, value=float(PADDING_PT), step=0.5)
        seed_val = st.number_input("Seed", min_value=0, value=SEED or 42, step=1)
        run_name = st.text_input("Run name", value="demo_01")
        curved_mode = st.checkbox("Curved label (Phase B)", value=False)
        seed: int | None = seed_val if SEED is not None else seed_val
    if st.button("Run"):
        report_dir = _run_placement(
            geom_path,
            label_text,
            font_size_pt,
            padding_pt,
            seed,
            run_name,
            curved_mode=curved_mode,
        )
        if report_dir is not None:
            st.session_state["last_report_dir"] = str(report_dir)
            st.session_state["last_run_name"] = run_name
            placement_path = report_dir / "placement.json"
            if placement_path.exists():
                import json
                data = json.loads(placement_path.read_text(encoding="utf-8"))
                mode_used = data.get("result", {}).get("mode", "—")
                st.caption(f"Mode used: {mode_used}")
            before_path = report_dir / "before.png"
            after_path = report_dir / "after.png"
            after_svg = report_dir / "after.svg"
            if before_path.exists() and after_path.exists():
                left, right = st.columns(2)
                left.image(str(before_path), caption="Before")
                right.image(str(after_path), caption="After")
            if after_svg.exists():
                st.subheader("Curved (SVG)")
                st.image(str(after_svg), caption="after.svg")
        else:
            st.error("Run failed. Check geometry path and inputs.")

with tab_debug:
    st.header("Debug")
    last_dir = st.session_state.get("last_report_dir")
    if last_dir:
        debug_path = Path(last_dir) / "debug.png"
        if debug_path.exists():
            st.image(str(debug_path), caption="Debug overlay")
        placement_path = Path(last_dir) / "placement.json"
        if placement_path.exists():
            import json
            data = json.loads(placement_path.read_text(encoding="utf-8"))
            st.subheader("Metrics")
            m = data.get("metrics", {})
            for k, v in m.items():
                st.text(f"{k}: {v}")
            st.subheader("Result")
            r = data.get("result", {})
            st.json(r)
    else:
        st.info("Run a demo first to see debug output.")

with tab_export:
    st.header("Export")
    last_dir = st.session_state.get("last_report_dir")
    if last_dir:
        report_dir = Path(last_dir)
        placement_path = report_dir / "placement.json"
        if placement_path.exists():
            st.download_button(
                "Download placement.json",
                data=placement_path.read_bytes(),
                file_name="placement.json",
                mime="application/json",
            )
        for name in ("before.png", "after.png", "debug.png"):
            p = report_dir / name
            if p.exists():
                st.download_button(
                    f"Download {name}",
                    data=p.read_bytes(),
                    file_name=name,
                    mime="image/png",
                )
        svg_path = report_dir / "after.svg"
        if svg_path.exists():
            st.download_button(
                "Download after.svg",
                data=svg_path.read_bytes(),
                file_name="after.svg",
                mime="image/svg+xml",
            )
    else:
        st.info("Run a demo first to export.")
