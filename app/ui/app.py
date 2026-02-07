# app/ui/app.py
"""
Streamlit UI: sidebar (geometry, inputs, determinism, modes, output), stable tabs, demo history.
See: docs/UI_SPEC.md, docs/AI_MODEL.md.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry.base import BaseGeometry

# Ensure repo root is on path when Streamlit loads this file
_repo_root = Path(__file__).resolve().parent.parent.parent

if not (_repo_root / "app" / "__init__.py").exists():
    _repo_root = Path.cwd().resolve()
    if not (_repo_root / "app" / "__init__.py").exists():
        raise RuntimeError(
            f"Cannot find repo root. Run from repo root directory.\n"
            f"Expected 'app/__init__.py' in: {_repo_root}\n"
            f"Current working directory: {Path.cwd()}"
        )

_repo_root_str = str(_repo_root.resolve())
sys.path = [p for p in sys.path if Path(p).resolve() != _repo_root.resolve()]
sys.path.insert(0, _repo_root_str)

import streamlit as st

from app.core.config import DEFAULT_GEOMETRY_PATH, DEFAULT_FONT_FAMILY, PADDING_PT, REPORTS_DIR, SEED
from app.core.candidates_a import generate_candidate_points
from app.core.io import (
    load_and_validate_river,
    parse_wkt,
    validate_geometry,
)
try:
    from app.core.io import extract_polygon_components, select_best_component
except ImportError:
    extract_polygon_components = None  # type: ignore[misc, assignment]
    select_best_component = None  # type: ignore[misc, assignment]
from app.core.placement import run_placement
try:
    from app.core.multiparts import merge_nearby_components, describe_components
except ImportError:
    merge_nearby_components = None  # type: ignore[misc, assignment]
    describe_components = None  # type: ignore[misc, assignment]
from app.core.preprocess import preprocess_river
from app.core.render import render_before, render_after, render_debug
from app.core.reporting import (
    ensure_report_dir,
    write_placement_json,
    write_run_metadata_json,
)
from app.core.types import LabelSpec
from app.ui.help_text import (
    WKT_HELP,
    CURVED_LABEL_HELP,
    LEARNED_RANKING_HELP,
    SYNTHETIC_HELP,
    GLOSSARY_MD,
    QUICK_TROUBLESHOOT_MD,
)
try:
    from app.ui.help_text import TOOLTIP_WKT, TOOLTIP_CURVED, TOOLTIP_LEARNED, TOOLTIP_EVAL
except ImportError:
    TOOLTIP_WKT = "Well-Known Text: POLYGON, MULTIPOLYGON, or GEOMETRYCOLLECTION."
    TOOLTIP_CURVED = "Place label along river centerline; falls back to straight if not feasible."
    TOOLTIP_LEARNED = "Blend heuristic score with trained model (when model is trained)."
    TOOLTIP_EVAL = "Run baselines vs heuristic vs heuristic+model on default + synthetic polygons."


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _sanitize_output_folder_name(name: str) -> str:
    """Sanitize to safe slug: only [A-Za-z0-9_-], max 40 chars. If empty, return timestamped name."""
    s = (name or "").strip()
    if not s:
        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
    return s[:40] if len(s) > 40 else s


def _validate_label_text(text: str) -> tuple[bool, str]:
    """Returns (ok, error_message). Label must be non-empty after strip and length <= 50."""
    t = (text or "").strip()
    if not t:
        return False, "Label text cannot be empty."
    if len(t) > 50:
        return False, "Label text must be 50 characters or fewer."
    return True, ""


def _load_geometry_from_source(
    source_type: str,
    default_path: Path | None = None,
    uploaded_file=None,
    pasted_wkt: str = "",
) -> tuple[BaseGeometry | None, str]:
    """Load geometry from default path, uploaded file, or pasted WKT. Returns (geometry, source_string)."""
    if source_type == "default" and default_path and default_path.exists():
        try:
            geom = load_and_validate_river(default_path, repo_root=None)
            return (
                geom,
                str(default_path.relative_to(_repo_root()))
                if default_path.is_relative_to(_repo_root())
                else str(default_path),
            )
        except Exception:
            return None, ""

    if source_type == "upload" and uploaded_file is not None:
        try:
            wkt_str = uploaded_file.read().decode("utf-8")
            geom = parse_wkt(wkt_str)
            geom = validate_geometry(geom)
            return geom, f"uploaded_{uploaded_file.name}"
        except Exception:
            return None, ""

    if source_type == "paste" and pasted_wkt.strip():
        try:
            geom = parse_wkt(pasted_wkt.strip())
            geom = validate_geometry(geom)
            return geom, "pasted_wkt"
        except Exception:
            return None, ""

    return None, ""


def _run_placement(
    river_geom: BaseGeometry,
    label_text: str,
    font_size_pt: float,
    padding_pt: float,
    seed: int | None,
    run_name: str,
    curved_mode: bool,
    use_learned_ranking: bool,
    geometry_source: str,
    render_scale: int = 1,
) -> tuple[Path | None, str | None]:
    """
    Run placement and write to reports/<run_name>.
    Returns (report_dir, error_key). error_key is "safe_poly_empty" or None.
    """
    repo_root = _repo_root()
    try:
        river_geom, safe_poly = preprocess_river(
            river_geom,
            simplify_tol_pt=0.0,
            padding_pt=padding_pt,
        )
        if safe_poly.is_empty:
            return None, "safe_poly_empty"
        label = LabelSpec(
            text=label_text,
            font_family=DEFAULT_FONT_FAMILY,
            font_size_pt=font_size_pt,
        )
        result = run_placement(
            river_geom,
            safe_poly,
            label,
            geometry_source=geometry_source,
            seed=seed,
            allow_phase_b=curved_mode,
            padding_pt=padding_pt,
            use_learned_ranking=use_learned_ranking,
        )
        report_dir = ensure_report_dir(repo_root, run_name)
        write_placement_json(report_dir, result)
        if result.mode == "phase_b_curved" and result.path_pt:
            try:
                from app.core.render_svg import export_curved_svg
                export_curved_svg(river_geom, label, result.path_pt, report_dir / "after.svg")
            except Exception:
                pass
        model_path_str: str | None = None
        if use_learned_ranking:
            try:
                from app.models.registry import get_model_metadata
                meta = get_model_metadata()
                if meta and meta.get("artifact_path"):
                    model_path_str = meta["artifact_path"]
            except Exception:
                pass
        write_run_metadata_json(
            report_dir,
            run_name,
            geometry_source,
            label_text,
            font_size_pt,
            padding_pt,
            seed,
            render_scale=render_scale,
            curved_mode=curved_mode,
            use_learned_ranking=use_learned_ranking,
            model_artifact_path=model_path_str,
        )
        def _render_with_scale():
            try:
                render_before(river_geom, report_dir / "before.png", scale=render_scale)
            except TypeError:
                render_before(river_geom, report_dir / "before.png")
            try:
                render_after(river_geom, result, report_dir / "after.png", scale=render_scale)
            except TypeError:
                render_after(river_geom, result, report_dir / "after.png")
            candidates = generate_candidate_points(safe_poly, seed=seed)
            try:
                render_debug(river_geom, safe_poly, candidates, result, report_dir / "debug.png", scale=render_scale)
            except TypeError:
                render_debug(river_geom, safe_poly, candidates, result, report_dir / "debug.png")
        _render_with_scale()
        if "last_run_error" in st.session_state:
            del st.session_state["last_run_error"]
        return report_dir, None
    except Exception as e:
        st.session_state["last_run_error"] = str(e)
        return None, None


# Session state defaults for demo history and active run
if "demo_history" not in st.session_state:
    st.session_state["demo_history"] = []
if "active_report_dir" not in st.session_state:
    st.session_state["active_report_dir"] = None

DEMO_HISTORY_MAX = 10

st.set_page_config(page_title="IntelliRiverLabel", layout="wide")
repo_root = _repo_root()
default_geom_path = repo_root / DEFAULT_GEOMETRY_PATH

# Model status
model_loaded = False
model_meta: dict | None = None
try:
    from app.models.registry import load_model, get_model_metadata
    model_loaded = load_model() is not None
    model_meta = get_model_metadata()
except Exception:
    pass

# ----- Sidebar -----
with st.sidebar:
    st.header("IntelliRiverLabel")

    # --- First run guide ---
    with st.expander("First run guide", expanded=False):
        st.markdown(
            "1. **Load geometry** — Use default path, upload a .wkt file, or paste WKT.\n"
            "2. Optionally enable **Curved label (Phase B)** and/or **Use learned ranking** (after training).\n"
            "3. Click **Run demo**.\n"
            "4. Open the **Demo** tab to see before/after and download results."
        )

    # --- Current run ---
    _active = st.session_state.get("active_report_dir") or st.session_state.get("last_report_dir")
    _run_name = st.session_state.get("last_run_name", "")
    if _active and _run_name:
        st.caption(f"**Current run:** {_run_name}")
    st.divider()

    # --- Geometry source ---
    st.subheader("Geometry source")
    source_type = st.radio(
        "Source",
        ["default", "upload", "paste"],
        format_func=lambda x: {
            "default": f"Default ({DEFAULT_GEOMETRY_PATH})",
            "upload": "Upload .wkt file",
            "paste": "Paste WKT",
        }[x],
        help=TOOLTIP_WKT,
    )
    uploaded_file = None
    pasted_wkt = ""
    if source_type == "default":
        if not default_geom_path.exists():
            st.warning(f"Default geometry not found: {DEFAULT_GEOMETRY_PATH}")
    elif source_type == "upload":
        uploaded_file = st.file_uploader("Upload WKT file", type=["wkt", "txt"])
    elif source_type == "paste":
        pasted_wkt = st.text_area("Paste WKT geometry", height=100, placeholder="POLYGON((...))")

    geom_for_run, geom_source = _load_geometry_from_source(
        source_type,
        default_path=default_geom_path if source_type == "default" else None,
        uploaded_file=uploaded_file,
        pasted_wkt=pasted_wkt,
    )
    components: list = []
    if geom_for_run is not None and extract_polygon_components is not None:
        components = extract_polygon_components(geom_for_run)

    merge_enabled = st.checkbox("Merge nearby components (braided rivers)", value=False)
    merge_distance_pt = 5.0
    if merge_enabled:
        merge_distance_pt = st.slider("Merge distance (pt)", 0.0, 20.0, 5.0)
    if merge_enabled and merge_nearby_components is not None and geom_for_run is not None:
        before_count = len(components) if components else (describe_components(geom_for_run)["component_count"] if describe_components else 0)
        geom_for_run = merge_nearby_components(geom_for_run, merge_distance_pt)
        if extract_polygon_components is not None:
            components = extract_polygon_components(geom_for_run)
        after_count = len(components) if components else (describe_components(geom_for_run)["component_count"] if describe_components else 0)
        st.caption(f"Components: before {before_count} → after {after_count}")

    component_choice = "Auto"
    if len(components) > 1:
        st.caption(f"Multi-part geometry: {len(components)} parts detected.")
        component_choice = st.selectbox(
            "Component",
            ["Auto"] + [str(i) for i in range(len(components))],
            help="Auto = best part by safe area; or pick part 0, 1, 2…",
        )

    # --- Inputs ---
    st.subheader("Inputs")
    label_text = st.text_input("Label text", value="ELBE", help="Text to place on the river.")
    font_size_pt = st.number_input("Font size (pt)", min_value=1.0, value=12.0, step=0.5)
    padding_pt = st.number_input(
        "Padding (pt)", min_value=0.0, value=float(PADDING_PT), step=0.5,
        help="Inward margin (pt) from river boundary; label must fit inside the shrunk polygon.",
    )

    # --- Determinism ---
    st.subheader("Determinism")
    use_seed = st.checkbox(
        "Deterministic (use seed)",
        value=True,
        help="Same seed = same result every time. Uncheck for random variation.",
    )
    seed_val = SEED or 42
    if use_seed:
        seed_val = st.number_input("Seed", min_value=0, value=seed_val, step=1)
    seed: int | None = seed_val if use_seed else None

    # --- Modes ---
    st.subheader("Modes")
    curved_mode = st.checkbox("Curved label (Phase B)", value=False, help=TOOLTIP_CURVED)
    use_learned_ranking = False
    if model_loaded:
        use_learned_ranking = st.checkbox("Use learned ranking", value=False, help=TOOLTIP_LEARNED)
    else:
        st.checkbox("Use learned ranking", value=False, disabled=True)
        st.caption("Train model to enable")

    # --- Output ---
    st.subheader("Output")
    render_scale_option = st.selectbox(
        "Render scale",
        [1, 2, 4],
        format_func=lambda x: f"{x}x",
        index=1,
        help="1x=800×600, 2x=1600×1200, 4x=3200×2400. Only affects PNG size, not placement.",
    )
    run_name_raw = st.text_input(
        "Output folder name",
        value="demo_01",
        help="Reports subfolder (e.g. reports/demo_01/). Sanitized to safe characters.",
    )
    run_name = _sanitize_output_folder_name(run_name_raw)
    if run_name != (run_name_raw or "").strip():
        st.caption(f"Using: {run_name}")
    # Run name uniqueness: avoid overwriting
    _reports_base = _repo_root() / REPORTS_DIR
    _run_dir = _reports_base / run_name
    if _run_dir.exists() and _run_dir.is_dir():
        st.warning(f"Folder already exists: {run_name}. Run will overwrite it, or choose a different name.")
    run_clicked = st.button("Run demo", type="primary")
    run_again_clicked = st.button("Run again (same options, new name)", help="Re-run with current geometry and options; uses a new folder name automatically.")

    # Run again: use new name and trigger run
    if run_again_clicked and geom_for_run is not None:
        base = st.session_state.get("last_run_name") or "demo_01"
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        st.session_state["run_again_name"] = f"{base}_{stamp}"
        st.session_state["trigger_run_again"] = True
        st.rerun()

    if run_clicked or st.session_state.pop("trigger_run_again", False):
        if st.session_state.get("run_again_name"):
            run_name = _sanitize_output_folder_name(st.session_state.pop("run_again_name", run_name))
        ok, err = _validate_label_text(label_text)
        if not ok:
            st.error(err)
        elif geom_for_run is None:
            st.error("Load geometry first (default path, upload, or paste WKT).")
        else:
            if len(components) > 1 and select_best_component is not None and component_choice != "Auto":
                river_geom = components[int(component_choice)]
            elif len(components) > 1 and select_best_component is not None:
                river_geom = select_best_component(geom_for_run, padding_pt)
            else:
                river_geom = geom_for_run
            report_dir, error_key = _run_placement(
                river_geom,
                label_text.strip(),
                font_size_pt,
                padding_pt,
                seed,
                run_name,
                curved_mode,
                use_learned_ranking,
                geom_source,
                render_scale=render_scale_option,
            )
            if report_dir is not None:
                st.session_state["active_report_dir"] = str(report_dir)
                st.session_state["show_demo_tab_banner"] = True  # prompt user to open Demo tab
                placement_path = report_dir / "placement.json"
                mode_used = ""
                if placement_path.exists():
                    data = json.loads(placement_path.read_text(encoding="utf-8"))
                    mode_used = data.get("result", {}).get("mode", "")
                ts = datetime.now(timezone.utc).isoformat()
                entry = {
                    "run_name": run_name,
                    "report_dir": str(report_dir),
                    "timestamp_utc": ts,
                    "mode_used": mode_used,
                    "geometry_source": geom_source,
                    "label_text": label_text.strip(),
                    "curved_mode": curved_mode,
                    "learned_ranking": use_learned_ranking,
                }
                history = st.session_state.get("demo_history", [])
                history.insert(0, entry)
                st.session_state["demo_history"] = history[:DEMO_HISTORY_MAX]
                st.session_state["last_report_dir"] = str(report_dir)
                st.session_state["last_run_name"] = run_name
                st.session_state["last_curved_mode"] = curved_mode
                st.rerun()
            else:
                if error_key == "safe_poly_empty":
                    st.error("Safe polygon became empty at this padding. Try reducing padding or font size.")
                else:
                    err_msg = st.session_state.pop("last_run_error", None)
                    st.error("Run failed. " + (err_msg if err_msg else "Check geometry and inputs."))

    # --- AI & Evaluation (expander) ---
    st.divider()
    with st.expander("AI & Evaluation"):
        if model_meta:
            st.success("Model: Available ✅")
            if model_meta.get("trained_timestamp_utc"):
                st.caption(f"Trained: {model_meta['trained_timestamp_utc']}")
        else:
            st.warning("Model: Missing ⚠️")
        st.caption("Train a regressor on synthetic river polygons to improve ranking. See Help & glossary.")
        if st.button("Train model (synthetic)"):
            try:
                from app.models.train import train_model
                with st.spinner("Training..."):
                    train_model(n_polygons=100, seed=seed)
                st.success("Model trained and saved.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        eval_run_name = st.text_input("Evaluation run name", value="eval_01", key="eval_run_name")
        n_synthetic = st.number_input("N synthetic polygons", min_value=5, value=20, step=5, key="n_synthetic")
        eval_seed = seed if use_seed else (SEED or 42)
        if st.button("Run evaluation", help=TOOLTIP_EVAL):
            try:
                from app.core.evaluate import run_evaluation
                with st.spinner("Running evaluation..."):
                    report_dir = run_evaluation(
                        run_name=eval_run_name,
                        geometry_path=DEFAULT_GEOMETRY_PATH,
                        n_synthetic=n_synthetic,
                        seed=eval_seed,
                        repo_root=repo_root,
                    )
                st.session_state["last_eval_dir"] = str(report_dir)
                st.session_state["last_eval_run_name"] = eval_run_name
                st.success("Evaluation complete.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # --- Status ---
    st.divider()
    st.subheader("Status")
    history = st.session_state.get("demo_history", [])
    if history:
        options = [f"{h['run_name']} — {h.get('mode_used', '')}" for h in history]
        current = st.session_state.get("active_report_dir") or history[0].get("report_dir")
        sel_idx = next((i for i, h in enumerate(history) if h.get("report_dir") == current), 0)
        active_sel = st.selectbox(
            "Active demo run",
            range(len(history)),
            format_func=lambda i: options[i],
            index=sel_idx,
            key="sidebar_active_demo",
        )
        st.session_state["active_report_dir"] = history[active_sel]["report_dir"]
        last = history[0]
        st.caption(f"Last: **{last.get('run_name', '')}** — {last.get('mode_used', '')}")
        if last.get("timestamp_utc"):
            st.caption(last["timestamp_utc"][:19].replace("T", " "))
    else:
        st.caption("No demo run yet")
    if st.session_state.get("last_eval_dir"):
        eval_dir = Path(st.session_state["last_eval_dir"])
        summary_path = eval_dir / "evaluation_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                n_cases = summary.get("n_cases", "")
                rates = summary.get("success_rate", {})
                st.caption(f"Last eval: **{st.session_state.get('last_eval_run_name', '')}** — {n_cases} cases")
                if rates:
                    st.caption("Success rates: " + ", ".join(f"{k}={v:.0%}" for k, v in rates.items()))
            except Exception:
                st.caption(f"Last eval: {st.session_state.get('last_eval_run_name', '')}")
        else:
            st.caption(f"Last eval: {st.session_state.get('last_eval_run_name', '')}")
    else:
        st.caption("No evaluation yet")
    if st.session_state.get("last_batch_dir"):
        batch_dir = Path(st.session_state["last_batch_dir"])
        idx = batch_dir / "index.csv"
        n_cases = ""
        if idx.exists():
            try:
                n_cases = str(sum(1 for _ in open(idx, encoding="utf-8")) - 1)  # minus header
            except Exception:
                n_cases = "—"
        st.caption(f"Last batch: **{st.session_state.get('last_batch_name', '')}** — {n_cases} cases")
    else:
        st.caption("No batch run yet")

    # --- Help & glossary ---
    with st.expander("Help & glossary"):
        st.markdown(GLOSSARY_MD)
        st.markdown("---")
        st.markdown("### Quick troubleshooting")
        st.markdown(QUICK_TROUBLESHOOT_MD)

# Resolve which report dir to show in Demo/Debug/Export (active run)
active_dir = st.session_state.get("active_report_dir") or st.session_state.get("last_report_dir")
history = st.session_state.get("demo_history", [])
if history and not active_dir:
    active_dir = history[0].get("report_dir")

def _placement_metrics(report_dir: Path) -> dict:
    """Read key metrics from placement.json or placements.json; return dict with defaults for missing keys."""
    # Multi-label: placements.json
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


def _compare_winner(metrics_a: dict, metrics_b: dict) -> str:
    """Simple winner: prefer phase_b_curved > phase_a_straight > external_fallback, then higher min_clearance_pt, then fit_margin_ratio."""
    mode_rank = {"phase_b_curved": 3, "phase_a_straight": 2, "external_fallback": 1}
    ra = mode_rank.get(metrics_a.get("mode"), 0)
    rb = mode_rank.get(metrics_b.get("mode"), 0)
    if ra != rb:
        return "A" if ra > rb else "B"
    ca = float(metrics_a.get("min_clearance_pt") or 0)
    cb = float(metrics_b.get("min_clearance_pt") or 0)
    if ca != cb:
        return "A" if ca > cb else "B"
    fa = float(metrics_a.get("fit_margin_ratio") or 0)
    fb = float(metrics_b.get("fit_margin_ratio") or 0)
    return "A" if fa >= fb else "B"


# ----- Main: 6 stable tabs -----
tab_demo, tab_debug, tab_export, tab_compare, tab_batch, tab_evaluate = st.tabs(["Demo", "Debug", "Export", "Compare", "Batch", "Evaluate"])

with tab_demo:
    st.header("Demo")
    if st.session_state.pop("show_demo_tab_banner", False):
        st.success("Results ready below. Use the download buttons to save images and check the Export tab for a single zip.")
    display_dir = active_dir
    full_width_images = st.toggle("Full-width images", value=False)
    if display_dir:
        report_dir = Path(display_dir)
        placement_path = report_dir / "placement.json"
        mode_used = ""
        if placement_path.exists():
            data = json.loads(placement_path.read_text(encoding="utf-8"))
            mode_used = data.get("result", {}).get("mode", "—")
            curved_attempted = mode_used == "phase_b_curved" or any("Phase B attempted but failed" in w for w in data.get("warnings", []))
            if curved_attempted:
                if mode_used == "phase_b_curved":
                    st.success("Phase B attempted: curved label placed successfully.")
                else:
                    st.warning("Phase B attempted: fallback to Phase A.")
                    phase_b_reasons = [w.replace("Phase B attempted but failed: ", "") for w in data.get("warnings", []) if "Phase B attempted but failed" in w]
                    if phase_b_reasons:
                        st.caption(f"Reason: {phase_b_reasons[0]}")
            st.success(f"Mode used: {mode_used}")
        before_path = report_dir / "before.png"
        after_path = report_dir / "after.png"
        if before_path.exists() and after_path.exists():
            if full_width_images:
                st.image(str(before_path), caption="Before", width="stretch")
                st.image(str(after_path), caption="After", width="stretch")
            else:
                left, right = st.columns(2)
                left.image(str(before_path), caption="Before")
                right.image(str(after_path), caption="After")
        for name, path in [("before.png", before_path), ("after.png", after_path), ("debug.png", report_dir / "debug.png")]:
            if path.exists():
                st.download_button(f"Download {name}", data=path.read_bytes(), file_name=name, mime="image/png", key=f"dl_demo_{name}")
        after_svg = report_dir / "after.svg"
        if after_svg.exists() and mode_used == "phase_b_curved":
            st.subheader("Curved label (SVG)")
            svg_text = after_svg.read_text(encoding="utf-8")
            if "<?xml" in svg_text:
                idx_svg = svg_text.find("<svg")
                if idx_svg >= 0:
                    svg_text = svg_text[idx_svg:]
            st.components.v1.html(f'<div style="overflow:auto;">{svg_text}</div>', height=700, scrolling=True)
    else:
        st.info("Run demo to see results.")

with tab_debug:
    st.header("Debug")
    debug_dir = st.session_state.get("active_report_dir") or active_dir
    if debug_dir:
        debug_path = Path(debug_dir) / "debug.png"
        if debug_path.exists():
            st.image(str(debug_path), caption="Debug overlay")
        placement_path = Path(debug_dir) / "placement.json"
        if placement_path.exists():
            data = json.loads(placement_path.read_text(encoding="utf-8"))
            st.subheader("Metrics")
            m = data.get("metrics", {})
            if m:
                st.dataframe(
                    [{"Metric": k, "Value": v} for k, v in m.items()],
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("Result JSON"):
                st.json(data.get("result", {}))
            warnings = data.get("warnings", [])
            if warnings:
                st.subheader("Warnings")
                for w in warnings:
                    st.warning(w)
    else:
        st.info("Run a demo first to see debug output.")

with tab_export:
    st.header("Export")
    export_dir = st.session_state.get("active_report_dir") or active_dir
    if export_dir:
        report_dir = Path(export_dir)
        # Download all as zip
        import zipfile
        import io
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("placement.json", "run_metadata.json", "before.png", "after.png", "debug.png", "after.svg"):
                p = report_dir / name
                if p.exists():
                    zf.write(p, name)
        zip_buffer.seek(0)
        st.download_button(
            "Download all (zip)",
            data=zip_buffer,
            file_name=f"{report_dir.name}_export.zip",
            mime="application/zip",
            key="dl_export_all_zip",
        )
        st.caption("Contains placement.json, run_metadata.json, before/after/debug PNGs, and after.svg if present.")
        st.divider()
        # Individual files with size and last modified
        def _file_info(path: Path) -> tuple[str, int, str]:
            if not path.exists():
                return ("—", 0, "—")
            try:
                stat = path.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                return (f"{size:,} B", size, mtime)
            except Exception:
                return ("—", 0, "—")
        files_to_show = [
            ("placement.json", report_dir / "placement.json", "application/json"),
            ("run_metadata.json", report_dir / "run_metadata.json", "application/json"),
            ("before.png", report_dir / "before.png", "image/png"),
            ("after.png", report_dir / "after.png", "image/png"),
            ("debug.png", report_dir / "debug.png", "image/png"),
            ("after.svg", report_dir / "after.svg", "image/svg+xml"),
        ]
        for label, p, mime in files_to_show:
            if p.exists():
                size_str, _, mtime_str = _file_info(p)
                st.download_button(
                    f"Download {label} ({size_str}, {mtime_str})",
                    data=p.read_bytes(),
                    file_name=label,
                    mime=mime,
                    key=f"dl_export_{label.replace('.', '_')}",
                )
    else:
        st.info("Run a demo first to export.")

with tab_compare:
    st.header("Compare")
    reports_path = _repo_root() / REPORTS_DIR
    run_folders = sorted([d.name for d in reports_path.iterdir() if d.is_dir()]) if reports_path.exists() else []
    # Exclude batch/eval case subdirs: only top-level report folders (demo_*, batch_*, eval_*)
    run_folders = [f for f in run_folders if not f.startswith("cases")]
    if len(run_folders) < 2:
        st.info("Need at least 2 run folders in reports/ to compare. Run demo or batch, then pick two.")
    else:
        idx_a = st.selectbox("Run A folder", run_folders, key="compare_sel_a")
        idx_b = st.selectbox("Run B folder", run_folders, key="compare_sel_b")
        dir_a = reports_path / idx_a
        dir_b = reports_path / idx_b
        # For batch runs, compare at batch level (index.csv); for single-run folders use the folder itself. Here we compare two top-level folders; if they are batch_*, we don't drill into cases.
        metrics_a = _placement_metrics(dir_a)
        metrics_b = _placement_metrics(dir_b)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"**Run A:** {idx_a}")
            after_a = dir_a / "after.png"
            if not after_a.exists() and (dir_a / "cases").is_dir():
                first_case = next((dir_a / "cases").iterdir(), None)
                if first_case and first_case.is_dir():
                    after_a = first_case / "after.png"
            if after_a.exists():
                st.image(str(after_a), caption="After", width="stretch")
            else:
                st.caption("after.png missing")
            st.text("Open folder: " + str(dir_a))
        with col_b:
            st.caption(f"**Run B:** {idx_b}")
            after_b = dir_b / "after.png"
            if not after_b.exists() and (dir_b / "cases").is_dir():
                first_case = next((dir_b / "cases").iterdir(), None)
                if first_case and first_case.is_dir():
                    after_b = first_case / "after.png"
            if after_b.exists():
                st.image(str(after_b), caption="After", width="stretch")
            else:
                st.caption("after.png missing")
            st.text("Open folder: " + str(dir_b))
        st.subheader("Metrics")
        keys = ["min_clearance_pt", "fit_margin_ratio", "mode", "success_count", "n_labels", "collisions_detected"]
        keys = [k for k in keys if metrics_a.get(k) is not None or metrics_b.get(k) is not None]
        if not keys:
            keys = ["min_clearance_pt", "fit_margin_ratio", "mode"]
        # Use string values so PyArrow doesn't fail on mixed types (e.g. mode=str vs clearance=float)
        def _str(v):
            return "" if v is None else str(v)
        rows = [{"Metric": k, "Run A": _str(metrics_a.get(k)), "Run B": _str(metrics_b.get(k))} for k in keys]
        st.dataframe(rows, width="stretch", hide_index=True)
        winner = _compare_winner(metrics_a, metrics_b) if metrics_a and metrics_b else None
        if winner:
            st.success(f"**Winner (heuristic):** Run {winner}")

with tab_batch:
    st.header("Batch")
    batch_run_name = st.text_input("Batch run name", value="batch_01", key="batch_run_name", help="Output: reports/batch_<name>/")
    batch_dir_path = st.text_input("Batch directory (path to folder with .wkt files)", value="", key="batch_dir_path", help="Relative to repo root or absolute path.")
    batch_manifest_upload = st.file_uploader("Or upload manifest CSV (path, labels)", type=["csv"], key="batch_manifest")
    batch_labels_text = st.text_input("Labels (single or comma-separated)", value="ELBE", key="batch_labels", help="e.g. ELBE or ELBE,MAIN")
    batch_limit = st.number_input("Limit cases", min_value=0, value=0, step=1, key="batch_limit", help="0 = no limit")
    run_batch_clicked = st.button("Run batch", type="primary", key="run_batch_btn")
    if run_batch_clicked:
        try:
            import tempfile
            from app.core.batch import run_batch
            root = _repo_root()
            batch_dir: Path | None = None
            manifest_path: Path | None = None
            if batch_manifest_upload:
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
                    tmp.write(batch_manifest_upload.getvalue())
                    manifest_path = Path(tmp.name)
            elif batch_dir_path.strip():
                p = Path(batch_dir_path.strip())
                batch_dir = (root / p) if not p.is_absolute() else p
            if batch_dir is None and manifest_path is None:
                st.error("Provide a batch directory path or upload a manifest CSV.")
            else:
                with st.spinner("Running batch..."):
                    report_dir = run_batch(
                        run_name=batch_run_name.strip() or "batch_01",
                        batch_dir=batch_dir,
                        manifest_path=manifest_path,
                        labels_text=batch_labels_text.strip() or "ELBE",
                        limit=batch_limit or None,
                        repo_root=root,
                        padding_pt=float(PADDING_PT),
                        seed=seed,
                        allow_phase_b=curved_mode,
                    )
                st.session_state["last_batch_dir"] = str(report_dir)
                st.session_state["last_batch_name"] = batch_run_name.strip() or "batch_01"
                st.success("Batch complete.")
                st.rerun()
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("last_batch_dir"):
        batch_report = Path(st.session_state["last_batch_dir"])
        index_csv = batch_report / "index.csv"
        if index_csv.exists():
            import csv as csv_module
            with open(index_csv, newline="", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
            st.subheader("Results")
            st.dataframe(rows, width="stretch", hide_index=True)
            st.download_button("Download index.csv", data=index_csv.read_bytes(), file_name="index.csv", mime="text/csv", key="dl_batch_index")
        st.caption("Report folder: " + str(batch_report))

with tab_evaluate:
    st.header("Evaluate")
    if st.session_state.get("last_eval_dir"):
        eval_dir = Path(st.session_state["last_eval_dir"])
        summary_path = eval_dir / "evaluation_summary.json"
        if summary_path.exists():
            with st.expander("Summary"):
                st.json(json.loads(summary_path.read_text(encoding="utf-8")))
        # Leaderboard
        lb_json = eval_dir / "leaderboard.json"
        lb_csv = eval_dir / "leaderboard.csv"
        if lb_json.exists():
            st.subheader("Leaderboard")
            lb = json.loads(lb_json.read_text(encoding="utf-8"))
            if lb.get("by_method"):
                rows_method = [{"method": k, **v} for k, v in lb["by_method"].items()]
                st.dataframe(rows_method, width="stretch", hide_index=True)
            if lb.get("by_family"):
                st.caption("By family")
                rows_family = [{"family": k, **v} for k, v in lb["by_family"].items()]
                st.dataframe(rows_family, width="stretch", hide_index=True)
        elif lb_csv.exists():
            st.subheader("Leaderboard")
            import csv as _csv
            with open(lb_csv, newline="", encoding="utf-8") as _f:
                _rows = list(_csv.DictReader(_f))
            st.dataframe(_rows, width="stretch", hide_index=True)
        plots_dir = eval_dir / "plots"
        if plots_dir.exists():
            for name, caption in [
                ("success_rate.png", "Success rate by method"),
                ("min_clearance_pt.png", "min_clearance_pt by method"),
                ("success_by_family.png", "Success rate by family"),
                ("collision_rate_by_method.png", "Collision rate by method"),
            ]:
                if (plots_dir / name).exists():
                    st.image(str(plots_dir / name), caption=caption)
        csv_path = eval_dir / "evaluation_results.csv"
        if csv_path.exists():
            st.download_button(
                "Download evaluation_results.csv",
                data=csv_path.read_bytes(),
                file_name="evaluation_results.csv",
                mime="text/csv",
                key="dl_eval_csv",
            )
        if lb_csv.exists():
            st.download_button("Download leaderboard.csv", data=lb_csv.read_bytes(), file_name="leaderboard.csv", mime="text/csv", key="dl_lb_csv")
        if lb_json.exists():
            st.download_button("Download leaderboard.json", data=lb_json.read_bytes(), file_name="leaderboard.json", mime="application/json", key="dl_lb_json")
        if summary_path.exists():
            st.download_button(
                "Download evaluation_summary.json",
                data=summary_path.read_bytes(),
                file_name="evaluation_summary.json",
                mime="application/json",
                key="dl_eval_json",
            )
    else:
        st.info("Run evaluation from the sidebar to see results.")
