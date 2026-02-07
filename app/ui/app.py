# app/ui/app.py
"""
Streamlit UI: sidebar (geometry, inputs, modes, output), tabs Demo/Debug/Export/All results/Compare/Batch/Evaluate.
See: docs/UI_SPEC.md, docs/AI_MODEL.md.

UI + State contract: run types, active_ref rules, what opens where — see app/ui/run_registry.py docstring.
Demo/Debug/Export render only from active_ref (no per-tab run picker). Run selection: Demo tab Run dropdown and All Results hub; Switch to run and Run Again removed from sidebar.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Configure logging from env (e.g. LOG_LEVEL=DEBUG for development)
_log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level_name, logging.INFO))

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

from app.core.config import (
    COLLISION_MAX_AREA,
    COLLISION_WEIGHT,
    DEFAULT_GEOMETRY_PATH,
    DEFAULT_FONT_FAMILY,
    K_TOP_CLEARANCE,
    N_SAMPLE_POINTS,
    PADDING_PT,
    REPORTS_DIR,
    SEED,
)
from app.core.error_codes import user_message
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
    EVAL_SHORT,
    TOOLTIP_RENDER_SCALE,
    TOOLTIP_RUN_AGAIN,
    TOOLTIP_OVERRIDE_NK,
    TOOLTIP_TRAIN_SYNTHETIC,
)
from app.ui.run_registry import (
    RunRecord,
    active_ref_from_path,
    get_record_by_path,
    records_with_placement,
    resolve_display_path,
    scan_reports,
)
from app.ui import components as ui_components
try:
    from app.ui.help_text import TOOLTIP_WKT, TOOLTIP_CURVED, TOOLTIP_LEARNED, TOOLTIP_EVAL, TOOLTIP_MERGE_NEARBY
except ImportError:
    TOOLTIP_WKT = "Well-Known Text: POLYGON, MULTIPOLYGON, or GEOMETRYCOLLECTION."
    TOOLTIP_CURVED = "Place label along river centerline; falls back to straight if not feasible."
    TOOLTIP_LEARNED = "Blend heuristic score with trained model (when model is trained)."
    TOOLTIP_EVAL = "Run baselines vs heuristic vs heuristic+model on default + synthetic polygons."
    TOOLTIP_MERGE_NEARBY = "Merge multiple polygon parts (e.g. braided rivers) within a distance; only affects multi-part geometry. Example WKT in Help & glossary."


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
) -> tuple[BaseGeometry | None, str, str]:
    """Load geometry from default path, uploaded file, or pasted WKT. Returns (geometry, source_string, error_message)."""
    if source_type == "default" and default_path and default_path.exists():
        try:
            geom = load_and_validate_river(default_path, repo_root=None)
            try:
                src = str(default_path.relative_to(_repo_root()))
            except ValueError:
                src = str(default_path)
            return (geom, src, "")
        except Exception as e:
            return None, "", str(e).strip() or "Failed to load geometry."

    if source_type == "upload" and uploaded_file is not None:
        try:
            wkt_str = uploaded_file.read().decode("utf-8")
            geom = parse_wkt(wkt_str)
            geom = validate_geometry(geom)
            return geom, f"uploaded_{uploaded_file.name}", ""
        except Exception as e:
            return None, "", str(e).strip() or "Invalid WKT in uploaded file."

    if source_type == "paste" and pasted_wkt.strip():
        try:
            geom = parse_wkt(pasted_wkt.strip())
            geom = validate_geometry(geom)
            return geom, "pasted_wkt", ""
        except Exception as e:
            return None, "", str(e).strip() or "Invalid WKT: check syntax (e.g. POLYGON((x y,...)))."

    return None, "", ""


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
    learned_alpha: float | None = None,
    n_sample: int | None = None,
    k_top: int | None = None,
    collision_weight: float | None = None,
    collision_max_area: float | None = None,
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
            learned_alpha=learned_alpha,
            n_sample=n_sample,
            k_top=k_top,
            collision_weight=collision_weight or COLLISION_WEIGHT,
            collision_max_area=collision_max_area if collision_max_area is not None else COLLISION_MAX_AREA,
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
        try:
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
                learned_alpha=learned_alpha if use_learned_ranking else None,
                model_artifact_path=model_path_str,
            )
        except TypeError:
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


# Session state: active_ref is the single source of truth. See run_registry.py "UI + State contract".
# Only All Results and Sidebar (Run demo) set active_ref; Run Again and Switch to run removed from sidebar.
# Demo, Debug, Export only read active_ref; they do not have their own run pickers.
if "demo_history" not in st.session_state:
    st.session_state["demo_history"] = []
if "active_report_dir" not in st.session_state:
    st.session_state["active_report_dir"] = None
if "active_ref" not in st.session_state:
    st.session_state["active_ref"] = None

DEMO_HISTORY_MAX = 5


def _set_active_run(path_str: str) -> None:
    """Set current run: active_report_dir, last_report_dir, and active_ref (single source of truth)."""
    st.session_state["active_report_dir"] = path_str
    st.session_state["last_report_dir"] = path_str
    ref = active_ref_from_path(path_str, _repo_root(), REPORTS_DIR)
    st.session_state["active_ref"] = ref if ref else {"kind": "single", "path": path_str, "case_id": None}

st.set_page_config(page_title="IntelliRiverLabel", layout="wide")

# Layout: sidebar and tabs at top (minimal gap). Selectors may need adjustment for Streamlit upgrades.
st.markdown("""
<style>
  /* Sidebar: IntelliRiverLabel as high as possible */
  [data-testid="stSidebar"] { padding-top: 0 !important; }
  [data-testid="stSidebar"] > div { padding-top: 0 !important; }
  section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
  [data-testid="stSidebar"] header { padding-top: 0 !important; margin-top: 0 !important; margin-bottom: 0.2rem !important; }
  [data-testid="stSidebar"] .stMarkdown { margin-bottom: 0.15rem !important; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { padding-top: 0.05rem !important; margin-bottom: 0.05rem !important; }
  [data-testid="stSidebar"] [data-testid="stExpander"] { margin-bottom: 0.2rem !important; }
  [data-testid="stSidebar"] hr { margin: 0.35rem 0 !important; }
  [data-testid="stSidebar"] .stCaptionContainer { margin-bottom: 0.1rem !important; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label { font-weight: 700 !important; font-size: 1.1rem !important; }
  /* Main: no gap above tab bar */
  [data-testid="stAppViewContainer"] { padding-top: 0 !important; }
  .main .block-container { padding-top: 0 !important; margin-top: 0 !important; }
  [data-testid="stTabs"] { margin-top: 0 !important; padding-top: 0 !important; }
  [data-testid="stTabs"] > div > div { margin-top: 0 !important; }
  [data-testid="stTabs"] button { font-size: 1.1rem !important; font-weight: 800 !important; padding: 0.45rem 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

repo_root = _repo_root()
default_geom_path = repo_root / DEFAULT_GEOMETRY_PATH
# Single scan for entire app; no tab rescans independently.
all_records = scan_reports(repo_root, REPORTS_DIR)
records_with_placement_list = records_with_placement(all_records)
records_placement_only = [r for r in records_with_placement_list if r.run_type != "eval"]

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
    if not default_geom_path.exists():
        st.caption("⚠️ Default geometry path not found; use upload or paste.")
    st.divider()

    # --- Geometry Source: same help style as Merge nearby / Label text (native Streamlit (?) ) ---
    source_type = st.radio(
        "Geometry Source",
        ["default", "upload", "paste"],
        format_func=lambda x: {"default": "Default", "upload": "Upload .wkt file", "paste": "Paste WKT"}[x],
        help=f"{TOOLTIP_WKT} Default file: {DEFAULT_GEOMETRY_PATH}",
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

    geom_for_run, geom_source, geom_error = _load_geometry_from_source(
        source_type,
        default_path=default_geom_path if source_type == "default" else None,
        uploaded_file=uploaded_file,
        pasted_wkt=pasted_wkt,
    )
    if geom_error:
        st.error(geom_error)
    components: list = []
    if geom_for_run is not None and extract_polygon_components is not None:
        components = extract_polygon_components(geom_for_run)
    _n_comp = len(components) if components else (describe_components(geom_for_run)["component_count"] if (geom_for_run and describe_components) else 0)

    merge_enabled = st.session_state.get("merge_enabled", False)
    merge_distance_pt = st.session_state.get("merge_distance_pt", 5.0)
    if merge_enabled and merge_nearby_components is not None and geom_for_run is not None:
        before_count = len(components) if components else (describe_components(geom_for_run)["component_count"] if describe_components else 0)
        geom_for_run = merge_nearby_components(geom_for_run, merge_distance_pt)
        if extract_polygon_components is not None:
            components = extract_polygon_components(geom_for_run)
        after_count = len(components) if components else (describe_components(geom_for_run)["component_count"] if describe_components else 0)
        if before_count != after_count or before_count > 1:
            st.caption(f"Components: before {before_count} → after {after_count}")

    component_choice = "Auto"
    if len(components) > 1:
        st.caption(f"Multi-part geometry: **{len(components)} parts** detected. Auto = largest by safe area after padding.")
        component_choice = st.selectbox(
            "Component",
            ["Auto"] + [str(i) for i in range(len(components))],
            help="Auto = best part by safe area; or pick part 0, 1, 2…",
        )

    # --- Inputs ---
    st.subheader("Inputs")
    label_text = st.text_input("Label text", value="ELBE", help="Text to place on the river.")
    font_size_pt = st.number_input("Font size (pt)", min_value=1.0, value=12.0, step=0.5)
    padding_pt = st.session_state.get("padding_pt", float(PADDING_PT))

    # --- Determinism ---
    st.subheader("Determinism")
    use_seed = st.checkbox(
        "Deterministic",
        value=True,
        help="Same seed = same result every time. Uncheck for random variation.",
    )
    seed_val = SEED or 42
    if use_seed:
        seed_val = st.number_input("Seed", min_value=0, value=max(0, seed_val), step=1, help="0 or positive. Same seed = same result; Preset does not change seed.")
    seed: int | None = seed_val if use_seed else None

    use_learned_ranking = False
    learned_alpha = 0.6
    if model_loaded:
        use_learned_ranking = st.checkbox("Use learned ranking", value=st.session_state.get("use_learned_ranking", False), key="use_learned_ranking", help=TOOLTIP_LEARNED)
        learned_alpha = st.slider(
            "Heuristic vs model blend (alpha)",
            0.3,
            0.9,
            0.6,
            0.1,
            help="Score = alpha × heuristic + (1−alpha) × model. Higher = more heuristic (safer).",
            key="learned_alpha_slider",
        )
    else:
        st.checkbox("Use learned ranking", value=False, disabled=True)
        st.caption("Train model to enable")
    # Preload model when learned ranking is checked so first run is not slow
    if use_learned_ranking and not model_loaded:
        try:
            from app.models.registry import load_model
            load_model()
        except Exception:
            pass

    # --- Output ---
    st.subheader("Output")
    _scale_opts = [1, 2, 4]
    _scale_val = st.session_state.get("render_scale_option", 2)
    _scale_idx = _scale_opts.index(_scale_val) if _scale_val in _scale_opts else 1
    render_scale_option = st.selectbox(
        "Render scale",
        _scale_opts,
        format_func=lambda x: f"{x}x",
        index=_scale_idx,
        key="render_scale_option",
        help=TOOLTIP_RENDER_SCALE,
    )
    run_name_raw = st.text_input(
        "Output folder name",
        value="demo_01",
        help="Reports subfolder (e.g. reports/demo_01/). Sanitized to safe characters.",
    )
    run_name = _sanitize_output_folder_name(run_name_raw)
    if run_name != (run_name_raw or "").strip():
        st.caption(f"Using: {run_name}")
    overwrite_output = st.checkbox(
        "Overwrite output folder",
        value=False,
        key="overwrite_output_folder",
        help="If checked, use the folder name as-is and overwrite existing files. If unchecked, a new folder (e.g. demo_01_2) is used when the name already exists.",
    )
    curved_mode = st.session_state.get("curved_mode", False)
    _reports_base = _repo_root() / REPORTS_DIR
    if not overwrite_output:
        _candidate = run_name
        _n = 2
        while (_reports_base / _candidate).exists():
            _candidate = f"{run_name}_{_n}"
            _n += 1
        if _candidate != run_name:
            st.caption(f"Folder **{run_name}** exists → using **{_candidate}**")
            run_name = _candidate
    # Advanced: candidate sampling and collision (applies to Run demo)
    n_sample_ui: int | None = None
    k_top_ui: int | None = None
    collision_weight_ui: float | None = None
    collision_max_area_ui: float | None = None
    with st.expander("Advanced (candidate sampling & collision)", expanded=False):
        use_advanced = st.checkbox("Override N sample / K top", value=False, key="use_advanced_sampling", help=TOOLTIP_OVERRIDE_NK)
        if use_advanced:
            n_sample_ui = st.number_input("N sample points", min_value=50, max_value=2000, value=int(N_SAMPLE_POINTS), step=50, key="n_sample_ui", help="More = slower but denser candidates.")
            k_top_ui = st.number_input("K top by clearance", min_value=10, max_value=200, value=int(K_TOP_CLEARANCE), step=10, key="k_top_ui", help="Candidates kept after sorting by clearance.")
        use_collision_override = st.checkbox("Override collision weight / max area", value=False, key="use_collision_override", help="For multi-label; penalty weight and max overlap area.")
        if use_collision_override:
            collision_weight_ui = st.number_input("Collision weight", min_value=0.0, value=float(COLLISION_WEIGHT), step=0.5, key="collision_weight_ui")
            collision_max_area_ui = st.number_input("Collision max area", min_value=0.0, value=float(COLLISION_MAX_AREA), step=0.1, key="collision_max_area_ui")
    run_clicked = st.button("Run demo", type="primary")
    # Run Again button hidden; trigger_run_again still used by All Results "Re-run with learned ranking"

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
                learned_alpha=learned_alpha if use_learned_ranking else None,
                n_sample=n_sample_ui if use_advanced else None,
                k_top=k_top_ui if use_advanced else None,
                collision_weight=collision_weight_ui if use_collision_override else None,
                collision_max_area=collision_max_area_ui if use_collision_override else None,
            )
            if report_dir is not None:
                _set_active_run(str(report_dir))
                st.session_state["show_demo_tab_banner"] = True  # prompt user to open Demo tab
                st.session_state["last_run_demo_report_dir"] = str(report_dir)  # so curved fallback message only shows for this run
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
                st.session_state["last_run_name"] = run_name
                st.session_state["last_curved_mode"] = curved_mode
                st.rerun()
            else:
                msg = user_message(error_key) if error_key else (st.session_state.pop("last_run_error", None) or "Check geometry and inputs.")
                st.error(msg)

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
        st.caption("**Deploy note:** The trained model is saved locally and is **not** pushed to GitHub (`app/models/artifacts/` is in .gitignore). On Streamlit Cloud, train after deploy or commit the model file if you need it there.")
        if st.button("Train model (synthetic)", help=TOOLTIP_TRAIN_SYNTHETIC):
            try:
                from app.models.train import train_model
                with st.spinner("Training..."):
                    train_model(n_polygons=100, seed=seed)
                st.success("Model trained and saved.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        st.caption(EVAL_SHORT)
        eval_run_name = st.text_input(
            "Evaluation run name",
            value="eval_01",
            key="eval_run_name",
            help="Reports subfolder for this evaluation run (e.g. reports/eval_01/).",
        )
        n_synthetic = st.number_input("N synthetic polygons", min_value=5, value=20, step=5, key="n_synthetic")
        eval_use_current_geom = st.checkbox(
            "Include current geometry",
            value=False,
            help="Add the currently loaded/pasted geometry as an extra test case (source: current_geometry).",
        )
        eval_seed = seed if use_seed else (SEED or 42)
        if st.button("Run evaluation", help=TOOLTIP_EVAL):
            try:
                from app.core.evaluate import run_evaluation
                current_geom = (geom_for_run if eval_use_current_geom else None)
                with st.spinner("Running evaluation..."):
                    report_dir = run_evaluation(
                        run_name=eval_run_name,
                        geometry_path=DEFAULT_GEOMETRY_PATH,
                        n_synthetic=n_synthetic,
                        seed=eval_seed,
                        repo_root=repo_root,
                        current_geometry=current_geom,
                    )
                st.session_state["last_eval_dir"] = str(report_dir)
                st.session_state["last_eval_run_name"] = eval_run_name
                st.success("Evaluation complete.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # --- Phase 2 (expander) ---
    def _on_preset_change():
        p = st.session_state.get("preset_sel", "None")
        if "High quality" in p:
            st.session_state["curved_mode"] = True
            st.session_state["use_learned_ranking"] = True
            st.session_state["render_scale_option"] = 2
        elif p == "Fast (straight only)":
            st.session_state["curved_mode"] = False
            st.session_state["use_learned_ranking"] = False
            st.session_state["render_scale_option"] = 1
        st.session_state["_preset_rerun"] = True

    preset_opts = ["None", "High quality (2× scale, curved, learned)", "Fast (straight only)"]
    with st.expander("Phase 2", expanded=False):
        st.subheader("Modes")
        st.number_input(
            "Padding (pt)", min_value=0.0, value=st.session_state.get("padding_pt", float(PADDING_PT)), step=0.5, key="padding_pt",
            help="Inward margin (pt) from river boundary; label must fit inside the shrunk polygon.",
        )
        st.checkbox("Merge nearby components (braided rivers)", value=st.session_state.get("merge_enabled", False), key="merge_enabled", help=TOOLTIP_MERGE_NEARBY)
        if st.session_state.get("merge_enabled", False):
            st.slider("Merge distance (pt)", 0.0, 20.0, value=st.session_state.get("merge_distance_pt", 5.0), step=0.5, key="merge_distance_pt")
        st.checkbox("Curved", value=st.session_state.get("curved_mode", False), key="curved_mode", help=TOOLTIP_CURVED)
        st.selectbox("Preset", preset_opts, key="preset_sel", on_change=_on_preset_change, help="Apply a preset to scale, curved, and learned options.")
        if st.session_state.pop("_preset_rerun", False):
            st.rerun()

    # --- Status ---
    st.divider()
    st.subheader("Status")
    # Current run = what Demo/Debug/Export tabs show (from active_ref).
    _status_repo = _repo_root()
    _status_ref = st.session_state.get("active_ref")
    if _status_ref is None and st.session_state.get("active_report_dir"):
        _status_ref = active_ref_from_path(st.session_state["active_report_dir"], _status_repo, REPORTS_DIR)
    _current_display_path = resolve_display_path(_status_ref, _status_repo, REPORTS_DIR) if _status_ref else None
    if _current_display_path:
        st.caption(f"Current run: **{_current_display_path.name}**")
    else:
        st.caption("No run selected")
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

    # --- Help & glossary ---
    with st.expander("Help & glossary"):
        st.markdown(GLOSSARY_MD)
        st.markdown("---")
        st.markdown("### Quick troubleshooting")
        st.markdown(QUICK_TROUBLESHOOT_MD)

# Resolve display path only from active_ref. One scan already done above (all_records, records_placement_only).
history = st.session_state.get("demo_history", [])
if st.session_state.get("active_ref") is None and st.session_state.get("active_report_dir"):
    st.session_state["active_ref"] = active_ref_from_path(st.session_state["active_report_dir"], repo_root, REPORTS_DIR)
if st.session_state.get("active_ref") is None and records_placement_only:
    _set_active_run(str(records_placement_only[0].path.resolve()))
display_path: Path | None = resolve_display_path(st.session_state.get("active_ref"), repo_root, REPORTS_DIR)
if display_path is not None:
    display_path = display_path.resolve()
effective_display_path: Path | None = display_path
if effective_display_path is None and records_placement_only:
    effective_display_path = records_placement_only[0].path.resolve()
is_fallback_display = display_path is None and effective_display_path is not None
active_dir = str(effective_display_path) if effective_display_path else (st.session_state.get("active_report_dir") or (history[0].get("report_dir") if history else None) or st.session_state.get("last_report_dir"))


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


def _compare_winner_reason(metrics_a: dict, metrics_b: dict) -> tuple[str | None, str]:
    """Return (winner letter or None, reason string)."""
    mode_rank = {"phase_b_curved": 3, "phase_a_straight": 2, "external_fallback": 1}
    ra = mode_rank.get(metrics_a.get("mode"), 0)
    rb = mode_rank.get(metrics_b.get("mode"), 0)
    if ra != rb:
        better = "A" if ra > rb else "B"
        reason = f"Better placement mode: {metrics_a.get('mode') or '—'} (A) vs {metrics_b.get('mode') or '—'} (B)."
        return better, reason
    ca = float(metrics_a.get("min_clearance_pt") or 0)
    cb = float(metrics_b.get("min_clearance_pt") or 0)
    if ca != cb:
        better = "A" if ca > cb else "B"
        reason = f"Higher min_clearance_pt: {ca} (Run {better}) vs {cb}."
        return better, reason
    fa = float(metrics_a.get("fit_margin_ratio") or 0)
    fb = float(metrics_b.get("fit_margin_ratio") or 0)
    better = "A" if fa >= fb else "B"
    reason = f"Higher or equal fit_margin_ratio: {fa} vs {fb}."
    return better, reason


# ----- Main: 5 tabs (Demo, Debug, All results, Compare, Evaluate) -----
tab_demo, tab_debug, tab_all_results, tab_compare, tab_evaluate = st.tabs(["Demo", "Debug", "All results", "Compare", "Evaluate"])

with tab_demo:
    if records_placement_only:
        _demo_current = str(effective_display_path.resolve()) if effective_display_path else ""
        _demo_idx = next((i for i, r in enumerate(records_placement_only) if str(r.path.resolve()) == _demo_current), 0)
        _demo_sel = st.selectbox(
            "Run",
            range(len(records_placement_only)),
            format_func=lambda i: records_placement_only[i].display_name if i < len(records_placement_only) else "",
            index=_demo_idx,
            key="demo_page_run_sel",
        )
        if 0 <= _demo_sel < len(records_placement_only) and str(records_placement_only[_demo_sel].path.resolve()) != _demo_current:
            _set_active_run(str(records_placement_only[_demo_sel].path.resolve()))
            st.rerun()
    if st.session_state.pop("show_demo_tab_banner", False):
        st.success("**Run complete.** Your result is shown below. Use the download buttons or **Download all** at the end.")
    full_width_images = st.toggle("Full-width images", value=False)
    if effective_display_path:
        report_dir = effective_display_path
        rec = get_record_by_path(report_dir, repo_root, REPORTS_DIR)
        run_meta = {}
        _meta_path = report_dir / "run_metadata.json"
        if _meta_path.exists():
            try:
                run_meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        learned = run_meta.get("use_learned_ranking", False)
        alpha = run_meta.get("learned_alpha")
        ai_badge = "Learned ranking: **ON**" + (f" (α={alpha})" if alpha is not None else "") if learned else "Learned ranking: **OFF**"
        st.markdown(f"<p style='text-align: center;'><strong>Run: {report_dir.name}</strong> — {ai_badge}</p>", unsafe_allow_html=True)
        placement_path = report_dir / "placement.json"
        mode_used = ""
        if placement_path.exists():
            data = json.loads(placement_path.read_text(encoding="utf-8"))
            mode_used = data.get("result", {}).get("mode", "—")
            # Only show curved fallback message when this run was just produced by Run demo, not when selecting a past run.
            _just_ran_this = str(report_dir.resolve()) == str(Path(st.session_state.get("last_run_demo_report_dir", "")).resolve())
            curved_attempted = mode_used == "phase_b_curved" or any("Phase B attempted but failed" in w for w in data.get("warnings", []))
            if _just_ran_this and curved_attempted and mode_used != "phase_b_curved":
                phase_b_reasons = [w.replace("Phase B attempted but failed: ", "") for w in data.get("warnings", []) if "Phase B attempted but failed" in w]
                reason = phase_b_reasons[0].strip() if phase_b_reasons else ""
                msg = "Curved attempted: fallback to straight." + (f" Reason: {reason}" if reason else "")
                st.warning(msg)
        ui_components.render_before_after_with_downloads(report_dir, full_width_images, "dl_demo")
        after_svg = report_dir / "after.svg"
        if after_svg.exists() and mode_used == "phase_b_curved":
            st.subheader("Curved (SVG)")
            ui_components.render_svg_viewer(after_svg)
            _col1, _col2, _col3 = st.columns([1, 1, 1])
            with _col2:
                st.download_button("Download after.svg", data=after_svg.read_bytes(), file_name="after.svg", mime="image/svg+xml", key="dl_demo_after_svg")
        st.divider()
        st.subheader("Download all")
        _zip_buf = io.BytesIO()
        with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
            for _name in ("placement.json", "run_metadata.json", "before.png", "after.png", "debug.png", "after.svg"):
                _p = report_dir / _name
                if _p.exists():
                    _zf.writestr(_name, _p.read_bytes())
        _zip_buf.seek(0)
        st.download_button(
            "Download all (zip)",
            data=_zip_buf.getvalue(),
            file_name=f"{report_dir.name}_export.zip",
            mime="application/zip",
            key="dl_demo_all_zip",
        )
        st.caption("Zip contains placement, metadata, and images for this run.")
    else:
        st.info("Run demo to see results.")

with tab_all_results:
    st.markdown("<p style='text-align: center;'><strong>All results</strong></p>", unsafe_allow_html=True)
    with st.expander("Help: types and re-processing"):
        st.markdown(
            "Runs live under **reports/**. Use the **Run** dropdown on the **Demo** page to switch runs. **Eval** runs open in the **Evaluate** tab only.\n\n"
            "**Run types:** **Demo/CLI** = single runs. **Batch** = many rivers. **Eval** = leaderboards and plots.\n\n"
            "**Re-run with learned ranking:** Expand on a run to see its geometry path; load in sidebar, enable Use learned ranking, then Run demo with a new name."
        )
    reports_root = repo_root / REPORTS_DIR
    if not reports_root.exists():
        st.info("No reports folder yet. Run a demo or batch to see results here.")
    else:
        top_level_all = [r for r in all_records if r.path.parent == reports_root]
        type_filter = st.selectbox("Filter by type", ["All", "Demo / CLI", "Batch", "Eval"], key="all_results_filter")
        if type_filter == "Eval":
            top_level_records = [r for r in top_level_all if r.run_type == "eval"]
        elif type_filter == "Batch":
            top_level_records = [r for r in top_level_all if r.run_type == "batch_root"]
        elif type_filter == "Demo / CLI":
            top_level_records = [r for r in top_level_all if r.run_type == "single"]
        else:
            top_level_records = [r for r in top_level_all if r.run_type != "eval"]
        if not top_level_records:
            st.info("No runs match the filter.")
        else:
            for i in range(0, len(top_level_records), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(top_level_records):
                        break
                    rec = top_level_records[idx]
                    with col:
                        if rec.thumbnail_path and rec.thumbnail_path.exists():
                            st.image(str(rec.thumbnail_path), width="stretch")
                        else:
                            st.caption("(no image)")
                        mode = (rec.metrics_summary or {}).get("mode") or ""
                        use_learned = rec.metadata.get("use_learned_ranking", False)
                        badges = []
                        if rec.run_type == "eval":
                            badges.append("Eval")
                        elif rec.run_type in ("batch_root", "batch_case"):
                            badges.append("Batch")
                        if mode == "phase_b_curved":
                            badges.append("Curved")
                        elif mode == "external_fallback":
                            badges.append("Fallback")
                        elif mode:
                            badges.append("Straight")
                        if use_learned:
                            badges.append("Learned")
                        if badges:
                            st.caption(" · ".join(badges))
                        if rec.run_type == "eval":
                            if st.button("Open in Evaluate tab", key=f"all_eval_{idx}"):
                                st.session_state["last_eval_dir"] = str(rec.path.resolve())
                                st.session_state["selected_eval_dir"] = str(rec.path.resolve())
                                st.success("Open the **Evaluate** tab to view leaderboards and plots.")
                                st.rerun()
                        else:
                            geom_path = rec.metadata.get("geometry_path", "")
                            if geom_path:
                                with st.expander("Re-run with learned ranking"):
                                    st.caption("Load this path in the sidebar, enable Use learned ranking, same seed, then Run demo with a new name.")
                                    st.code(geom_path, language=None)
                                    if st.button("Show path below", key=f"all_rerun_hint_{idx}"):
                                        st.session_state["rerun_geometry_path"] = geom_path
                                        st.rerun()
            if st.session_state.get("rerun_geometry_path"):
                st.info("Re-run hint: geometry path **" + st.session_state.get("rerun_geometry_path", "") + "** — load it in the sidebar and run with Use learned ranking.")
                if st.button("Clear re-run hint", key="clear_rerun_hint"):
                    st.session_state.pop("rerun_geometry_path", None)
                    st.rerun()

with tab_debug:
    if effective_display_path:
        report_dir = effective_display_path
        rec = get_record_by_path(report_dir, repo_root, REPORTS_DIR)
        st.markdown(f"<p style='text-align: center;'><strong>Run: {report_dir.name}</strong></p>", unsafe_allow_html=True)
        ui_components.render_debug_image(report_dir / "debug.png")
        placement_path = report_dir / "placement.json"
        if placement_path.exists():
            data = json.loads(placement_path.read_text(encoding="utf-8"))
            ui_components.render_warnings(data.get("warnings", []))
            # Input & criteria (all blocks that drove this run)
            st.subheader("Input & criteria")
            label_block = data.get("label", {})
            input_block = data.get("input", {})
            meta_path = report_dir / "run_metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            crit = []
            if label_block:
                crit.append(("Label", f"text={label_block.get('text', '')}, font_size_pt={label_block.get('font_size_pt')}, font_family={label_block.get('font_family', '')}"))
            if input_block:
                crit.append(("Input", f"geometry_source={input_block.get('geometry_source', '')}, units={input_block.get('units', '')}"))
            for k in ("run_name", "geometry_path", "label_text", "font_size_pt", "padding_pt", "seed", "curved_mode", "use_learned_ranking", "render_scale"):
                if k in meta and meta[k] is not None:
                    crit.append((k, str(meta[k])))
            for label, val in crit:
                st.caption(f"**{label}:** {val}")
            if meta.get("config"):
                with st.expander("Config (weights & constants)"):
                    st.json(meta["config"])
            st.subheader("Metrics")
            metrics = dict(data.get("metrics", {}))
            result = data.get("result", {})
            if result.get("mode") is not None:
                metrics["mode"] = result["mode"]
            if result.get("confidence") is not None:
                metrics["confidence"] = result["confidence"]
                metrics["score"] = result["confidence"]  # placement score (heuristic; 0–1)
            ui_components.render_metrics(metrics)
            with st.expander("Result JSON"):
                st.json(result)
    else:
        st.info("Run a demo first to see debug output. Debug shows candidate points and placement overlay for the selected run.")

with tab_compare:
    # Use registry only; placement runs (single + batch_case), no eval.
    compare_records = records_placement_only
    if len(compare_records) < 2:
        st.info("Need at least 2 placement runs to compare. Use **Use as A** / **Use as B** in **All results**, or run more demos.")
    compare_paths = [str(r.path.resolve()) for r in compare_records]
    path_a = st.session_state.get("compare_path_a", "")
    path_b = st.session_state.get("compare_path_b", "")
    idx_a = next((i for i, r in enumerate(compare_records) if str(r.path.resolve()) == path_a), 0)
    idx_b = next((i for i, r in enumerate(compare_records) if str(r.path.resolve()) == path_b), 1 if len(compare_records) > 1 else 0)
    sel_a = idx_a
    sel_b = idx_b
    if compare_records:
        sel_a = st.selectbox(
            "Run A",
            range(len(compare_records)),
            format_func=lambda i: compare_records[i].display_name if i < len(compare_records) else "",
            index=idx_a,
            key="compare_sel_a",
        )
        sel_b = st.selectbox(
            "Run B",
            range(len(compare_records)),
            format_func=lambda i: compare_records[i].display_name if i < len(compare_records) else "",
            index=idx_b,
            key="compare_sel_b",
        )
        st.session_state["compare_path_a"] = str(compare_records[sel_a].path.resolve())
        st.session_state["compare_path_b"] = str(compare_records[sel_b].path.resolve())
    dir_a = compare_records[sel_a].path.resolve() if compare_records and 0 <= sel_a < len(compare_records) else None
    dir_b = compare_records[sel_b].path.resolve() if compare_records and 0 <= sel_b < len(compare_records) else None
    if dir_a and not dir_a.exists():
        dir_a = None
    if dir_b and not dir_b.exists():
        dir_b = None
    label_a = compare_records[sel_a].display_name if compare_records and 0 <= sel_a < len(compare_records) else (dir_a.name if dir_a else "")
    label_b = compare_records[sel_b].display_name if compare_records and 0 <= sel_b < len(compare_records) else (dir_b.name if dir_b else "")
    if dir_a and dir_b and dir_a.exists() and dir_b.exists():
        # For batch runs, compare at batch level (index.csv); for single-run folders use the folder itself.
        metrics_a = _placement_metrics(dir_a)
        metrics_b = _placement_metrics(dir_b)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"<p style='text-align: center;'><strong>{label_a}</strong></p>", unsafe_allow_html=True)
            after_a = dir_a / "after.png"
            if not after_a.exists() and (dir_a / "cases").is_dir():
                first_case = next((dir_a / "cases").iterdir(), None)
                if first_case and first_case.is_dir():
                    after_a = first_case / "after.png"
            if after_a.exists():
                st.image(str(after_a), width="stretch")
            else:
                st.caption("after.png missing")
            try:
                rel_a = dir_a.relative_to(repo_root)
            except ValueError:
                rel_a = dir_a
            st.markdown(f"<p style='text-align: center;'>Loc: {rel_a}</p>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"<p style='text-align: center;'><strong>{label_b}</strong></p>", unsafe_allow_html=True)
            after_b = dir_b / "after.png"
            if not after_b.exists() and (dir_b / "cases").is_dir():
                first_case = next((dir_b / "cases").iterdir(), None)
                if first_case and first_case.is_dir():
                    after_b = first_case / "after.png"
            if after_b.exists():
                st.image(str(after_b), width="stretch")
            else:
                st.caption("after.png missing")
            try:
                rel_b = dir_b.relative_to(repo_root)
            except ValueError:
                rel_b = dir_b
            st.markdown(f"<p style='text-align: center;'>Loc: {rel_b}</p>", unsafe_allow_html=True)
        winner, reason = _compare_winner_reason(metrics_a, metrics_b) if metrics_a and metrics_b else (None, "")
        if winner:
            st.markdown(f"**Winner (heuristic):** Run {winner}. {reason}")
        st.subheader("Metrics")
        priority_keys = ["mode", "min_clearance_pt", "fit_margin_ratio", "confidence", "curvature_total_deg", "straightness_ratio", "success_count", "n_labels", "collisions_detected"]
        keys = [k for k in priority_keys if metrics_a.get(k) is not None or metrics_b.get(k) is not None]
        extra = sorted(set(metrics_a.keys()) | set(metrics_b.keys()) - set(keys))
        keys = keys + extra
        if not keys:
            keys = ["min_clearance_pt", "fit_margin_ratio", "mode"]
        def _str(v):
            return "" if v is None else str(v)
        rows = [{"Metric": k, "Run A": _str(metrics_a.get(k)), "Run B": _str(metrics_b.get(k))} for k in keys]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        if not (dir_a and dir_b):
            st.info("Select two runs from the dropdowns above.")

with tab_evaluate:
    eval_records = [r for r in all_records if r.run_type == "eval"]
    eval_dir = None
    _eval_dir_str = st.session_state.get("selected_eval_dir") or st.session_state.get("last_eval_dir")
    if _eval_dir_str and Path(_eval_dir_str).exists():
        eval_dir = Path(_eval_dir_str).resolve()
    if eval_records and not eval_dir and eval_records[0].path.exists():
        eval_dir = eval_records[0].path.resolve()
    if eval_records and eval_dir:
        _idx = next((i for i, r in enumerate(eval_records) if r.path.resolve() == eval_dir), 0)
        _sel = st.selectbox(
            "Evaluation run",
            range(len(eval_records)),
            format_func=lambda i: eval_records[i].display_name if i < len(eval_records) else "",
            index=_idx,
            key="eval_run_sel",
        )
        if 0 <= _sel < len(eval_records):
            st.session_state["selected_eval_dir"] = str(eval_records[_sel].path.resolve())
            eval_dir = eval_records[_sel].path.resolve()
    if eval_dir and eval_dir.exists():
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
            by_method = lb.get("by_method") or {}
            if by_method:
                rows_method = [{"method": k, **{kk: (str(vv) if vv is not None else "") for kk, vv in v.items()}} for k, v in by_method.items()]
                st.dataframe(rows_method, width="stretch", hide_index=True)
            if "heuristic_plus_model" in by_method and "heuristic_only" in by_method:
                st.subheader("AI vs heuristic-only (Δ)")
                h = by_method["heuristic_only"]
                m = by_method["heuristic_plus_model"]
                delta_sr = (m.get("success_rate") or 0) - (h.get("success_rate") or 0)
                delta_cl = (m.get("avg_clearance") or 0) - (h.get("avg_clearance") or 0)
                _delta_rows = [
                    {"Metric": "Success rate Δ", "heuristic_only": f"{(h.get('success_rate') or 0):.2%}", "heuristic_plus_model": f"{(m.get('success_rate') or 0):.2%}", "Δ": f"{delta_sr:+.2%}"},
                    {"Metric": "Avg clearance Δ", "heuristic_only": f"{h.get('avg_clearance', 0):.2f}", "heuristic_plus_model": f"{m.get('avg_clearance', 0):.2f}", "Δ": f"{delta_cl:+.2f}"},
                ]
                st.dataframe(_delta_rows, width="stretch", hide_index=True)
                if delta_sr > 0 or delta_cl > 0:
                    st.caption("AI-assisted ranking improves over heuristic-only on this evaluation suite.")
                else:
                    st.caption("AI-assisted ranking integrated with safety blending; improvements vary by suite or case.")
            if lb.get("by_family"):
                st.subheader("Per-family success")
                rows_family = [{"family": str(k), "success_count": str(v.get("success_count", 0)), "n": str(v.get("n", 0)), "success_rate": f"{(v.get('success_count', 0) / v.get('n', 1) * 100):.1f}%" if v.get("n") else "—"} for k, v in lb["by_family"].items()]
                st.dataframe(rows_family, width="stretch", hide_index=True)
        elif lb_csv.exists():
            st.subheader("Leaderboard")
            import csv as _csv
            with open(lb_csv, newline="", encoding="utf-8") as _f:
                _rows = list(_csv.DictReader(_f))
            _rows_str = [{k: str(v) for k, v in r.items()} for r in _rows]
            st.dataframe(_rows_str, width="stretch", hide_index=True)
        plots_dir = eval_dir / "plots"
        if plots_dir.exists():
            for name, caption in [
                ("success_rate.png", "Success rate by method"),
                ("min_clearance_pt.png", "min_clearance_pt by method"),
                ("success_by_family.png", "Success rate by family"),
                ("collision_rate_by_method.png", "Collision rate by method"),
                ("min_clearance_by_method_family.png", "min_clearance_pt by method and family"),
            ]:
                if (plots_dir / name).exists():
                    st.image(str(plots_dir / name), caption=caption, width="stretch")
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
        # Download all evaluation as zip
        def _make_eval_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in ["evaluation_summary.json", "evaluation_results.csv", "leaderboard.json", "leaderboard.csv"]:
                    p = eval_dir / name
                    if p.exists():
                        zf.writestr(name, p.read_bytes())
                if (eval_dir / "plots").exists():
                    for f in (eval_dir / "plots").iterdir():
                        if f.is_file():
                            zf.writestr(f"plots/{f.name}", f.read_bytes())
            buf.seek(0)
            return buf.getvalue()

        if summary_path.exists() or csv_path.exists() or lb_json.exists():
            st.download_button(
                "Download all evaluation (zip)",
                data=_make_eval_zip(),
                file_name=f"evaluation_{eval_dir.name}.zip",
                mime="application/zip",
                key="dl_eval_all_zip",
            )
    else:
        st.info("No evaluation run yet. Run evaluation to see leaderboards, plots, and per-method metrics here.")
        st.markdown("**How to run:** Open the **sidebar** → **AI & Evaluation** → set *Evaluation run name* and *N synthetic polygons* → click **Run evaluation**.")
