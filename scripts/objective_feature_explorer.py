"""DOAR -- Objective Feature Explorer (standalone demo, feature/supervisor-demo-v2).

Proves that DOAR's objective/formal drawing-feature layer
(`src/doar/formal_features.py`, Phase G0 -- Appendix G) works completely
independently of the rest of the system: no AI model call, no rule
engine, no psychological interpretation, no evidence-aggregation/concern
scoring, no internet access.

This script imports ONLY:
  - `doar.formal_features`      (the objective/formal CV measurement layer)
  - `doar.case_artifacts`       (a tiny path-resolution helper, no logic)
It deliberately does NOT import (directly or transitively) any of:
  doar.rule_engine_v2, doar.case_interpretation, doar.human_interaction
  (Ask DOAR), doar.chat, doar.case_presentation, doar.drawing_synthesis,
  any Gemini/google.generativeai client, or the concern/evidence
  aggregation pipeline. See tests/test_objective_feature_explorer.py's
  `ImportGuardTests` for a structural check of this claim.

Run:
    .venv\\Scripts\\python.exe -m streamlit run scripts\\objective_feature_explorer.py

Nothing here writes to, or reads from, the main app's case directories'
mutable review/rule artifacts -- it only reads each demo case's already
-computed `analysis.json` (produced once, earlier, by the real
segmentation/colour pipeline) and the drawing image itself, then calls
the SAME `compute_formal_features` the main app's "Formal/graphic
measurements" button calls. No new feature, no new algorithm, no
psychological claim is introduced by this file.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The ONLY DOAR imports this script makes: the objective/formal feature
# layer itself, plus the tiny case-relative-path resolver it needs to
# open a saved case's artifacts (foreground mask etc.) portably. Neither
# module imports the rule engine, Ask DOAR, Gemini, or evidence
# aggregation -- see the module docstring above and the import-guard test.
from doar import formal_features as ff  # noqa: E402
from doar.case_artifacts import resolve_analysis_artifacts  # noqa: E402
from doar.features import FeatureValue  # noqa: E402  (dataclass shape only)

CASES_DIR = ROOT / "outputs" / "prototype_cases"
DEFINITION_TABLE_PATH = ROOT / "FORMAL_FEATURE_DEFINITION_TABLE.csv"

# ---------------------------------------------------------------------------
# The 3 selected examples (development/train+validation material only --
# every basename below was checked against
# experiments/E1_visual_representation/raw/e1a_test_paths_LOCKED_DO_NOT_USE.csv
# and confirmed NOT part of the Locked Test split; see
# FEATURE_EXPLORER_GUIDE.md for the full verification note). Reused from
# the existing SUPERVISOR_DEMO_CASES set already vetted in a prior
# session (doar_prototype_app.py) -- picked here purely for visual/
# technical suitability (plausible feature values, foreground/QC not
# obviously failing), never by psychological label.
# ---------------------------------------------------------------------------
EXAMPLES: dict[str, dict] = {
    "Example Drawing 1": {
        "case_id": "e2e_check_1786239075",
        "kind": "Simple line/pencil drawing",
        "why": "Sparse, mostly monochrome linework (colour.diversity = 1, low chromatic coverage) -- "
               "the clearest 'line drawing' of the three candidates.",
    },
    "Example Drawing 2": {
        "case_id": "a111_1787479358",
        "kind": "Colourful drawing",
        "why": "Highest chromatic coverage and colour diversity of the three candidates (a saturated, "
               "multi-colour painted drawing) -- the clearest 'colourful' example.",
    },
    "Example Drawing 3": {
        "case_id": "h38_1786305027",
        "kind": "Complex / detailed drawing",
        "why": "Many distinct drawn objects (vehicle, figures, sky elements, multiple small repeated "
               "shapes) -- highest direction-cluster count and orientation entropy of the three "
               "candidates, consistent with a visually busier scene.",
    },
}

# ---------------------------------------------------------------------------
# Human-readable grouping (matches src/doar/formal_features.py's own
# `compute_formal_features` output keys exactly -- nothing invented here).
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "1. Lines & Stroke Quality": [
        "line.mean_width",
        "line.width_variability",
        "line.darkness_mean",
        "line.darkness_variability",
        "line.continuity",
        "line.fragmentation_rate",
        "line.straightness",
    ],
    "2. Direction & Organization": [
        "line.dominant_orientation_code",
        "line.orientation_entropy",
        "line.orientation_coherence",
        "line.direction_cluster_count",
        "stroke.junction_corner_density_proxy",
        "stroke.density_mean",
        "stroke.scribble_candidate_score",
    ],
    "3. Colour": [
        "colour.chromatic_coverage",
        "colour.diversity",
        "colour.brightness",
        "colour.saturation",
    ],
    "4. Spatial / Global Structure": [
        "fill.global_spatial_density_uniformity",
        "symmetry.bilateral",
        "symmetry.spacing_regularity",
        "scene.detail_density",
    ],
}
ALL_FEATURE_IDS = [fid for group in FEATURE_GROUPS.values() for fid in group]

# Fallback human labels, used only if FORMAL_FEATURE_DEFINITION_TABLE.csv
# is unavailable for some reason -- the CSV's own `display_name` column
# (read below) is the primary source, per the task's own instruction to
# treat the CSV as a wording reference, not a second source of truth for
# behaviour.
FALLBACK_LABELS: dict[str, str] = {
    "line.mean_width": "Mean apparent line width",
    "line.width_variability": "Line width variability",
    "line.darkness_mean": "Line darkness (mean)",
    "line.darkness_variability": "Darkness variability",
    "line.continuity": "Line continuity",
    "line.fragmentation_rate": "Fragmentation rate",
    "line.straightness": "Straightness",
    "line.dominant_orientation_code": "Dominant orientation",
    "line.orientation_entropy": "Orientation entropy",
    "line.orientation_coherence": "Orientation coherence",
    "line.direction_cluster_count": "Direction cluster count",
    "stroke.junction_corner_density_proxy": "Junction/corner density (proxy)",
    "stroke.density_mean": "Stroke density",
    "stroke.scribble_candidate_score": "Scribble candidate score",
    "colour.chromatic_coverage": "Chromatic colour coverage",
    "colour.diversity": "Colour diversity",
    "colour.brightness": "Brightness",
    "colour.saturation": "Saturation",
    "fill.global_spatial_density_uniformity": "Global spatial density uniformity",
    "symmetry.bilateral": "Bilateral symmetry",
    "symmetry.spacing_regularity": "Spacing regularity",
    "scene.detail_density": "Scene/detail density",
}

# Which internal formal_features.py helper actually produces each value --
# read directly off compute_formal_features's body, for the "Technical
# details" panel only (documentation, not a second computation).
PRODUCING_FUNCTION: dict[str, str] = {
    "line.mean_width": "_line_width_stats",
    "line.width_variability": "_line_width_stats",
    "line.darkness_mean": "_darkness_stats",
    "line.darkness_variability": "_darkness_stats",
    "line.continuity": "_continuity_and_fragmentation",
    "line.fragmentation_rate": "_continuity_and_fragmentation",
    "line.straightness": "_straightness",
    "line.dominant_orientation_code": "_orientation_stats",
    "line.orientation_entropy": "_orientation_stats",
    "line.orientation_coherence": "_orientation_stats",
    "line.direction_cluster_count": "_orientation_stats",
    "stroke.junction_corner_density_proxy": "_junction_corner_density_proxy",
    "stroke.density_mean": "compute_formal_features (mask.mean())",
    "stroke.scribble_candidate_score": "compute_formal_features (density_mean * orientation_entropy)",
    "colour.chromatic_coverage": "_chromatic_colour_coverage",
    "colour.diversity": "reused from analysis.py's colour_diversity (not recomputed)",
    "colour.brightness": "compute_formal_features (gray.mean() / 255)",
    "colour.saturation": "compute_formal_features (mean HSV saturation over foreground)",
    "fill.global_spatial_density_uniformity": "_fill_uniformity",
    "symmetry.bilateral": "_bilateral_symmetry",
    "symmetry.spacing_regularity": "_spacing_regularity",
    "scene.detail_density": "compute_formal_features (Canny edge density over foreground)",
}

METHOD_EXPLANATIONS = [
    ("Line width & darkness",
     "Line width is a deterministic stroke-width measurement: the mask is separated into "
     "stroke-like vs. filled regions, and width is read off a distance transform over the "
     "stroke-like pixels only. Darkness is pixel intensity (255 - greyscale value) relative to "
     "the drawing's own foreground/background representation -- an appearance proxy, never a "
     "measurement of physical pencil pressure."),
    ("Orientation",
     "Stroke/edge directions are detected with Sobel gradients (rotated 90 degrees from the "
     "gradient's own edge-normal direction to the stroke's tangent direction) and summarized as "
     "a dominant direction, an entropy value (how spread across directions), and a structure-"
     "tensor coherence value (how locally consistent nearby strokes are)."),
    ("Colour",
     "Chromatic coverage, colour diversity, brightness, and saturation are computed from the "
     "image's own RGB/HSV pixel values over the foreground mask (or the whole image, for "
     "brightness) -- fixed, documented thresholds, no learned model."),
    ("Composition / spatial structure",
     "Global spatial density uniformity and detail density describe how evenly marks occupy an "
     "8x8 grid of the page and how much fine edge detail is present -- spatial occupancy "
     "statistics, not an object count."),
    ("Symmetry",
     "Bilateral symmetry compares the foreground mask against its own horizontal mirror image; "
     "spacing regularity compares the distances between separate connected components. Both are "
     "measures of visual similarity/structure, not a claim about drawing intent or meaning."),
]


# ---------------------------------------------------------------------------
# Data loading (cached -- the CV computation is the same deterministic
# work the main app's "Compute formal/graphic measurements" button does).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_definition_table() -> dict[str, dict]:
    """feature_id -> {display_name, exact_definition, ...}, from the
    committed FORMAL_FEATURE_DEFINITION_TABLE.csv -- used only for human
    wording/method description, never as a source of numeric values."""
    table: dict[str, dict] = {}
    if DEFINITION_TABLE_PATH.exists():
        with DEFINITION_TABLE_PATH.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                table[row["feature_id"]] = row
    return table


@st.cache_data(show_spinner="Computing objective drawing features...")
def _load_case(case_id: str) -> dict:
    case_dir = CASES_DIR / case_id
    analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    image_path = case_dir / Path(analysis["image_path"]).name
    resolved = resolve_analysis_artifacts(analysis, case_dir)
    features_row = ff.compute_formal_features(str(image_path), resolved)
    features_serialized = {name: fv_asdict(fv) for name, fv in features_row.items()}
    return {
        "case_id": case_id,
        "image_path": str(image_path),
        "analysis": analysis,
        "resolved_analysis": resolved,
        "features": features_serialized,
    }


def fv_asdict(fv: FeatureValue) -> dict:
    return {
        "value": fv.value, "confidence": fv.confidence, "missing": fv.missing,
        "method": fv.method, "evidence_id": fv.evidence_id,
    }


@dataclass
class QCInfo:
    """Only fields that formal_features.py's own inputs (analysis.json's
    `segmentation`/`quality`/`composition` sections, produced by the real
    segmentation pipeline) actually provide -- nothing invented here."""
    status: str | None
    confidence: float | None
    foreground_ratio: float | None
    selected_strategy: str | None
    candidate_disagreement: float | None
    background_stability: float | None
    background_rgb: list | None
    quality_status: str | None
    is_reliable: bool


def _build_qc_info(analysis: dict) -> QCInfo:
    seg = analysis.get("segmentation", {})
    comp = analysis.get("composition", {})
    quality = analysis.get("quality", {})
    status = seg.get("status")
    quality_status = quality.get("quality_status")
    # Same rule the pipeline itself uses to set segmentation.status
    # ("verified" if confidence >= 0.6 else "uncertain" -- src/doar/
    # analysis.py::analyze_image) -- displayed here, not reinvented.
    is_reliable = (status == "verified") and (quality_status == "supported")
    return QCInfo(
        status=status,
        confidence=seg.get("confidence"),
        foreground_ratio=comp.get("foreground_coverage"),
        selected_strategy=seg.get("selected_strategy"),
        candidate_disagreement=seg.get("candidate_disagreement"),
        background_stability=seg.get("background_stability"),
        background_rgb=seg.get("background_rgb"),
        quality_status=quality_status,
        is_reliable=is_reliable,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _format_value(fid: str, fv: dict) -> str:
    if fv["missing"]:
        return "Not measurable for this image"
    value = fv["value"]
    if fid == "line.dominant_orientation_code":
        code_names = {0.0: "Horizontal", 1.0: "Vertical", 2.0: "Diagonal"}
        return code_names.get(round(value), "Unknown")
    if fid in ("stroke.junction_corner_density_proxy",):
        return f"{value:.4f}"
    if fid in ("line.direction_cluster_count",):
        return f"{value:.0f}"
    return f"{value:.3f}"


def render_qc_panel(qc: QCInfo) -> None:
    st.subheader("Measurement Quality / QC")
    if qc.status is None:
        st.info("No segmentation/QC information is available for this case.")
        return

    if not qc.is_reliable:
        st.warning(
            "**Measurement quality warning.** DOAR's own segmentation step flagged this image's "
            f"foreground/background separation as **{qc.status}**"
            + (f" and the image quality gate as **{qc.quality_status}**"
               if qc.quality_status != "supported" else "")
            + f" (segmentation confidence {qc.confidence:.2f}). Preprocessing may be less reliable "
            "for this image, so the feature values below should be treated with more caution than "
            "for a confidently segmented drawing."
        )
    else:
        st.success(
            f"Foreground/background separation was **verified** (confidence {qc.confidence:.2f}); "
            "the image quality gate reported **supported**."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Foreground/mask status", qc.status or "n/a")
        st.metric("Foreground ratio", f"{qc.foreground_ratio:.3f}" if qc.foreground_ratio is not None else "n/a")
        st.metric("Segmentation confidence", f"{qc.confidence:.2f}" if qc.confidence is not None else "n/a")
    with c2:
        st.metric("Segmentation strategy used", qc.selected_strategy or "n/a")
        st.metric("Candidate-mask disagreement", f"{qc.candidate_disagreement:.3f}"
                   if qc.candidate_disagreement is not None else "n/a")
        st.metric("Background stability", f"{qc.background_stability:.3f}"
                   if qc.background_stability is not None else "n/a")
    if qc.background_rgb:
        st.caption(f"Detected background/support colour (RGB): {tuple(qc.background_rgb)}")


def render_feature_groups(features: dict, definitions: dict) -> None:
    st.subheader("Feature Results")
    tabs = st.tabs(list(FEATURE_GROUPS.keys()))
    for tab, (group_name, feature_ids) in zip(tabs, FEATURE_GROUPS.items()):
        with tab:
            cols = st.columns(2)
            for i, fid in enumerate(feature_ids):
                fv = features[fid]
                label = definitions.get(fid, {}).get("display_name") or FALLBACK_LABELS.get(fid, fid)
                with cols[i % 2]:
                    st.metric(label, _format_value(fid, fv))
                    higher = definitions.get(fid, {}).get("higher_value_means")
                    if higher:
                        st.caption(f"Higher = {higher.lower()}")

    with st.expander("Technical details (raw feature IDs, values, and method descriptions)", expanded=False):
        rows = []
        for fid in ALL_FEATURE_IDS:
            fv = features[fid]
            defn = definitions.get(fid, {})
            rows.append({
                "feature_id": fid,
                "human_label": defn.get("display_name") or FALLBACK_LABELS.get(fid, fid),
                "raw_value": round(fv["value"], 6),
                "missing": fv["missing"],
                "confidence": round(fv["confidence"], 3),
                "producing_module": "src/doar/formal_features.py",
                "producing_function": PRODUCING_FUNCTION.get(fid, "?"),
                "method_description": defn.get("exact_definition", ""),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_visual_aids(case_data: dict) -> None:
    st.subheader("Visual Aids")
    analysis = case_data["analysis"]
    resolved = case_data["resolved_analysis"]
    features = case_data["features"]

    aid_cols = st.columns(2)

    # 1) Foreground/mask preview -- the extractor already produces this
    # exact file (analysis["artifacts"]["foreground_mask"]).
    with aid_cols[0]:
        mask_path = resolved.get("artifacts", {}).get("foreground_mask")
        if mask_path and Path(mask_path).exists():
            st.image(mask_path, caption="Foreground mask used for every measurement above", width="stretch")
        else:
            st.caption("No foreground mask artifact available for this case.")

    # 2) Orientation summary -- built only from the summary values the
    # extractor already returns (entropy, coherence, cluster count),
    # never from a re-derived pixel histogram.
    with aid_cols[1]:
        entropy = features["line.orientation_entropy"]["value"]
        coherence = features["line.orientation_coherence"]["value"]
        clusters_norm = features["line.direction_cluster_count"]["value"] / 18.0
        orientation_series = pd.Series(
            {"Orientation entropy": entropy, "Orientation coherence": coherence,
             "Direction clusters (normalized)": clusters_norm})
        st.bar_chart(orientation_series)
        st.caption("Summary of the already-computed orientation statistics (0-1 scale).")

    aid_cols2 = st.columns(2)

    # 3) Colour palette preview -- reuses analysis["colour"]["colour_proportions"],
    # already computed by analysis.py, not recomputed here.
    with aid_cols2[0]:
        proportions = analysis.get("colour", {}).get("colour_proportions")
        if proportions:
            st.bar_chart(pd.Series(proportions))
            st.caption("Colour-bin proportions already computed by DOAR's colour analysis.")
        else:
            st.caption("No colour-proportion data available for this case.")

    # 4) Left/right split for bilateral symmetry -- pure crop/mirror of
    # the existing image, no new CV algorithm.
    with aid_cols2[1]:
        try:
            img = Image.open(case_data["image_path"]).convert("RGB")
            w, h = img.size
            left_half = img.crop((0, 0, w // 2, h))
            right_half = img.crop((w // 2, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)
            split = Image.new("RGB", (left_half.width + right_half.width, h))
            split.paste(left_half, (0, 0))
            split.paste(right_half, (left_half.width, 0))
            symmetry_value = features["symmetry.bilateral"]["value"]
            st.image(split, caption=f"Left half | mirrored right half (bilateral symmetry = {symmetry_value:.3f})",
                      width="stretch")
        except Exception:  # noqa: BLE001 -- presentation-only, never fatal
            st.caption("Left/right symmetry split could not be rendered for this image.")


def render_method_explanation() -> None:
    with st.expander("How are these measurements calculated?", expanded=False):
        st.markdown(
            "This layer is intentionally deterministic and interpretable. Instead of asking an AI "
            "whether the drawing looks chaotic or symmetric, it quantifies properties such as line "
            "continuity, orientation entropy, colour diversity, and bilateral symmetry using plain "
            "computer-vision statistics -- the same code path for every image, every time."
        )
        for title, text in METHOD_EXPLANATIONS:
            st.markdown(f"**{title}.** {text}")
        st.caption(
            "These measurements remain objective until their psychological relevance is separately "
            "validated. None of them claims physical pencil pressure or psychological meaning."
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="DOAR - Objective Feature Explorer", layout="wide")

    st.title("DOAR — Objective Feature Explorer")
    st.caption("Independent validation of measurable drawing properties")
    st.info(
        "These measurements describe visible properties of the drawing. They do not automatically "
        "carry psychological meaning."
    )

    example_name = st.radio(
        "Choose an example drawing", list(EXAMPLES.keys()),
        horizontal=True, key="ofe_example_picker",
    )
    example = EXAMPLES[example_name]
    st.caption(f"{example['kind']} -- {example['why']}")

    case_data = _load_case(example["case_id"])
    definitions = _load_definition_table()
    qc = _build_qc_info(case_data["analysis"])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Drawing")
        st.image(case_data["image_path"], width="stretch")
    with right:
        render_qc_panel(qc)

    st.divider()
    render_feature_groups(case_data["features"], definitions)

    st.divider()
    render_visual_aids(case_data)

    st.divider()
    render_method_explanation()


if __name__ == "__main__":
    main()
else:
    # Streamlit's `streamlit run` executes the module as __main__, but
    # AppTest.from_file also runs the module top-level as __main__ via
    # exec, so this branch is effectively unused in practice; kept only
    # so a plain `import` of this module (e.g. from a future test) never
    # renders anything by side effect.
    pass
