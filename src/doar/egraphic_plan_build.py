"""Phase G0.2, Step 8/10/11 -- builds E_GRAPHIC_EXPERIMENT_PLAN.json: a
RESEARCH-ONLY planning file for the next (not-yet-started) E-Graphic
experiment phase. No experiment is run here, no method is chosen as a
winner -- every `status` is "PLANNED_NOT_STARTED" or
"CURRENT_BASELINE_ONLY", and no new dependency (SAM2/DeepLSD/SOLD2/
sketch-vectorization) is installed or imported anywhere in this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
E_GRAPHIC_EXPERIMENT_PLAN_PATH = ROOT / "E_GRAPHIC_EXPERIMENT_PLAN.json"
E_GRAPHIC_PLAN_SCHEMA_VERSION = "e_graphic_experiment_plan_v1"


def _task(**kwargs: Any) -> dict[str, Any]:
    task = {
        "experiment_id": None, "feature_group": None, "target_observable": None,
        "current_method": None, "candidate_methods_to_compare_later": [],
        "annotation_required": True, "ground_truth_type": None, "suggested_metric": None,
        "requires_segmentation": False, "requires_new_dependency": None, "status": "PLANNED_NOT_STARTED",
    }
    task.update(kwargs)
    return task


def build_egraphic_experiment_plan() -> dict[str, Any]:
    tasks = [
        # ---- GROUP A: LINE WIDTH / LINE QUALITY ---------------------------
        _task(
            experiment_id="EG_A1_LINE_WIDTH", feature_group="A_LINE_WIDTH_QUALITY",
            target_observable="line.mean_width",
            current_method="CURRENT_BASELINE_ONLY: connected-component morphological granulometry "
                            "(open with kernel radius = 2% of min(H,W); a component fully erased is "
                            "stroke-like, kept whole) + Euclidean distance transform over stroke-like pixels "
                            "(mean_width = 2*mean(distance)). See G0.1 Step 4A / line.mean_width limitation "
                            "(Step 11 below).",
            candidate_methods_to_compare_later=[
                "skeleton (medial-axis) + distance-transform-at-skeleton method (needs a thinning "
                "implementation this OpenCV build lacks -- no ximgproc/scikit-image installed)",
                "contour-pair half-width estimation (match opposing contour points along the local normal)",
                "adaptive granulometry (per-component kernel size instead of one fixed global radius)",
            ],
            ground_truth_type="Human-ranked ordinal width categories (thin/medium/thick) on real "
                               "stroke crops, plus a ruler-measured physical-pixel reference set",
            suggested_metric="Spearman rank correlation against human ordinal ranking; MAE in pixels "
                              "against the ruler-measured reference set",
            requires_new_dependency="scikit-image (skimage.morphology.skeletonize) for the skeleton "
                                     "candidate method only -- NOT installed this phase",
        ),
        _task(
            experiment_id="EG_A2_LINE_WIDTH_VARIABILITY", feature_group="A_LINE_WIDTH_QUALITY",
            target_observable="line.width_variability",
            current_method="CURRENT_BASELINE_ONLY: std of the same distance-transform values as EG_A1",
            candidate_methods_to_compare_later=["Whatever EG_A1 selects, applied identically"],
            ground_truth_type="Human-ranked ordinal consistency categories (uniform/somewhat "
                               "variable/highly variable) on real stroke crops",
            suggested_metric="Spearman rank correlation against human ordinal ranking",
        ),
        _task(
            experiment_id="EG_A3_LINE_DARKNESS", feature_group="A_LINE_WIDTH_QUALITY",
            target_observable="line.darkness_mean / line.darkness_variability",
            current_method="CURRENT_BASELINE_ONLY: mean/std of (255-greyscale)/255 over the whole foreground mask",
            candidate_methods_to_compare_later=[
                "Restrict to stroke-like pixels only (mirroring EG_A1's fix) to test whether darkness "
                "should ALSO exclude filled regions, unlike G0.1's deliberate choice to keep it whole-mask",
            ],
            ground_truth_type="Human-ranked ordinal darkness categories (faint/medium/dark)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_A4_LINE_CONTINUITY", feature_group="A_LINE_WIDTH_QUALITY",
            target_observable="line.continuity", current_method="CURRENT_BASELINE_ONLY: largest 8-connected component / total foreground area",
            candidate_methods_to_compare_later=["Skeleton-graph-based continuity (longest connected path / total skeleton length)"],
            ground_truth_type="Human-ranked ordinal continuity categories (one continuous mark / mostly "
                               "connected / fragmented)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_A5_FRAGMENTATION_RATE", feature_group="A_LINE_WIDTH_QUALITY",
            target_observable="line.fragmentation_rate",
            current_method="CURRENT_BASELINE_ONLY: normalized connected-component count, capped at 1.0 "
                            "(the /50 normalization constant is documented, uncalibrated)",
            candidate_methods_to_compare_later=["Calibrate the normalization constant against the same human-labelled set as EG_A4"],
            ground_truth_type="Same human-ranked continuity/fragmentation categories as EG_A4",
            suggested_metric="Spearman rank correlation",
        ),
        # ---- GROUP B: LINE ORIENTATION / ORGANIZATION ----------------------
        _task(
            experiment_id="EG_B1_DOMINANT_ORIENTATION", feature_group="B_LINE_ORIENTATION_ORGANIZATION",
            target_observable="line.dominant_orientation_code",
            current_method="CURRENT_BASELINE_ONLY: magnitude-weighted Sobel-gradient histogram (18 bins), "
                            "+90-degree gradient-to-stroke correction (see FORMAL_FEATURE_DEFINITION_TABLE.csv)",
            candidate_methods_to_compare_later=["Structure-tensor dominant-orientation estimate (single global tensor instead of a histogram peak)"],
            ground_truth_type="Human-labelled dominant direction (horizontal/vertical/diagonal/none) on real drawings",
            suggested_metric="Categorical accuracy against human label",
        ),
        _task(
            experiment_id="EG_B2_ORIENTATION_ENTROPY", feature_group="B_LINE_ORIENTATION_ORGANIZATION",
            target_observable="line.orientation_entropy",
            current_method="CURRENT_BASELINE_ONLY: normalized Shannon entropy of the same weighted histogram",
            candidate_methods_to_compare_later=["Circular-statistics dispersion measure (circular variance) as an alternative to histogram entropy"],
            ground_truth_type="Human-ranked ordinal direction-diversity categories (LOW/MEDIUM/HIGH)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_B3_LOCAL_ORIENTATION_COHERENCE", feature_group="B_LINE_ORIENTATION_ORGANIZATION",
            target_observable="line.orientation_coherence",
            current_method="CURRENT_BASELINE_ONLY: mean structure-tensor coherence (lambda1-lambda2)/(lambda1+lambda2)",
            candidate_methods_to_compare_later=["Multi-scale coherence (average over several Gaussian-blur scales instead of one fixed 5x5)"],
            ground_truth_type="Human-ranked ordinal local-agreement categories (LOW/MEDIUM/HIGH/CANNOT_ASSESS -- "
                               "see E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv's orientation_consistency example)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_B4_DIRECTION_CLUSTER_COUNT", feature_group="B_LINE_ORIENTATION_ORGANIZATION",
            target_observable="line.direction_cluster_count",
            current_method="CURRENT_BASELINE_ONLY: count of histogram bins > 30% of the max bin's weight "
                            "(documented, uncalibrated threshold)",
            candidate_methods_to_compare_later=["Circular k-means / mean-shift clustering on the weighted angle distribution"],
            ground_truth_type="Human count of visually distinct dominant directions",
            suggested_metric="Exact-match accuracy and MAE against human count",
        ),
        # ---- GROUP C: SCRIBBLE / CROSSING / COMPLEXITY ---------------------
        _task(
            experiment_id="EG_C1_SCRIBBLE_CANDIDATE_SCORE", feature_group="C_SCRIBBLE_CROSSING_COMPLEXITY",
            target_observable="stroke.scribble_candidate_score",
            current_method="CURRENT_BASELINE_ONLY: exploratory composite = stroke_density_mean * orientation_entropy "
                            "(no other weighting invented -- see G0.1 Step 4C)",
            candidate_methods_to_compare_later=[
                "Full composite per the PDF's own Appendix G note: local coherence + short-stroke density + "
                "curvature + intersections + spatial irregularity, weighting TBD via regression against human labels",
            ],
            ground_truth_type="Human binary/ternary label: scribble PRESENT / NOT_PRESENT / UNCERTAIN "
                               "(see E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv)",
            suggested_metric="ROC-AUC / F1 against human PRESENT-vs-NOT_PRESENT label",
        ),
        _task(
            experiment_id="EG_C2_JUNCTION_CORNER_DENSITY", feature_group="C_SCRIBBLE_CROSSING_COMPLEXITY",
            target_observable="stroke.junction_corner_density_proxy",
            current_method="CURRENT_BASELINE_ONLY: Harris corner-response density (cannot distinguish "
                            "true crossings from corners/sharp bends -- see G0.1 Step 4D)",
            candidate_methods_to_compare_later=[
                "Skeletonize then count true skeleton-graph branch points (needs a thinning algorithm "
                "this OpenCV build lacks)",
                "Learned junction classifier trained on the human-annotated crossing/corner distinction",
            ],
            ground_truth_type="Human-marked crossing points vs. corner/bend points on real stroke crops "
                               "(a labelled point set distinguishing the two)",
            suggested_metric="Precision/recall of detected response peaks against human-marked TRUE crossings only",
            requires_new_dependency="scikit-image or opencv-contrib (ximgproc.thinning) for the skeleton candidate -- NOT installed this phase",
        ),
        _task(
            experiment_id="EG_C3_STRAIGHTNESS", feature_group="C_SCRIBBLE_CROSSING_COMPLEXITY",
            target_observable="line.straightness",
            current_method="CURRENT_BASELINE_ONLY: fraction of Canny edge pixels covered by a Hough-detected line segment",
            candidate_methods_to_compare_later=["Vectorize each stroke and measure its own path curvature directly (needs a stroke vectorization method)"],
            ground_truth_type="Human-ranked ordinal straightness categories (curved/mixed/straight)",
            suggested_metric="Spearman rank correlation",
            requires_new_dependency="A sketch-vectorization package for the candidate method -- NOT installed this phase",
        ),
        _task(
            experiment_id="EG_C4_TRUE_CROSSING_DETECTION", feature_group="C_SCRIBBLE_CROSSING_COMPLEXITY",
            target_observable="(not yet implemented -- future replacement for stroke.junction_corner_density_proxy)",
            current_method="NOT_YET_IMPLEMENTED -- no current method exists; stroke.junction_corner_density_proxy is the interim proxy",
            candidate_methods_to_compare_later=[
                "Skeletonization (Zhang-Suen or morphological thinning) + skeleton-graph branch-point "
                "detection (degree-3+ nodes = true crossings)",
            ],
            ground_truth_type="Human-marked true crossing points (X/T junctions only, excluding corners/bends)",
            suggested_metric="Precision/recall against human-marked crossings",
            requires_new_dependency="scikit-image (skimage.morphology.skeletonize) or opencv-contrib (ximgproc) -- NOT installed this phase",
            status="PLANNED_NOT_STARTED",
        ),
        # ---- GROUP D: COLOUR / SHADING --------------------------------------
        _task(
            experiment_id="EG_D1_CHROMATIC_COVERAGE", feature_group="D_COLOUR_SHADING",
            target_observable="colour.chromatic_coverage",
            current_method="CURRENT_BASELINE_ONLY: HSV thresholds (saturation>0.15, 0.08<value<0.98), "
                            "documented as EXPLORATORY/uncalibrated -- see G0.1 Step 4B",
            candidate_methods_to_compare_later=["Perceptual colourfulness metrics (e.g. Hasler-Susstrunk colourfulness index) instead of a fixed HSV threshold"],
            ground_truth_type="Human-ranked ordinal colour-use categories (mostly achromatic/mixed/strongly coloured)",
            suggested_metric="Spearman rank correlation; threshold calibration via ROC on a human PRESENT/NOT_PRESENT chromatic-content label",
        ),
        _task(
            experiment_id="EG_D2_COLOUR_DIVERSITY", feature_group="D_COLOUR_SHADING",
            target_observable="colour.diversity",
            current_method="CURRENT_BASELINE_ONLY: reused from analysis.py -- count of 5 fixed colour bins (red/green/blue/yellow/dark) present above 2%",
            candidate_methods_to_compare_later=["K-means colour clustering with a data-driven cluster count instead of 5 fixed bins"],
            ground_truth_type="Human count of visually distinct colours used",
            suggested_metric="MAE against human count",
        ),
        _task(
            experiment_id="EG_D3_SATURATION", feature_group="D_COLOUR_SHADING", target_observable="colour.saturation",
            current_method="CURRENT_BASELINE_ONLY: mean HSV saturation over the foreground mask",
            candidate_methods_to_compare_later=["No change expected -- a direct, well-defined pixel statistic; candidate is validating against scan/lighting-corrected images"],
            ground_truth_type="Human-ranked ordinal vividness categories (muted/moderate/vivid)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_D4_BRIGHTNESS", feature_group="D_COLOUR_SHADING", target_observable="colour.brightness",
            current_method="CURRENT_BASELINE_ONLY: mean whole-image greyscale brightness",
            candidate_methods_to_compare_later=["Foreground-only brightness as an alternative to whole-image (currently dominated by background/paper)"],
            ground_truth_type="Human-ranked ordinal lightness categories",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_D5_DARKNESS_SHADING", feature_group="D_COLOUR_SHADING",
            target_observable="whole_drawing_darkness / whole_drawing_shading_density canonical observables "
                               "(colour.chromatic_coverage's dark-value exclusion + line.darkness_mean together)",
            current_method="CURRENT_BASELINE_ONLY: composed from line.darkness_mean (whole-foreground darkness) "
                            "-- no separate whole_drawing_shading_density measurement is implemented yet "
                            "(registered LITERATURE_CANDIDATE, not yet in formal_features.py)",
            candidate_methods_to_compare_later=["A dedicated local-shading-texture measurement (e.g. local binary patterns / GLCM contrast) distinct from plain darkness mean"],
            ground_truth_type="Human-ranked ordinal shading-density categories",
            suggested_metric="Spearman rank correlation",
        ),
        # ---- GROUP E: SPATIAL / ORGANIZATION ---------------------------------
        _task(
            experiment_id="EG_E1_BILATERAL_SYMMETRY", feature_group="E_SPATIAL_ORGANIZATION",
            target_observable="symmetry.bilateral",
            current_method="CURRENT_BASELINE_ONLY: 1 - mean(XOR(mask, fliplr(mask))) -- left-right only, "
                            "position-sensitive (an intrinsically symmetric shape drawn off-center scores low)",
            candidate_methods_to_compare_later=["Symmetry measured about the shape's OWN centroid axis (translation-invariant) instead of the image's vertical midline"],
            ground_truth_type="Human-ranked ordinal symmetry categories (asymmetric/somewhat symmetric/highly symmetric)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_E2_SPACING_REGULARITY", feature_group="E_SPATIAL_ORGANIZATION",
            target_observable="symmetry.spacing_regularity",
            current_method="CURRENT_BASELINE_ONLY: inverse coefficient-of-variation of nearest-neighbour "
                            "component-centroid distances (requires >=3 components)",
            candidate_methods_to_compare_later=["Full pairwise-distance regularity (not just nearest-neighbour) for richer spacing patterns"],
            ground_truth_type="Human-ranked ordinal spacing-regularity categories on repeated-element drawings",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_E3_DETAIL_DENSITY", feature_group="E_SPATIAL_ORGANIZATION",
            target_observable="scene.detail_density",
            current_method="CURRENT_BASELINE_ONLY: Canny edge-pixel fraction within the foreground mask "
                            "(fixed 50/150 thresholds)",
            candidate_methods_to_compare_later=["Multi-scale edge density (combine several Canny threshold pairs) for scale-robustness"],
            ground_truth_type="Human-ranked ordinal detail categories (sparse/moderate/highly detailed)",
            suggested_metric="Spearman rank correlation",
        ),
        _task(
            experiment_id="EG_E4_GLOBAL_SPATIAL_DENSITY_UNIFORMITY", feature_group="E_SPATIAL_ORGANIZATION",
            target_observable="fill.global_spatial_density_uniformity",
            current_method="CURRENT_BASELINE_ONLY: inverse-variance uniformity over a fixed 8x8 whole-image grid",
            candidate_methods_to_compare_later=["Adaptive grid resolution scaled to drawing complexity instead of a fixed 8x8"],
            ground_truth_type="Human-ranked ordinal spatial-distribution categories (concentrated/mixed/evenly spread)",
            suggested_metric="Spearman rank correlation",
        ),
        # ---- GROUP F: REGION-SPECIFIC FUTURE EXPERIMENT ----------------------
        _task(
            experiment_id="EG_F1_REGION_COLOURING_DIRECTION_COHERENCE", feature_group="F_REGION_SPECIFIC_FUTURE",
            target_observable="region_colouring_direction_coherence (canonical observable, EXPERIMENT_CANDIDATE)",
            current_method="NOT_YET_IMPLEMENTED -- see E_GRAPHIC_EXPERIMENT_PLAN's dedicated REGION_COLOURING_DIRECTION spec below",
            candidate_methods_to_compare_later=["Structure-tensor coherence restricted to a segmentation mask, once one exists"],
            ground_truth_type="Human-ranked ordinal within-region directional consistency (see Step 10 spec)",
            suggested_metric="Spearman rank correlation, computed only once masks exist",
            requires_segmentation=True,
            requires_new_dependency="A segmentation method (SAM2-class or otherwise) -- explicitly NOT added this phase",
        ),
        _task(
            experiment_id="EG_F2_REGION_ORIENTATION_ENTROPY", feature_group="F_REGION_SPECIFIC_FUTURE",
            target_observable="region_colouring_orientation_entropy (canonical observable, EXPERIMENT_CANDIDATE)",
            current_method="NOT_YET_IMPLEMENTED",
            candidate_methods_to_compare_later=["Same weighted-histogram entropy as line.orientation_entropy, restricted to a segmentation mask"],
            ground_truth_type="Human-ranked ordinal within-region direction-diversity",
            suggested_metric="Spearman rank correlation, computed only once masks exist",
            requires_segmentation=True,
            requires_new_dependency="A segmentation method -- explicitly NOT added this phase",
        ),
        _task(
            experiment_id="EG_F3_FILL_COMPLETENESS", feature_group="F_REGION_SPECIFIC_FUTURE",
            target_observable="fill_completeness (canonical observable, EXPERIMENT_CANDIDATE)",
            current_method="NOT_YET_IMPLEMENTED",
            candidate_methods_to_compare_later=["filled-pixel-count / object-mask-area, once a mask exists"],
            ground_truth_type="Human-ranked ordinal fill-completeness categories (mostly unfilled/partial/fully filled)",
            suggested_metric="Spearman rank correlation, computed only once masks exist",
            requires_segmentation=True,
            requires_new_dependency="A segmentation method -- explicitly NOT added this phase",
        ),
        _task(
            experiment_id="EG_F4_REGION_FILL_UNIFORMITY", feature_group="F_REGION_SPECIFIC_FUTURE",
            target_observable="region_fill_uniformity (distinct from the already-implemented "
                               "fill.global_spatial_density_uniformity, which is whole-drawing)",
            current_method="NOT_YET_IMPLEMENTED",
            candidate_methods_to_compare_later=["Grid-occupancy uniformity (same algorithm as fill.global_spatial_density_uniformity) restricted to one object's mask"],
            ground_truth_type="Human-ranked ordinal within-region fill-evenness categories",
            suggested_metric="Spearman rank correlation, computed only once masks exist",
            requires_segmentation=True,
            requires_new_dependency="A segmentation method -- explicitly NOT added this phase",
        ),
        _task(
            experiment_id="EG_F5_BOUNDARY_OVERFLOW", feature_group="F_REGION_SPECIFIC_FUTURE",
            target_observable="boundary_overflow (canonical observable, EXPERIMENT_CANDIDATE)",
            current_method="NOT_YET_IMPLEMENTED",
            candidate_methods_to_compare_later=["pixels-outside-mask-boundary / total-coloured-pixels, once a mask exists"],
            ground_truth_type="Human-ranked ordinal boundary-control categories (stays within lines/minor "
                               "overflow/major overflow)",
            suggested_metric="Spearman rank correlation, computed only once masks exist",
            requires_segmentation=True,
            requires_new_dependency="A segmentation method -- explicitly NOT added this phase",
        ),
        _task(
            experiment_id="EG_F6_LOCAL_SYMMETRY", feature_group="F_REGION_SPECIFIC_FUTURE",
            target_observable="local_symmetry (canonical observable, EXPERIMENT_CANDIDATE)",
            current_method="NOT_YET_IMPLEMENTED -- symmetry.bilateral (implemented) is WHOLE_DRAWING scope only",
            candidate_methods_to_compare_later=["Same XOR-mirror algorithm as symmetry.bilateral, restricted to one object's own mask/centroid"],
            ground_truth_type="Human-ranked ordinal per-object symmetry categories",
            suggested_metric="Spearman rank correlation, computed only once masks exist",
            requires_segmentation=True,
            requires_new_dependency="A segmentation method -- explicitly NOT added this phase",
        ),
    ]

    # ---- Step 10: the doctor's own worked example, specified in full ------
    region_colouring_direction_experiment = {
        "experiment_name": "REGION_COLOURING_DIRECTION",
        "goal": (
            "Given a segmented coloured region (e.g. a shirt, a tree crown, a house wall, trousers), "
            "measure the stroke-direction character of the colouring INSIDE that region only -- never "
            "approximated by a bounding box, which would include background/other-object contamination."
        ),
        "measurements_planned": [
            "horizontal_stroke_proportion", "vertical_stroke_proportion", "diagonal_stroke_proportion",
            "orientation_entropy_within_region", "local_orientation_coherence_within_region",
            "direction_cluster_count_within_region", "cross_hatching_intersection_behaviour_within_region",
            "fill_uniformity_within_region", "boundary_overflow_within_region",
        ],
        "reuses_existing_algorithms_from": (
            "The horizontal/vertical/diagonal proportions, orientation entropy, local coherence, and "
            "direction-cluster-count math is IDENTICAL to formal_features.py's already-implemented "
            "line.dominant_orientation_code/line.orientation_entropy/line.orientation_coherence/"
            "line.direction_cluster_count -- only the INPUT PIXEL SET changes (a region mask instead of "
            "the whole-image foreground mask). No new algorithm is invented for this phase; only the "
            "masking scope is new."
        ),
        "current_status": "NEEDS_SEGMENTATION_MASK",
        "why_not_implemented_now": (
            "DOAR's entities (visual_evidence.py::VisualEntity) carry only a bounding box, never a "
            "per-object pixel mask. Running any of the above measurements over a bounding box would "
            "silently include background paper and/or neighbouring objects, producing a number that "
            "LOOKS like a region measurement but is contaminated -- explicitly forbidden by this phase's "
            "instructions ('Do NOT approximate these by using the entire bounding box')."
        ),
        "mask_quality_required": {
            "minimum_requirement": (
                "A pixel-accurate (or near-pixel-accurate) binary segmentation mask per target object "
                "instance, aligned to the same coordinate space as the existing foreground mask."
            ),
            "acceptable_sources": [
                "A validated automatic segmentation method (e.g. SAM2-class) -- NOT added this phase, "
                "would be its own separate E-Graphic sub-experiment with its own accuracy validation "
                "before being trusted for measurement.",
                "Human-drawn segmentation masks (a researcher/psychologist traces the region boundary) "
                "for an initial small validation/calibration set.",
            ],
            "quality_bar": (
                "Boundary error should be small relative to the region's own stroke width (a boundary "
                "that clips or includes a stroke-width's worth of extra pixels will bias the fill-"
                "uniformity/boundary-overflow measurements specifically) -- an exact numeric IoU "
                "threshold is deliberately NOT set here; that is itself an E-Graphic calibration task."
            ),
        },
        "ground_truth_annotation_the_psychologist_would_provide": [
            "Region boundary trace (for the initial calibration set, before/instead of an automatic segmenter)",
            "Ordinal judgment: orientation_consistency (LOW/MEDIUM/HIGH/CANNOT_ASSESS) -- see "
            "E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv's own worked example for this exact field",
            "Ordinal judgment: fill_uniformity_within_region (LOW/MEDIUM/HIGH/CANNOT_ASSESS)",
            "Binary/ternary judgment: cross_hatching_present (PRESENT/NOT_PRESENT/UNCERTAIN)",
            "Binary/ternary judgment: boundary_overflow_present (PRESENT/NOT_PRESENT/UNCERTAIN)",
            "Free-text note on any visual confound observed (e.g. two colours in one region, region "
            "only partially coloured, tool/material visible)",
        ],
        "explicitly_not_implemented_this_phase": [
            "No segmentation mask (SAM2 or otherwise) is added.",
            "No bounding-box approximation is used as a stand-in.",
            "No psychological interpretation (e.g. 'messy colouring = dysregulation') is attached -- "
            "this experiment produces an OBJECTIVE_ONLY measurement only; the doctor's proposed meaning "
            "remains CLINICIAN_CANDIDATE pending a SEPARATE, later validation experiment against "
            "reviewed cases.",
        ],
        "related_experiment_plan_tasks": ["EG_F1_REGION_COLOURING_DIRECTION_COHERENCE", "EG_F2_REGION_ORIENTATION_ENTROPY",
                                          "EG_F4_REGION_FILL_UNIFORMITY", "EG_F5_BOUNDARY_OVERFLOW"],
    }

    # ---- Step 11: preserve the G0.1-documented line-width limitation ------
    line_width_limitation = {
        "feature_id": "line.mean_width",
        "current_method_status": "BASELINE_METHOD",
        "limitation": (
            "The current connected-component morphological-granulometry stroke/blob separation "
            "(G0.1 Step 4A) can misclassify a genuinely thick stroke as a filled blob if its width is "
            "near or above the documented kernel radius (2% of the image's shorter side) -- such a "
            "stroke is then honestly reported as MISSING rather than measured, never as a fabricated "
            "value. This limitation is NOT hidden: it is documented in formal_features.py's own "
            "_stroke_like_mask docstring, in FORMAL_FEATURE_DEFINITION_TABLE.csv's known_confounders "
            "column, and here."
        ),
        "candidate_methods_for_later_comparison": [
            "skeleton (medial-axis) + distance-transform-at-skeleton method",
            "contour-pair half-width estimation",
            "adaptive per-component granulometry kernel size",
        ],
        "no_new_dependency_installed_this_phase": True,
    }

    return {
        "schema_version": E_GRAPHIC_PLAN_SCHEMA_VERSION,
        "region_colouring_direction_experiment": region_colouring_direction_experiment,
        "line_width_limitation": line_width_limitation,
        "status": (
            "RESEARCH-ONLY planning file -- no experiment has been run, no method has been chosen as a "
            "winner, and no new dependency (SAM2/DeepLSD/SOLD2/sketch-vectorization/scikit-image) has "
            "been installed or imported anywhere in this codebase. Every task's status is "
            "PLANNED_NOT_STARTED. This organizes the NEXT experiment phase (E-Graphic), not a decision."
        ),
        "task_count": len(tasks),
        "task_count_by_group": {
            g: sum(1 for t in tasks if t["feature_group"] == g)
            for g in sorted({t["feature_group"] for t in tasks})
        },
        "tasks_requiring_segmentation": [t["experiment_id"] for t in tasks if t["requires_segmentation"]],
        "tasks_requiring_a_new_dependency": [t["experiment_id"] for t in tasks if t["requires_new_dependency"]],
        "tasks": tasks,
    }


def write_egraphic_experiment_plan() -> Path:
    document = build_egraphic_experiment_plan()
    E_GRAPHIC_EXPERIMENT_PLAN_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return E_GRAPHIC_EXPERIMENT_PLAN_PATH


if __name__ == "__main__":
    path = write_egraphic_experiment_plan()
    print(f"wrote {path}")
