"""Phase G0.2, Step 9 -- builds E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv: what
will later be SHOWN TO THE PSYCHOLOGIST to collect ground truth for the
E-Graphic experiments planned in E_GRAPHIC_EXPERIMENT_PLAN.json.

Every row asks ONLY about WHAT IS VISUALLY PRESENT (line width, darkness,
direction, symmetry, fill, crossings, scribble appearance, colour) --
NEVER a psychological judgment ("unstable home", "dysregulation", etc.).
Psychological relevance is explicitly a LATER, separate experiment (see
E_GRAPHIC_EXPERIMENT_PLAN.json's region_colouring_direction_experiment
note making this same point for the doctor's own worked example).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
E_GRAPHIC_ANNOTATION_SPEC_PATH = ROOT / "E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv"

FIELDNAMES = [
    "annotation_id", "canonical_observable_id", "human_label", "plain_definition",
    "positive_example_description", "negative_example_description", "ambiguous_case_description",
    "annotation_type", "allowed_values", "region_annotation_required", "task_requirement",
    "age_context_required", "material_context_required", "notes",
]


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {k: "" for k in FIELDNAMES}
    row["region_annotation_required"] = "no"
    row["task_requirement"] = "ANY_DRAWING"
    row["age_context_required"] = "no"
    row["material_context_required"] = "no"
    row.update(kwargs)
    return row


def build_egraphic_annotation_spec() -> list[dict[str, Any]]:
    return [
        _row(
            annotation_id="ANNOT_01_LINE_WIDTH", canonical_observable_id="mean_line_width",
            human_label="Apparent line width", plain_definition="How thick the drawn strokes appear overall.",
            positive_example_description="A marker drawing with clearly thick, bold strokes throughout.",
            negative_example_description="A fine-tip pen drawing with thin, delicate strokes throughout.",
            ambiguous_case_description="A drawing mixing very thin outline strokes with thick filled areas -- rate the STROKES, not the fills.",
            annotation_type="ordinal", allowed_values="THIN|MEDIUM|THICK|CANNOT_ASSESS",
            notes="Ground truth for line.mean_width (EG_A1).",
        ),
        _row(
            annotation_id="ANNOT_02_LINE_WIDTH_CONSISTENCY", canonical_observable_id="mean_line_width",
            human_label="Line width consistency", plain_definition="How consistent stroke thickness is across the drawing.",
            positive_example_description="Every stroke is roughly the same thickness throughout.",
            negative_example_description="Some strokes are very thin, others very thick, within the same drawing.",
            ambiguous_case_description="Thickness varies only slightly, within normal hand-drawing wobble.",
            annotation_type="ordinal", allowed_values="UNIFORM|SOMEWHAT_VARIABLE|HIGHLY_VARIABLE|CANNOT_ASSESS",
            notes="Ground truth for line.width_variability (EG_A2).",
        ),
        _row(
            annotation_id="ANNOT_03_LINE_DARKNESS", canonical_observable_id="line_darkness_intensity",
            human_label="Line darkness", plain_definition="How dark/faint the marks appear overall.",
            positive_example_description="Heavy, dark, saturated marks (e.g. pressed-hard pencil or thick marker).",
            negative_example_description="Very light, faint marks (e.g. light pencil sketch).",
            ambiguous_case_description="A drawing with both dark outlines and faint shading -- rate the OVERALL impression.",
            annotation_type="ordinal", allowed_values="FAINT|MEDIUM|DARK|CANNOT_ASSESS",
            notes="Ground truth for line.darkness_mean (EG_A3). Never asks about physical pencil pressure.",
        ),
        _row(
            annotation_id="ANNOT_04_LINE_CONTINUITY", canonical_observable_id="line_continuity",
            human_label="Line continuity", plain_definition="Whether the main marks form one continuous shape or many separate pieces.",
            positive_example_description="One continuous outline drawn without lifting the pen.",
            negative_example_description="Many small, disconnected dots/dashes making up the image.",
            ambiguous_case_description="Mostly connected but with a few small gaps.",
            annotation_type="ordinal", allowed_values="ONE_CONTINUOUS_MARK|MOSTLY_CONNECTED|FRAGMENTED|CANNOT_ASSESS",
            notes="Ground truth for line.continuity/line.fragmentation_rate (EG_A4/EG_A5).",
        ),
        _row(
            annotation_id="ANNOT_05_DOMINANT_DIRECTION", canonical_observable_id="dominant_orientation",
            human_label="Dominant stroke direction", plain_definition="The single most common direction strokes run in.",
            positive_example_description="A drawing made almost entirely of vertical strokes.",
            negative_example_description="A drawing with no clearly dominant direction (mixed evenly).",
            ambiguous_case_description="Two directions appear almost equally often.",
            annotation_type="categorical", allowed_values="HORIZONTAL|VERTICAL|DIAGONAL|NONE_DOMINANT|CANNOT_ASSESS",
            notes="Ground truth for line.dominant_orientation_code (EG_B1).",
        ),
        _row(
            annotation_id="ANNOT_06_DIRECTION_DIVERSITY", canonical_observable_id="orientation_entropy",
            human_label="Direction diversity", plain_definition="How many different stroke directions are present overall.",
            positive_example_description="Strokes going in many different directions across the whole drawing.",
            negative_example_description="Nearly all strokes run in the same direction.",
            ambiguous_case_description="A few directions dominate but with some scattered outliers.",
            annotation_type="ordinal", allowed_values="LOW|MEDIUM|HIGH|CANNOT_ASSESS",
            notes="Ground truth for line.orientation_entropy (EG_B2).",
        ),
        _row(
            annotation_id="ANNOT_07_ORIENTATION_CONSISTENCY", canonical_observable_id="local_orientation_coherence",
            human_label="Orientation consistency",
            plain_definition="How consistently strokes inside the assessed region follow similar directions to their immediate neighbours.",
            positive_example_description="Neighbouring strokes all run the same way, even if the drawing as a whole uses several directions elsewhere.",
            negative_example_description="Strokes cross and change direction abruptly within a small area.",
            ambiguous_case_description="A region with two clearly separate direction bands, each internally consistent.",
            annotation_type="ordinal", allowed_values="LOW|MEDIUM|HIGH|CANNOT_ASSESS",
            notes="Ground truth for line.orientation_coherence (EG_B3) -- the exact worked example given in this "
                  "phase's own instructions.",
        ),
        _row(
            annotation_id="ANNOT_08_DIRECTION_CLUSTER_COUNT", canonical_observable_id="direction_cluster_count",
            human_label="Number of distinct directions", plain_definition="Count of visually distinct dominant stroke directions.",
            positive_example_description="A cross-hatched region with exactly two clearly distinct directions.",
            negative_example_description="A region with only one direction present.",
            ambiguous_case_description="A gradually curving stroke that spans a continuous range of angles rather than discrete clusters.",
            annotation_type="count", allowed_values="0|1|2|3|4_OR_MORE|CANNOT_ASSESS",
            notes="Ground truth for line.direction_cluster_count (EG_B4).",
        ),
        _row(
            annotation_id="ANNOT_09_SCRIBBLE_APPEARANCE", canonical_observable_id="stroke_density_map",
            human_label="Scribble appearance", plain_definition="Whether the marks in an area look like a scribble (dense, multidirectional, unplanned-looking marks).",
            positive_example_description="A dense tangle of criss-crossing lines with no clear single shape.",
            negative_example_description="A single clean outline or a few clearly separate deliberate strokes.",
            ambiguous_case_description="Deliberate cross-hatching used as a shading technique, not a scribble.",
            annotation_type="categorical", allowed_values="PRESENT|NOT_PRESENT|UNCERTAIN",
            notes="Ground truth for stroke.scribble_candidate_score (EG_C1) -- the exact worked example given in "
                  "this phase's own instructions.",
        ),
        _row(
            annotation_id="ANNOT_10_TRUE_CROSSINGS", canonical_observable_id="crossing_density",
            human_label="Line crossings (mark the points)",
            plain_definition="Points where two separate strokes genuinely cross one another (X/T junctions) -- "
                              "distinct from a single stroke's own corner or sharp bend.",
            positive_example_description="A clear X where two strokes cross.",
            negative_example_description="A single stroke making a sharp 90-degree turn (a corner, not a crossing).",
            ambiguous_case_description="A T-junction where a short stroke touches but does not fully cross a longer one.",
            annotation_type="point_annotation", allowed_values="mark each point as CROSSING|CORNER|UNCERTAIN",
            notes="Ground truth for stroke.junction_corner_density_proxy / future true crossing detection (EG_C2/EG_C4). "
                  "This is a POINT annotation task, not a whole-image ordinal rating.",
        ),
        _row(
            annotation_id="ANNOT_11_STRAIGHTNESS", canonical_observable_id="straightness",
            human_label="Line straightness", plain_definition="How straight vs. curved the main strokes are.",
            positive_example_description="Ruler-straight lines throughout.",
            negative_example_description="Freehand curved, wavy strokes throughout.",
            ambiguous_case_description="Mostly straight strokes with a few deliberately curved elements (e.g. a curved smile).",
            annotation_type="ordinal", allowed_values="CURVED|MIXED|STRAIGHT|CANNOT_ASSESS",
            notes="Ground truth for line.straightness (EG_C3).",
        ),
        _row(
            annotation_id="ANNOT_12_CHROMATIC_COLOUR_USE", canonical_observable_id="colour_coverage",
            human_label="Chromatic colour use", plain_definition="How much of the drawing uses visibly saturated colour (not just black/grey/white).",
            positive_example_description="A drawing filled with bright red, blue, and green.",
            negative_example_description="A drawing done entirely in pencil (greyscale).",
            ambiguous_case_description="A drawing using only very pale/washed-out colour.",
            annotation_type="ordinal", allowed_values="MOSTLY_ACHROMATIC|MIXED|STRONGLY_COLOURED|CANNOT_ASSESS",
            notes="Ground truth for colour.chromatic_coverage (EG_D1). Never conflated with how much of the "
                  "PAGE has any mark on it at all.",
        ),
        _row(
            annotation_id="ANNOT_13_COLOUR_COUNT", canonical_observable_id="colour_diversity_low",
            human_label="Number of distinct colours", plain_definition="Count of visibly distinct colours used.",
            positive_example_description="A drawing using 5+ clearly different colours.",
            negative_example_description="A drawing using a single colour throughout.",
            ambiguous_case_description="Several very close shades of the same hue (e.g. light and dark blue).",
            annotation_type="count", allowed_values="0|1|2|3|4|5_OR_MORE|CANNOT_ASSESS",
            notes="Ground truth for colour.diversity (EG_D2).",
        ),
        _row(
            annotation_id="ANNOT_14_SHADING_DENSITY", canonical_observable_id="whole_drawing_shading_density",
            human_label="Shading density", plain_definition="How much local shading/hatching texture is present.",
            positive_example_description="Areas densely filled with hatching or scribble-shading.",
            negative_example_description="Only clean outlines, no shading at all.",
            ambiguous_case_description="Light, sparse shading in only a small part of the drawing.",
            annotation_type="ordinal", allowed_values="NONE|LIGHT|HEAVY|CANNOT_ASSESS",
            notes="Ground truth for a future whole_drawing_shading_density measurement (EG_D5) -- not yet "
                  "implemented in formal_features.py.",
        ),
        _row(
            annotation_id="ANNOT_15_SYMMETRY", canonical_observable_id="bilateral_symmetry",
            human_label="Left-right symmetry", plain_definition="How mirror-symmetric the drawing (or a specific shape) looks left-to-right.",
            positive_example_description="A face or house drawn as a near-perfect mirror image left-to-right.",
            negative_example_description="A scene where the left and right sides look nothing alike.",
            ambiguous_case_description="A symmetric shape drawn off-center on the page.",
            annotation_type="ordinal", allowed_values="ASYMMETRIC|SOMEWHAT_SYMMETRIC|HIGHLY_SYMMETRIC|CANNOT_ASSESS",
            notes="Ground truth for symmetry.bilateral (EG_E1). Rate the SHAPE's own symmetry, not its position on the page.",
        ),
        _row(
            annotation_id="ANNOT_16_SPACING_REGULARITY", canonical_observable_id="spacing_regularity",
            human_label="Spacing regularity", plain_definition="How evenly spaced repeated elements are (e.g. a row of dots, stars, or trees).",
            positive_example_description="A neat grid of evenly spaced repeated shapes.",
            negative_example_description="Repeated shapes scattered at random, irregular distances apart.",
            ambiguous_case_description="Fewer than 3 repeated elements present (cannot meaningfully assess spacing pattern).",
            annotation_type="ordinal", allowed_values="REGULAR|SOMEWHAT_IRREGULAR|IRREGULAR|CANNOT_ASSESS",
            notes="Ground truth for symmetry.spacing_regularity (EG_E2). Only applicable with >=3 repeated elements.",
        ),
        _row(
            annotation_id="ANNOT_17_DETAIL_LEVEL", canonical_observable_id="details_of_objects_environment",
            human_label="Level of detail", plain_definition="How much fine detail is present relative to the marked area.",
            positive_example_description="Densely detailed drawing with many small added elements.",
            negative_example_description="A very simple, minimal drawing with almost no fine detail.",
            ambiguous_case_description="High detail concentrated in only one part of the drawing.",
            annotation_type="ordinal", allowed_values="SPARSE|MODERATE|HIGHLY_DETAILED|CANNOT_ASSESS",
            notes="Ground truth for scene.detail_density (EG_E3).",
        ),
        _row(
            annotation_id="ANNOT_18_SPATIAL_DISTRIBUTION", canonical_observable_id="space",
            human_label="Spatial distribution across the page", plain_definition="How evenly marks are distributed across the whole page vs. concentrated in one area.",
            positive_example_description="Marks spread fairly evenly across the entire page.",
            negative_example_description="All marks concentrated in one small corner or area.",
            ambiguous_case_description="Two separate concentrated clusters with empty space between them.",
            annotation_type="ordinal", allowed_values="CONCENTRATED|MIXED|EVENLY_SPREAD|CANNOT_ASSESS",
            notes="Ground truth for fill.global_spatial_density_uniformity (EG_E4).",
        ),
        # ---- Region-specific (Step 10) -- explicitly region_annotation_required=yes ----
        _row(
            annotation_id="ANNOT_19_REGION_ORIENTATION_CONSISTENCY", canonical_observable_id="region_colouring_direction_coherence",
            human_label="Within-region orientation consistency (Step 10 worked example)",
            plain_definition="How consistently the colouring strokes INSIDE one segmented region (e.g. a shirt) follow similar directions.",
            positive_example_description="A shirt coloured with all-vertical strokes.",
            negative_example_description="A shirt coloured with strokes going in many different directions.",
            ambiguous_case_description="A region only partially coloured, with the rest left blank.",
            annotation_type="ordinal", allowed_values="LOW|MEDIUM|HIGH|CANNOT_ASSESS",
            region_annotation_required="yes",
            notes="Requires a segmented region -- see E_GRAPHIC_EXPERIMENT_PLAN.json's "
                  "region_colouring_direction_experiment. Deferred until a segmentation mask exists.",
        ),
        _row(
            annotation_id="ANNOT_20_REGION_FILL_UNIFORMITY", canonical_observable_id="fill_uniformity",
            human_label="Within-region fill uniformity",
            plain_definition="How evenly the colouring fills the segmented region (vs. patchy/uneven coverage).",
            positive_example_description="A shirt coloured solidly and evenly throughout.",
            negative_example_description="A shirt with large uncoloured gaps or very patchy, uneven fill.",
            ambiguous_case_description="Deliberate texture/pattern (e.g. stripes) that is uneven by design, not by inconsistent effort.",
            annotation_type="ordinal", allowed_values="LOW|MEDIUM|HIGH|CANNOT_ASSESS",
            region_annotation_required="yes",
            notes="Requires a segmented region -- deferred (EG_F4).",
        ),
        _row(
            annotation_id="ANNOT_21_REGION_CROSSHATCH_PRESENT", canonical_observable_id="cross_hatching",
            human_label="Cross-hatching present within region",
            plain_definition="Whether the colouring inside the region shows visible cross-hatching (two or more crossing stroke directions used as a shading technique).",
            positive_example_description="A region shaded with visible criss-crossed pencil strokes.",
            negative_example_description="A region filled with a single smooth colour or single-direction strokes.",
            ambiguous_case_description="A small area of incidental overlap that may not be deliberate cross-hatching.",
            annotation_type="categorical", allowed_values="PRESENT|NOT_PRESENT|UNCERTAIN",
            region_annotation_required="yes",
            notes="Requires a segmented region -- deferred (related to EG_F1/EG_C2).",
        ),
        _row(
            annotation_id="ANNOT_22_REGION_BOUNDARY_OVERFLOW", canonical_observable_id="boundary_overflow",
            human_label="Boundary overflow (colouring outside the lines)",
            plain_definition="Whether the colouring extends visibly beyond the region's own outline boundary.",
            positive_example_description="Colour clearly extends past the drawn outline into the background or another object.",
            negative_example_description="Colouring stays neatly within the outline.",
            ambiguous_case_description="A very thin, barely-visible overflow at only one edge.",
            annotation_type="categorical", allowed_values="PRESENT|NOT_PRESENT|UNCERTAIN",
            region_annotation_required="yes",
            notes="Requires a segmented region -- deferred (EG_F5).",
        ),
    ]


def write_egraphic_annotation_spec() -> Path:
    rows = build_egraphic_annotation_spec()
    with open(E_GRAPHIC_ANNOTATION_SPEC_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return E_GRAPHIC_ANNOTATION_SPEC_PATH


if __name__ == "__main__":
    path = write_egraphic_annotation_spec()
    print(f"wrote {path}")
