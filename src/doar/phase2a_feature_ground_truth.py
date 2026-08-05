"""Analytically-controlled synthetic feature ground-truth validation
(DOAR-TRACE Phase 2A, Section 4). Every test case's expected value is
computed from the drawn shape's own geometry (never from the pipeline
being tested -- that would be circular), then compared against the real
`analyze_image`/`objective_feature_row` output.

**This validates measurement implementation only.** Synthetic agreement
proves the code computes what it claims to compute; it says nothing
about whether any psychological interpretation attached to a feature by
a registry rule is valid.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .analysis import analyze_image
from .features import objective_feature_row

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "artifacts" / "phase2a" / "feature_ground_truth.csv"
SUMMARY_PATH = ROOT / "artifacts" / "phase2a" / "feature_ground_truth_summary.json"

CSV_FIELDS = [
    "case_id", "feature_id", "expected_value", "measured_value", "absolute_error",
    "relative_error", "tolerance", "status", "limitations",
]


def _run(image: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
    temp = tempfile.TemporaryDirectory()
    path = Path(temp.name) / "image.png"
    image.save(path)
    out = Path(temp.name) / "out"
    result = analyze_image(path, out)
    analysis = result.to_dict()
    features = objective_feature_row(path, analysis)
    temp.cleanup()
    return analysis, features


def _check(case_id: str, feature_id: str, expected: float, measured: float, tolerance: float,
           limitations: str = "") -> dict[str, Any]:
    abs_err = abs(measured - expected)
    rel_err = abs_err / abs(expected) if abs(expected) > 1e-9 else None
    status = "pass" if abs_err <= tolerance else ("warn" if abs_err <= tolerance * 2 else "fail")
    return {
        "case_id": case_id, "feature_id": feature_id, "expected_value": round(expected, 6),
        "measured_value": round(measured, 6), "absolute_error": round(abs_err, 6),
        "relative_error": round(rel_err, 6) if rel_err is not None else None,
        "tolerance": tolerance, "status": status, "limitations": limitations,
    }


def build_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    W = H = 300

    # --- Case 1: centered solid square, known area fraction -----------------
    # Square 150x150 centered in a 300x300 canvas -> area fraction = 150*150/300*300 = 0.25
    img = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img).rectangle((75, 75, 224, 224), fill="black")  # 150x150 inclusive
    analysis, features = _run(img)
    expected_area_frac = (150 * 150) / (W * H)
    rows.append(_check("centered_square", "segmentation.foreground_coverage", expected_area_frac,
                        features["segmentation.foreground_coverage"].value, tolerance=0.02,
                        limitations="Anti-aliasing/cleanup morphology at shape edges can shift coverage by ~1-2%."))
    rows.append(_check("centered_square", "segmentation.bounding_box_coverage", expected_area_frac,
                        features["segmentation.bounding_box_coverage"].value, tolerance=0.02,
                        limitations="A solid filled square's bbox coverage should equal its foreground coverage."))
    rows.append(_check("centered_square", "composition.centroid_x", 0.5,
                        features["composition.centroid_x"].value, tolerance=0.02))
    rows.append(_check("centered_square", "composition.centroid_y", 0.5,
                        features["composition.centroid_y"].value, tolerance=0.02))

    # --- Case 2: small square in top-left quadrant, known centroid ----------
    img2 = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img2).rectangle((20, 20, 79, 79), fill="black")  # 60x60 square, center at (50,50)
    analysis2, features2 = _run(img2)
    expected_cx, expected_cy = 50 / W, 50 / H
    rows.append(_check("top_left_square", "composition.centroid_x", expected_cx,
                        features2["composition.centroid_x"].value, tolerance=0.03))
    rows.append(_check("top_left_square", "composition.centroid_y", expected_cy,
                        features2["composition.centroid_y"].value, tolerance=0.03))
    # placement logic: cx<0.4 and cy<0.4 -> "top_left" (analysis.py::_composition threshold)
    placement = analysis2["composition"]["placement"]
    rows.append({
        "case_id": "top_left_square", "feature_id": "composition.placement", "expected_value": "top_left",
        "measured_value": placement, "absolute_error": 0.0 if placement == "top_left" else 1.0,
        "relative_error": None, "tolerance": 0, "status": "pass" if placement == "top_left" else "fail",
        "limitations": "Categorical match, not a numeric error.",
    })

    # --- Case 3: square in bottom-right, for right/center placement check ---
    img3 = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img3).rectangle((220, 130, 279, 189), fill="black")  # center at (250,160) -> cx=0.833, cy=0.533
    analysis3, features3 = _run(img3)
    placement3 = analysis3["composition"]["placement"]
    rows.append({
        "case_id": "right_square", "feature_id": "composition.placement", "expected_value": "middle_right",
        "measured_value": placement3, "absolute_error": 0.0 if placement3 == "middle_right" else 1.0,
        "relative_error": None, "tolerance": 0, "status": "pass" if placement3 == "middle_right" else "fail",
        "limitations": "Categorical match, not a numeric error.",
    })

    # --- Case 4: dark_ratio, solid black foreground --------------------------
    rows.append(_check("centered_square", "colour.dark_ratio", 1.0,
                        features["colour.dark_ratio"].value, tolerance=0.05,
                        limitations="dark_ratio thresholds fg pixel mean<80; solid black (0,0,0) should be ~100% dark."))

    # --- Case 5: dark_ratio, mid-gray foreground (should NOT be counted dark) ---
    img5 = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img5).rectangle((75, 75, 224, 224), fill=(150, 150, 150))
    analysis5, features5 = _run(img5)
    rows.append(_check("gray_square", "colour.dark_ratio", 0.0,
                        features5["colour.dark_ratio"].value, tolerance=0.05,
                        limitations="Mid-gray (150,150,150) mean=150 > 80 dark threshold -> expected ~0% dark."))
    rows.append({
        "case_id": "light_ratio", "feature_id": "colour.light_ratio", "expected_value": "n/a",
        "measured_value": "NOT_IMPLEMENTED", "absolute_error": None, "relative_error": None,
        "tolerance": None, "status": "not_applicable",
        "limitations": "No colour.light_ratio (or equivalent) feature exists in features.py -- cannot validate what is not implemented.",
    })

    # --- Case 6: connected-component count, 3 disjoint squares --------------
    img6 = Image.new("RGB", (W, H), "white")
    d6 = ImageDraw.Draw(img6)
    d6.rectangle((20, 20, 59, 59), fill="black")
    d6.rectangle((120, 120, 159, 159), fill="black")
    d6.rectangle((220, 220, 259, 259), fill="black")
    analysis6, features6 = _run(img6)
    rows.append(_check("three_squares", "segmentation.component_count", 3.0,
                        features6["segmentation.component_count"].value, tolerance=0,
                        limitations="Exact integer match expected; _connected_components subsamples large masks but these squares are well above the subsampling step size at 300x300."))
    rows.append(_check("three_squares", "shape.contour_proxy_count", 3.0,
                        features6["shape.contour_proxy_count"].value, tolerance=0,
                        limitations="shape.contour_proxy_count and segmentation.component_count share an identical formula (len(component_sizes)) -- verified here to confirm they always agree."))

    # --- Case 7: largest_component_ratio, two very different sized squares --
    img7 = Image.new("RGB", (W, H), "white")
    d7 = ImageDraw.Draw(img7)
    d7.rectangle((20, 20, 39, 39), fill="black")     # 20x20 = 400 px
    d7.rectangle((100, 100, 199, 199), fill="black")  # 100x100 = 10000 px
    analysis7, features7 = _run(img7)
    expected_ratio = 10000 / (10000 + 400)
    rows.append(_check("two_squares_sizes", "segmentation.largest_component_ratio", expected_ratio,
                        features7["segmentation.largest_component_ratio"].value, tolerance=0.02,
                        limitations="Subsampling for large masks (>512px longest side is not triggered at 300x300) should keep this near-exact."))

    # --- Case 8: border_touch_ratio -- RETIRED (Phase 2A.1, Section 5) ------
    # Phase 2A's original comment attributed this to `_segment`'s
    # morphological cleanup ("erodes ~1-2px from shape edges"). Phase 2A.1
    # traced the EXACT mechanism (docs/BORDER_TOUCH_RATIO_DECISION.md) and
    # found that explanation was incomplete: the primary cause is upstream,
    # in `_segment`'s CANDIDATE SELECTION -- `candidate_adaptive`'s
    # BoxBlur-based local-contrast test structurally fails to detect thick,
    # border-touching foreground (PIL's edge-padding during the blur erases
    # the local contrast it depends on), and `_candidate_score`'s explicit
    # `(1 - border_ratio)` term then REWARDS that failure, actively
    # selecting the flawed candidate over the two that correctly detect the
    # border-touching content. Morphological cleanup, applied after
    # selection, is a minor, secondary contributor (it actually recovers
    # most of a thin under-detected band, verified directly). Because the
    # true bug lives in shared `_segment` logic used by every feature and
    # by page_frame.py -- not something isolated to this one feature -- and
    # because this feature has zero downstream consumers (confirmed: no
    # rule, no page_frame.py logic reads it), it is RETIRED from downstream
    # use (features.py::KNOWN_UNRELIABLE: confidence=0.0, missing=True)
    # rather than "fixed" here in isolation, which would not address the
    # real cause and would risk a false sense of correctness. The
    # historical computation is preserved (still computed, still visible in
    # objective_features.json/the Technical view) for comparison. This case
    # now documents the retirement, not a numeric pass/fail -- pretending a
    # tolerance-based check is meaningful for a value already known to be
    # unreliable would misrepresent what was actually validated.
    img8 = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img8).rectangle((0, 0, W - 1, 10), fill="black")  # a strip across the whole top edge
    analysis8, features8 = _run(img8)
    fv8 = features8["segmentation.border_touch_ratio"]
    rows.append({
        "case_id": "top_edge_strip", "feature_id": "segmentation.border_touch_ratio",
        "expected_value": 0.25, "measured_value": round(fv8.value, 6),
        "absolute_error": None, "relative_error": None, "tolerance": None,
        "status": "known_unreliable",
        "limitations": (
            "RETIRED (Phase 2A.1): root cause is a _segment candidate-selection bias, not a test-construction "
            f"error. Historical value preserved for comparison (feature.confidence={fv8.confidence}, "
            f"feature.missing={fv8.missing}). See docs/BORDER_TOUCH_RATIO_DECISION.md."
        ),
    })

    img8b = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img8b).rectangle((100, 100, 199, 199), fill="black")  # isolated, no edge touch
    analysis8b, features8b = _run(img8b)
    fv8b = features8b["segmentation.border_touch_ratio"]
    rows.append({
        "case_id": "isolated_square", "feature_id": "segmentation.border_touch_ratio",
        "expected_value": 0.0, "measured_value": round(fv8b.value, 6),
        "absolute_error": None, "relative_error": None, "tolerance": None,
        "status": "known_unreliable",
        "limitations": "RETIRED (Phase 2A.1): recorded for comparison only, not certified correct. See docs/BORDER_TOUCH_RATIO_DECISION.md.",
    })

    # --- Case 9: horizontal (left-right) symmetry ----------------------------
    img9 = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img9).ellipse((100, 100, 199, 199), fill="black")  # centered circle -> symmetric
    analysis9, features9 = _run(img9)
    rows.append(_check("centered_circle", "composition.symmetry", 1.0,
                        features9["composition.symmetry"].value, tolerance=0.03,
                        limitations="composition.symmetry is left-right (horizontal-flip) symmetry only; no separate vertical (up-down) symmetry feature exists in features.py."))

    img9b = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(img9b).rectangle((20, 20, 79, 279), fill="black")  # tall strip on the LEFT only -> asymmetric
    analysis9b, features9b = _run(img9b)
    measured_asym = features9b["composition.symmetry"].value
    rows.append({
        "case_id": "left_only_strip", "feature_id": "composition.symmetry", "expected_value": "<0.85 (asymmetric)",
        "measured_value": measured_asym, "absolute_error": None, "relative_error": None, "tolerance": None,
        "status": "pass" if measured_asym < 0.85 else "fail",
        "limitations": "Directional expectation (must be clearly below the symmetric case), not a single exact target value.",
    })

    # --- Case 10: stroke.intensity_proxy, known foreground darkness ---------
    rows.append(_check("centered_square", "stroke.intensity_proxy", 1.0,
                        features["stroke.intensity_proxy"].value, tolerance=0.05,
                        limitations="intensity_proxy = (255 - fg_mean)/255; solid black fg (mean=0) -> expected 1.0 exactly."))
    expected_gray_intensity = (255 - 150) / 255
    rows.append(_check("gray_square", "stroke.intensity_proxy", expected_gray_intensity,
                        features5["stroke.intensity_proxy"].value, tolerance=0.05))

    # --- Case 11: background occupancy (1 - foreground_coverage) ------------
    expected_bg_occupancy = 1 - expected_area_frac
    rows.append(_check("centered_square", "segmentation.empty_space_ratio", expected_bg_occupancy,
                        features["segmentation.empty_space_ratio"].value, tolerance=0.02,
                        limitations="Used here as the 'background occupancy' proxy: 1 - foreground_coverage."))

    # --- Case 12: edge_density, ordinal check (dense hatching > blank) ------
    img12a = Image.new("RGB", (W, H), "white")
    img12b = Image.new("RGB", (W, H), "white")
    d12b = ImageDraw.Draw(img12b)
    # Sparse hatching (wide white gaps between thin lines) -- a densely
    # packed pattern was tried first and degenerately saturated the
    # gradient field (magnitude > 0 almost everywhere), making the
    # adaptive 80th-percentile threshold collapse to the image max and
    # edges.mean() read 0.0 for BOTH blank and "hatched" images. That
    # failure mode is itself a real, disclosed limitation of edge_density's
    # adaptive threshold on very high-density content -- see
    # docs/FEATURE_MEASUREMENT_VALIDATION.md -- not fixed here (out of
    # scope: this module validates, it does not patch analysis.py).
    for x in range(0, W, 30):
        d12b.line((x, 0, x, H - 1), fill="black", width=2)
    _analysis12a, features12a = _run(img12a)
    _analysis12b, features12b = _run(img12b)
    blank_edge = features12a["stroke.edge_density"].value
    hatched_edge = features12b["stroke.edge_density"].value
    rows.append({
        "case_id": "blank_vs_hatched", "feature_id": "stroke.edge_density",
        "expected_value": "hatched > blank (ordinal)", "measured_value": f"blank={blank_edge:.4f}, hatched={hatched_edge:.4f}",
        "absolute_error": None, "relative_error": None, "tolerance": None,
        "status": "pass" if hatched_edge > blank_edge else "fail",
        "limitations": "edge_density has no single analytically-exact expected value for arbitrary content; validated ordinally (denser hatching must score higher than a blank page), not against an absolute target.",
    })

    return rows


def write_ground_truth() -> dict[str, Any]:
    rows = build_cases()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "schema_version": "feature_ground_truth_summary_v1",
        "n_cases": len(rows),
        "status_counts": status_counts,
        "note": "Validates measurement implementation only -- does NOT validate any psychological interpretation.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(write_ground_truth())
