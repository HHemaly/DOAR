"""Generate artifacts/feature_audit.csv by actually running the existing
feature extractor, not by transcribing its source from memory.

Counts and names come from executing `features.py::objective_feature_row`
on a real synthetic image (mirrors what `docs/DOAR_V2_FEATURE_AUDIT.md`
reports: 59 features, not the "~22" figure the task explicitly warned
might be stale) plus the 6 canonical features `evidence_adapter.py`
already wraps and the 10 canonical categories it explicitly declares
`UNIMPLEMENTED_FEATURES` for.

Per-feature judgment columns (status, deterministic, full_page_required,
interpretation_mixed_in) are assigned by category below -- see
docs/DOAR_V2_FEATURE_AUDIT.md for the reasoning. This script only
guarantees the *names and counts* are real; the judgment columns are
authored, not computed.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw

from doar.analysis import analyze_image
from doar.evidence_adapter import UNIMPLEMENTED_FEATURES
from doar.features import objective_feature_row

OUTPUT = ROOT / "artifacts" / "feature_audit.csv"

CATEGORY_JUDGMENT = {
    "quality": dict(status="ADAPT", deterministic="yes", full_page_required="no",
                    interpretation_mixed_in="no",
                    reliability_concerns="Blur/exposure/noise thresholds are engineering defaults, not validated against this dataset."),
    "segmentation": dict(status="ADAPT", deterministic="yes", full_page_required="yes",
                          interpretation_mixed_in="no",
                          reliability_concerns="Foreground/background split only, not object-aware; cannot separate multiple figures from one large one."),
    "composition": dict(status="ADAPT", deterministic="yes", full_page_required="yes",
                         interpretation_mixed_in="no",
                         reliability_concerns="Placement/balance measures assume a single foreground blob; undefined on a blank mask."),
    "colour": dict(status="ADAPT", deterministic="yes", full_page_required="no",
                   interpretation_mixed_in="no",
                   reliability_concerns="Colour-bin classification is a coarse fixed heuristic, not perceptual/calibrated colour science; legacy audit also flagged page-pixel leakage risk in the older implementation this one replaced."),
    "stroke": dict(status="ADAPT", deterministic="yes", full_page_required="no",
                   interpretation_mixed_in="no",
                   reliability_concerns="Edge/gradient-based proxies, sensitive to image resolution and scan quality; not validated against hand-labeled stroke ground truth."),
    "shape": dict(status="ADAPT", deterministic="yes", full_page_required="no",
                  interpretation_mixed_in="no",
                  reliability_concerns="contour_proxy_count is a real connected-component count; enclosed_shape_count/repetition_score are honest NaN placeholders (no shape-semantic detector exists) -- already correctly marked missing=True, not fabricated."),
}

CANONICAL_KEPT = {
    "composition.bounding_box_coverage": "global_composition",
    "composition.centroid_normalized": "size_placement",
    "composition.centroid_x": "size_placement",
    "composition.centroid_y": "size_placement",
    "colour.dominant": "colour",
    "quality.overall_status": "image_page_quality",
}


def _features_py_rows() -> list[dict]:
    tmp = Path(tempfile.mkdtemp())
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    path = tmp / "image.png"
    image.save(path)
    analysis = analyze_image(path, tmp / "out").to_dict()
    row = objective_feature_row(path, analysis)

    rows = []
    for name, fv in sorted(row.items()):
        category = name.split(".")[0]
        judgment = CATEGORY_JUDGMENT[category]
        rows.append({
            "feature_id": name,
            "source_module": "src/doar/features.py::objective_feature_row",
            "computes": "see docs/DOAR_V2_FEATURE_AUDIT.md",
            "output_type": "continuous" if not fv.missing else "continuous_but_unavailable",
            "deterministic": judgment["deterministic"],
            "maps_to_v2_ontology": "not_yet_in_config/feature_ontology.yaml",
            "status": judgment["status"],
            "reliability_concerns": judgment["reliability_concerns"],
            "full_page_visibility_required": judgment["full_page_required"],
            "interpretation_mixed_into_extraction": judgment["interpretation_mixed_in"],
            "currently_honestly_reports_missing": "yes" if fv.missing else "n/a",
        })
    return rows


def _canonical_rows() -> list[dict]:
    rows = []
    for feature_id, category in CANONICAL_KEPT.items():
        rows.append({
            "feature_id": feature_id,
            "source_module": "src/doar/evidence_adapter.py::build_evidence_set",
            "computes": f"category={category}",
            "output_type": "see config/feature_ontology.yaml",
            "deterministic": "yes",
            "maps_to_v2_ontology": "yes (config/feature_ontology.yaml)",
            "status": "KEEP",
            "reliability_concerns": "Already wrapped as EvidenceItem/FeatureObservation with explicit assessability; end-to-end tested.",
            "full_page_visibility_required": "yes" if category in ("global_composition", "size_placement") else "no",
            "interpretation_mixed_into_extraction": "no",
            "currently_honestly_reports_missing": "n/a",
        })
    return rows


def _planned_rows() -> list[dict]:
    rows = []
    for feature_id, info in UNIMPLEMENTED_FEATURES.items():
        rows.append({
            "feature_id": feature_id,
            "source_module": "none",
            "computes": info["reason"],
            "output_type": "undefined_no_extractor",
            "deterministic": "n/a",
            "maps_to_v2_ontology": "yes (config/feature_ontology.yaml, status=PLANNED)",
            "status": "PLANNED",
            "reliability_concerns": "No extractor exists; category is honestly represented as unavailable everywhere it is referenced.",
            "full_page_visibility_required": "unknown_pending_detector",
            "interpretation_mixed_into_extraction": "no",
            "currently_honestly_reports_missing": "yes",
        })
    return rows


def main() -> None:
    rows = _features_py_rows() + _canonical_rows() + _planned_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT} "
          f"({len(_features_py_rows())} features.py + {len(_canonical_rows())} canonical KEEP + {len(_planned_rows())} PLANNED)")


if __name__ == "__main__":
    main()
