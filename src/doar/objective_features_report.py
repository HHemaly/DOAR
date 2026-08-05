"""Builds the dedicated `objective_features.json` document (DOAR-TRACE 4B).

Consumes `Analysis.to_dict()["objective_features"]` -- the already-serialized
dict of `feature_id -> asdict(FeatureValue)` that `analysis.py::analyze_image`
now populates for every run (see `CURRENT_TO_TARGET_GAP_V2.md`). Does not
change `features.py::FeatureValue` or `serialize_feature_row` -- this module
only re-wraps that same real data into the specific per-feature envelope
this increment's report needs: feature_id, value, unit, method, missing,
evidence_id, judge_status, limitations.

`unit` is honestly `None` for every feature: `features.py` does not track
physical units for any of its 59 features today, and inventing units here
(e.g. guessing "pixels" or "ratio") would be a fabricated addition beyond
what the extractor actually records -- flagged as a real, known gap rather
than silently patched.

`judge_status` is `"not_evaluated"` for every feature in this phase: no
per-feature judge exists yet (Section 4G's `feature_judge` is schema-only
in this increment) -- this is not a placeholder bug, it is the honest
current state.

`missing` values keep their raw (possibly NaN) `value` rather than
collapsing it to `None`, per the task's explicit "preserve honest
NaN/missing values" requirement -- NaN is a valid, if non-standard, JSON
token that Python's `json.dumps` already emits by default, matching this
codebase's existing convention (`case_output.py::write_versioned` never
passes `allow_nan=False`).
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "objective_features_report_v1"

# The only features.py family with a currently-known, real limitation
# (no shape/enclosed-region detector exists). Extending this per-family map
# is the honest way to record limitations without inventing one for every
# feature that has none.
_FAMILY_LIMITATIONS: dict[str, list[str]] = {
    "shape.enclosed_shape_count": ["No shape/enclosed-region detector exists in this release."],
    "shape.repetition_score": ["No shape-repetition detector exists in this release."],
}


def build_objective_features_document(objective_features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`objective_features` is `Analysis.to_dict()["objective_features"]`:
    {feature_id: {value, valid_min, valid_max, confidence, method,
    evidence_id, missing, version}}."""
    features = []
    for feature_id, fv in objective_features.items():
        features.append({
            "feature_id": feature_id,
            "value": fv["value"],
            "unit": None,
            "method": fv["method"],
            "missing": fv["missing"],
            "evidence_id": fv["evidence_id"],
            "judge_status": "not_evaluated",
            "limitations": list(_FAMILY_LIMITATIONS.get(feature_id, [])),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_count": len(features),
        "missing_count": sum(1 for f in features if f["missing"]),
        "features": features,
    }
