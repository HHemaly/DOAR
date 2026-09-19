"""Generate artifacts/rule_feature_matrix.csv from the real rule registry.

Single source of truth: `src/doar/rule_schema.py` (RULE_PROVENANCE,
REQUIRED_FEATURE_IDS, load_rules_v2). This script derives the matrix
rather than hand-transcribing it, so it cannot drift from the actual
registry the evidence pipeline evaluates against.

`implemented`/`image_only` columns are cross-checked against
`src/doar/evidence_adapter.py::UNIMPLEMENTED_FEATURES` (also real code,
not a second hand-maintained list) -- a feature not in that dict and not
one of the two real composition features is flagged for manual review
rather than guessed.

Run: python3 scripts/generate_rule_feature_matrix.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.evidence_adapter import UNIMPLEMENTED_FEATURES
from doar.rule_schema import load_rules_v2

IMPLEMENTED_FEATURE_IDS = {
    "composition.bounding_box_coverage",
    "composition.centroid_normalized",
    "composition.centroid_x",
    "composition.centroid_y",
}

OUTPUT = ROOT / "artifacts" / "rule_feature_matrix.csv"


def main() -> None:
    rows = []
    for rule in load_rules_v2():
        for feature_id in rule.required_feature_ids:
            if feature_id in IMPLEMENTED_FEATURE_IDS:
                implemented, image_only = "yes", "yes"
            elif feature_id in UNIMPLEMENTED_FEATURES:
                implemented, image_only = "no", "unknown_pending_detector"
            else:
                implemented, image_only = "REVIEW_REQUIRED", "REVIEW_REQUIRED"
            rows.append({
                "rule_id": rule.rule_id,
                "feature_id": feature_id,
                "requirement_type": "required",
                "tier": rule.tier,
                "framework": "CLASSICAL_PROJECTIVE",
                "implemented": implemented,
                "image_only": image_only,
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rule_id", "feature_id", "requirement_type", "tier", "framework", "implemented", "image_only",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
