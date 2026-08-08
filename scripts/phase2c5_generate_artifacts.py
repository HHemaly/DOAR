#!/usr/bin/env python
"""Phase 2C.5 Stage 10: writes non-private, committed machine-readable
artifacts from the rule-traceability, expansion-sampling, and
quality-metrics modules. Does NOT read any private image or store the
feasibility pilot's own raw proposal CSV verbatim (that stays under
outputs/, gitignored) -- the feasibility summary below is a hand-recorded
qualitative table (visual review), not an automated export.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import expansion_sampling as exp_mod  # noqa: E402
from doar.phase2c5 import rule_traceability as trace_mod  # noqa: E402
from doar.phase2c5.ontology import PART_TARGETS  # noqa: E402

OUT_DIR = ROOT / "artifacts/phase2c5"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUT_DIR / "rule_annotation_traceability.csv", trace_mod.to_rows())

    schema = {
        "schema_version": "phase2c5_part_annotation_schema_v1",
        "part_ontology_version": "phase2c5_part_ontology_v1",
        "targets": list(PART_TARGETS),
        "object_statuses": ["present", "absent", "uncertain", "not_assessable"],
        "bbox_sources": ["model_proposed", "human_accepted", "human_edited", "human_drawn"],
        "attributes": {
            "eye": {"eye_state": ["open", "closed", "uncertain", "not_assessable"],
                    "eye_detail": ["detailed", "undetailed", "missing", "uncertain", "not_assessable"]},
            "mouth": {}, "hand": {}, "face": {}, "person": {},
        },
        "provenance_fields_per_instance": [
            "instance_index", "bbox", "bbox_source", "proposal_model", "proposal_checkpoint",
            "proposal_revision", "proposal_prompt", "proposal_threshold", "proposal_timestamp",
        ],
        "notes": "One row = one (pilot_id, target_name, annotator_id) judgment; instances/attributes "
                 "serialized as JSON cells. See src/doar/phase2c5/schema.py for full validation rules.",
    }
    (OUT_DIR / "annotation_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Feasibility summary -- hand-recorded from visual review of
    # outputs/phase2c5/feasibility_pilot_private/*.png (private, gitignored)
    # against outputs/phase2c5/feasibility_pilot_raw_proposals_private.csv.
    # Counts are from the 15-image dev-eligible pilot; "go_no_go" reflects
    # the qualitative visual read documented in
    # PHASE2C5_RULE_CRITICAL_ANNOTATION_REPORT.md section 5.
    feasibility_rows = [
        {"target": "person", "model": "grounding_dino", "n_proposal_boxes": 41,
         "n_images_with_proposal": 14, "qualitative_read": "mostly correct, strong on multi-figure scenes",
         "go_no_go": "go"},
        {"target": "person", "model": "owlv2", "n_proposal_boxes": 13,
         "n_images_with_proposal": 5, "qualitative_read": "correct where offered, lower recall than grounding_dino",
         "go_no_go": "go"},
        {"target": "face", "model": "grounding_dino", "n_proposal_boxes": 29,
         "n_images_with_proposal": 13, "qualitative_read": "consistently reasonable, single and multi-figure",
         "go_no_go": "go"},
        {"target": "face", "model": "owlv2", "n_proposal_boxes": 9,
         "n_images_with_proposal": 5, "qualitative_read": "reasonable where offered, lower recall",
         "go_no_go": "go"},
        {"target": "eye", "model": "grounding_dino", "n_proposal_boxes": 16,
         "n_images_with_proposal": 9, "qualitative_read": "tight/correct on clear drawn faces, false positives "
         "on heart shapes in one image (p2b_0023)",
         "go_no_go": "go_with_review"},
        {"target": "eye", "model": "owlv2", "n_proposal_boxes": 5,
         "n_images_with_proposal": 2, "qualitative_read": "tight/correct in every inspected case, "
         "including 2 correct boxes on a two-face image (p2b_0010)",
         "go_no_go": "go_with_review"},
        {"target": "mouth", "model": "grounding_dino", "n_proposal_boxes": 5,
         "n_images_with_proposal": 3, "qualitative_read": "excellent on 2 clean/simple drawings "
         "(p2b_0010, p2b_0049), false positive on a hair ribbon in a third (p2b_0000)",
         "go_no_go": "go_with_heavy_review"},
        {"target": "mouth", "model": "owlv2", "n_proposal_boxes": 5,
         "n_images_with_proposal": 4, "qualitative_read": "excellent on one clean drawing (p2b_0049), "
         "false positive on a crossed-arms/clothing region in another (p2b_0033)",
         "go_no_go": "go_with_heavy_review"},
        {"target": "hand", "model": "grounding_dino", "n_proposal_boxes": 5,
         "n_images_with_proposal": 4, "qualitative_read": "small sample, plausible placements where offered",
         "go_no_go": "go_with_review"},
        {"target": "hand", "model": "owlv2", "n_proposal_boxes": 5,
         "n_images_with_proposal": 2, "qualitative_read": "4 boxes in one image (p2b_0043) reasonably "
         "placed near actual hand/wrist regions",
         "go_no_go": "go_with_review"},
    ]
    write_csv(OUT_DIR / "proposal_feasibility_summary.csv", feasibility_rows)

    (OUT_DIR / "expansion_sampling_protocol.json").write_text(
        json.dumps(exp_mod.to_protocol_dict(), indent=2), encoding="utf-8")

    readiness = {
        "note": "No live human annotation round has been run yet this phase -- the app/schema/store "
                "were built and tested (synthetic fixtures + a real feasibility pilot's model proposals), "
                "but a human annotator has not yet used phase2c5_part_annotation_app.py to produce a "
                "trusted-review pass. All counts below are therefore 0 by construction, not a data-quality "
                "problem -- this is the honest state at the end of this phase.",
        "human_reviewed_images": 0,
        "human_reviewed_target_judgments": 0,
        "targets_defined": list(PART_TARGETS),
        "single_annotator_caveat": "N/A yet -- no annotator has used the tool. When one does, "
                                    "src/doar/phase2c5/quality.py::SINGLE_ANNOTATOR_CAVEAT applies.",
    }
    (OUT_DIR / "annotation_readiness_summary.json").write_text(
        json.dumps(readiness, indent=2), encoding="utf-8")

    print(f"wrote artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
