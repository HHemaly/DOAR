#!/usr/bin/env python
"""Phase 2C.6 Stage 10: writes non-private, committed machine-readable
artifacts from the first-50-image batch checkpoint/proposals (private,
gitignored inputs) and the fine-tuning-readiness module. Only opaque
pilot_ids, counts, and aggregate stats are written -- never a path or
image_id.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.ontology import PART_TARGETS  # noqa: E402
from doar.phase2c6 import finetuning_readiness as ftr  # noqa: E402

OUT_DIR = ROOT / "artifacts/phase2c6"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ROOT / "outputs/phase2c6/proposal_batch_checkpoint_private.csv"
    proposals_path = ROOT / "outputs/phase2c6/expansion_raw_proposals_private.csv"
    checkpoint = list(csv.DictReader(checkpoint_path.open(encoding="utf-8")))
    proposals = list(csv.DictReader(proposals_path.open(encoding="utf-8")))

    n_images = len(checkpoint)
    by_status = Counter(r["status"] for r in checkpoint)
    by_target_boxes = Counter(r["target"] for r in proposals)
    by_target_images = defaultdict(set)
    for r in proposals:
        by_target_images[r["target"]].add(r["pilot_id"])

    batch_summary_rows = [
        {"target": t, "n_proposal_boxes": by_target_boxes.get(t, 0),
         "n_images_with_proposal": len(by_target_images.get(t, set())),
         "n_images_processed": n_images,
         "proposal_rate": round(len(by_target_images.get(t, set())) / n_images, 3) if n_images else None}
        for t in PART_TARGETS
    ]
    with (OUT_DIR / "proposal_batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(batch_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(batch_summary_rows)

    elapsed = [float(r["elapsed_s"]) for r in checkpoint]
    n_fallback = sum(1 for r in checkpoint if r["fallback_used"] == "True")
    mean_s = sum(elapsed) / len(elapsed) if elapsed else None
    runtime_summary = {
        "n_images_processed": n_images,
        "n_completed": by_status.get("completed", 0),
        "n_failed": by_status.get("failed", 0),
        "n_skipped_not_assessable": by_status.get("skipped_not_assessable", 0),
        "mean_s_per_image": mean_s,
        "min_s_per_image": min(elapsed) if elapsed else None,
        "max_s_per_image": max(elapsed) if elapsed else None,
        "n_images_used_owlv2_fallback": n_fallback,
        "fallback_rate": round(n_fallback / n_images, 3) if n_images else None,
        "total_proposal_boxes": len(proposals),
        "mean_proposal_boxes_per_image": round(len(proposals) / n_images, 2) if n_images else None,
        "projected_total_s_for_300_images": round(mean_s * 300, 1) if mean_s else None,
        "projected_total_hours_for_300_images": round(mean_s * 300 / 3600, 2) if mean_s else None,
        "note": "These are PROPOSAL counts, not accuracy -- accuracy requires human-reviewed ground "
                "truth, which does not yet exist for this expansion batch (see "
                "finetuning_readiness_criteria.json's n_reviewed_images=0).",
    }
    (OUT_DIR / "proposal_runtime_summary.json").write_text(
        json.dumps(runtime_summary, indent=2), encoding="utf-8")

    ui_spec = {
        "canvas_dependency": {"package": "streamlit-drawable-canvas", "version": "0.9.3",
                               "license": "MIT", "verified_interactively_in_browser": False},
        "display_size": {"min_dim": 400, "max_dim": 800, "aspect_ratio_preserving": True},
        "bbox_sources": ["model_proposed", "human_accepted", "human_edited", "human_drawn"],
        "targets": list(PART_TARGETS),
        "eye_attributes": ["eye_state", "eye_detail"],
        "navigation": ["previous/next image", "jump to image #", "previous/next target"],
        "undo": "provided by the canvas's own native toolbar (Fabric.js), not a separate app-level stack",
    }
    (OUT_DIR / "annotation_ui_spec.json").write_text(json.dumps(ui_spec, indent=2), encoding="utf-8")

    empty_store = {}
    readiness = ftr.evaluate_finetuning_readiness(empty_store, set(), targets=PART_TARGETS)
    (OUT_DIR / "finetuning_readiness_criteria.json").write_text(
        json.dumps(ftr.to_dict(readiness), indent=2), encoding="utf-8")

    print(f"wrote artifacts to {OUT_DIR}")
    print(json.dumps(runtime_summary, indent=2))


if __name__ == "__main__":
    main()
