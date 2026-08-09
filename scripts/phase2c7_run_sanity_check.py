#!/usr/bin/env python
"""Phase 2C.7 Stage 7: runtime/integration sanity check ONLY -- runs the
frozen `visual_detector.analyze_image` on a small blinded sample of real
drawings and prints the structured output. This is NOT another accuracy
claim (no ground truth is consulted here at all) and NOT another human
annotation task -- purely "does the frozen, image-only API run end to end
on real images and produce sane-looking structured records."

Sample: 5 deterministic dev-eligible Phase 2C.1 pilot images (p2b_ prefix)
-- chosen specifically because they are NOT part of the Phase 2C.6 eye
dev/holdout split at all (a completely different pilot_id namespace), so
this sanity check cannot leak into or be confused with the frozen eye
evaluation's own holdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c2.cohorts import DEV_ELIGIBLE, split_pilot_ids_by_cohort  # noqa: E402
from doar.phase2c7 import detector_policy as pol  # noqa: E402
from doar.phase2c7 import runtime as rt  # noqa: E402
from doar.phase2c7 import visual_detector as vd  # noqa: E402

N_SAMPLE = 5


def build_real_model_predict_fns(eye_best_model: str) -> dict:
    """Thin, print-annotated wrapper around the shared
    `phase2c7.runtime.build_real_model_predict_fns` (moved there during the
    DOAR MVP build so the app can reuse the exact same routing logic) --
    kept here only to preserve this script's progress-reporting prints."""
    print("loading grounding_dino (object classes)...", flush=True)
    print("loading owlv2 (object classes)...", flush=True)
    if "grounding_dino_parts" in eye_best_model:
        print("loading grounding_dino (parts)...", flush=True)
    if "owlv2_parts" in eye_best_model:
        print("loading owlv2 (parts)...", flush=True)
    return rt.build_real_model_predict_fns(eye_best_model)


def main() -> None:
    policy_path = ROOT / "artifacts/phase2c7/visual_detector_policy.json"
    policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
    eye_row = next(r for r in policy_data if r["target"] == "eye")

    eye_entry = pol.build_eye_policy_entry(
        status=eye_row["status"], best_model=eye_row["best_model"],
        best_model_checkpoint=eye_row["best_model_checkpoint"], prompt=eye_row["prompt"],
        threshold=eye_row["threshold"], precision=eye_row["precision"], recall=eye_row["recall"],
        balanced_accuracy=eye_row["balanced_accuracy"],
        n_ground_truth_present=eye_row["n_ground_truth_present"],
        localization_validated=eye_row["localization_validated"], rationale=eye_row["rationale"])
    policy = pol.full_policy(eye_entry)

    mapping_path = ROOT / "outputs/phase2c1/private_pilot_mapping.csv"
    images_dir = ROOT / "outputs/phase2c1/private_images"
    mapping_rows = workspace_mod.load_pilot_mapping(mapping_path)
    dev_ids = sorted(split_pilot_ids_by_cohort(mapping_rows)[DEV_ELIGIBLE])
    sample_ids = dev_ids[::max(1, len(dev_ids) // N_SAMPLE)][:N_SAMPLE]
    print(f"sanity-check sample ({len(sample_ids)} images, Phase 2C.1 dev-eligible, "
          f"disjoint from the eye dev/holdout split): {sample_ids}")

    model_predict_fns = build_real_model_predict_fns(eye_entry.best_model)

    results = []
    for pilot_id in sample_ids:
        image_path = workspace_mod.image_path_for_pilot_id(images_dir, pilot_id)
        records = vd.analyze_image(str(image_path), policy, model_predict_fns=model_predict_fns)
        results.append({"pilot_id": pilot_id, "records": [r.to_dict() for r in records]})
        n_present = sum(1 for r in records if r.present)
        print(f"{pilot_id}: {len(records)} evidence records, {n_present} present")

    out_path = ROOT / "artifacts/phase2c7/automatic_inference_example.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
