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
from doar.phase2c4 import detectors as det4_mod  # noqa: E402
from doar.phase2c5 import proposals as prop5_mod  # noqa: E402
from doar.phase2c5.ontology import PART_TARGETS  # noqa: E402
from doar.phase2c7 import detector_policy as pol  # noqa: E402
from doar.phase2c7 import visual_detector as vd  # noqa: E402

N_SAMPLE = 5


def _object_class_predict_fn(detector):
    """`detector.predict_image` (phase2c4.detectors.OpenVocabDetector)
    already returns {class_name: (detected, score)} -- no bbox is tracked
    for the object-class pathway (Phase 2C.4 never needed one), so this
    reports bbox=None honestly rather than fabricating one."""
    def predict(image_path: str) -> dict:
        presence = detector.predict_image(image_path)
        return {cls: (detected, score, None) for cls, (detected, score) in presence.items()}
    return predict


def _part_predict_fn(raw_predict_fn):
    """Wraps a phase2c5.proposals raw predict_fn (returns
    list[RawBoxDetection]) directly -- bypasses `build_part_proposals`
    (which builds annotation-schema `PartInstance`s with no confidence
    field, the wrong shape for live evidence) so the real per-detection
    confidence score is preserved in the output."""
    label_to_targets = prop5_mod.make_label_to_target(PART_TARGETS)

    def predict(image_path: str) -> dict:
        raw = raw_predict_fn(image_path)
        best: dict[str, tuple[bool, float, tuple]] = {t: (False, 0.0, None) for t in PART_TARGETS}
        for d in raw:
            for target in label_to_targets(d.label):
                if target in best and d.score > best[target][1]:
                    best[target] = (True, d.score, d.bbox_xywh_normalized)
        return best
    return predict


def _combined_fallback_predict_fn(primary_predict, fallback_predict):
    """Reproduces the exact deployed eye policy (Phase 2C.6/2C.7): primary
    result per target, falling back to the secondary model ONLY for
    targets the primary found nothing for -- same semantics as
    src/doar/phase2c6/proposal_batch.py::process_one_image, just at
    single-image, live-inference granularity instead of a batch CSV."""
    def predict(image_path: str) -> dict:
        primary = primary_predict(image_path)
        fallback = fallback_predict(image_path)
        merged = dict(primary)
        for target, (detected, _score, _bbox) in primary.items():
            if not detected and fallback.get(target, (False, 0.0, None))[0]:
                merged[target] = fallback[target]
        return merged
    return predict


def build_real_model_predict_fns(eye_best_model: str) -> dict:
    """Loads exactly the real models the frozen policy actually needs --
    never more. `eye_best_model` is one of 'owlv2_parts',
    'grounding_dino_parts', or the composite
    'grounding_dino_parts+owlv2_parts_fallback' (Phase 2C.7's actual
    frozen eye config -- Grounding DINO primary, OWLv2 fallback for
    targets Grounding DINO missed, exactly reproducing the deployed
    evaluation policy for this live demo)."""
    fns = {}

    print("loading grounding_dino (object classes)...", flush=True)
    gd_object = det4_mod.load_real_grounding_dino()
    fns["grounding_dino_object_classes"] = _object_class_predict_fn(gd_object)

    print("loading owlv2 (object classes)...", flush=True)
    owl_object = det4_mod.load_real_owlv2()
    fns["owlv2_object_classes"] = _object_class_predict_fn(owl_object)

    needs_gd_parts = "grounding_dino_parts" in eye_best_model
    needs_owl_parts = "owlv2_parts" in eye_best_model
    gd_parts_fn = owl_parts_fn = None
    if needs_gd_parts:
        print("loading grounding_dino (parts)...", flush=True)
        gd_parts_predict, _meta = prop5_mod.load_real_grounding_dino_parts()
        gd_parts_fn = _part_predict_fn(gd_parts_predict)
    if needs_owl_parts:
        print("loading owlv2 (parts)...", flush=True)
        owl_parts_predict, _meta = prop5_mod.load_real_owlv2_parts()
        owl_parts_fn = _part_predict_fn(owl_parts_predict)

    if "+" in eye_best_model and gd_parts_fn and owl_parts_fn:
        fns[eye_best_model] = _combined_fallback_predict_fn(gd_parts_fn, owl_parts_fn)
    elif gd_parts_fn:
        fns[eye_best_model] = gd_parts_fn
    elif owl_parts_fn:
        fns[eye_best_model] = owl_parts_fn
    return fns


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
