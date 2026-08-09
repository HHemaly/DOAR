#!/usr/bin/env python
"""Phase 2C.7 Stages 1-4: loads the real eye export + real detector
proposals, splits dev/holdout, evaluates the eye detector configuration(s)
on the DEV split only, freezes the winning configuration, and applies it
ONCE to the holdout. Writes every artifact this phase's report cites.
Never reruns inference -- reads only what
scripts/phase2c6_run_proposal_batch.py has already produced.

IMPORTANT data-collection fact, discovered and handled explicitly (not
silently): the stored proposals reflect the DEPLOYED fallback policy
(Grounding DINO primary; OWLv2 invoked, per image, only for whichever
targets Grounding DINO found nothing for -- src/doar/phase2c6/
proposal_batch.py::process_one_image). This means:

- "grounding_dino" is always run as primary -> its presence/absence is
  fairly observed on every processed image. Fully fair.
- "combination" (Grounding DINO OR the OWLv2 fallback result) is exactly
  what the deployed pipeline produced -> also fully fair, full coverage.
- A true independent "OWLv2 alone, system-wide" evaluation is NOT
  recoverable from this data: OWLv2's own finding for 'eye' is only ever
  written when Grounding DINO found NOTHING for eye on that image (the
  fallback only overwrites empty targets) -- so a naive "owlv2" filter
  would silently evaluate OWLv2 only on Grounding DINO's failure cases,
  which is a biased, unfair sample, not a system-wide OWLv2 result. This
  script does NOT report that biased number as if it were a fair
  system-wide metric -- it reports it separately, explicitly labeled, as
  a "recovery rate" (of Grounding DINO's dev-set misses, how many did the
  OWLv2 fallback catch) -- itself a real and useful number, just not
  "OWLv2 alone accuracy."
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c2.evaluation import compute_specificity  # noqa: E402
from doar.phase2c5 import store as store_mod  # noqa: E402
from doar.phase2c7 import detector_policy as pol  # noqa: E402
from doar.phase2c7 import dev_holdout_split as split_mod  # noqa: E402
from doar.phase2c7 import eye_evaluation as ev  # noqa: E402
from doar.phase2c7 import eye_export_summary as summ  # noqa: E402

EXPORT_PATH = ROOT / "outputs/phase2c5/exports/phase2c5_part_annotations.csv"
PROPOSALS_PATH = ROOT / "outputs/phase2c6/expansion_raw_proposals_private.csv"
MANIFEST_PATH = ROOT / "artifacts/phase2c6/expansion_manifest_summary.json"
OUT_DIR = ROOT / "artifacts/phase2c7"
MIN_PRECISION = 0.6
MIN_BAL_ACC = 0.6


def load_gt(store, pilot_ids: set[str]) -> tuple[dict, dict, dict]:
    """Returns (status_by_pilot, boxes_by_pilot[only present], count_by_pilot[only present])."""
    rows = {r.pilot_id: r for r in store.values() if r.target_name == "eye" and r.pilot_id in pilot_ids}
    status = {pid: r.status for pid, r in rows.items()}
    boxes = {pid: [i.bbox for i in r.instances] for pid, r in rows.items() if r.status == "present"}
    counts = {pid: len(r.instances) for pid, r in rows.items() if r.status == "present"}
    return status, boxes, counts


def _load_raw_rows() -> list[dict]:
    if not PROPOSALS_PATH.exists():
        return []
    return list(csv.DictReader(PROPOSALS_PATH.open(encoding="utf-8")))


def evaluate_predictions(model_label: str, pred_positive: dict, pred_boxes: dict, gt_status: dict,
                          gt_boxes: dict, gt_counts: dict) -> dict:
    usable_ids = set(pred_positive) & set(gt_status)
    gt_status_usable = {pid: gt_status[pid] for pid in usable_ids}
    presence = ev.presence_metrics(gt_status_usable, pred_positive)
    specificity = compute_specificity(presence.tn, presence.fp)
    bal_acc = ((presence.recall + specificity) / 2
               if presence.recall is not None and specificity is not None else None)
    gt_boxes_usable = {pid: b for pid, b in gt_boxes.items() if pid in usable_ids}
    pred_boxes_usable = {pid: b for pid, b in pred_boxes.items() if pid in usable_ids}
    localization = ev.localization_metrics(gt_boxes_usable, pred_boxes_usable)
    gt_counts_usable = {pid: c for pid, c in gt_counts.items() if pid in usable_ids}
    pred_counts_usable = {pid: len(pred_boxes.get(pid, [])) for pid in gt_counts_usable}
    instance_diag = ev.instance_count_diagnostics(gt_counts_usable, pred_counts_usable)
    return {
        "model": model_label, "n_usable_images": len(usable_ids),
        "n_present": presence.n_present, "n_absent": presence.n_absent,
        "n_uncertain": presence.n_uncertain, "n_not_assessable": presence.n_not_assessable,
        "tp": presence.tp, "fp": presence.fp, "fn": presence.fn, "tn": presence.tn,
        "precision": presence.precision, "recall": presence.recall, "specificity": specificity,
        "balanced_accuracy": bal_acc, "localization": localization, "instance_diagnostics": instance_diag,
    }


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = store_mod.load_store(EXPORT_PATH)

    # ---- Stage 1 ----------------------------------------------------
    export_summary = summ.summarize_target_export(store, "eye")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    completion = summ.completion_against_manifest(set(export_summary["reviewed_pilot_ids_sorted"]),
                                                    set(manifest["pilot_ids"]))
    raw_rows = _load_raw_rows()
    reviewed_ids = set(export_summary["reviewed_pilot_ids_sorted"])
    n_offered = sum(1 for r in raw_rows if r["target"] == "eye" and r["pilot_id"] in reviewed_ids)
    efficiency = ev.annotation_efficiency_diagnostics(export_summary["bbox_source_counts"], n_offered)
    (OUT_DIR / "eye_annotation_summary.json").write_text(
        json.dumps({"export": export_summary, "completion_vs_manifest": completion,
                    "annotation_efficiency": efficiency}, indent=2), encoding="utf-8")

    # ---- Stage 3: dev/holdout split -----------------------------------
    reviewed_pilot_ids = export_summary["reviewed_pilot_ids_sorted"]
    dh_split = split_mod.split_dev_holdout(reviewed_pilot_ids)
    split_mod.assert_no_overlap(dh_split)
    gt_status_all, gt_boxes_all, gt_counts_all = load_gt(store, set(reviewed_pilot_ids))

    # ---- Stage 2/4: evaluate on DEV only -------------------------------
    dev_ids = set(dh_split.dev_pilot_ids)
    gd_pred, gd_boxes = ev.load_grounding_dino_predictions(raw_rows, dev_ids)
    combo_pred, combo_boxes = ev.load_combination_predictions(raw_rows, dev_ids)
    dev_results = {
        "grounding_dino": evaluate_predictions("grounding_dino", gd_pred, gd_boxes, gt_status_all,
                                                gt_boxes_all, gt_counts_all),
        "combination": evaluate_predictions("combination", combo_pred, combo_boxes, gt_status_all,
                                             gt_boxes_all, gt_counts_all),
    }
    fallback_recovery_dev = ev.owlv2_fallback_recovery(raw_rows, dev_ids, gt_status_all)

    comparison_rows = []
    for m, r in dev_results.items():
        comparison_rows.append({
            "model": m, "n_usable_images": r["n_usable_images"], "n_present": r["n_present"],
            "precision": r["precision"], "recall": r["recall"], "balanced_accuracy": r["balanced_accuracy"],
            "mean_iou": r["localization"]["mean_iou"],
            "success_rate_iou_0.3": r["localization"]["success_rate_by_threshold"].get("0.3"),
            "success_rate_iou_0.5": r["localization"]["success_rate_by_threshold"].get("0.5"),
            "correct_count_rate": r["instance_diagnostics"]["correct_count_rate"],
        })
    with (OUT_DIR / "eye_detector_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    # ---- freeze: grounding_dino alone vs the combination -----------------
    candidates = [r for r in dev_results.values()
                  if r["precision"] is not None and r["precision"] >= MIN_PRECISION
                  and r["balanced_accuracy"] is not None and r["balanced_accuracy"] >= MIN_BAL_ACC]
    if candidates:
        frozen = max(candidates, key=lambda r: r["balanced_accuracy"])
        status = pol.VALIDATED_AUTOMATIC
    else:
        frozen = max(dev_results.values(), key=lambda r: (r["balanced_accuracy"] or 0.0))
        status = pol.EXPERIMENTAL_AUTOMATIC

    frozen_uses_combination = frozen["model"] == "combination"
    eye_entry = pol.build_eye_policy_entry(
        status=status,
        best_model=("grounding_dino_parts+owlv2_parts_fallback" if frozen_uses_combination
                    else "grounding_dino_parts"),
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny"
        + (" + google/owlv2-base-patch16-ensemble (fallback)" if frozen_uses_combination else ""),
        prompt="eye. mouth. hand. face. person.", threshold=0.25, precision=frozen["precision"],
        recall=frozen["recall"], balanced_accuracy=frozen["balanced_accuracy"],
        n_ground_truth_present=frozen["n_present"],
        localization_validated=(frozen["localization"]["mean_iou"] is not None
                                 and status == pol.VALIDATED_AUTOMATIC),
        rationale=f"Frozen on dev split (n={frozen['n_usable_images']} usable images) before any holdout "
                  f"contact: {frozen['model']} precision={frozen['precision']}, "
                  f"balanced_accuracy={frozen['balanced_accuracy']}, mean_iou="
                  f"{frozen['localization']['mean_iou']}. Grounding DINO-vs-OWLv2-alone comparison could "
                  f"not be made fairly system-wide from stored fallback data (see script docstring); "
                  f"OWLv2's role evaluated instead as a fallback recovery contribution (see "
                  f"fallback_recovery in eye_holdout_metrics.json).")

    # ---- Stage 3: apply frozen config to holdout ONCE, descriptive -----
    split_mod.assert_holdout_untouched_by(dev_ids, dh_split)
    holdout_ids = set(dh_split.holdout_pilot_ids)
    if frozen_uses_combination:
        h_pred, h_boxes = ev.load_combination_predictions(raw_rows, holdout_ids)
    else:
        h_pred, h_boxes = ev.load_grounding_dino_predictions(raw_rows, holdout_ids)
    holdout_result = evaluate_predictions(frozen["model"], h_pred, h_boxes, gt_status_all, gt_boxes_all,
                                           gt_counts_all)
    fallback_recovery_holdout = ev.owlv2_fallback_recovery(raw_rows, holdout_ids, gt_status_all)

    (OUT_DIR / "eye_holdout_metrics.json").write_text(json.dumps({
        "frozen_model": frozen["model"], "frozen_status": status,
        "dev_split_n": len(dh_split.dev_pilot_ids), "holdout_split_n": len(dh_split.holdout_pilot_ids),
        "seed": dh_split.seed, "dev_fraction": dh_split.dev_fraction,
        "dev_result": frozen, "holdout_result": holdout_result,
        "owlv2_fallback_recovery_dev": fallback_recovery_dev,
        "owlv2_fallback_recovery_holdout": fallback_recovery_holdout,
    }, indent=2), encoding="utf-8")

    policy = pol.full_policy(eye_entry)
    (OUT_DIR / "visual_detector_policy.json").write_text(
        json.dumps(pol.to_rows(policy), indent=2), encoding="utf-8")
    rows_out = pol.to_rows(policy)
    with (OUT_DIR / "target_validation_status.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "eye_status": status, "eye_frozen_config": frozen["model"],
        "n_reviewed": len(reviewed_pilot_ids), "n_dev": len(dh_split.dev_pilot_ids),
        "n_holdout": len(dh_split.holdout_pilot_ids),
        "dev_precision": frozen["precision"], "dev_recall": frozen["recall"],
        "dev_balanced_accuracy": frozen["balanced_accuracy"],
        "holdout_precision": holdout_result["precision"], "holdout_recall": holdout_result["recall"],
        "holdout_balanced_accuracy": holdout_result["balanced_accuracy"],
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
