"""Phase 2C.7 Stage 2: eye detector evaluation against the real,
human-reviewed Phase 2C.5/2C.6 export.

Presence-level metrics reuse Phase 2B's own, unmodified
`compute_class_metrics`/`ranking_separation` (never reimplemented, same
convention every phase since 2B has followed). Localization (IoU) and
instance-count diagnostics are new this phase -- Phase 2C.4 explicitly
never computed IoU (0% human bbox ground truth coverage at the time); this
module is the first place in the project real bounding-box ground truth
exists to compare against.

IMPORTANT data-collection fact this module's prediction-loading functions
handle explicitly (not silently): the stored proposals reflect the
DEPLOYED fallback policy (Grounding DINO primary; OWLv2 invoked, per
image, only for whichever targets Grounding DINO found nothing for --
src/doar/phase2c6/proposal_batch.py::process_one_image). Grounding DINO's
own presence/absence is fairly observed on every processed image (always
run as primary); the deployed "combination" (Grounding DINO OR the OWLv2
fallback result) is also fairly observed (it is exactly what got
written). A true independent "OWLv2 alone, system-wide" evaluation is NOT
recoverable from this data -- OWLv2's own finding for a target is only
ever written when Grounding DINO found NOTHING for that target, so a
naive per-model filter would silently evaluate OWLv2 only on Grounding
DINO's failure cases. `owlv2_fallback_recovery` reports that restricted
subset explicitly labeled as a recovery-rate diagnostic, never as a
system-wide OWLv2 metric.
"""
from __future__ import annotations

from ..phase2b.evaluation import ClassMetrics, compute_class_metrics

Bbox = tuple[float, float, float, float]


def iou(a: Bbox, b: Bbox) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes: list[Bbox], pred_boxes: list[Bbox]) -> list[tuple[int, int, float]]:
    """Greedy best-IoU one-to-one matching. Returns (gt_index, pred_index,
    iou) for every GT box that found ANY overlapping (iou > 0) predicted
    box -- callers apply their own success threshold downstream, this
    function does not discard low-IoU matches itself."""
    remaining_preds = list(enumerate(pred_boxes))
    matches = []
    for gi, gt_box in enumerate(gt_boxes):
        best_pi, best_iou = None, 0.0
        for pi, pred_box in remaining_preds:
            score = iou(gt_box, pred_box)
            if score > best_iou:
                best_pi, best_iou = pi, score
        if best_pi is not None and best_iou > 0.0:
            remaining_preds = [(pi, b) for pi, b in remaining_preds if pi != best_pi]
            matches.append((gi, best_pi, best_iou))
    return matches


def presence_metrics(ground_truth: dict[str, str], predicted_positive: dict[str, bool]) -> ClassMetrics:
    """Image-level 'does this drawing contain >=1 eye' -- identical
    convention to every other class in this project (compute_class_metrics
    excludes uncertain/not_assessable ground truth from precision/recall,
    counting them separately)."""
    return compute_class_metrics("eye", ground_truth, predicted_positive)


def localization_metrics(gt_boxes_by_image: dict[str, list[Bbox]],
                          pred_boxes_by_image: dict[str, list[Bbox]],
                          iou_thresholds: tuple[float, ...] = (0.3, 0.5)) -> dict:
    """Only defined over images where BOTH genuine ground truth boxes and
    predicted boxes exist (status='present' with >=1 human box, and the
    detector proposed >=1 box) -- an image with no predicted boxes at all
    contributes 0 successes at the GT-box level (already reflected in
    n_gt_boxes_total) but is not itself a 'localization attempt' with an
    IoU value; that is a presence-recall failure, already captured by
    presence_metrics, not double-counted here."""
    all_ious = []
    n_gt_boxes_total = 0
    n_gt_boxes_with_any_predictions_image = 0
    success_counts = dict.fromkeys(iou_thresholds, 0)
    for pilot_id, gt_boxes in gt_boxes_by_image.items():
        n_gt_boxes_total += len(gt_boxes)
        pred_boxes = pred_boxes_by_image.get(pilot_id, [])
        if not pred_boxes:
            continue
        n_gt_boxes_with_any_predictions_image += len(gt_boxes)
        matches = match_boxes(gt_boxes, pred_boxes)
        matched_ious = {gi: sc for gi, _, sc in matches}
        for gi in range(len(gt_boxes)):
            score = matched_ious.get(gi, 0.0)
            all_ious.append(score)
            for t in iou_thresholds:
                if score >= t:
                    success_counts[t] += 1

    sorted_ious = sorted(all_ious)
    n = len(sorted_ious)
    return {
        "n_gt_boxes_total": n_gt_boxes_total,
        "n_gt_boxes_in_images_with_predictions": n_gt_boxes_with_any_predictions_image,
        "n_iou_values": n,
        "mean_iou": (sum(sorted_ious) / n) if n else None,
        "median_iou": (sorted_ious[n // 2] if n % 2 else (sorted_ious[n // 2 - 1] + sorted_ious[n // 2]) / 2) if n else None,
        "min_iou": sorted_ious[0] if n else None,
        "max_iou": sorted_ious[-1] if n else None,
        "success_rate_by_threshold": {
            str(t): (success_counts[t] / n_gt_boxes_with_any_predictions_image
                     if n_gt_boxes_with_any_predictions_image else None)
            for t in iou_thresholds
        },
    }


def instance_count_diagnostics(gt_counts: dict[str, int], pred_counts: dict[str, int]) -> dict:
    """Only over images where genuine ground truth status is 'present'
    (gt_counts only contains those pilot_ids, by caller convention) --
    compares instance COUNT, not localization."""
    correct = under = over = 0
    n_no_prediction = 0
    for pilot_id, gt_n in gt_counts.items():
        pred_n = pred_counts.get(pilot_id, 0)
        if pred_n == 0:
            n_no_prediction += 1
        if pred_n == gt_n:
            correct += 1
        elif pred_n < gt_n:
            under += 1
        else:
            over += 1
    total = len(gt_counts)
    return {
        "n_present_images": total,
        "n_correct_count": correct,
        "n_under_detected": under,
        "n_over_detected": over,
        "n_zero_predictions": n_no_prediction,
        "correct_count_rate": (correct / total) if total else None,
        "under_detection_rate": (under / total) if total else None,
        "over_detection_rate": (over / total) if total else None,
    }


def annotation_efficiency_diagnostics(bbox_source_counts: dict[str, int],
                                       n_proposal_boxes_offered_in_reviewed_images: int) -> dict:
    """`bbox_source_counts`: totals across all reviewed eye instances
    (e.g. {'human_drawn': 331, 'human_accepted': 0, 'human_edited': 0}).
    `n_proposal_boxes_offered_in_reviewed_images`: the number of raw
    detector proposal boxes that existed for the SAME reviewed images
    (from the raw-proposals CSV, independent of the final store) -- used
    only to report how many offered proposals are NOT reflected as
    accepted/edited in the final store. This cannot distinguish a human
    explicitly reviewing-and-rejecting a proposal from a proposal simply
    never being displayed (e.g. a misconfigured proposals-path environment
    variable) -- both look identical in the final store, and this function
    does not claim to tell them apart."""
    total = sum(bbox_source_counts.values())
    accepted = bbox_source_counts.get("human_accepted", 0)
    edited = bbox_source_counts.get("human_edited", 0)
    drawn = bbox_source_counts.get("human_drawn", 0)
    incorporated_from_proposal = accepted + edited
    return {
        "total_saved_instances": total,
        "n_accepted_as_is": accepted,
        "n_edited": edited,
        "n_human_drawn": drawn,
        "acceptance_rate": (accepted / total) if total else None,
        "edit_rate": (edited / total) if total else None,
        "human_drawn_rate": (drawn / total) if total else None,
        "n_proposal_boxes_offered_in_reviewed_images": n_proposal_boxes_offered_in_reviewed_images,
        "proposal_incorporation_rate": (
            incorporated_from_proposal / n_proposal_boxes_offered_in_reviewed_images
            if n_proposal_boxes_offered_in_reviewed_images else None),
        "caveat": "Cannot distinguish an explicit human rejection of a shown proposal from a "
                  "proposal that was never displayed to the annotator -- see this session's "
                  "report for the concrete evidence pointing at the latter.",
    }


def load_grounding_dino_predictions(raw_rows: list[dict], pilot_ids: set[str]
                                     ) -> tuple[dict[str, bool], dict[str, list[Bbox]]]:
    """Fair, full coverage: Grounding DINO is always run as primary, so
    its presence/absence is genuinely observed on every image the batch
    pipeline processed. `raw_rows`: already-loaded rows from a
    phase2c6 raw-proposals CSV (each with 'model'/'pilot_id'/'target'/
    'bbox_x'/'bbox_y'/'bbox_w'/'bbox_h' keys) -- the 'model' column is the
    FULL provenance string (e.g.
    'grounding_dino:IDEA-Research/grounding-dino-tiny'), matched by
    prefix, not equality."""
    have_inference = {r["pilot_id"] for r in raw_rows}
    predicted_positive = {pid: False for pid in pilot_ids if pid in have_inference}
    boxes: dict[str, list[Bbox]] = {pid: [] for pid in pilot_ids if pid in have_inference}
    for r in raw_rows:
        if (not r["model"].startswith("grounding_dino:") or r["target"] != "eye"
                or r["pilot_id"] not in pilot_ids):
            continue
        pid = r["pilot_id"]
        predicted_positive[pid] = True
        boxes.setdefault(pid, []).append(
            (float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])))
    return predicted_positive, boxes


def load_combination_predictions(raw_rows: list[dict], pilot_ids: set[str]
                                  ) -> tuple[dict[str, bool], dict[str, list[Bbox]]]:
    """Fair, full coverage: exactly the deployed pipeline's output (any
    stored row for (pilot_id, target), regardless of which model wrote it
    -- Grounding DINO's own finding, or the OWLv2 fallback's, whichever
    is present)."""
    have_inference = {r["pilot_id"] for r in raw_rows}
    predicted_positive = {pid: False for pid in pilot_ids if pid in have_inference}
    boxes: dict[str, list[Bbox]] = {pid: [] for pid in pilot_ids if pid in have_inference}
    for r in raw_rows:
        if r["target"] != "eye" or r["pilot_id"] not in pilot_ids:
            continue
        pid = r["pilot_id"]
        predicted_positive[pid] = True
        boxes.setdefault(pid, []).append(
            (float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])))
    return predicted_positive, boxes


def owlv2_fallback_recovery(raw_rows: list[dict], pilot_ids: set[str], gt_status: dict[str, str]) -> dict:
    """NOT a system-wide OWLv2 metric (see module docstring) -- of the
    images where Grounding DINO found nothing for eye AND genuine ground
    truth says an eye is present (Grounding DINO's real misses), what
    fraction did the OWLv2 fallback recover? Also reports OWLv2 fallback
    precision on the restricted subset where it actually fired (fallback
    only ever fires on Grounding DINO's empty images)."""
    gd_positive, _ = load_grounding_dino_predictions(raw_rows, pilot_ids)
    gd_misses = {pid for pid, present in gd_positive.items()
                 if not present and gt_status.get(pid) == "present"}
    owlv2_hit_pilots = {r["pilot_id"] for r in raw_rows
                         if r["model"].startswith("owlv2:") and r["target"] == "eye"
                         and r["pilot_id"] in pilot_ids}
    recovered = gd_misses & owlv2_hit_pilots
    owlv2_fired_pilots = {r["pilot_id"] for r in raw_rows
                           if r["model"].startswith("owlv2:") and r["pilot_id"] in pilot_ids}
    tp = sum(1 for pid in owlv2_hit_pilots if gt_status.get(pid) == "present")
    fp = sum(1 for pid in owlv2_hit_pilots if gt_status.get(pid) == "absent")
    return {
        "caveat": "NOT a system-wide OWLv2 metric -- see module docstring. Restricted to images where "
                  "Grounding DINO found nothing for eye (the only subset where the fallback's own eye "
                  "outcome was ever recorded).",
        "n_grounding_dino_misses_with_real_eye_present": len(gd_misses),
        "n_recovered_by_owlv2_fallback": len(recovered),
        "recovery_rate": (len(recovered) / len(gd_misses)) if gd_misses else None,
        "n_owlv2_fallback_fired": len(owlv2_fired_pilots),
        "owlv2_fallback_precision_on_fired_subset": (tp / (tp + fp)) if (tp + fp) else None,
    }
