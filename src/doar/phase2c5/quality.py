"""Phase 2C.5 Stage 8: annotation quality / readiness metrics.

These are plain descriptive counts, never an inter-rater-reliability
statistic -- this project has a single annotator per (pilot_id, target)
in this phase (unlike Phase 2C.1, which has a documented second-reviewer
path). No Cohen's kappa or any other agreement statistic is computed
anywhere in this module; if a caller wants one, they need a second
independent annotator's rows first, which this module does not assume
exist.
"""
from __future__ import annotations

from .ontology import PART_TARGETS
from .schema import PartAnnotationRecord

SINGLE_ANNOTATOR_CAVEAT = (
    "Metrics below describe ONE annotator's judgments per (pilot_id, target). "
    "This is not inter-rater reliability -- no second independent annotator "
    "pass has been run for these targets, and no agreement statistic "
    "(e.g. Cohen's kappa) is computed here."
)


def completion_rate(store: dict[str, PartAnnotationRecord], pilot_ids: list[str],
                     annotator_id: str, targets: tuple[str, ...] = PART_TARGETS) -> dict:
    done = 0
    for pid in pilot_ids:
        recorded = {r.target_name for r in store.values()
                    if r.pilot_id == pid and r.annotator_id == annotator_id}
        if set(targets) <= recorded:
            done += 1
    total = len(pilot_ids)
    return {"complete_images": done, "total_images": total,
            "incomplete_images": total - done,
            "completion_rate": (done / total) if total else None}


def support_counts(store: dict[str, PartAnnotationRecord], target_name: str) -> dict:
    rows = [r for r in store.values() if r.target_name == target_name]
    counts = {"present": 0, "absent": 0, "uncertain": 0, "not_assessable": 0}
    for r in rows:
        counts[r.status] += 1
    return {"target_name": target_name, "n_reviewed": len(rows), **counts}


def bbox_coverage_among_present(store: dict[str, PartAnnotationRecord], target_name: str) -> dict:
    present_rows = [r for r in store.values() if r.target_name == target_name and r.status == "present"]
    with_bbox = [r for r in present_rows if len(r.instances) >= 1]
    n_present = len(present_rows)
    return {"target_name": target_name, "n_present": n_present, "n_with_bbox": len(with_bbox),
            "bbox_coverage": (len(with_bbox) / n_present) if n_present else None}


def instance_count_stats(store: dict[str, PartAnnotationRecord], target_name: str) -> dict:
    counts = [len(r.instances) for r in store.values()
              if r.target_name == target_name and r.status == "present"]
    return {"target_name": target_name, "n_present_rows": len(counts),
            "total_instances": sum(counts),
            "mean_instances_per_present_row": (sum(counts) / len(counts)) if counts else None,
            "max_instances": max(counts) if counts else None}


def proposal_review_outcomes(store: dict[str, PartAnnotationRecord], target_name: str,
                              n_proposal_boxes_offered: int) -> dict:
    """Acceptance/edit/manual/rejection rates. `n_proposal_boxes_offered`
    must come from the raw proposals data (a rejected proposal leaves no
    trace in the final store by design -- rejecting a suggestion is not
    itself an annotation event), so this function cannot compute a
    rejection rate from `store` alone."""
    instances = [inst for r in store.values() if r.target_name == target_name for inst in r.instances]
    by_source = {"model_proposed": 0, "human_accepted": 0, "human_edited": 0, "human_drawn": 0}
    from_a_proposal = 0
    for inst in instances:
        by_source[inst.bbox_source] = by_source.get(inst.bbox_source, 0) + 1
        if inst.proposal_model:
            from_a_proposal += 1
    accepted_or_edited = by_source["human_accepted"] + by_source["human_edited"]
    n_rejected = max(0, n_proposal_boxes_offered - from_a_proposal)
    return {
        "target_name": target_name,
        "n_proposal_boxes_offered": n_proposal_boxes_offered,
        "n_accepted_as_is": by_source["human_accepted"],
        "n_edited": by_source["human_edited"],
        "n_manually_added": by_source["human_drawn"],
        "n_rejected_or_unreviewed": n_rejected,
        "acceptance_rate": (accepted_or_edited / n_proposal_boxes_offered) if n_proposal_boxes_offered else None,
        "edit_rate": (by_source["human_edited"] / n_proposal_boxes_offered) if n_proposal_boxes_offered else None,
        "manual_add_rate": ((by_source["human_drawn"] / len(instances)) if instances else None),
    }


def class_support_imbalance(store: dict[str, PartAnnotationRecord],
                             targets: tuple[str, ...] = PART_TARGETS) -> list[dict]:
    return [support_counts(store, t) for t in targets]


def readiness_summary(store: dict[str, PartAnnotationRecord], pilot_ids: list[str],
                       annotator_id: str, targets: tuple[str, ...] = PART_TARGETS) -> dict:
    return {
        "single_annotator_caveat": SINGLE_ANNOTATOR_CAVEAT,
        "completion": completion_rate(store, pilot_ids, annotator_id, targets),
        "support_by_target": class_support_imbalance(store, targets),
        "bbox_coverage_by_target": [bbox_coverage_among_present(store, t) for t in targets],
        "instance_counts_by_target": [instance_count_stats(store, t) for t in targets],
    }
