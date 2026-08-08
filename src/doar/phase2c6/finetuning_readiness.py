"""Phase 2C.6 Stage 8: fine-tuning readiness criteria -- DESIGN ONLY.

No fine-tuning, PEFT, adapter, training, or hyperparameter search happens
anywhere in this module or anywhere else in Phase 2C.6. This defines a
deterministic, documented criterion for deciding WHEN enough real
human-reviewed evidence exists to justify starting FT-1 (Phase 2C.4's own
feasibility ladder, `PHASE2C4_DETECTOR_BENCHMARK_REPORT.md` section 10),
and evaluates that criterion against whatever the current store contains
-- honestly reporting "not ready yet" rather than lowering the bar to
manufacture a "ready" result.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..phase2c5.ontology import PART_TARGETS
from ..phase2c5.quality import bbox_coverage_among_present, instance_count_stats, support_counts

# Predeclared thresholds -- not tuned against any specific dataset size,
# set from general practice for adapting a pretrained open-vocabulary
# detector's detection head (far less data-hungry than training from
# scratch) plus this project's own MIN_POSITIVE_SUPPORT=5 floor
# (src/doar/phase2b/evaluation.py) as an absolute lower bound, scaled up
# by an order of magnitude for a threshold that must generalize to unseen
# images, not just clear Phase 2B's own descriptive-support bar.
MIN_REVIEWED_IMAGES = 150            # at least half of the planned 300-image Stage B expansion
MIN_POSITIVE_INSTANCES_PER_TARGET = 40
MIN_BBOX_COVERAGE_AMONG_PRESENT = 0.95   # near-100%; a schema invariant already enforces this structurally
MAX_SUPPORT_IMBALANCE_RATIO = 8.0    # most-supported target's positive count / least-supported's
MIN_VALIDATION_SPLIT_IMAGES = 30     # held out, never touched during any threshold/hyperparameter choice


@dataclass(frozen=True)
class TargetReadiness:
    target_name: str
    n_present: int
    n_positive_instances: int
    bbox_coverage: float | None
    meets_instance_floor: bool
    meets_bbox_coverage_floor: bool


@dataclass(frozen=True)
class ReadinessReport:
    n_reviewed_images: int
    meets_image_floor: bool
    per_target: tuple[TargetReadiness, ...]
    support_imbalance_ratio: float | None
    meets_imbalance_ceiling: bool
    recommended_validation_split_images: int
    overall_ready: bool
    blocking_reasons: tuple[str, ...]


def evaluate_finetuning_readiness(store: dict, reviewed_pilot_ids: set[str],
                                   targets: tuple[str, ...] = PART_TARGETS) -> ReadinessReport:
    """`store`: a phase2c5.schema.PartAnnotationRecord store (any mix of
    Stage-A pilot + Stage-B expansion rows is fine -- this function counts
    whatever human-reviewed rows are present, it does not care which
    round they came from). `reviewed_pilot_ids`: the set of pilot_ids that
    have at least one human-reviewed row, used only for the image-count
    floor."""
    per_target = []
    positive_counts = []
    for target in targets:
        support = support_counts(store, target)
        coverage = bbox_coverage_among_present(store, target)
        inst_stats = instance_count_stats(store, target)
        n_present = support["present"]
        n_instances = inst_stats["total_instances"] or 0
        positive_counts.append(n_present)
        per_target.append(TargetReadiness(
            target_name=target, n_present=n_present, n_positive_instances=n_instances,
            bbox_coverage=coverage["bbox_coverage"],
            meets_instance_floor=n_instances >= MIN_POSITIVE_INSTANCES_PER_TARGET,
            meets_bbox_coverage_floor=(coverage["bbox_coverage"] is not None
                                        and coverage["bbox_coverage"] >= MIN_BBOX_COVERAGE_AMONG_PRESENT),
        ))

    n_reviewed = len(reviewed_pilot_ids)
    meets_image_floor = n_reviewed >= MIN_REVIEWED_IMAGES

    nonzero = [c for c in positive_counts if c > 0]
    imbalance_ratio = (max(positive_counts) / min(nonzero)) if nonzero and min(nonzero) > 0 else None
    meets_imbalance_ceiling = imbalance_ratio is not None and imbalance_ratio <= MAX_SUPPORT_IMBALANCE_RATIO

    blocking = []
    if not meets_image_floor:
        blocking.append(f"only {n_reviewed} images human-reviewed, need >= {MIN_REVIEWED_IMAGES}")
    for t in per_target:
        if not t.meets_instance_floor:
            blocking.append(f"{t.target_name}: only {t.n_positive_instances} positive instances, "
                             f"need >= {MIN_POSITIVE_INSTANCES_PER_TARGET}")
    if imbalance_ratio is None:
        blocking.append("cannot compute support imbalance -- at least one target has zero positives")
    elif not meets_imbalance_ceiling:
        blocking.append(f"support imbalance ratio {imbalance_ratio:.1f} exceeds ceiling "
                         f"{MAX_SUPPORT_IMBALANCE_RATIO}")
    if n_reviewed < MIN_REVIEWED_IMAGES + MIN_VALIDATION_SPLIT_IMAGES:
        blocking.append(f"not enough reviewed images to also reserve a held-out validation split "
                         f"of >= {MIN_VALIDATION_SPLIT_IMAGES} disjoint from training")

    return ReadinessReport(
        n_reviewed_images=n_reviewed, meets_image_floor=meets_image_floor,
        per_target=tuple(per_target), support_imbalance_ratio=imbalance_ratio,
        meets_imbalance_ceiling=meets_imbalance_ceiling,
        recommended_validation_split_images=MIN_VALIDATION_SPLIT_IMAGES,
        overall_ready=(not blocking), blocking_reasons=tuple(blocking),
    )


def to_dict(report: ReadinessReport) -> dict:
    return {
        "n_reviewed_images": report.n_reviewed_images,
        "meets_image_floor": report.meets_image_floor,
        "min_reviewed_images_threshold": MIN_REVIEWED_IMAGES,
        "min_positive_instances_per_target_threshold": MIN_POSITIVE_INSTANCES_PER_TARGET,
        "min_bbox_coverage_among_present_threshold": MIN_BBOX_COVERAGE_AMONG_PRESENT,
        "max_support_imbalance_ratio_threshold": MAX_SUPPORT_IMBALANCE_RATIO,
        "recommended_validation_split_images": report.recommended_validation_split_images,
        "support_imbalance_ratio": report.support_imbalance_ratio,
        "meets_imbalance_ceiling": report.meets_imbalance_ceiling,
        "overall_ready": report.overall_ready,
        "blocking_reasons": list(report.blocking_reasons),
        "per_target": [
            {"target_name": t.target_name, "n_present": t.n_present,
             "n_positive_instances": t.n_positive_instances, "bbox_coverage": t.bbox_coverage,
             "meets_instance_floor": t.meets_instance_floor,
             "meets_bbox_coverage_floor": t.meets_bbox_coverage_floor}
            for t in report.per_target
        ],
    }
