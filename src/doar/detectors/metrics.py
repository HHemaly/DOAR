"""
metrics.py -- scoring harness for Phase 3 detector pilots against the
held-out annotation sample (docs/PHASE3_ANNOTATION_SCHEMA.md).

Pure stdlib + the DetectorResult contract from schema.py. No model, no
download, no dependency on any specific detector implementation -- this is
deliberately generic so it can score a classical-CV pilot, a QuickDraw-style
classifier, or a bbox detector with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import DetectorResult


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x_min, y_min, x_max, y_max) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass(frozen=True)
class ClassMetrics:
    class_label: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def meets_threshold(self, min_precision: float, min_recall: float) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall


def match_classifications_by_component_id(
    predictions: list[DetectorResult], ground_truth: list[dict],
) -> dict[str, ClassMetrics]:
    """Score classification-only predictions (shapes/symbols pilot) joined by
    `component_id` -- no bbox/IoU matching needed since DOAR's existing
    connected-component extraction already did localization.

    `ground_truth` entries: {"component_id": str, "class_label": str}.
    A ground-truth class of "none" or "other" is a valid label, not skipped.
    """
    gt_by_id = {g["component_id"]: g["class_label"] for g in ground_truth}
    pred_by_id = {p.component_id: p.class_label for p in predictions if p.component_id is not None}

    all_classes = {c for c in gt_by_id.values()} | {c for c in pred_by_id.values()}
    counts: dict[str, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in all_classes}

    all_ids = set(gt_by_id) | set(pred_by_id)
    for component_id in all_ids:
        true_class = gt_by_id.get(component_id)
        pred_class = pred_by_id.get(component_id)
        if pred_class is not None and pred_class == true_class:
            counts[pred_class]["tp"] += 1
        else:
            if pred_class is not None:
                counts[pred_class]["fp"] += 1
            if true_class is not None:
                counts[true_class]["fn"] += 1

    return {
        cls: ClassMetrics(cls, n["tp"], n["fp"], n["fn"])
        for cls, n in counts.items()
    }


def match_detections_by_iou(
    predictions: list[DetectorResult], ground_truth: list[dict], iou_threshold: float = 0.5,
) -> dict[str, ClassMetrics]:
    """Score bbox-based detections (face/animal/vehicle pilots) by greedy IoU
    matching, per class label.

    `ground_truth` entries: {"bbox": (x1,y1,x2,y2), "class_label": str}.
    Each ground-truth box may be matched at most once (no double-counting).
    """
    all_classes = {p.class_label for p in predictions} | {g["class_label"] for g in ground_truth}
    counts: dict[str, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in all_classes}

    matched_gt: set[int] = set()
    # Highest-confidence predictions matched first (standard greedy detection scoring).
    for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
        if pred.bbox is None:
            continue
        best_iou, best_idx = 0.0, None
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt or gt["class_label"] != pred.class_label:
                continue
            score = iou(pred.bbox, gt["bbox"])
            if score > best_iou:
                best_iou, best_idx = score, idx
        if best_idx is not None and best_iou >= iou_threshold:
            counts[pred.class_label]["tp"] += 1
            matched_gt.add(best_idx)
        else:
            counts[pred.class_label]["fp"] += 1

    for idx, gt in enumerate(ground_truth):
        if idx not in matched_gt:
            counts[gt["class_label"]]["fn"] += 1

    return {
        cls: ClassMetrics(cls, n["tp"], n["fp"], n["fn"])
        for cls, n in counts.items()
    }
