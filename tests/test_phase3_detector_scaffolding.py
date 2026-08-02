"""Tests for the Phase 3 detector-evaluation scaffolding
(src/doar/detectors/). No real detector, model, or download is involved --
purely synthetic data exercising the contract and metrics harness that a
future detector pilot will use. Nothing here is imported by, or affects,
any user-facing code path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class DetectorResultSchemaTests(unittest.TestCase):
    def test_valid_result_constructs(self):
        from doar.detectors.schema import DetectorResult
        r = DetectorResult(
            detector_name="classical_cv_circularity", detector_version="v0_pilot",
            domain="classical_cv", task_type="classification", class_label="circle",
            confidence=0.9, bbox=None, component_id="comp_1",
        )
        self.assertFalse(r.validated)   # default must be False
        self.assertEqual(r.to_dict()["class_label"], "circle")

    def test_confidence_out_of_range_rejected(self):
        from doar.detectors.schema import DetectorResult
        with self.assertRaises(ValueError):
            DetectorResult("d", "v1", "photo", "detection", "face", 1.5, None, None)

    def test_validated_result_requires_limitations(self):
        # A detector cannot claim validated=True while hiding its limitations
        # -- "validated" means "cleared its documented bar", not "flawless".
        from doar.detectors.schema import DetectorResult
        with self.assertRaises(ValueError):
            DetectorResult("d", "v1", "photo", "detection", "face", 0.9, None, None,
                           validated=True, limitations=[])
        # With limitations stated, it's fine.
        r = DetectorResult("d", "v1", "photo", "detection", "face", 0.9, None, None,
                           validated=True, limitations=["untested on line drawings"])
        self.assertTrue(r.validated)


class ClassificationMatchingTests(unittest.TestCase):
    def test_perfect_agreement(self):
        from doar.detectors.schema import DetectorResult
        from doar.detectors.metrics import match_classifications_by_component_id
        preds = [
            DetectorResult("d", "v1", "sketch", "classification", "circle", 0.9, None, "c1"),
            DetectorResult("d", "v1", "sketch", "classification", "star", 0.8, None, "c2"),
        ]
        gt = [{"component_id": "c1", "class_label": "circle"},
              {"component_id": "c2", "class_label": "star"}]
        metrics = match_classifications_by_component_id(preds, gt)
        self.assertEqual(metrics["circle"].precision, 1.0)
        self.assertEqual(metrics["circle"].recall, 1.0)
        self.assertEqual(metrics["star"].f1, 1.0)

    def test_false_positive_and_false_negative(self):
        from doar.detectors.schema import DetectorResult
        from doar.detectors.metrics import match_classifications_by_component_id
        preds = [
            # Predicted "heart" for a component that is really "circle" (FP for
            # heart, FN for circle).
            DetectorResult("d", "v1", "sketch", "classification", "heart", 0.6, None, "c1"),
        ]
        gt = [{"component_id": "c1", "class_label": "circle"},
              {"component_id": "c2", "class_label": "star"}]   # c2 never predicted -> FN for star
        metrics = match_classifications_by_component_id(preds, gt)
        self.assertEqual(metrics["heart"].false_positives, 1)
        self.assertEqual(metrics["heart"].precision, 0.0)
        self.assertEqual(metrics["circle"].false_negatives, 1)
        self.assertEqual(metrics["star"].false_negatives, 1)

    def test_none_and_other_are_valid_classes_not_skipped(self):
        # Per the annotation schema, "none"/"other"/"ambiguous" are legitimate
        # ground-truth answers, not omissions.
        from doar.detectors.schema import DetectorResult
        from doar.detectors.metrics import match_classifications_by_component_id
        preds = [DetectorResult("d", "v1", "sketch", "classification", "none", 0.5, None, "c1")]
        gt = [{"component_id": "c1", "class_label": "none"}]
        metrics = match_classifications_by_component_id(preds, gt)
        self.assertEqual(metrics["none"].true_positives, 1)


class DetectionMatchingTests(unittest.TestCase):
    def test_iou_perfect_overlap_is_one(self):
        from doar.detectors.metrics import iou
        box = (0.0, 0.0, 10.0, 10.0)
        self.assertEqual(iou(box, box), 1.0)

    def test_iou_no_overlap_is_zero(self):
        from doar.detectors.metrics import iou
        self.assertEqual(iou((0, 0, 1, 1), (5, 5, 6, 6)), 0.0)

    def test_bbox_match_above_threshold_counts_as_tp(self):
        from doar.detectors.schema import DetectorResult
        from doar.detectors.metrics import match_detections_by_iou
        preds = [DetectorResult("d", "v1", "photo", "detection", "face", 0.9,
                                (0, 0, 10, 10), None)]
        gt = [{"bbox": (1, 1, 11, 11), "class_label": "face"}]  # heavy overlap
        metrics = match_detections_by_iou(preds, gt, iou_threshold=0.5)
        self.assertEqual(metrics["face"].true_positives, 1)

    def test_bbox_match_below_threshold_is_fp_and_fn(self):
        from doar.detectors.schema import DetectorResult
        from doar.detectors.metrics import match_detections_by_iou
        preds = [DetectorResult("d", "v1", "photo", "detection", "face", 0.9,
                                (0, 0, 2, 2), None)]
        gt = [{"bbox": (50, 50, 60, 60), "class_label": "face"}]  # no overlap
        metrics = match_detections_by_iou(preds, gt, iou_threshold=0.5)
        self.assertEqual(metrics["face"].false_positives, 1)
        self.assertEqual(metrics["face"].false_negatives, 1)

    def test_ground_truth_matched_at_most_once(self):
        # Two overlapping predictions for the same ground-truth box must not
        # both count as true positives -- only the higher-confidence one.
        from doar.detectors.schema import DetectorResult
        from doar.detectors.metrics import match_detections_by_iou
        preds = [
            DetectorResult("d", "v1", "photo", "detection", "face", 0.95, (0, 0, 10, 10), None),
            DetectorResult("d", "v1", "photo", "detection", "face", 0.60, (1, 1, 11, 11), None),
        ]
        gt = [{"bbox": (0, 0, 10, 10), "class_label": "face"}]
        metrics = match_detections_by_iou(preds, gt, iou_threshold=0.5)
        self.assertEqual(metrics["face"].true_positives, 1)
        self.assertEqual(metrics["face"].false_positives, 1)


class AcceptanceThresholdTests(unittest.TestCase):
    def test_meets_threshold_true_and_false_cases(self):
        from doar.detectors.metrics import ClassMetrics
        strong = ClassMetrics("circle", true_positives=9, false_positives=1, false_negatives=1)
        weak = ClassMetrics("fox", true_positives=1, false_positives=5, false_negatives=5)
        self.assertTrue(strong.meets_threshold(min_precision=0.75, min_recall=0.65))
        self.assertFalse(weak.meets_threshold(min_precision=0.75, min_recall=0.65))


if __name__ == "__main__":
    unittest.main(verbosity=2)
