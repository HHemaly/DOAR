"""Phase 2C.7 eye-evaluation tests -- synthetic data only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7 import eye_evaluation as ev


class IouTests(unittest.TestCase):
    def test_identical_boxes_iou_one(self):
        box = (0.1, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(ev.iou(box, box), 1.0)

    def test_disjoint_boxes_iou_zero(self):
        self.assertEqual(ev.iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)), 0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 0.2, 0.2)
        b = (0.1, 0.1, 0.2, 0.2)
        # intersection: 0.1x0.1=0.01, union: 0.04+0.04-0.01=0.07
        self.assertAlmostEqual(ev.iou(a, b), 0.01 / 0.07, places=6)


class MatchBoxesTests(unittest.TestCase):
    def test_one_to_one_greedy_matching(self):
        gt = [(0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)]
        pred = [(0.5, 0.5, 0.1, 0.1), (0.0, 0.0, 0.1, 0.1)]
        matches = ev.match_boxes(gt, pred)
        self.assertEqual(len(matches), 2)
        by_gi = {gi: (pi, score) for gi, pi, score in matches}
        self.assertEqual(by_gi[0][0], 1)
        self.assertEqual(by_gi[1][0], 0)
        self.assertAlmostEqual(by_gi[0][1], 1.0)

    def test_no_predictions_gives_no_matches(self):
        self.assertEqual(ev.match_boxes([(0.0, 0.0, 0.1, 0.1)], []), [])

    def test_a_predicted_box_is_never_matched_twice(self):
        gt = [(0.0, 0.0, 0.1, 0.1), (0.01, 0.01, 0.1, 0.1)]  # both near the same spot
        pred = [(0.0, 0.0, 0.1, 0.1)]  # only one predicted box
        matches = ev.match_boxes(gt, pred)
        self.assertEqual(len(matches), 1)  # only one GT can claim the single pred box


class PresenceMetricsTests(unittest.TestCase):
    def test_matches_phase2b_compute_class_metrics_semantics(self):
        gt = {"a": "present", "b": "present", "c": "absent"}
        pred = {"a": True, "b": False, "c": False}
        m = ev.presence_metrics(gt, pred)
        self.assertEqual(m.tp, 1)
        self.assertEqual(m.fn, 1)
        self.assertEqual(m.tn, 1)


class LocalizationMetricsTests(unittest.TestCase):
    def test_perfect_localization(self):
        gt = {"p1": [(0.1, 0.1, 0.2, 0.2)]}
        pred = {"p1": [(0.1, 0.1, 0.2, 0.2)]}
        result = ev.localization_metrics(gt, pred, iou_thresholds=(0.5,))
        self.assertEqual(result["n_gt_boxes_total"], 1)
        self.assertAlmostEqual(result["mean_iou"], 1.0)
        self.assertEqual(result["success_rate_by_threshold"]["0.5"], 1.0)

    def test_no_predictions_excluded_from_iou_stats_but_counted_in_total(self):
        gt = {"p1": [(0.1, 0.1, 0.2, 0.2)]}
        pred = {}
        result = ev.localization_metrics(gt, pred)
        self.assertEqual(result["n_gt_boxes_total"], 1)
        self.assertEqual(result["n_iou_values"], 0)
        self.assertIsNone(result["mean_iou"])

    def test_poor_localization_counted_as_zero_iou_not_dropped(self):
        gt = {"p1": [(0.0, 0.0, 0.1, 0.1)]}
        pred = {"p1": [(0.8, 0.8, 0.1, 0.1)]}  # present but nowhere near
        result = ev.localization_metrics(gt, pred, iou_thresholds=(0.3,))
        self.assertEqual(result["n_iou_values"], 1)
        self.assertEqual(result["mean_iou"], 0.0)
        self.assertEqual(result["success_rate_by_threshold"]["0.3"], 0.0)

    def test_multiple_thresholds_computed_independently(self):
        gt = {"p1": [(0.1, 0.1, 0.2, 0.2)]}
        pred = {"p1": [(0.12, 0.12, 0.2, 0.2)]}  # decent but not perfect overlap
        result = ev.localization_metrics(gt, pred, iou_thresholds=(0.3, 0.9))
        self.assertEqual(result["success_rate_by_threshold"]["0.3"], 1.0)
        self.assertEqual(result["success_rate_by_threshold"]["0.9"], 0.0)


class InstanceCountDiagnosticsTests(unittest.TestCase):
    def test_correct_under_over_counts(self):
        gt_counts = {"p1": 2, "p2": 2, "p3": 2}
        pred_counts = {"p1": 2, "p2": 1, "p3": 3}
        result = ev.instance_count_diagnostics(gt_counts, pred_counts)
        self.assertEqual(result["n_correct_count"], 1)
        self.assertEqual(result["n_under_detected"], 1)
        self.assertEqual(result["n_over_detected"], 1)

    def test_missing_prediction_counts_as_zero(self):
        gt_counts = {"p1": 1}
        pred_counts = {}
        result = ev.instance_count_diagnostics(gt_counts, pred_counts)
        self.assertEqual(result["n_zero_predictions"], 1)
        self.assertEqual(result["n_under_detected"], 1)


class AnnotationEfficiencyDiagnosticsTests(unittest.TestCase):
    def test_all_human_drawn_gives_zero_acceptance_and_edit_rate(self):
        result = ev.annotation_efficiency_diagnostics(
            {"human_drawn": 331, "human_accepted": 0, "human_edited": 0}, 98)
        self.assertEqual(result["acceptance_rate"], 0.0)
        self.assertEqual(result["edit_rate"], 0.0)
        self.assertEqual(result["human_drawn_rate"], 1.0)
        self.assertEqual(result["proposal_incorporation_rate"], 0.0)

    def test_zero_proposals_offered_gives_none_incorporation_rate(self):
        result = ev.annotation_efficiency_diagnostics({"human_drawn": 5}, 0)
        self.assertIsNone(result["proposal_incorporation_rate"])

    def test_caveat_text_present(self):
        result = ev.annotation_efficiency_diagnostics({"human_drawn": 1}, 1)
        self.assertIn("cannot distinguish", result["caveat"].lower())


def _row(model, pilot_id, target="eye", bbox=(0.1, 0.1, 0.1, 0.1)):
    return {"model": model, "pilot_id": pilot_id, "target": target,
            "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3]}


class LoadGroundingDinoPredictionsTests(unittest.TestCase):
    def test_matches_full_provenance_string_by_prefix(self):
        rows = [_row("grounding_dino:IDEA-Research/grounding-dino-tiny", "p1")]
        pred, boxes = ev.load_grounding_dino_predictions(rows, {"p1"})
        self.assertTrue(pred["p1"])
        self.assertEqual(len(boxes["p1"]), 1)

    def test_owlv2_rows_never_count_as_grounding_dino(self):
        rows = [_row("owlv2:google/owlv2-base-patch16-ensemble", "p1")]
        pred, _ = ev.load_grounding_dino_predictions(rows, {"p1"})
        self.assertFalse(pred["p1"])  # p1 IS in have_inference (a row exists) but not a GD row

    def test_pilot_with_no_row_at_all_excluded_not_false(self):
        rows = [_row("grounding_dino:x", "p1")]
        pred, _ = ev.load_grounding_dino_predictions(rows, {"p1", "p2"})
        self.assertIn("p1", pred)
        self.assertNotIn("p2", pred)  # never processed -- excluded, not silently False

    def test_other_target_rows_ignored(self):
        rows = [_row("grounding_dino:x", "p1", target="mouth")]
        pred, _ = ev.load_grounding_dino_predictions(rows, {"p1"})
        self.assertFalse(pred["p1"])  # image was processed (row exists) but not for eye


class LoadCombinationPredictionsTests(unittest.TestCase):
    def test_owlv2_only_row_counts_in_combination(self):
        rows = [_row("owlv2:google/owlv2-base-patch16-ensemble", "p1")]
        pred, _ = ev.load_combination_predictions(rows, {"p1"})
        self.assertTrue(pred["p1"])

    def test_grounding_dino_only_row_counts_in_combination(self):
        rows = [_row("grounding_dino:x", "p1")]
        pred, _ = ev.load_combination_predictions(rows, {"p1"})
        self.assertTrue(pred["p1"])

    def test_no_row_for_eye_but_image_processed_gives_false(self):
        rows = [_row("grounding_dino:x", "p1", target="mouth")]
        pred, _ = ev.load_combination_predictions(rows, {"p1"})
        self.assertFalse(pred["p1"])


class Owlv2FallbackRecoveryTests(unittest.TestCase):
    def test_recovery_rate_counts_only_grounding_dino_misses(self):
        # p1: GD missed (no GD row), OWLv2 fallback found it, GT present -> recovered
        # p2: GD missed, OWLv2 also found nothing, GT present -> NOT recovered
        # p3: GD found it directly -> not a "miss" at all, irrelevant to recovery
        rows = [
            _row("owlv2:m", "p1"),
            _row("grounding_dino:x", "p3"),
        ]
        gt_status = {"p1": "present", "p2": "present", "p3": "present"}
        # p2 must appear as "processed" (some row) but with no eye row from either model
        rows.append(_row("grounding_dino:x", "p2", target="mouth"))
        result = ev.owlv2_fallback_recovery(rows, {"p1", "p2", "p3"}, gt_status)
        self.assertEqual(result["n_grounding_dino_misses_with_real_eye_present"], 2)  # p1, p2
        self.assertEqual(result["n_recovered_by_owlv2_fallback"], 1)  # p1 only
        self.assertAlmostEqual(result["recovery_rate"], 0.5)

    def test_caveat_present(self):
        result = ev.owlv2_fallback_recovery([], set(), {})
        self.assertIn("not a system-wide", result["caveat"].lower())

    def test_no_misses_gives_none_recovery_rate(self):
        rows = [_row("grounding_dino:x", "p1")]
        gt_status = {"p1": "present"}
        result = ev.owlv2_fallback_recovery(rows, {"p1"}, gt_status)
        self.assertEqual(result["n_grounding_dino_misses_with_real_eye_present"], 0)
        self.assertIsNone(result["recovery_rate"])


if __name__ == "__main__":
    unittest.main()
