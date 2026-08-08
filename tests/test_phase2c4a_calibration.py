"""Phase 2C.4A calibration tests -- synthetic scores only, no real model
inference. Pins: threshold sweeps never go below a model's original
operating threshold, the predeclared precision-bar selection rule, and
that applying a frozen threshold to a new cohort never re-derives it."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c4 import calibration as cal


class CandidateThresholdsNeverBelowOriginalTests(unittest.TestCase):
    def test_owlv2_grid_starts_at_original_threshold(self):
        self.assertEqual(min(cal.CANDIDATE_THRESHOLDS["owlv2"]), cal.OWLV2_ORIGINAL_THRESHOLD)

    def test_grounding_dino_grid_starts_at_original_threshold(self):
        self.assertEqual(min(cal.CANDIDATE_THRESHOLDS["grounding_dino"]), cal.GROUNDING_DINO_ORIGINAL_THRESHOLD)

    def test_no_candidate_threshold_is_below_original_for_either_model(self):
        self.assertTrue(all(t >= cal.OWLV2_ORIGINAL_THRESHOLD for t in cal.CANDIDATE_THRESHOLDS["owlv2"]))
        self.assertTrue(all(t >= cal.GROUNDING_DINO_ORIGINAL_THRESHOLD
                             for t in cal.CANDIDATE_THRESHOLDS["grounding_dino"]))


class SweepThresholdsTests(unittest.TestCase):
    def test_higher_threshold_never_increases_recall(self):
        gt = {f"p{i}": ("present" if i < 5 else "absent") for i in range(10)}
        scores = {f"p{i}": (1.0 - i / 10) for i in range(10)}
        rows = cal.sweep_thresholds("cls", gt, scores, (0.1, 0.3, 0.5, 0.7))
        recalls = [r["recall"] for r in rows]
        self.assertEqual(recalls, sorted(recalls, reverse=True))


class SelectOperatingPointTests(unittest.TestCase):
    def test_picks_smallest_threshold_clearing_precision_bar(self):
        rows = [
            {"threshold": 0.1, "precision": 0.4, "recall": 0.9, "balanced_accuracy": 0.6},
            {"threshold": 0.2, "precision": 0.7, "recall": 0.6, "balanced_accuracy": 0.65},
            {"threshold": 0.3, "precision": 0.9, "recall": 0.3, "balanced_accuracy": 0.6},
        ]
        result = cal.select_operating_point(rows, min_precision=0.6)
        self.assertEqual(result["decision"], "calibrated")
        self.assertEqual(result["threshold"], 0.2)

    def test_abstains_when_no_threshold_clears_bar(self):
        rows = [
            {"threshold": 0.1, "precision": 0.2, "recall": 0.9, "balanced_accuracy": 0.5},
            {"threshold": 0.2, "precision": 0.3, "recall": 0.5, "balanced_accuracy": 0.5},
        ]
        result = cal.select_operating_point(rows, min_precision=0.6)
        self.assertEqual(result["decision"], "abstain")
        self.assertIsNone(result["threshold"])

    def test_none_precision_never_selected(self):
        rows = [{"threshold": 0.1, "precision": None, "recall": 0.0, "balanced_accuracy": 0.5}]
        result = cal.select_operating_point(rows, min_precision=0.6)
        self.assertEqual(result["decision"], "abstain")


class FreezeAndApplyTests(unittest.TestCase):
    def _dev_data(self):
        gt = {"a": {"p1": "present", "p2": "present", "p3": "absent", "p4": "absent"}}
        scores = {"a": {"p1": 0.9, "p2": 0.2, "p3": 0.1, "p4": 0.05}}
        return gt, scores

    def test_freeze_then_apply_uses_frozen_threshold_verbatim(self):
        gt, scores = self._dev_data()
        frozen = cal.freeze_operating_points(
            "owlv2", cal.OWLV2_ORIGINAL_THRESHOLD, gt, scores,
            thresholds=(0.1, 0.15, 0.2), min_positive_support=1, min_precision=0.5)
        self.assertEqual(len(frozen), 1)
        chosen_threshold = frozen[0].frozen_threshold
        locked_gt = {"a": {"p5": "present", "p6": "absent"}}
        locked_scores = {"a": {"p5": 0.5, "p6": 0.01}}
        applied = cal.apply_frozen_thresholds(frozen, locked_gt, locked_scores)
        self.assertEqual(applied[0]["frozen_threshold"], chosen_threshold)

    def test_apply_frozen_thresholds_reports_abstain_unchanged(self):
        gt, scores = self._dev_data()
        # min_precision impossible to reach -> abstain
        frozen = cal.freeze_operating_points(
            "owlv2", cal.OWLV2_ORIGINAL_THRESHOLD, gt, scores,
            thresholds=(0.1,), min_positive_support=1, min_precision=1.1)
        applied = cal.apply_frozen_thresholds(frozen, gt, scores)
        self.assertEqual(applied[0]["decision"], "abstain")
        self.assertIsNone(applied[0]["frozen_threshold"])


class ApplyFrozenThresholdsNeverRederivesTests(unittest.TestCase):
    """Structural guard: apply_frozen_thresholds must not call
    sweep_thresholds or select_operating_point -- the locked-test
    application path can only consume an already-frozen decision, never
    recompute one."""

    def test_apply_frozen_thresholds_does_not_call_sweep_or_select(self):
        src = (ROOT / "src/doar/phase2c4/calibration.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "apply_frozen_thresholds")
        called_names = {n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn("sweep_thresholds", called_names)
        self.assertNotIn("select_operating_point", called_names)


if __name__ == "__main__":
    unittest.main()
