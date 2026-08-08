"""Phase 2C.4 generalized-evaluation tests -- synthetic data only, no
dependency on any real detector model or drawings."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import store as store_mod
from doar.phase2c1.schema import AnnotationRecord
from doar.phase2c2.cohorts import DEV_ELIGIBLE, FULL, LOCKED_TEST
from doar.phase2c4 import evaluation as eval_mod


def _rec(**overrides):
    defaults = dict(
        pilot_id="p2b_0000", image_id="abc", source_image_group="grp_1",
        class_name="person", status="present", annotator_id="ann1",
        annotation_timestamp="t", instance_count=1, annotator_type="human",
    )
    defaults.update(overrides)
    return AnnotationRecord(**defaults)


class BalancedAccuracyTests(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(eval_mod.balanced_accuracy(0.8, 0.6), 0.7)

    def test_none_recall_propagates_none(self):
        self.assertIsNone(eval_mod.balanced_accuracy(None, 0.6))

    def test_none_specificity_propagates_none(self):
        self.assertIsNone(eval_mod.balanced_accuracy(0.8, None))


class EvaluateModelClassCohortTests(unittest.TestCase):
    def test_hand_computed_metrics_and_balanced_accuracy(self):
        gt = {f"p2b_{i:04d}": s for i, s in enumerate(
            ["present", "present", "present", "absent", "absent", "absent"])}
        pred = {f"p2b_{i:04d}": p for i, p in enumerate(
            [True, True, False, True, False, False])}
        sim = {f"p2b_{i:04d}": v for i, v in enumerate([0.9, 0.8, 0.3, 0.6, 0.2, 0.1])}
        cohort_ids = list(gt.keys())
        result = eval_mod.evaluate_model_class_cohort(
            "test_model", "person", gt, pred, sim, FULL, cohort_ids)
        self.assertEqual(result["model"], "test_model")
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 2)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["specificity"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 2 / 3)

    def test_none_similarity_gives_none_ranking_separation(self):
        gt = {"p2b_0000": "present", "p2b_0001": "absent"}
        pred = {"p2b_0000": True, "p2b_0001": False}
        result = eval_mod.evaluate_model_class_cohort(
            "test_model", "person", gt, pred, None, FULL, ["p2b_0000", "p2b_0001"])
        self.assertIsNone(result["ranking_separation"])


class EvaluateModelTests(unittest.TestCase):
    def _store_and_predictions(self):
        from doar.phase2b.ontology import CLASS_NAMES
        st = {}
        pred_by_class = {}
        for i in range(6):
            pid = f"p2b_{i:04d}"
            status = "present" if i < 3 else "absent"
            for cls in CLASS_NAMES:
                store_mod.upsert(st, _rec(pilot_id=pid, class_name=cls, status=status,
                                           instance_count=1 if status == "present" else 0))
                pred_by_class.setdefault(cls, {})[pid] = (i % 2 == 0)
        return st, pred_by_class

    def test_only_requested_classes_evaluated(self):
        st, pred_by_class = self._store_and_predictions()
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train"} for i in range(6)]
        subset = {"person": pred_by_class["person"]}
        results = eval_mod.evaluate_model("mymodel", st, mapping_rows, subset)
        self.assertEqual({r["class_name"] for r in results}, {"person"})
        self.assertEqual(len(results), 3)  # 1 class x 3 cohorts

    def test_all_classes_produce_three_cohorts_each(self):
        st, pred_by_class = self._store_and_predictions()
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train" if i < 5 else "test"}
                         for i in range(6)]
        results = eval_mod.evaluate_model("mymodel", st, mapping_rows, pred_by_class)
        from doar.phase2b.ontology import CLASS_NAMES
        self.assertEqual(len(results), len(CLASS_NAMES) * 3)
        cohorts_seen = {r["cohort"] for r in results}
        self.assertEqual(cohorts_seen, {FULL, DEV_ELIGIBLE, LOCKED_TEST})

    def test_model_name_recorded_on_every_row(self):
        st, pred_by_class = self._store_and_predictions()
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train"} for i in range(6)]
        results = eval_mod.evaluate_model("owlv2_base", st, mapping_rows, pred_by_class)
        self.assertTrue(all(r["model"] == "owlv2_base" for r in results))

    def test_ground_truth_never_legacy_provisional(self):
        st, pred_by_class = self._store_and_predictions()
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", class_name="person", status="absent",
                                   instance_count=0, annotator_id="legacy",
                                   annotator_type="legacy_provisional_human"))
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train"} for i in range(6)]
        results = eval_mod.evaluate_model("mymodel", st, mapping_rows, pred_by_class)
        person_full = next(r for r in results if r["class_name"] == "person" and r["cohort"] == FULL)
        self.assertEqual(person_full["n_present"], 3)  # unaffected by the disagreeing legacy row


if __name__ == "__main__":
    unittest.main()
