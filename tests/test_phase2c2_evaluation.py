"""Phase 2C.2 evaluation tests -- synthetic data only, no dependency on
real predictions or drawings. Verifies metrics against hand-computed
values and confirms no threshold is ever re-derived from these results."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import store as store_mod
from doar.phase2c1.schema import AnnotationRecord
from doar.phase2c2 import evaluation as eval_mod
from doar.phase2c2.cohorts import DEV_ELIGIBLE, FULL


def _rec(**overrides):
    defaults = dict(
        pilot_id="p2b_0000", image_id="abc", source_image_group="grp_1",
        class_name="person", status="present", annotator_id="ann1",
        annotation_timestamp="t", instance_count=1, annotator_type="human",
    )
    defaults.update(overrides)
    return AnnotationRecord(**defaults)


class SpecificityTests(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(eval_mod.compute_specificity(tn=8, fp=2), 0.8)

    def test_zero_denominator_returns_none(self):
        self.assertIsNone(eval_mod.compute_specificity(tn=0, fp=0))


class FpFnPilotIdsTests(unittest.TestCase):
    def test_identifies_both_error_types(self):
        gt = {"p2b_0000": "present", "p2b_0001": "absent", "p2b_0002": "present", "p2b_0003": "absent"}
        pred = {"p2b_0000": False, "p2b_0001": True, "p2b_0002": True, "p2b_0003": False}
        result = eval_mod.find_fp_fn_pilot_ids(gt, pred)
        self.assertEqual(result["false_positive_pilot_ids"], ["p2b_0001"])
        self.assertEqual(result["false_negative_pilot_ids"], ["p2b_0000"])

    def test_uncertain_ground_truth_excluded(self):
        gt = {"p2b_0000": "uncertain"}
        pred = {"p2b_0000": True}
        result = eval_mod.find_fp_fn_pilot_ids(gt, pred)
        self.assertEqual(result["false_positive_pilot_ids"], [])
        self.assertEqual(result["false_negative_pilot_ids"], [])

    def test_missing_prediction_excluded(self):
        gt = {"p2b_0000": "present"}
        pred = {}
        result = eval_mod.find_fp_fn_pilot_ids(gt, pred)
        self.assertEqual(result["false_negative_pilot_ids"], [])


class ClipExtractionTests(unittest.TestCase):
    def _raw(self):
        return {
            "p2b_0000": {"person__status": "detected", "person__similarity": "0.31"},
            "p2b_0001": {"person__status": "not_detected", "person__similarity": "0.20"},
            "p2b_0002": {"person__status": "uncertain", "person__similarity": "0.28"},
        }

    def test_predicted_positive_only_detected_counts(self):
        result = eval_mod.predicted_positive_from_clip(self._raw(), "person")
        self.assertEqual(result, {"p2b_0000": True, "p2b_0001": False, "p2b_0002": False})

    def test_similarity_extracted_as_float(self):
        result = eval_mod.similarity_from_clip(self._raw(), "person")
        self.assertEqual(result["p2b_0000"], 0.31)


class ClassicalCvExtractionTests(unittest.TestCase):
    def test_predicted_positive_parses_bool_strings(self):
        raw = {
            "p2b_0000": {"circle_classical__detected": "True", "circle_classical__circularity": "0.9"},
            "p2b_0001": {"circle_classical__detected": "False", "circle_classical__circularity": "0.1"},
        }
        result = eval_mod.predicted_positive_from_classical_cv(raw)
        self.assertEqual(result, {"p2b_0000": True, "p2b_0001": False})

    def test_similarity_is_circularity(self):
        raw = {"p2b_0000": {"circle_classical__detected": "True", "circle_classical__circularity": "0.87"}}
        result = eval_mod.similarity_from_classical_cv(raw)
        self.assertEqual(result["p2b_0000"], 0.87)


class EvaluateOneTests(unittest.TestCase):
    def test_hand_computed_metrics(self):
        # 2 TP, 1 FP, 1 FN, 2 TN.
        gt = {f"p2b_{i:04d}": s for i, s in enumerate(
            ["present", "present", "present", "absent", "absent", "absent"])}
        pred = {f"p2b_{i:04d}": p for i, p in enumerate(
            [True, True, False, True, False, False])}
        sim = {f"p2b_{i:04d}": v for i, v in enumerate([0.9, 0.8, 0.3, 0.6, 0.2, 0.1])}
        cohort_ids = list(gt.keys())
        result = eval_mod.evaluate_one("person", "clip_zero_shot", gt, pred, sim, FULL, cohort_ids)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 2)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["specificity"], 2 / 3)
        self.assertEqual(result["false_positive_pilot_ids"], ["p2b_0003"])
        self.assertEqual(result["false_negative_pilot_ids"], ["p2b_0002"])

    def test_restricts_to_cohort_pilot_ids_only(self):
        gt = {"p2b_0000": "present", "p2b_0001": "absent"}
        pred = {"p2b_0000": True, "p2b_0001": True}
        sim = {"p2b_0000": 0.9, "p2b_0001": 0.9}
        # Cohort excludes p2b_0001 -- its false positive must not count.
        result = eval_mod.evaluate_one("person", "clip_zero_shot", gt, pred, sim, DEV_ELIGIBLE, ["p2b_0000"])
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["n_images_in_cohort"], 1)


class EvaluateAllTests(unittest.TestCase):
    def _build_store_and_predictions(self):
        from doar.phase2b.ontology import CLASS_NAMES
        st = {}
        raw = {}
        for i in range(6):
            pid = f"p2b_{i:04d}"
            status = "present" if i < 3 else "absent"
            row = {}
            for cls in CLASS_NAMES:
                store_mod.upsert(st, _rec(pilot_id=pid, class_name=cls, status=status,
                                           instance_count=1 if status == "present" else 0))
                row[f"{cls}__status"] = "detected" if i % 2 == 0 else "not_detected"
                row[f"{cls}__similarity"] = str(0.5 if i % 2 == 0 else 0.1)
            row["circle_classical__detected"] = "True" if i % 2 == 0 else "False"
            row["circle_classical__circularity"] = "0.9" if i % 2 == 0 else "0.1"
            row["circle_classical__count"] = "1"
            raw[pid] = row
        return st, raw

    def test_produces_rows_for_every_class_and_cohort(self):
        st, raw = self._build_store_and_predictions()
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train" if i < 5 else "test"}
                         for i in range(6)]
        results = eval_mod.evaluate_all(st, raw, mapping_rows)
        from doar.phase2b.ontology import CLASS_NAMES
        # 10 classes x 3 cohorts (CLIP) + 1 class (circle) x 3 cohorts (classical CV) = 33.
        self.assertEqual(len(results), len(CLASS_NAMES) * 3 + 3)

    def test_classical_cv_only_for_circle(self):
        st, raw = self._build_store_and_predictions()
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train"} for i in range(6)]
        results = eval_mod.evaluate_all(st, raw, mapping_rows)
        classical = [r for r in results if r["baseline"] == "classical_cv_circularity"]
        self.assertTrue(all(r["class_name"] == "circle" for r in classical))
        self.assertEqual(len(classical), 3)

    def test_ground_truth_is_never_legacy_provisional(self):
        """If a legacy_provisional_human row exists with a DIFFERENT status
        than the genuine human row for the same (pilot_id, class), the
        provisional one must never leak into the evaluated ground truth."""
        st, raw = self._build_store_and_predictions()
        # Add a disagreeing legacy row for p2b_0000/person.
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", class_name="person", status="absent",
                                   instance_count=0, annotator_id="legacy",
                                   annotator_type="legacy_provisional_human"))
        mapping_rows = [{"pilot_id": f"p2b_{i:04d}", "original_split": "train"} for i in range(6)]
        results = eval_mod.evaluate_all(st, raw, mapping_rows)
        person_full = next(r for r in results if r["class_name"] == "person"
                            and r["cohort"] == FULL and r["baseline"] == "clip_zero_shot")
        # p2b_0000 genuine human status is "present" (3 of first 6 are present) --
        # n_present must still be 3, unaffected by the disagreeing legacy row.
        self.assertEqual(person_full["n_present"], 3)


class CompareToPhase2B20Tests(unittest.TestCase):
    def test_matches_by_class_and_computes_deltas(self):
        new_results = [
            {"class_name": "person", "baseline": "clip_zero_shot", "cohort": FULL,
             "n_present": 59, "precision": 1.0, "recall": 0.03, "ranking_separation": 0.53,
             "sufficient_support": True},
        ]
        old_rows = [
            {"class": "person", "n_present": "9", "precision": "", "recall": "0.0",
             "ranking_separation_vs_random_0.5": "0.7037037037037037", "sufficient_support": "True"},
        ]
        result = eval_mod.compare_to_phase2b_20(new_results, old_rows)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["old_n_present_20"], 9)
        self.assertEqual(row["new_n_present_80"], 59)
        self.assertIsNone(row["old_precision"])  # blank string -> None
        self.assertAlmostEqual(row["old_ranking_separation"], 0.7037037037037037)
        self.assertAlmostEqual(row["new_ranking_separation"], 0.53)

    def test_class_missing_from_old_data_skipped(self):
        new_results = [{"class_name": "person", "baseline": "clip_zero_shot", "cohort": FULL,
                         "n_present": 1, "precision": None, "recall": None,
                         "ranking_separation": None, "sufficient_support": False}]
        result = eval_mod.compare_to_phase2b_20(new_results, [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
