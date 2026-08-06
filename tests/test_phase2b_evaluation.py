from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.evaluation import (
    MIN_POSITIVE_SUPPORT, compute_class_metrics, ranking_separation,
    threshold_sensitivity, write_csv,
)


class ComputeClassMetricsTests(unittest.TestCase):
    def test_perfect_predictions_yield_precision_recall_1(self):
        gt = {"a": "present", "b": "present", "c": "absent", "d": "absent"}
        pred = {"a": True, "b": True, "c": False, "d": False}
        m = compute_class_metrics("x", gt, pred)
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 1.0)
        self.assertEqual(m.f1, 1.0)

    def test_uncertain_and_not_assessable_are_excluded_from_metrics(self):
        gt = {"a": "present", "b": "uncertain", "c": "not_assessable", "d": "absent"}
        pred = {"a": True, "b": True, "c": True, "d": False}
        m = compute_class_metrics("x", gt, pred)
        self.assertEqual(m.n_usable, 2)  # only a and d
        self.assertEqual(m.n_uncertain, 1)
        self.assertEqual(m.n_not_assessable, 1)

    def test_sufficient_support_threshold(self):
        gt = {f"img{i}": "present" for i in range(MIN_POSITIVE_SUPPORT)}
        gt["neg"] = "absent"
        pred = {k: False for k in gt}
        m = compute_class_metrics("x", gt, pred)
        self.assertTrue(m.sufficient_support)

        gt2 = {f"img{i}": "present" for i in range(MIN_POSITIVE_SUPPORT - 1)}
        gt2["neg"] = "absent"
        pred2 = {k: False for k in gt2}
        m2 = compute_class_metrics("x", gt2, pred2)
        self.assertFalse(m2.sufficient_support)

    def test_no_positive_examples_yields_none_precision_and_recall(self):
        gt = {"a": "absent", "b": "absent"}
        pred = {"a": False, "b": True}
        m = compute_class_metrics("x", gt, pred)
        self.assertIsNone(m.recall)  # no positives to recall


class RankingSeparationTests(unittest.TestCase):
    def test_perfect_separation_is_1(self):
        gt = {"p1": "present", "p2": "present", "n1": "absent", "n2": "absent"}
        sim = {"p1": 0.9, "p2": 0.8, "n1": 0.2, "n2": 0.1}
        self.assertEqual(ranking_separation(gt, sim), 1.0)

    def test_inverted_separation_is_0(self):
        gt = {"p1": "present", "n1": "absent"}
        sim = {"p1": 0.1, "n1": 0.9}
        self.assertEqual(ranking_separation(gt, sim), 0.0)

    def test_no_usable_pairs_returns_none(self):
        gt = {"a": "uncertain", "b": "not_assessable"}
        sim = {"a": 0.5, "b": 0.5}
        self.assertIsNone(ranking_separation(gt, sim))

    def test_ties_count_as_half(self):
        gt = {"p1": "present", "n1": "absent"}
        sim = {"p1": 0.5, "n1": 0.5}
        self.assertEqual(ranking_separation(gt, sim), 0.5)


class ThresholdSensitivityTests(unittest.TestCase):
    def test_returns_one_row_per_threshold(self):
        gt = {"a": "present", "b": "absent"}
        sim = {"a": 0.6, "b": 0.2}
        rows = threshold_sensitivity(gt, sim, [0.1, 0.5, 0.9])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["threshold"], 0.1)


class WriteCsvTests(unittest.TestCase):
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_csv(Path(d) / "out.csv", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            text = path.read_text(encoding="utf-8")
            self.assertIn("a,b", text)
            self.assertIn("1,2", text)

    def test_empty_rows_writes_empty_file_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_csv(Path(d) / "out.csv", [])
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
