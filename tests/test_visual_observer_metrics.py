"""DOAR Visual Observer: the minimal reusable metrics schema for LATER
provider comparison -- proves precision/recall/F1/count-accuracy compute
correctly from given counts. No live provider call, no benchmark claim."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_observer_metrics import ObserverEvaluationMetrics  # noqa: E402


class ObserverEvaluationMetricsTests(unittest.TestCase):
    def test_precision_recall_f1_computed_correctly(self):
        m = ObserverEvaluationMetrics(provider="openai", model="gpt-4o",
                                       true_positives=8, false_positives=2, false_negatives=2)
        self.assertEqual(m.precision, 0.8)
        self.assertEqual(m.recall, 0.8)
        self.assertEqual(m.f1, 0.8)

    def test_zero_true_positives_returns_none_not_zero_division_error(self):
        m = ObserverEvaluationMetrics(provider="openai", model="gpt-4o",
                                       true_positives=0, false_positives=0, false_negatives=0)
        self.assertIsNone(m.precision)
        self.assertIsNone(m.recall)
        self.assertIsNone(m.f1)

    def test_count_accuracy_none_until_evaluated(self):
        m = ObserverEvaluationMetrics(provider="openai", model="gpt-4o",
                                       true_positives=1, false_positives=0, false_negatives=0)
        self.assertIsNone(m.count_accuracy)

    def test_count_accuracy_computed_when_present(self):
        m = ObserverEvaluationMetrics(provider="openai", model="gpt-4o",
                                       true_positives=1, false_positives=0, false_negatives=0,
                                       count_accuracy_matches=3, count_accuracy_total=4)
        self.assertEqual(m.count_accuracy, 0.75)

    def test_to_dict_is_flat_and_json_serializable(self):
        import json
        m = ObserverEvaluationMetrics(provider="openai", model="gpt-4o",
                                       true_positives=5, false_positives=1, false_negatives=0,
                                       correct_abstentions=2)
        json.dumps(m.to_dict())  # must not raise
        self.assertEqual(m.to_dict()["correct_abstentions"], 2)


if __name__ == "__main__":
    unittest.main()
