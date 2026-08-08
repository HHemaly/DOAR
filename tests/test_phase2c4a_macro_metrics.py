"""Phase 2C.4A macro-metric correction tests -- synthetic data only.

Pins the exact bug found on audit: a class a detector never fires on must
not silently disappear from the macro average."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c4 import macro_metrics as mm


class F1ForMacroTests(unittest.TestCase):
    def test_defined_f1_passes_through(self):
        self.assertEqual(mm.f1_for_macro(0.75, n_present=10), 0.75)

    def test_undefined_f1_with_positives_scores_zero(self):
        self.assertEqual(mm.f1_for_macro(None, n_present=5), 0.0)

    def test_undefined_f1_with_no_positives_excluded(self):
        self.assertIsNone(mm.f1_for_macro(None, n_present=0))


class BalancedAccuracyForMacroTests(unittest.TestCase):
    def test_defined_passes_through(self):
        self.assertEqual(mm.balanced_accuracy_for_macro(0.6, n_present=5, n_absent=5), 0.6)

    def test_undefined_with_no_positives_excluded(self):
        self.assertIsNone(mm.balanced_accuracy_for_macro(None, n_present=0, n_absent=5))

    def test_undefined_with_no_negatives_excluded(self):
        self.assertIsNone(mm.balanced_accuracy_for_macro(None, n_present=5, n_absent=0))


def _row(class_name, n_present, n_absent, f1, balanced_accuracy, model="m", cohort="dev_eligible_excl_test"):
    return {"model": model, "cohort": cohort, "class_name": class_name,
            "n_present": n_present, "n_absent": n_absent, "f1": f1, "balanced_accuracy": balanced_accuracy}


class MacroAverageForModelCohortTests(unittest.TestCase):
    _CLASSES = ("a", "b", "c")

    def test_raises_on_missing_class(self):
        rows = [_row("a", 5, 5, 0.5, 0.5), _row("b", 5, 5, 0.5, 0.5)]
        with self.assertRaises(ValueError):
            mm.macro_average_for_model_cohort(rows, class_names=self._CLASSES)

    def test_naive_drop_vs_corrected_macro_f1(self):
        """Reproduces the real YOLO-World-shaped bug: a class the detector
        never fires on (f1=None, n_present>0) must pull the macro average
        DOWN, not be excluded from it."""
        rows = [
            _row("a", n_present=10, n_absent=10, f1=0.8, balanced_accuracy=0.7),
            _row("b", n_present=10, n_absent=10, f1=None, balanced_accuracy=0.5),  # never fired
            _row("c", n_present=10, n_absent=10, f1=None, balanced_accuracy=0.5),  # never fired
        ]
        naive_drop_average = 0.8  # what averaging only the defined values would give
        summary = mm.macro_average_for_model_cohort(rows, class_names=self._CLASSES)
        self.assertEqual(summary.n_classes_included_f1, 3)
        self.assertEqual(summary.n_classes_excluded_f1, 0)
        self.assertAlmostEqual(summary.macro_f1, (0.8 + 0.0 + 0.0) / 3)
        self.assertLess(summary.macro_f1, naive_drop_average)

    def test_class_with_no_positive_ground_truth_excluded_from_f1(self):
        rows = [
            _row("a", n_present=10, n_absent=10, f1=0.8, balanced_accuracy=0.7),
            _row("b", n_present=10, n_absent=10, f1=0.6, balanced_accuracy=0.6),
            _row("c", n_present=0, n_absent=20, f1=None, balanced_accuracy=None),  # no positives at all
        ]
        summary = mm.macro_average_for_model_cohort(rows, class_names=self._CLASSES)
        self.assertEqual(summary.n_classes_included_f1, 2)
        self.assertEqual(summary.excluded_f1_classes, ["c"])
        self.assertAlmostEqual(summary.macro_f1, (0.8 + 0.6) / 2)

    def test_low_support_classes_flagged(self):
        rows = [
            _row("a", n_present=10, n_absent=10, f1=0.8, balanced_accuracy=0.7),
            _row("b", n_present=3, n_absent=10, f1=0.5, balanced_accuracy=0.5),  # below MIN_POSITIVE_SUPPORT
            _row("c", n_present=10, n_absent=10, f1=0.5, balanced_accuracy=0.5),
        ]
        summary = mm.macro_average_for_model_cohort(rows, class_names=self._CLASSES)
        self.assertEqual(summary.low_support_classes, ["b"])


class MacroSummariesByModelTests(unittest.TestCase):
    def test_sorted_by_balanced_accuracy_descending(self):
        rows = [
            _row("a", 10, 10, 0.5, 0.5, model="weak", cohort="dev_eligible_excl_test"),
            _row("a", 10, 10, 0.9, 0.9, model="strong", cohort="dev_eligible_excl_test"),
            _row("a", 10, 10, 0.5, 0.5, model="weak", cohort="full_80"),  # different cohort, must be excluded
        ]
        summaries = mm.macro_summaries_by_model(rows, "dev_eligible_excl_test", class_names=("a",))
        self.assertEqual([s.model for s in summaries], ["strong", "weak"])
        self.assertTrue(all(s.cohort == "dev_eligible_excl_test" for s in summaries))


if __name__ == "__main__":
    unittest.main()
