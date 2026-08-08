"""Phase 2C.4A bootstrap-CI tests -- synthetic data only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c4 import uncertainty as unc


def _synthetic(n=20):
    pilot_ids = [f"p{i}" for i in range(n)]
    gt = {"a": {pid: ("present" if i < n // 2 else "absent") for i, pid in enumerate(pilot_ids)}}
    pred = {"a": {pid: (i % 3 == 0) for i, pid in enumerate(pilot_ids)}}
    return pilot_ids, gt, pred


class BootstrapDeterminismTests(unittest.TestCase):
    def test_same_seed_same_inputs_gives_identical_output(self):
        pilot_ids, gt, pred = _synthetic()
        r1 = unc.bootstrap_macro_ci(pilot_ids, gt, pred, class_names=("a",), n_bootstrap=200, seed=42)
        r2 = unc.bootstrap_macro_ci(pilot_ids, gt, pred, class_names=("a",), n_bootstrap=200, seed=42)
        self.assertEqual(r1, r2)

    def test_different_seed_can_differ(self):
        pilot_ids, gt, pred = _synthetic()
        r1 = unc.bootstrap_macro_ci(pilot_ids, gt, pred, class_names=("a",), n_bootstrap=200, seed=1)
        r2 = unc.bootstrap_macro_ci(pilot_ids, gt, pred, class_names=("a",), n_bootstrap=200, seed=2)
        # not asserting inequality (could coincide), just that both run and return valid bounds
        for r in (r1, r2):
            self.assertLessEqual(r["macro_balanced_accuracy_ci_low"], r["macro_balanced_accuracy_ci_high"])


class BootstrapBoundsTests(unittest.TestCase):
    def test_ci_brackets_reasonable_range_for_perfect_predictor(self):
        n = 30
        pilot_ids = [f"p{i}" for i in range(n)]
        gt = {"a": {pid: ("present" if i < n // 2 else "absent") for i, pid in enumerate(pilot_ids)}}
        pred = {"a": {pid: (status == "present") for pid, status in gt["a"].items()}}
        result = unc.bootstrap_macro_ci(pilot_ids, gt, pred, class_names=("a",), n_bootstrap=500, seed=7)
        self.assertAlmostEqual(result["macro_balanced_accuracy_ci_low"], 1.0, places=6)
        self.assertAlmostEqual(result["macro_balanced_accuracy_ci_high"], 1.0, places=6)

    def test_records_seed_and_n_bootstrap(self):
        pilot_ids, gt, pred = _synthetic()
        result = unc.bootstrap_macro_ci(pilot_ids, gt, pred, class_names=("a",), n_bootstrap=100, seed=99)
        self.assertEqual(result["seed"], 99)
        self.assertEqual(result["n_bootstrap"], 100)
        self.assertEqual(result["n_images_in_cohort"], len(pilot_ids))


if __name__ == "__main__":
    unittest.main()
