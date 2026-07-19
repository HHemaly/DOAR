"""D2 — runnable late-fusion / stacking arm (validation-only selection)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def _write_base(path, rng, n_valid=40, n_train=60, signal=3.0):
    """Synthetic base-model export: labels correlate with argmax so fusion works."""
    n = n_valid + n_train
    labels = rng.randint(0, 4, size=n)
    logits = rng.randn(n, 4)
    logits[np.arange(n), labels] += signal
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    splits = np.array(["valid"] * n_valid + ["train"] * n_train)
    np.savez(path, probabilities=probs, labels=labels, splits=splits,
             sample_ids=np.array([f"s{i}" for i in range(n)]),
             fold_ids=(np.arange(n) % 3))
    return labels


class LateFusionTests(unittest.TestCase):
    def _two_bases(self, d):
        rng = np.random.RandomState(0)
        # identical labels across models (aligned exports)
        n_valid, n_train = 40, 60
        n = n_valid + n_train
        labels = rng.randint(0, 4, size=n)
        splits = np.array(["valid"] * n_valid + ["train"] * n_train)
        paths = []
        for k in range(2):
            r = np.random.RandomState(k + 1)
            logits = r.randn(n, 4)
            logits[np.arange(n), labels] += 3.0
            probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
            p = Path(d) / f"base{k}.npz"
            np.savez(p, probabilities=probs, labels=labels, splits=splits,
                     sample_ids=np.array([f"s{i}" for i in range(n)]),
                     fold_ids=(np.arange(n) % 3))
            paths.append(str(p))
        return paths

    def test_equal_late_fusion_runs_and_locks_test(self):
        from doar.fusion.late import train_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two_bases(d)
            res = train_late_fusion(base, Path(d) / "out", method="equal_late_fusion")
            self.assertFalse(res["test_used"])
            self.assertEqual(res["selection_split"], "valid")
            self.assertGreater(res["validation_macro_f1"], 0.5)
            self.assertIn("ensemble_uncertainty", res)
            self.assertTrue((Path(d) / "out" / "late_fusion_result.json").exists())

    def test_validation_weighted_fusion_selects_weights(self):
        from doar.fusion.late import train_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two_bases(d)
            res = train_late_fusion(base, Path(d) / "out",
                                    method="validation_weighted_late_fusion")
            self.assertEqual(len(res["weights"]), 2)
            self.assertAlmostEqual(sum(res["weights"]), 1.0, places=6)

    def test_requires_two_bases(self):
        from doar.fusion.late import train_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two_bases(d)
            with self.assertRaises(ValueError):
                train_late_fusion(base[:1], Path(d) / "out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
