"""Item 3 & 8 — shared metrics, common prediction format, sample_id alignment."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        from doar.evaluation import compute_metrics
        y = np.array([0, 1, 2, 3, 0, 1])
        proba = np.eye(4)[y]
        m = compute_metrics(y, y, proba)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)
        self.assertAlmostEqual(m["ece"], 0.0, places=6)
        self.assertEqual(len(m["confusion_matrix"]), 4)
        self.assertEqual(len(m["normalized_confusion_matrix"]), 4)
        self.assertIn("per_class", m)

    def test_metrics_have_full_key_set(self):
        from doar.evaluation import compute_metrics
        rng = np.random.RandomState(0)
        y = rng.randint(0, 4, 40)
        proba = rng.dirichlet(np.ones(4), size=40)
        m = compute_metrics(y, proba.argmax(1), proba)
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                    "log_loss", "multiclass_brier", "ece", "per_class",
                    "confusion_matrix", "normalized_confusion_matrix"):
            self.assertIn(key, m)
        # AUC keys present (None when sklearn absent) — never crash.
        self.assertIn("macro_ovr_roc_auc", m)


class ExportAlignTests(unittest.TestCase):
    def _export(self, path, model_id, ids, labels, seed=0, split="valid"):
        from doar.evaluation import save_probability_export
        rng = np.random.RandomState(seed)
        proba = rng.dirichlet(np.ones(4), size=len(ids))
        return save_probability_export(
            path, sample_ids=ids, splits=[split] * len(ids),
            y_true=labels, proba=proba, model_id=model_id,
            checkpoint_hash="abc123", calibration_status="uncalibrated")

    def test_export_roundtrip(self):
        from doar.evaluation import load_probability_export
        with tempfile.TemporaryDirectory() as d:
            p = self._export(Path(d) / "e.json", "m1", ["s0", "s1", "s2"], [0, 1, 2])
            exp = load_probability_export(p)
            self.assertEqual(exp["format"], "doar_probability_export_v1")
            self.assertEqual(exp["count"], 3)
            self.assertEqual(len(exp["predictions"][0]["probabilities"]), 4)

    def test_align_by_sample_id_not_row_order(self):
        from doar.evaluation import load_probability_export, align_exports
        with tempfile.TemporaryDirectory() as d:
            # Same ids + labels, DIFFERENT order in the two exports.
            e1 = load_probability_export(
                self._export(Path(d) / "a.json", "m1", ["s0", "s1", "s2"], [0, 1, 2], seed=1))
            e2 = load_probability_export(
                self._export(Path(d) / "b.json", "m2", ["s2", "s0", "s1"], [2, 0, 1], seed=2))
            ids, y_true, mats, folds = align_exports([e1, e2], "valid")
            self.assertEqual(ids, ["s0", "s1", "s2"])
            self.assertEqual(y_true.tolist(), [0, 1, 2])
            self.assertEqual(len(mats), 2)
            self.assertEqual(mats[0].shape, (3, 4))

    def test_align_fails_on_missing_id(self):
        from doar.evaluation import load_probability_export, align_exports
        with tempfile.TemporaryDirectory() as d:
            e1 = load_probability_export(
                self._export(Path(d) / "a.json", "m1", ["s0", "s1", "s2"], [0, 1, 2]))
            e2 = load_probability_export(
                self._export(Path(d) / "b.json", "m2", ["s0", "s1"], [0, 1]))
            with self.assertRaises(ValueError):
                align_exports([e1, e2], "valid")

    def test_align_fails_on_label_mismatch(self):
        from doar.evaluation import load_probability_export, align_exports
        with tempfile.TemporaryDirectory() as d:
            e1 = load_probability_export(
                self._export(Path(d) / "a.json", "m1", ["s0", "s1"], [0, 1]))
            e2 = load_probability_export(
                self._export(Path(d) / "b.json", "m2", ["s0", "s1"], [0, 2]))  # s1 label differs
            with self.assertRaises(ValueError):
                align_exports([e1, e2], "valid")

    def test_export_rejects_duplicate_ids(self):
        from doar.evaluation import save_probability_export
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_probability_export(
                    Path(d) / "e.json", sample_ids=["s0", "s0"], splits=["valid", "valid"],
                    y_true=[0, 1], proba=np.eye(4)[[0, 1]], model_id="m",
                    checkpoint_hash="h", calibration_status="uncalibrated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
