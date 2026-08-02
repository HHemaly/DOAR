"""Item 8 — late fusion + stacking over common-format exports (CPU).

Uses the common probability export format (evaluation.save_probability_export).
equal / validation_weighted paths are pure-numpy; stacking needs sklearn (skipped
if absent). All alignment is by sample_id.
"""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def _make_export(path, model_id, ids, splits, labels, seed, signal=3.0, folds=None):
    from doar.evaluation import save_probability_export
    rng = np.random.RandomState(seed)
    logits = rng.randn(len(ids), 4)
    for i, lab in enumerate(labels):
        logits[i, lab] += signal
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return save_probability_export(
        path, sample_ids=ids, splits=splits, y_true=labels, proba=proba,
        model_id=model_id, checkpoint_hash=f"hash_{model_id}",
        calibration_status="uncalibrated", fold_ids=folds)


class LateFusionTests(unittest.TestCase):
    def _two(self, d, with_train=False, folds=False):
        n_valid = 30
        rng = np.random.RandomState(7)
        vids = [f"v{i}" for i in range(n_valid)]
        vlab = rng.randint(0, 4, n_valid).tolist()
        paths = []
        for k in range(2):
            ids, splits, labels = list(vids), ["valid"] * n_valid, list(vlab)
            fold_arg = None
            if with_train:
                n_train = 40
                tids = [f"t{i}" for i in range(n_train)]
                tlab = rng.randint(0, 4, n_train).tolist()
                ids = tids + vids
                splits = ["train"] * n_train + ["valid"] * n_valid
                labels = tlab + vlab
                if folds:
                    fold_arg = [(i % 3) for i in range(n_train)] + [None] * n_valid
            p = str(Path(d) / f"m{k}.json")
            _make_export(p, f"m{k}", ids, splits, labels, seed=k + 1, folds=fold_arg)
            paths.append(p)
        return paths

    def test_equal_fusion_runs_and_saves_model(self):
        from doar.fusion.late import train_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two(d)
            res = train_late_fusion(base, Path(d) / "out", method="equal_late_fusion")
            self.assertFalse(res["test_used"])
            self.assertEqual(res["selection_split"], "valid")
            self.assertIn("validation_metrics", res)
            self.assertTrue((Path(d) / "out" / "late_fusion_model.json").exists())
            self.assertTrue((Path(d) / "out" / "per_sample_uncertainty.json").exists())

    def test_validation_weighted_saves_weights_and_applies(self):
        from doar.fusion.late import train_late_fusion, load_late_fusion, apply_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two(d)
            res = train_late_fusion(base, Path(d) / "out",
                                    method="validation_weighted_late_fusion")
            self.assertEqual(len(res["weights"]), 2)
            model = load_late_fusion(str(Path(d) / "out" / "late_fusion_model.json"))
            ids, fused = apply_late_fusion(model, base, "valid")
            self.assertEqual(len(ids), 30)
            np.testing.assert_allclose(fused.sum(1), np.ones(30), atol=1e-6)

    def test_stacking_requires_oof_folds(self):
        from doar.fusion.late import train_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two(d, with_train=True, folds=False)  # train present but no folds
            with self.assertRaises(Exception):  # ValueError (no OOF) before sklearn import
                train_late_fusion(base, Path(d) / "out", method="logistic_probability_meta")

    def test_per_sample_uncertainty_is_per_sample(self):
        from doar.fusion.late import train_late_fusion
        import json
        with tempfile.TemporaryDirectory() as d:
            base = self._two(d)
            train_late_fusion(base, Path(d) / "out", method="equal_late_fusion")
            us = json.loads((Path(d) / "out" / "per_sample_uncertainty.json").read_text())
            self.assertEqual(len(us), 30)                 # one entry per sample_id
            self.assertTrue(all("entropy" in v for v in us.values()))

    def test_requires_two_exports(self):
        from doar.fusion.late import train_late_fusion
        with tempfile.TemporaryDirectory() as d:
            base = self._two(d)
            with self.assertRaises(ValueError):
                train_late_fusion(base[:1], Path(d) / "out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
