"""C1 — OOF probability generation for genuine stacking."""

from __future__ import annotations
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

try:
    import sklearn  # noqa
    _SK = True
except Exception:
    _SK = False


class StratifiedFoldTests(unittest.TestCase):
    def test_folds_cover_all_and_are_deterministic(self):
        from doar.fusion.oof import stratified_folds
        y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        f1 = stratified_folds(y, 4, seed=1)
        f2 = stratified_folds(y, 4, seed=1)
        self.assertEqual(f1, f2)                       # deterministic
        self.assertEqual(set(f1), {0, 1, 2, 3})        # all folds used
        self.assertEqual(len(f1), len(y))

    def test_min_folds(self):
        from doar.fusion.oof import stratified_folds
        with self.assertRaises(ValueError):
            stratified_folds([0, 1], 1)


@unittest.skipUnless(_SK, "sklearn not installed")
class GenerateOofTests(unittest.TestCase):
    def _features(self, d, n=40):
        fp = Path(d) / "features.csv"
        rng = np.random.RandomState(0)
        with open(fp, "w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["image_id", "path", "split", "class", "f0", "f1"])
            for i in range(n):
                cls = ["Angry", "Fear", "Happy", "Sad"][i % 4]
                w.writerow([f"s{i}", "", "train", cls, rng.randn(), rng.randn()])
        return fp

    def test_each_sample_predicted_exactly_once_with_fold_id(self):
        from doar.fusion.oof import generate_oof
        with tempfile.TemporaryDirectory() as d:
            fp = self._features(d)
            res = generate_oof(str(fp), Path(d) / "oof", n_folds=4, seed=1)
            self.assertTrue(res["each_sample_predicted_once"])
            data = json.loads((Path(d) / "oof" / "oof_export.json").read_text())
            ids = [r["sample_id"] for r in data["predictions"]]
            self.assertEqual(len(ids), len(set(ids)))            # each once
            self.assertTrue(all(r["fold_id"] is not None for r in data["predictions"]))
            self.assertTrue(all(r["split"] == "train" for r in data["predictions"]))

    def test_oof_export_is_usable_for_stacking(self):
        # The OOF export must satisfy validate_oof_folds (unique fold per sample).
        from doar.fusion.oof import generate_oof
        from doar.fusion.probability import validate_oof_folds
        with tempfile.TemporaryDirectory() as d:
            fp = self._features(d)
            generate_oof(str(fp), Path(d) / "oof", n_folds=4, seed=1)
            data = json.loads((Path(d) / "oof" / "oof_export.json").read_text())
            ids = [r["sample_id"] for r in data["predictions"]]
            folds = [r["fold_id"] for r in data["predictions"]]
            validate_oof_folds(ids, folds)   # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
