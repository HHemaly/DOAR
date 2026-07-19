"""C5 / Item 3 — smoke experiment: limited subset for BOTH stages, never test."""

from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import sklearn  # noqa
    _SK = True
except Exception:
    _SK = False
try:
    import torch  # noqa
    _TORCH = True
except Exception:
    _TORCH = False


def _synthetic_dataset(root, per_class=6, size=48):
    from PIL import Image
    import numpy as np
    for si, split in enumerate(("train", "valid", "test")):
        for ci, cls in enumerate(("Angry", "Fear", "Happy", "Sad")):
            d = Path(root) / split / cls
            d.mkdir(parents=True)
            for i in range(per_class):
                seed = si * 100000 + ci * 1000 + i
                arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{i}.png")


@unittest.skipUnless(_SK, "sklearn not installed")
class SmokeExperimentTests(unittest.TestCase):
    def test_limited_subset_and_no_test(self):
        from doar.smoke import run_smoke_experiment
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"
            _synthetic_dataset(data, per_class=6)
            out = Path(d) / "out"
            s = run_smoke_experiment(str(data), str(out), max_samples_per_class=2,
                                     device="cpu", skip_deep=True)
            self.assertFalse(s["test_used"])
            self.assertIn(s["status"], ("PASS", "WARN", "FAIL"))
            # exactly 2 per (train/valid, class); NO test split present
            self.assertEqual(set(s["samples_per_split_class"]), {"train", "valid"})
            for split in ("train", "valid"):
                for cls in ("Angry", "Fear", "Happy", "Sad"):
                    self.assertEqual(s["samples_per_split_class"][split][cls], 2)
            # the limited dataset dir must contain no test folder
            self.assertFalse((out / "limited_dataset" / "test").exists())
            # manifest of the limited dataset has no test rows
            import csv
            rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
            self.assertTrue(all(r["split"] != "test" for r in rows))

    def test_skip_deep_is_warn_not_fail(self):
        from doar.smoke import run_smoke_experiment
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"
            _synthetic_dataset(data, per_class=4)
            s = run_smoke_experiment(str(data), str(Path(d) / "out"),
                                     max_samples_per_class=2, device="cpu", skip_deep=True)
            self.assertIn("small_cnn", s["skipped_steps"])
            self.assertEqual(s["status"], "WARN")

    @unittest.skipUnless(_TORCH, "torch not installed")
    def test_require_deep_runs_cnn_on_limited_subset_and_passes(self):
        from doar.smoke import run_smoke_experiment
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"
            _synthetic_dataset(data, per_class=5)
            out = Path(d) / "out"
            s = run_smoke_experiment(str(data), str(out), max_samples_per_class=2,
                                     device="cpu", require_deep=True)
            self.assertEqual(s["steps"]["small_cnn_one_epoch"], "PASS")
            self.assertEqual(s["device_used"], "cpu")
            self.assertFalse(s["cuda_used"])
            self.assertEqual(s["status"], "PASS")
            self.assertIsNotNone(s["checkpoint"])
            self.assertIsNotNone(s["validation_metrics"])
            # CNN used the SAME limited dataset (2 per class), so the small_cnn
            # training result reflects the limited counts.
            res = json.loads((out / "small_cnn" / "training_result.json").read_text())
            self.assertEqual(res["executed_config"]["physical_batch_size"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
