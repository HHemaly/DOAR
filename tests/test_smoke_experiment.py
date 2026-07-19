"""C5 — end-to-end smoke experiment on a tiny SYNTHETIC dataset (never test)."""

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


def _synthetic_dataset(root, per_class=5, size=48):
    from PIL import Image
    import numpy as np
    for si, split in enumerate(("train", "valid", "test")):
        for ci, cls in enumerate(("Angry", "Fear", "Happy", "Sad")):
            d = Path(root) / split / cls
            d.mkdir(parents=True)
            for i in range(per_class):
                # distinct seed per split so no cross-split (near-)duplicates
                seed = si * 100000 + ci * 1000 + i
                arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{i}.png")


@unittest.skipUnless(_SK, "sklearn not installed")
class SmokeExperimentTests(unittest.TestCase):
    def test_end_to_end_smoke_runs_and_never_uses_test(self):
        from doar.smoke import run_smoke_experiment
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"; _synthetic_dataset(data)
            out = Path(d) / "out"
            summary = run_smoke_experiment(str(data), str(out), max_samples_per_class=3,
                                           device="cpu")
            self.assertEqual(summary["status"], "ok")
            self.assertFalse(summary["test_used"])
            self.assertIn("validation_metrics", summary["steps"])
            self.assertTrue((out / "smoke_summary.json").exists())
            # limited manifest must contain NO test rows
            import csv
            rows = list(csv.DictReader(open(out / "manifest_limited.csv", encoding="utf-8")))
            self.assertTrue(all(r["split"] != "test" for r in rows))
            # export must be validation-only
            exp = json.loads((out / "export_valid.json").read_text())
            self.assertTrue(all(r["split"] == "valid" for r in exp["predictions"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
