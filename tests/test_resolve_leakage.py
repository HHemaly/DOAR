"""resolve-leakage — materialize a clean ImageFolder dataset (no override)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _dataset(root, leak=False):
    from PIL import Image
    import numpy as np
    for si, split in enumerate(("train", "valid", "test")):
        for ci, cls in enumerate(("Angry", "Fear", "Happy", "Sad")):
            d = Path(root) / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                # if leak: only image 0 is shared between train and test (partial
                # leakage) so clean train/valid still retain images after removal.
                shared = leak and split in ("train", "test") and i == 0
                seed = (ci * 10) if shared else (si * 1000 + ci * 10 + i)
                arr = (np.random.RandomState(seed).rand(40, 40, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{i}.png")


class ResolveLeakageTests(unittest.TestCase):
    def test_clean_dataset_no_materialize_needed(self):
        from doar.leakage import resolve_leakage
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"
            _dataset(data, leak=False)
            r = resolve_leakage(str(data), str(Path(d) / "out"))
            self.assertTrue(r["leakage_ok"])
            self.assertEqual(r["quarantined_images"], 0)
            self.assertNotIn("clean_dataset", r)   # nothing to materialize

    def test_leaky_dataset_materializes_clean_imagefolder(self):
        from doar.leakage import resolve_leakage
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"
            _dataset(data, leak=True)
            r = resolve_leakage(str(data), str(Path(d) / "out"))
            self.assertFalse(r["leakage_ok"])
            self.assertGreater(r["quarantined_images"], 0)
            self.assertIn("clean_dataset", r)
            clean = Path(r["clean_dataset"])
            # clean dataset is a real ImageFolder (train/valid at least)
            self.assertTrue((clean / "train").exists())
            self.assertTrue((clean / "valid").exists())
            # re-assessing the clean dataset must be leakage-free
            r2 = resolve_leakage(str(clean), str(Path(d) / "out2"))
            self.assertTrue(r2["leakage_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
