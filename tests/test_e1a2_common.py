"""Tests for experiments/E1_visual_representation/scripts/e1a2_common.py --
the portable E1-A2 GPU pipeline library. Never downloads a pretrained
model: cache-key/dataset/device/schema/resume behavior is tested with
either the real (already-local) T0 manifest CSV or synthetic in-memory
embeddings, never a real image or a real backbone forward pass."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E1_SCRIPTS = ROOT / "experiments" / "E1_visual_representation" / "scripts"
sys.path.insert(0, str(E1_SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

try:
    import e1a2_common as ec
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment without torch
    ec = None
    _IMPORT_ERROR = exc

T0_MANIFEST = ROOT / "outputs" / "t0_automated" / "final_partition" / "partition_manifest.csv"
EXPECTED_SHA = "4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d"


@unittest.skipIf(ec is None, f"e1a2_common import failed: {_IMPORT_ERROR}")
class DatasetRootRemappingTests(unittest.TestCase):
    def test_remapping_preserves_image_ids_and_splits(self):
        if not T0_MANIFEST.exists():
            self.skipTest("T0 manifest not present in this checkout")
        with open(T0_MANIFEST, encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        raw_by_id = {r["image_id"]: r for r in raw_rows}

        fake_root = "/fake/dataset/root"
        by_split = ec.load_split_rows(fake_root)
        for split in ("train", "valid", "test"):
            for row in by_split[split]:
                raw = raw_by_id[row["image_id"]]
                self.assertEqual(row["class"], raw["class"])
                self.assertEqual(row["split"], raw["new_split"])
                self.assertEqual(row["relative_path"], raw["relative_path"])
                # path = dataset_root / relative_path -- never the T0 manifest's
                # own machine-specific `path` column.
                self.assertEqual(row["path"], str(Path(fake_root) / raw["relative_path"]))
                self.assertNotEqual(row["path"], raw["path"])

    def test_split_sizes_match_frozen_spec(self):
        if not T0_MANIFEST.exists():
            self.skipTest("T0 manifest not present in this checkout")
        by_split = ec.load_split_rows("/fake/root")
        self.assertEqual(len(by_split["train"]), 2599)
        self.assertEqual(len(by_split["valid"]), 284)
        self.assertEqual(len(by_split["test"]), 512)

    def test_no_test_rows_ever_appear_in_train_or_valid(self):
        if not T0_MANIFEST.exists():
            self.skipTest("T0 manifest not present in this checkout")
        by_split = ec.load_split_rows("/fake/root")
        train_ids = {r["image_id"] for r in by_split["train"]}
        valid_ids = {r["image_id"] for r in by_split["valid"]}
        test_ids = {r["image_id"] for r in by_split["test"]}
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(valid_ids & test_ids, set())


@unittest.skipIf(ec is None, f"e1a2_common import failed: {_IMPORT_ERROR}")
class FrozenT0ShaCheckTests(unittest.TestCase):
    def test_real_manifest_matches_expected_sha(self):
        if not T0_MANIFEST.exists():
            self.skipTest("T0 manifest not present in this checkout")
        self.assertEqual(ec.verify_t0_manifest_sha256(), EXPECTED_SHA)

    def test_mismatched_manifest_raises(self):
        original = ec.T0_MANIFEST
        try:
            with tempfile.TemporaryDirectory() as d:
                fake = Path(d) / "partition_manifest.csv"
                fake.write_text("image_id,new_split\nfake,train\n", encoding="utf-8")
                ec.T0_MANIFEST = fake
                with self.assertRaises(RuntimeError):
                    ec.verify_t0_manifest_sha256()
        finally:
            ec.T0_MANIFEST = original


@unittest.skipIf(ec is None, f"e1a2_common import failed: {_IMPORT_ERROR}")
class CacheKeyInvalidationTests(unittest.TestCase):
    """Pure cache-key math -- no model download, no image I/O."""

    def test_cache_key_changes_with_any_identity_field(self):
        base = ec._cache_key("resnet18", "sha_a", "train", "weights_a", "pp_hash_a")
        variants = [
            ec._cache_key("mobilenet_v3_small", "sha_a", "train", "weights_a", "pp_hash_a"),
            ec._cache_key("resnet18", "sha_b", "train", "weights_a", "pp_hash_a"),
            ec._cache_key("resnet18", "sha_a", "valid", "weights_a", "pp_hash_a"),
            ec._cache_key("resnet18", "sha_a", "train", "weights_b", "pp_hash_a"),
            ec._cache_key("resnet18", "sha_a", "train", "weights_a", "pp_hash_b"),
        ]
        for v in variants:
            self.assertNotEqual(base, v)

    def test_cache_key_deterministic(self):
        a = ec._cache_key("resnet18", "sha_a", "train", "weights_a", "pp_hash_a")
        b = ec._cache_key("resnet18", "sha_a", "train", "weights_a", "pp_hash_a")
        self.assertEqual(a, b)

    def test_preprocessing_hash_changes_with_spec(self):
        h1 = ec.preprocessing_hash({"family": "torchvision", "resize": 224})
        h2 = ec.preprocessing_hash({"family": "torchvision", "resize": 256})
        self.assertNotEqual(h1, h2)


@unittest.skipIf(ec is None, f"e1a2_common import failed: {_IMPORT_ERROR}")
class DeviceResolutionTests(unittest.TestCase):
    def test_cpu_always_resolves_to_cpu(self):
        self.assertEqual(ec.resolve_device("cpu"), "cpu")

    def test_auto_resolves_without_error(self):
        result = ec.resolve_device("auto")
        self.assertIn(result, ("cpu", "cuda"))

    def test_cuda_requested_without_cuda_raises(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this machine -- cannot test the failure path")
        with self.assertRaises(RuntimeError):
            ec.resolve_device("cuda")


@unittest.skipIf(ec is None, f"e1a2_common import failed: {_IMPORT_ERROR}")
class ProbabilityOutputSchemaAndResumeTests(unittest.TestCase):
    """Synthetic embeddings only -- no real image, no real backbone."""

    def _synthetic_data(self, seed=0):
        import numpy as np
        rng = np.random.RandomState(seed)
        x_train = rng.randn(40, 8).astype("float32")
        y_train = np.tile(np.arange(4), 10)
        x_valid = rng.randn(16, 8).astype("float32")
        y_valid = np.tile(np.arange(4), 4)
        valid_ids = [f"img_{i}" for i in range(16)]
        return x_train, y_train, x_valid, y_valid, valid_ids

    def test_probability_csv_schema_and_row_sums(self):
        x_train, y_train, x_valid, y_valid, valid_ids = self._synthetic_data()
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "ckpt"
            ec.run_linear_probe(
                x_train, y_train, x_valid, y_valid, valid_ids, device="cpu", max_epochs=3, patience=5,
                head_lr=1e-2, weight_decay=0.0, class_weights=None, output_dir=out_dir, model_key="synthetic_test",
            )
            pred_path = out_dir / "best_valid_predictions.csv"
            self.assertTrue(pred_path.exists())
            with open(pred_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), [
                "image_id", "true_class", "predicted_class",
                "prob_Angry", "prob_Fear", "prob_Happy", "prob_Sad",
            ])
            self.assertEqual(len(rows), 16)
            for row in rows:
                self.assertIn(row["true_class"], ec.CLASSES)
                self.assertIn(row["predicted_class"], ec.CLASSES)
                total = sum(float(row[f"prob_{c}"]) for c in ec.CLASSES)
                self.assertAlmostEqual(total, 1.0, places=3)

    def test_resume_continues_from_last_epoch_not_from_scratch(self):
        x_train, y_train, x_valid, y_valid, valid_ids = self._synthetic_data()
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "ckpt"
            history1, _, _ = ec.run_linear_probe(
                x_train, y_train, x_valid, y_valid, valid_ids, device="cpu", max_epochs=3, patience=100,
                head_lr=1e-2, weight_decay=0.0, class_weights=None, output_dir=out_dir, model_key="synthetic_test",
            )
            self.assertEqual(len(history1), 3)
            history2, _, _ = ec.run_linear_probe(
                x_train, y_train, x_valid, y_valid, valid_ids, device="cpu", max_epochs=6, patience=100,
                head_lr=1e-2, weight_decay=0.0, class_weights=None, output_dir=out_dir, model_key="synthetic_test",
                resume=True,
            )
            self.assertEqual(len(history2), 6)
            # First 3 epochs of history2 must be the SAME records already
            # written by the first call (proof it resumed, not restarted).
            for i in range(3):
                self.assertEqual(history2[i]["epoch"], history1[i]["epoch"])

    def test_resume_false_always_restarts_from_epoch_zero(self):
        x_train, y_train, x_valid, y_valid, valid_ids = self._synthetic_data()
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "ckpt"
            ec.run_linear_probe(
                x_train, y_train, x_valid, y_valid, valid_ids, device="cpu", max_epochs=3, patience=100,
                head_lr=1e-2, weight_decay=0.0, class_weights=None, output_dir=out_dir, model_key="synthetic_test",
            )
            history2, _, _ = ec.run_linear_probe(
                x_train, y_train, x_valid, y_valid, valid_ids, device="cpu", max_epochs=2, patience=100,
                head_lr=1e-2, weight_decay=0.0, class_weights=None, output_dir=out_dir, model_key="synthetic_test",
                resume=False,
            )
            self.assertEqual(len(history2), 2)
            self.assertEqual(history2[0]["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
