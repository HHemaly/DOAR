from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.hog_features import (
    DESCRIPTOR_LENGTH, compute_hog_descriptor, extract_hog_features,
    hog_feature_row, serialize_hog_row,
)


class HogDescriptorCorrectnessTests(unittest.TestCase):
    def test_flat_image_has_all_zero_descriptor(self):
        flat = np.full((64, 64), 128.0)
        d = compute_hog_descriptor(flat)
        self.assertEqual(d.shape, (DESCRIPTOR_LENGTH,))
        self.assertEqual(float(np.abs(d).max()), 0.0)

    def test_edge_image_has_nonzero_bins(self):
        edge = np.zeros((64, 64))
        edge[:, 32:] = 255.0
        d = compute_hog_descriptor(edge)
        self.assertGreater(int((d != 0).sum()), 0)

    def test_deterministic_for_identical_input(self):
        edge = np.zeros((64, 64))
        edge[:, 32:] = 255.0
        d1 = compute_hog_descriptor(edge)
        d2 = compute_hog_descriptor(edge)
        self.assertTrue(np.array_equal(d1, d2))

    def test_block_norm_bounded(self):
        # L2 block normalization -> every value is in [-1, 1] give or take
        # the numerical stabilizer; sanity bound, not an exact theoretical one.
        rng = np.random.RandomState(0)
        img = (rng.rand(64, 64) * 255)
        d = compute_hog_descriptor(img)
        self.assertLessEqual(float(np.abs(d).max()), 1.0 + 1e-6)

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            compute_hog_descriptor(np.zeros((32, 32)))


class HogFeatureRowTests(unittest.TestCase):
    def test_real_image_produces_correct_row_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "drawing.png"
            img = Image.new("RGB", (100, 100), "white")
            ImageDraw.Draw(img).ellipse((20, 20, 80, 80), fill="black")
            img.save(path)
            row = hog_feature_row(path)
            self.assertEqual(len(row), DESCRIPTOR_LENGTH)
            self.assertEqual(list(row.keys())[0], "hog.bin_0000")
            self.assertEqual(list(row.keys())[-1], f"hog.bin_{DESCRIPTOR_LENGTH - 1:04d}")
            for fv in row.values():
                self.assertFalse(fv.missing)
                self.assertEqual(fv.confidence, 1.0)

    def test_serialize_hog_row_round_trips_to_plain_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "drawing.png"
            Image.new("RGB", (60, 60), "white").save(path)
            row = hog_feature_row(path)
            serialized = serialize_hog_row(row)
            self.assertIsInstance(serialized["hog.bin_0000"], dict)
            self.assertIn("value", serialized["hog.bin_0000"])


class ExtractHogFeaturesCliLevelTests(unittest.TestCase):
    def test_extract_hog_features_writes_csv_matching_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            img_dir = root / "imgs"
            img_dir.mkdir()
            rows = []
            for i, cls in enumerate(["Angry", "Fear", "Happy", "Sad"]):
                p = img_dir / f"{i}.png"
                Image.new("RGB", (50, 50), "white").save(p)
                rows.append({"image_id": f"id{i}", "path": str(p), "split": "train",
                            "class": cls, "readable": "True"})
            manifest = root / "manifest.csv"
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            out = root / "out"
            summary = extract_hog_features(manifest, out)
            self.assertEqual(summary["processed"], 4)
            self.assertEqual(summary["failures"], 0)
            self.assertEqual(summary["descriptor_length"], DESCRIPTOR_LENGTH)
            csv_rows = list(csv.DictReader(open(out / "hog_features.csv", encoding="utf-8")))
            self.assertEqual(len(csv_rows), 4)
            self.assertIn("hog.bin_0000", csv_rows[0])

    def test_unreadable_row_is_recorded_as_failure_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{"image_id": "bad", "path": "does_not_exist.png", "split": "train",
                    "class": "Angry", "readable": "False"}]
            manifest = root / "manifest.csv"
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            summary = extract_hog_features(manifest, root / "out")
            self.assertEqual(summary["processed"], 0)
            self.assertEqual(summary["failures"], 1)


if __name__ == "__main__":
    unittest.main()
