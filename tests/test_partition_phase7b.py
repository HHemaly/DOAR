"""Phase 7B -- duplicate-policy validation and final partition refinement.

Covers: the dHash near-dup method (compute_dhash_column, hash_field
parameter on compute_duplicate_groups), previously-exposed-image group
exclusion from valid/test, and end-to-end determinism/verification of
run_partition_design under the Phase 7B policy (dhash, threshold, exposed
images)."""

from __future__ import annotations
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _row(image_id, split, cls, sha256, phash="0f0f0f0f0f0f0f0f", dhash="", path=None):
    return {
        "image_id": image_id, "path": path or f"/x/{image_id}.png",
        "relative_path": f"{split}/{cls}/{image_id}.png",
        "split": split, "class": cls, "sha256": sha256, "phash": phash, "dhash": dhash,
    }


class DHashTests(unittest.TestCase):
    def test_difference_hash_is_deterministic_and_16_hex_chars(self):
        from doar.dataset import _difference_hash
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image
            import numpy as np
            path = Path(d) / "img.png"
            arr = (np.arange(64 * 64) % 256).reshape(64, 64).astype("uint8")
            Image.fromarray(arr).save(path)
            h1 = _difference_hash(path)
            h2 = _difference_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 16)
            int(h1, 16)  # must parse as hex

    def test_compute_dhash_column_attaches_field_without_touching_source(self):
        from doar.partition import compute_dhash_column
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image
            path = Path(d) / "img.png"
            Image.new("L", (32, 32), color=128).save(path)
            original_bytes = path.read_bytes()
            rows = [{"image_id": "a", "path": str(path)}]
            compute_dhash_column(rows)
            self.assertIn("dhash", rows[0])
            self.assertTrue(rows[0]["dhash"])
            self.assertEqual(path.read_bytes(), original_bytes)  # untouched

    def test_compute_duplicate_groups_uses_requested_hash_field(self):
        from doar.partition import compute_duplicate_groups
        # phash says "far apart" (no near-dup), dhash says "identical" (near-dup)
        rows = [
            _row("a", "train", "Happy", "h1", phash="0000000000000000", dhash="0f0f0f0f0f0f0f0f"),
            _row("b", "valid", "Happy", "h2", phash="ffffffffffffffff", dhash="0f0f0f0f0f0f0f0e"),
        ]
        by_phash = compute_duplicate_groups(rows, near_dup_threshold=5, hash_field="phash")
        by_dhash = compute_duplicate_groups(rows, near_dup_threshold=5, hash_field="dhash")
        self.assertNotEqual(by_phash["group_of"]["a"], by_phash["group_of"]["b"])
        self.assertEqual(by_dhash["group_of"]["a"], by_dhash["group_of"]["b"])


class ExposedGroupTests(unittest.TestCase):
    def test_identify_exposed_groups_maps_image_to_its_group(self):
        from doar.partition import compute_duplicate_groups, identify_exposed_groups
        rows = [
            _row("a", "train", "Happy", "h1"),
            _row("b", "test", "Happy", "h1"),   # exact dup of a -- same group
            _row("c", "valid", "Sad", "h2"),
        ]
        g = compute_duplicate_groups(rows, near_dup_threshold=0)
        exposed = identify_exposed_groups(g["group_of"], {"a"})
        self.assertEqual(exposed, frozenset({g["group_of"]["a"]}))
        self.assertEqual(g["group_of"]["a"], g["group_of"]["b"])  # a and b share a group

    def test_exposed_clean_group_forced_to_train(self):
        from doar.partition import (
            compute_duplicate_groups, assess_group_label_conflicts,
            identify_exposed_groups, build_group_disjoint_partition,
        )
        rows = [_row(f"img_{i}", "train", "Happy", f"h{i}") for i in range(20)]
        g = compute_duplicate_groups(rows, near_dup_threshold=0)
        conflicts = assess_group_label_conflicts(rows, g["group_of"])
        exposed = identify_exposed_groups(g["group_of"], {"img_0"})
        part = build_group_disjoint_partition(
            rows, g["group_of"], set(conflicts["conflict_group_ids"]), seed=42,
            split_ratios=(0.5, 0.25, 0.25), exposed_group_ids=exposed)
        exposed_gid = g["group_of"]["img_0"]
        self.assertEqual(part["assignment"][exposed_gid], "train")

    def test_exposed_conflict_group_stays_excluded_not_train(self):
        # Conflict status takes precedence over exposed status.
        from doar.partition import (
            compute_duplicate_groups, assess_group_label_conflicts,
            identify_exposed_groups, build_group_disjoint_partition,
        )
        rows = [
            _row("a", "train", "Happy", "h1"),
            _row("b", "test", "Sad", "h1"),  # same bytes, different label -> conflict
        ]
        g = compute_duplicate_groups(rows, near_dup_threshold=0)
        conflicts = assess_group_label_conflicts(rows, g["group_of"])
        exposed = identify_exposed_groups(g["group_of"], {"a"})
        conflict_ids = set(conflicts["conflict_group_ids"])
        self.assertTrue(conflict_ids)  # fixture must produce a conflict
        part = build_group_disjoint_partition(
            rows, g["group_of"], conflict_ids, seed=1, exposed_group_ids=exposed)
        gid = g["group_of"]["a"]
        self.assertEqual(part["assignment"][gid], "excluded_conflict")

    def test_verify_no_exposed_group_in_valid_or_test(self):
        from doar.partition import verify_no_exposed_group_in_valid_or_test
        rows = [
            {"image_id": "a", "exposed_status": "previously_exposed", "new_split": "train"},
            {"image_id": "b", "exposed_status": "previously_exposed", "new_split": "valid"},
            {"image_id": "c", "exposed_status": "not_exposed", "new_split": "test"},
        ]
        result = verify_no_exposed_group_in_valid_or_test(rows)
        self.assertFalse(result["ok"])
        self.assertEqual(result["violations"], ["b"])

    def test_verify_passes_when_no_exposed_group_leaks(self):
        from doar.partition import verify_no_exposed_group_in_valid_or_test
        rows = [
            {"image_id": "a", "exposed_status": "previously_exposed", "new_split": "train"},
            {"image_id": "b", "exposed_status": "not_exposed", "new_split": "valid"},
            {"image_id": "c", "exposed_status": "not_exposed", "new_split": "test"},
        ]
        result = verify_no_exposed_group_in_valid_or_test(rows)
        self.assertTrue(result["ok"])


class EndToEndPhase7BTests(unittest.TestCase):
    def _write_manifest(self, d, rows):
        p = Path(d) / "manifest.csv"
        fields = ["image_id", "path", "relative_path", "split", "class", "sha256", "phash"]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return p

    def test_run_partition_design_with_dhash_and_exposed_images_is_deterministic(self):
        from doar.partition import run_partition_design
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image
            import random as _random
            rng = _random.Random(3)
            rows = []
            classes = ("Angry", "Fear", "Happy", "Sad")
            for i in range(60):
                cls = classes[i % 4]
                img_dir = Path(d) / "imgs"
                img_dir.mkdir(exist_ok=True)
                path = img_dir / f"img_{i}.png"
                color = rng.randint(0, 255)
                Image.new("L", (32, 32), color=color).save(path)
                rows.append({
                    "image_id": f"img_{i:03d}", "path": str(path),
                    "relative_path": f"train/{cls}/img_{i}.png",
                    "split": "train", "class": cls,
                    "sha256": f"sha_{i}", "phash": "0f0f0f0f0f0f0f0f",
                })
            manifest = self._write_manifest(d, rows)

            out1 = Path(d) / "out1"
            out2 = Path(d) / "out2"
            r1 = run_partition_design(
                manifest, out1, seed=7, near_dup_threshold=6, hash_field="dhash",
                exposed_image_ids={"img_000"})
            r2 = run_partition_design(
                manifest, out2, seed=7, near_dup_threshold=6, hash_field="dhash",
                exposed_image_ids={"img_000"})

            self.assertEqual(r1["artifact_hashes"], r2["artifact_hashes"])
            self.assertTrue(r1["verification"]["no_exposed_group_in_valid_or_test"]["ok"])

            # img_000's row in the manifest must be in train, never valid/test
            with open(out1 / "partition_manifest.csv", newline="", encoding="utf-8") as f:
                out_rows = list(csv.DictReader(f))
            img0 = next(r for r in out_rows if r["image_id"] == "img_000")
            self.assertEqual(img0["exposed_status"], "previously_exposed")
            self.assertEqual(img0["new_split"], "train")


if __name__ == "__main__":
    unittest.main(verbosity=2)
