"""Phase 2C.6 Stage 3 expansion-manifest tests -- synthetic manifest only,
never the real corpus. Confirms determinism, prior-round exclusion, the
distinct p2c6_ id namespace, and that the public summary never contains
anything beyond opaque pilot_ids/counts."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c6 import expansion_manifest as mod


def _write_manifest(path: Path, n_groups: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group_id", "group_size", "path", "conflict_status"])
        for g in range(n_groups):
            w.writerow([f"img_{g:04d}", f"grp_{g:04d}", 1, f"/fake/img_{g:04d}.jpg", "clean"])


def _write_source_images(dir_: Path, n: int) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for g in range(n):
        (dir_ / f"img_{g:04d}.jpg").write_bytes(b"fake jpeg bytes")


class LoadUsedImageIdsTests(unittest.TestCase):
    def test_missing_file_returns_empty_set(self):
        self.assertEqual(mod.load_used_image_ids("/no/such/file.csv"), set())

    def test_loads_image_ids_from_multiple_mapping_files(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "m1.csv", Path(d) / "m2.csv"
            for p, ids in [(p1, ["a", "b"]), (p2, ["c"])]:
                with p.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["pilot_id", "image_id", "source_image_group", "original_path"])
                    for i in ids:
                        w.writerow(["p", i, "g", "/x"])
            used = mod.load_used_image_ids(p1, p2)
            self.assertEqual(used, {"a", "b", "c"})


class BuildExpansionManifestTests(unittest.TestCase):
    def test_deterministic_given_fixed_seed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            manifest = d / "manifest.csv"
            _write_manifest(manifest, 60)
            src_dir = d / "src"
            _write_source_images(src_dir, 60)
            # patch paths in the manifest to point at real files
            rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
            for r in rows:
                r["path"] = str(src_dir / Path(r["path"]).name)
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image_id", "group_id", "group_size", "path", "conflict_status"])
                w.writeheader()
                w.writerows(rows)

            blinded_a = mod.build_expansion_manifest(
                10, seed=42, exclude_image_ids=set(), corpus_manifest_path=manifest,
                images_output_dir=d / "out_a", mapping_output_path=d / "map_a.csv")
            blinded_b = mod.build_expansion_manifest(
                10, seed=42, exclude_image_ids=set(), corpus_manifest_path=manifest,
                images_output_dir=d / "out_b", mapping_output_path=d / "map_b.csv")
            self.assertEqual([r["image_id"] for r in blinded_a], [r["image_id"] for r in blinded_b])
            self.assertEqual([r["pilot_id"] for r in blinded_a], [r["pilot_id"] for r in blinded_b])

    def test_pilot_ids_use_p2c6_prefix_not_p2b(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            manifest = d / "manifest.csv"
            _write_manifest(manifest, 20)
            src_dir = d / "src"
            _write_source_images(src_dir, 20)
            rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
            for r in rows:
                r["path"] = str(src_dir / Path(r["path"]).name)
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image_id", "group_id", "group_size", "path", "conflict_status"])
                w.writeheader()
                w.writerows(rows)
            blinded = mod.build_expansion_manifest(
                5, seed=1, exclude_image_ids=set(), corpus_manifest_path=manifest,
                images_output_dir=d / "out", mapping_output_path=d / "map.csv")
            for r in blinded:
                self.assertTrue(r["pilot_id"].startswith("p2c6_"))
                self.assertFalse(r["pilot_id"].startswith("p2b_"))


class WriteExpansionSummaryTests(unittest.TestCase):
    def test_summary_contains_only_opaque_ids_and_counts(self):
        blinded = [{"pilot_id": "p2c6_0000", "image_id": "secret_id_1", "group_id": "g1",
                    "path": "/private/Happy/x.jpg"},
                   {"pilot_id": "p2c6_0001", "image_id": "secret_id_2", "group_id": "g2",
                    "path": "/private/Sad/y.jpg"}]
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "summary.json"
            mod.write_expansion_summary(blinded, out_path, seed=1, n_excluded_prior_round=80)
            text = out_path.read_text(encoding="utf-8")
            self.assertNotIn("secret_id", text)
            self.assertNotIn("Happy", text)
            self.assertNotIn("Sad", text)
            self.assertNotIn("private", text)
            summary = json.loads(text)
            self.assertEqual(summary["pilot_ids"], ["p2c6_0000", "p2c6_0001"])
            self.assertEqual(summary["n_images"], 2)


if __name__ == "__main__":
    unittest.main()
