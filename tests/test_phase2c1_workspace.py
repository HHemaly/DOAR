"""Workspace-building tests -- synthetic images and manifests only, no
dependency on the real dataset or real drawings."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import CLASS_NAMES
from doar.phase2c1 import workspace as workspace_mod


class _TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class ListWorkspaceImagesTests(_TempDirTestCase):
    def test_empty_dir_returns_empty_list(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        self.assertEqual(workspace_mod.list_workspace_images(d), [])

    def test_missing_dir_returns_empty_list_not_error(self):
        self.assertEqual(workspace_mod.list_workspace_images(self.tmp_dir / "nope"), [])

    def test_only_image_extensions_counted(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        (d / "p2b_0000.jpg").write_bytes(b"fake")
        (d / "p2b_0001.png").write_bytes(b"fake")
        (d / "readme.txt").write_text("not an image")
        self.assertEqual(workspace_mod.list_workspace_images(d), ["p2b_0000", "p2b_0001"])

    def test_sorted_order(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        for pid in ["p2b_0002", "p2b_0000", "p2b_0001"]:
            (d / f"{pid}.jpg").write_bytes(b"fake")
        self.assertEqual(workspace_mod.list_workspace_images(d), ["p2b_0000", "p2b_0001", "p2b_0002"])


class ImagePathForPilotIdTests(_TempDirTestCase):
    def test_finds_matching_stem_regardless_of_extension(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        (d / "p2b_0000.jpeg").write_bytes(b"fake")
        found = workspace_mod.image_path_for_pilot_id(d, "p2b_0000")
        self.assertEqual(found.name, "p2b_0000.jpeg")

    def test_missing_pilot_id_returns_none(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        self.assertIsNone(workspace_mod.image_path_for_pilot_id(d, "p2b_9999"))

    def test_missing_dir_returns_none(self):
        self.assertIsNone(workspace_mod.image_path_for_pilot_id(self.tmp_dir / "nope", "p2b_0000"))


class MigratePhase2BProvisionalTests(_TempDirTestCase):
    def _write_synthetic_manifest(self) -> Path:
        manifest_path = self.tmp_dir / "annotation_manifest.csv"
        fields = ["pilot_id", "image_id", "source_image_group", "original_split", "class",
                  "status", "bbox", "partial_or_occluded", "uncertain_reason",
                  "instance_count", "annotator", "review_status", "notes"]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(2):  # 2 synthetic images x 10 classes
                pid = f"p2b_{i:04d}"
                for cls in CLASS_NAMES:
                    status = "present" if cls == "person" else "absent"
                    w.writerow({
                        "pilot_id": pid, "image_id": f"imgid{i}", "source_image_group": f"grp_{i}",
                        "original_split": "train", "class": cls, "status": status, "bbox": "",
                        "partial_or_occluded": "False", "uncertain_reason": "",
                        "instance_count": 1 if status == "present" else 0,
                        "annotator": "single_annotator_session_2026-08-06",
                        "review_status": "unreviewed", "notes": "",
                    })
        return manifest_path

    def _write_synthetic_mapping(self) -> Path:
        mapping_path = self.tmp_dir / "mapping.csv"
        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pilot_id", "image_id", "source_image_group", "original_path", "blind_path"])
            for i in range(2):
                w.writerow([f"p2b_{i:04d}", f"imgid{i}", f"grp_{i}", f"/fake/path{i}.jpg",
                            f"/fake/blind/p2b_{i:04d}.jpg"])
        return mapping_path

    def test_migrates_expected_row_count(self):
        manifest = self._write_synthetic_manifest()
        mapping = self._write_synthetic_mapping()
        records = workspace_mod.migrate_phase2b_provisional(manifest, mapping)
        self.assertEqual(len(records), 2 * len(CLASS_NAMES))

    def test_preserves_original_annotator_id(self):
        manifest = self._write_synthetic_manifest()
        mapping = self._write_synthetic_mapping()
        records = workspace_mod.migrate_phase2b_provisional(manifest, mapping)
        self.assertTrue(all(r.annotator_id == "single_annotator_session_2026-08-06" for r in records))

    def test_never_writes_the_source_manifest(self):
        manifest = self._write_synthetic_manifest()
        mapping = self._write_synthetic_mapping()
        before = manifest.read_text(encoding="utf-8")
        workspace_mod.migrate_phase2b_provisional(manifest, mapping)
        after = manifest.read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_migrated_records_carry_real_provenance_from_mapping(self):
        manifest = self._write_synthetic_manifest()
        mapping = self._write_synthetic_mapping()
        records = workspace_mod.migrate_phase2b_provisional(manifest, mapping)
        p0 = [r for r in records if r.pilot_id == "p2b_0000"]
        self.assertTrue(all(r.image_id == "imgid0" for r in p0))
        self.assertTrue(all(r.source_image_group == "grp_0" for r in p0))


class RemainingUnannotatedTests(_TempDirTestCase):
    def test_returns_images_not_in_store_pilot_ids(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        for pid in ["p2b_0000", "p2b_0001", "p2b_0002"]:
            (d / f"{pid}.jpg").write_bytes(b"fake")
        remaining = workspace_mod.remaining_unannotated_pilot_ids(d, {"p2b_0000"})
        self.assertEqual(remaining, ["p2b_0001", "p2b_0002"])

    def test_all_annotated_returns_empty(self):
        d = self.tmp_dir / "images"
        d.mkdir()
        (d / "p2b_0000.jpg").write_bytes(b"fake")
        self.assertEqual(workspace_mod.remaining_unannotated_pilot_ids(d, {"p2b_0000"}), [])


if __name__ == "__main__":
    unittest.main()
