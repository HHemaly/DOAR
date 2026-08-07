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
            w.writerow(["pilot_id", "image_id", "source_image_group", "original_split",
                        "original_path", "blind_path"])
            for i in range(2):
                w.writerow([f"p2b_{i:04d}", f"imgid{i}", f"grp_{i}", "train", f"/fake/path{i}.jpg",
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

    def test_migrated_records_are_marked_legacy_provisional_human_not_human(self):
        """The hard requirement: migrated Phase 2B rows must NEVER look
        like a genuine Phase 2C.1 'human' annotation event -- that is what
        lets quality.py's agreement tooling exclude them from human-human
        Cohen's kappa without inspecting annotator_id strings."""
        manifest = self._write_synthetic_manifest()
        mapping = self._write_synthetic_mapping()
        records = workspace_mod.migrate_phase2b_provisional(manifest, mapping)
        self.assertTrue(all(r.annotator_type == "legacy_provisional_human" for r in records))
        self.assertFalse(any(r.annotator_type == "human" for r in records))

    def test_migrated_records_stay_unreviewed(self):
        manifest = self._write_synthetic_manifest()
        mapping = self._write_synthetic_mapping()
        records = workspace_mod.migrate_phase2b_provisional(manifest, mapping)
        self.assertTrue(all(r.review_status == "unreviewed" for r in records))


class LockedTestSplitGuardTests(_TempDirTestCase):
    def _write_mixed_split_mapping(self) -> Path:
        mapping_path = self.tmp_dir / "mapping.csv"
        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pilot_id", "image_id", "source_image_group", "original_split",
                        "original_path", "blind_path"])
            w.writerow(["p2b_0000", "id0", "grp_0", "train", "/fake/0.jpg", "/fake/blind/0.jpg"])
            w.writerow(["p2b_0001", "id1", "grp_1", "valid", "/fake/1.jpg", "/fake/blind/1.jpg"])
            w.writerow(["p2b_0002", "id2", "grp_2", "test", "/fake/2.jpg", "/fake/blind/2.jpg"])
            w.writerow(["p2b_0003", "id3", "grp_3", "test", "/fake/3.jpg", "/fake/blind/3.jpg"])
        return mapping_path

    def test_pilot_ids_from_locked_test_split_returns_only_test(self):
        mapping = self._write_mixed_split_mapping()
        result = workspace_mod.pilot_ids_from_locked_test_split(mapping)
        self.assertEqual(result, {"p2b_0002", "p2b_0003"})

    def test_assert_passes_when_no_test_split_pilot_ids_given(self):
        mapping = self._write_mixed_split_mapping()
        # Should not raise.
        workspace_mod.assert_pilot_ids_exclude_locked_test({"p2b_0000", "p2b_0001"}, mapping)

    def test_assert_raises_when_a_test_split_pilot_id_is_included(self):
        mapping = self._write_mixed_split_mapping()
        with self.assertRaises(ValueError):
            workspace_mod.assert_pilot_ids_exclude_locked_test({"p2b_0000", "p2b_0002"}, mapping)

    def test_assert_raises_and_names_all_violating_ids(self):
        mapping = self._write_mixed_split_mapping()
        with self.assertRaises(ValueError) as ctx:
            workspace_mod.assert_pilot_ids_exclude_locked_test(
                {"p2b_0000", "p2b_0002", "p2b_0003"}, mapping)
        self.assertIn("p2b_0002", str(ctx.exception))
        self.assertIn("p2b_0003", str(ctx.exception))

    def test_empty_pilot_id_set_never_raises(self):
        mapping = self._write_mixed_split_mapping()
        workspace_mod.assert_pilot_ids_exclude_locked_test(set(), mapping)


class BuildPilotWorkspaceRetainsOriginalSplitTests(_TempDirTestCase):
    """End-to-end test of build_pilot_workspace against a fully synthetic
    partition manifest + synthetic image files -- never the real dataset.
    Confirms original_split actually round-trips into the written mapping
    CSV, not just that the source mentions the column name."""

    def _write_synthetic_partition_manifest(self, n_images: int) -> Path:
        images_dir = self.tmp_dir / "fake_dataset"
        images_dir.mkdir()
        manifest_path = self.tmp_dir / "partition_manifest.csv"
        splits = ["train", "train", "train", "valid", "test"]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "path", "relative_path", "original_split", "class", "sha256",
                        "phash", "dhash", "group_id", "group_size", "conflict_status",
                        "exposed_status", "new_split", "include_in_supervised_training"])
            for i in range(n_images):
                img_path = images_dir / f"img_{i:04d}.jpg"
                img_path.write_bytes(b"fake image bytes " + str(i).encode())
                split = splits[i % len(splits)]
                w.writerow([f"imgid{i:04d}", str(img_path), f"{split}/Happy/img_{i:04d}.jpg", split,
                            "Happy", f"sha{i}", "0" * 16, "0" * 16, f"grp_{i:05d}", 1, "clean",
                            "not_exposed", split, True])
        return manifest_path

    def test_original_split_round_trips_into_mapping_csv(self):
        partition_manifest = self._write_synthetic_partition_manifest(n_images=10)
        images_dir = self.tmp_dir / "blind_images"
        mapping_path = self.tmp_dir / "mapping.csv"
        result = workspace_mod.build_pilot_workspace(
            n=10, images_dir=images_dir, mapping_path=mapping_path,
            partition_manifest_path=partition_manifest,
        )
        self.assertEqual(result["n_selected"], 10)
        rows = workspace_mod.load_pilot_mapping(mapping_path)
        self.assertEqual(len(rows), 10)
        splits_seen = {r["original_split"] for r in rows}
        self.assertIn("test", splits_seen)
        self.assertIn("train", splits_seen)
        # n_from_locked_test_split must match an independent recount from the mapping file.
        recounted = sum(1 for r in rows if r["original_split"] == "test")
        self.assertEqual(result["n_from_locked_test_split"], recounted)
        self.assertGreater(recounted, 0)

    def test_guard_functions_work_against_a_freshly_built_real_mapping(self):
        partition_manifest = self._write_synthetic_partition_manifest(n_images=10)
        images_dir = self.tmp_dir / "blind_images"
        mapping_path = self.tmp_dir / "mapping.csv"
        workspace_mod.build_pilot_workspace(
            n=10, images_dir=images_dir, mapping_path=mapping_path,
            partition_manifest_path=partition_manifest,
        )
        locked = workspace_mod.pilot_ids_from_locked_test_split(mapping_path)
        self.assertGreater(len(locked), 0)
        with self.assertRaises(ValueError):
            workspace_mod.assert_pilot_ids_exclude_locked_test(locked, mapping_path)
        all_pilot_ids = {r["pilot_id"] for r in workspace_mod.load_pilot_mapping(mapping_path)}
        non_locked = all_pilot_ids - locked
        workspace_mod.assert_pilot_ids_exclude_locked_test(non_locked, mapping_path)  # must not raise


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
