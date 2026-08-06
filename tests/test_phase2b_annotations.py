from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.annotations import (
    AnnotationRow, load_manifest, validate_manifest_completeness, write_manifest,
)
from doar.phase2b.ontology import CLASS_NAMES


def _full_row_set(pilot_id: str, image_id: str = "img1", group: str = "grp_0") -> list[AnnotationRow]:
    return [
        AnnotationRow(pilot_id=pilot_id, image_id=image_id, source_image_group=group,
                     original_split="train", class_name=cls, status="absent")
        for cls in CLASS_NAMES
    ]


class AnnotationRowInvariantTests(unittest.TestCase):
    def test_valid_row_constructs(self):
        row = AnnotationRow(pilot_id="p2b_0000", image_id="i1", source_image_group="g1",
                            original_split="train", class_name="person", status="present",
                            instance_count=2)
        self.assertEqual(row.status, "present")

    def test_unknown_class_raises(self):
        with self.assertRaises(ValueError):
            AnnotationRow(pilot_id="p2b_0000", image_id="i1", source_image_group="g1",
                          original_split="train", class_name="giraffe", status="present")

    def test_unknown_status_raises(self):
        with self.assertRaises(ValueError):
            AnnotationRow(pilot_id="p2b_0000", image_id="i1", source_image_group="g1",
                          original_split="train", class_name="person", status="maybe")

    def test_non_present_status_with_positive_count_raises(self):
        # An "absent"/"uncertain"/"not_assessable" row must never carry a
        # positive instance_count -- that would look like a confirmed count.
        with self.assertRaises(ValueError):
            AnnotationRow(pilot_id="p2b_0000", image_id="i1", source_image_group="g1",
                          original_split="train", class_name="person", status="absent",
                          instance_count=1)

    def test_uncertain_status_is_valid_and_never_auto_converted(self):
        row = AnnotationRow(pilot_id="p2b_0000", image_id="i1", source_image_group="g1",
                            original_split="train", class_name="person", status="uncertain")
        self.assertEqual(row.status, "uncertain")
        self.assertEqual(row.instance_count, 0)


class ManifestWriteLoadTests(unittest.TestCase):
    def test_write_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            rows = _full_row_set("p2b_0000")
            path = write_manifest(rows, Path(d) / "manifest.csv")
            loaded = load_manifest(path)
            self.assertEqual(len(loaded), len(CLASS_NAMES))
            self.assertEqual({r["class"] for r in loaded}, set(CLASS_NAMES))

    def test_deterministic_manifest_creation(self):
        with tempfile.TemporaryDirectory() as d:
            rows = _full_row_set("p2b_0000")
            p1 = write_manifest(rows, Path(d) / "a.csv")
            p2 = write_manifest(rows, Path(d) / "b.csv")
            self.assertEqual(p1.read_text(encoding="utf-8"), p2.read_text(encoding="utf-8"))

    def test_validate_completeness_passes_for_a_full_image(self):
        with tempfile.TemporaryDirectory() as d:
            rows = _full_row_set("p2b_0000")
            path = write_manifest(rows, Path(d) / "manifest.csv")
            validate_manifest_completeness(load_manifest(path))  # must not raise

    def test_validate_completeness_raises_when_a_class_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            rows = _full_row_set("p2b_0000")[:-1]  # drop the last class's row
            path = write_manifest(rows, Path(d) / "manifest.csv")
            with self.assertRaises(ValueError):
                validate_manifest_completeness(load_manifest(path))

    def test_validate_completeness_raises_on_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as d:
            rows = _full_row_set("p2b_0000") + [_full_row_set("p2b_0000")[0]]
            path = write_manifest(rows, Path(d) / "manifest.csv")
            with self.assertRaises(ValueError):
                validate_manifest_completeness(load_manifest(path))


if __name__ == "__main__":
    unittest.main()
