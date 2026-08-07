"""Store persistence tests: duplicate-row prevention, autosave/resume,
export, and corrupt-manifest handling. Synthetic fixtures only."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import store as store_mod
from doar.phase2c1.schema import AnnotationRecord


def _rec(**overrides):
    defaults = dict(
        pilot_id="p2b_0000", image_id="abc123", source_image_group="grp_00001",
        class_name="person", status="present", annotator_id="ann1",
        annotation_timestamp="2026-08-08T00:00:00+00:00", instance_count=1,
    )
    defaults.update(overrides)
    return AnnotationRecord(**defaults)


class _TempStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "store.csv"

    def tearDown(self):
        self._tmp.cleanup()


class LoadEmptyStoreTests(_TempStoreTestCase):
    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(store_mod.load_store(self.path), {})


class DuplicateRowPreventionTests(_TempStoreTestCase):
    def test_upsert_same_key_replaces_not_appends(self):
        st = {}
        store_mod.upsert(st, _rec(notes="first"))
        store_mod.upsert(st, _rec(notes="second"))
        self.assertEqual(len(st), 1)
        self.assertEqual(next(iter(st.values())).notes, "second")

    def test_distinct_class_names_produce_distinct_rows(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0))
        self.assertEqual(len(st), 2)

    def test_distinct_annotators_produce_distinct_rows_for_same_class(self):
        st = {}
        store_mod.upsert(st, _rec(annotator_id="ann1"))
        store_mod.upsert(st, _rec(annotator_id="ann2"))
        self.assertEqual(len(st), 2)

    def test_saved_csv_has_no_duplicate_annotation_ids(self):
        st = {}
        for _ in range(3):
            store_mod.upsert_and_save(self.path, st, _rec())
        rows = list(csv.DictReader(self.path.open(encoding="utf-8")))
        self.assertEqual(len(rows), 1)


class AutosaveResumeTests(_TempStoreTestCase):
    def test_save_then_load_round_trips(self):
        st = {}
        store_mod.upsert_and_save(self.path, st, _rec())
        reloaded = store_mod.load_store(self.path)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(next(iter(reloaded.values())).status, "present")

    def test_resume_after_simulated_interruption(self):
        st = {}
        store_mod.upsert_and_save(self.path, st, _rec(class_name="person"))
        # Simulate a fresh process picking the store back up.
        resumed = store_mod.load_store(self.path)
        store_mod.upsert_and_save(self.path, resumed, _rec(class_name="face", status="absent", instance_count=0))
        final = store_mod.load_store(self.path)
        self.assertEqual(len(final), 2)

    def test_every_save_is_a_complete_atomic_replace_no_partial_file(self):
        st = {}
        store_mod.upsert_and_save(self.path, st, _rec())
        # File must always be fully valid CSV after each save -- no .tmp
        # leftovers, no truncated content.
        tmp_files = list(self.path.parent.glob("*.tmp"))
        self.assertEqual(tmp_files, [])
        rows = list(csv.DictReader(self.path.open(encoding="utf-8")))
        self.assertEqual(len(rows), 1)


class ExportTests(_TempStoreTestCase):
    def test_export_csv_row_count(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0))
        out = Path(self._tmp.name) / "export.csv"
        n = store_mod.export_csv(st, out)
        self.assertEqual(n, 2)
        self.assertTrue(out.exists())

    def test_export_json_row_count_and_shape(self):
        import json
        st = {}
        store_mod.upsert(st, _rec())
        out = Path(self._tmp.name) / "export.json"
        n = store_mod.export_json(st, out)
        self.assertEqual(n, 1)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertIn("annotation_id", data[0])

    def test_export_is_deterministic_across_repeated_calls(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0))
        out1 = Path(self._tmp.name) / "export1.csv"
        out2 = Path(self._tmp.name) / "export2.csv"
        store_mod.export_csv(st, out1)
        store_mod.export_csv(st, out2)
        self.assertEqual(out1.read_text(encoding="utf-8"), out2.read_text(encoding="utf-8"))


class CorruptManifestTests(_TempStoreTestCase):
    def test_missing_required_column_raises_keyerror(self):
        self.path.write_text("annotation_id,pilot_id\nfoo,p2b_0000\n", encoding="utf-8")
        with self.assertRaises(KeyError):
            store_mod.load_store(self.path)

    def test_malformed_bbox_raises_valueerror(self):
        from doar.phase2c1.schema import CSV_FIELDS
        row = {f: "" for f in CSV_FIELDS}
        row.update({
            "annotation_id": "x", "pilot_id": "p2b_0000", "image_id": "i",
            "source_image_group": "g", "class_name": "person", "status": "present",
            "instance_count": "1", "bbox": "not,valid",
            "annotator_id": "a", "annotation_timestamp": "t",
        })
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerow(row)
        with self.assertRaises(ValueError):
            store_mod.load_store(self.path)

    def test_empty_file_with_header_only_loads_empty(self):
        from doar.phase2c1.schema import CSV_FIELDS
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        self.assertEqual(store_mod.load_store(self.path), {})


class ApplyReviewOutcomeTests(_TempStoreTestCase):
    def test_updates_primary_row_review_fields_in_place(self):
        st = {}
        store_mod.upsert_and_save(self.path, st, _rec(annotator_id="primary"))
        store_mod.apply_review_outcome(
            self.path, st, pilot_id="p2b_0000", class_name="person",
            primary_annotator_id="primary", reviewer_id="reviewer1",
            adjudication_status="agreement", review_timestamp="2026-08-08T01:00:00+00:00",
        )
        row = st[next(iter(st))]
        self.assertEqual(row.review_status, "reviewed")
        self.assertEqual(row.reviewer_id, "reviewer1")
        self.assertEqual(row.adjudication_status, "agreement")

    def test_does_not_create_a_new_row(self):
        st = {}
        store_mod.upsert_and_save(self.path, st, _rec(annotator_id="primary"))
        store_mod.apply_review_outcome(
            self.path, st, pilot_id="p2b_0000", class_name="person",
            primary_annotator_id="primary", reviewer_id="reviewer1",
            adjudication_status="agreement", review_timestamp="t",
        )
        self.assertEqual(len(st), 1)

    def test_missing_primary_row_raises(self):
        st = {}
        with self.assertRaises(KeyError):
            store_mod.apply_review_outcome(
                self.path, st, pilot_id="p2b_9999", class_name="person",
                primary_annotator_id="nobody", reviewer_id="reviewer1",
                adjudication_status="agreement", review_timestamp="t",
            )


class RecordsForPilotTests(_TempStoreTestCase):
    def test_filters_by_pilot_id(self):
        st = {}
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", class_name="person"))
        store_mod.upsert(st, _rec(pilot_id="p2b_0001", class_name="person"))
        rows = store_mod.records_for_pilot(st, "p2b_0000")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].pilot_id, "p2b_0000")


if __name__ == "__main__":
    unittest.main()
