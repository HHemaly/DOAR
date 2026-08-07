"""Schema invariant tests for Phase 2C.1's AnnotationRecord -- synthetic
data only, no dependency on the real dataset or private drawings."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import CLASS_NAMES
from doar.phase2c1.schema import (
    ADJUDICATION_STATUSES, ANNOTATOR_TYPES, OBJECT_STATUSES, REVIEW_STATUSES,
    AnnotationRecord, make_annotation_id, record_from_row, validate_image_complete,
)


def _rec(**overrides):
    defaults = dict(
        pilot_id="p2b_0000", image_id="abc123", source_image_group="grp_00001",
        class_name="person", status="present", annotator_id="ann1",
        annotation_timestamp="2026-08-08T00:00:00+00:00", instance_count=1,
    )
    defaults.update(overrides)
    return AnnotationRecord(**defaults)


class ValidStatusVocabularyTests(unittest.TestCase):
    def test_object_statuses_match_ontology(self):
        self.assertEqual(OBJECT_STATUSES, frozenset({"present", "absent", "uncertain", "not_assessable"}))

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            _rec(status="confirmed_negative", instance_count=0)

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            _rec(class_name="not_a_real_class", instance_count=0)


class UncertainNeverBecomesAbsentTests(unittest.TestCase):
    """The hard safety requirement: uncertain/not_assessable must never be
    representable, storable, or exportable as if they were 'absent'."""

    def test_uncertain_is_a_distinct_status_value(self):
        rec = _rec(status="uncertain", instance_count=0, uncertainty_reason="ambiguous mark")
        self.assertEqual(rec.status, "uncertain")
        self.assertNotEqual(rec.status, "absent")

    def test_not_assessable_is_a_distinct_status_value(self):
        rec = _rec(status="not_assessable", instance_count=0)
        self.assertEqual(rec.status, "not_assessable")
        self.assertNotEqual(rec.status, "absent")

    def test_uncertain_round_trips_through_csv_row_unchanged(self):
        rec = _rec(status="uncertain", instance_count=0, uncertainty_reason="sun vs star")
        row = rec.to_row()
        self.assertEqual(row["status"], "uncertain")
        restored = record_from_row(row)
        self.assertEqual(restored.status, "uncertain")

    def test_not_assessable_round_trips_through_csv_row_unchanged(self):
        rec = _rec(status="not_assessable", instance_count=0)
        row = rec.to_row()
        self.assertEqual(row["status"], "not_assessable")
        restored = record_from_row(row)
        self.assertEqual(restored.status, "not_assessable")

    def test_no_default_status_exists(self):
        """status has no dataclass default -- constructing a record without
        explicitly choosing one is a TypeError, not a silent 'absent'."""
        with self.assertRaises(TypeError):
            AnnotationRecord(
                pilot_id="p", image_id="i", source_image_group="g", class_name="person",
                annotator_id="a", annotation_timestamp="t",
            )  # type: ignore[call-arg]


class InstanceCountConstraintTests(unittest.TestCase):
    def test_present_requires_positive_count(self):
        with self.assertRaises(ValueError):
            _rec(status="present", instance_count=0)

    def test_absent_forbids_positive_count(self):
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=1)

    def test_uncertain_forbids_positive_count(self):
        with self.assertRaises(ValueError):
            _rec(status="uncertain", instance_count=1)

    def test_not_assessable_forbids_positive_count(self):
        with self.assertRaises(ValueError):
            _rec(status="not_assessable", instance_count=1)

    def test_present_with_count_two_is_valid(self):
        rec = _rec(status="present", instance_count=2)
        self.assertEqual(rec.instance_count, 2)


class BboxConstraintTests(unittest.TestCase):
    def test_bbox_requires_present_status(self):
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=0, bbox=(0.1, 0.1, 0.2, 0.2))

    def test_bbox_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            _rec(status="present", instance_count=1, bbox=(1.5, 0.1, 0.2, 0.2))

    def test_bbox_nonpositive_width_rejected(self):
        with self.assertRaises(ValueError):
            _rec(status="present", instance_count=1, bbox=(0.1, 0.1, 0.0, 0.2))

    def test_valid_bbox_round_trips(self):
        rec = _rec(status="present", instance_count=1, bbox=(0.1, 0.2, 0.3, 0.4))
        row = rec.to_row()
        self.assertEqual(row["bbox"], "0.100000,0.200000,0.300000,0.400000")
        restored = record_from_row(row)
        self.assertEqual(restored.bbox, (0.1, 0.2, 0.3, 0.4))

    def test_no_bbox_round_trips_as_none(self):
        rec = _rec(status="present", instance_count=1, bbox=None)
        row = rec.to_row()
        self.assertEqual(row["bbox"], "")
        self.assertIsNone(record_from_row(row).bbox)


class VocabularyValidationTests(unittest.TestCase):
    def test_annotator_type_vocabulary_is_human_and_legacy_provisional_only(self):
        self.assertEqual(ANNOTATOR_TYPES, frozenset({"human", "legacy_provisional_human"}))
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=0, annotator_type="ai")

    def test_legacy_provisional_human_is_a_valid_annotator_type(self):
        rec = _rec(status="absent", instance_count=0, annotator_type="legacy_provisional_human")
        self.assertEqual(rec.annotator_type, "legacy_provisional_human")

    def test_empty_annotator_id_rejected(self):
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=0, annotator_id="   ")

    def test_review_status_vocabulary(self):
        self.assertEqual(REVIEW_STATUSES, frozenset({"unreviewed", "in_review", "reviewed"}))
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=0, review_status="approved")

    def test_adjudication_status_vocabulary(self):
        self.assertEqual(
            ADJUDICATION_STATUSES,
            frozenset({"not_applicable", "agreement", "disagreement_unresolved", "disagreement_resolved"}),
        )
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=0, adjudication_status="resolved_somehow")

    def test_adjudication_without_review_is_inconsistent(self):
        with self.assertRaises(ValueError):
            _rec(status="absent", instance_count=0, review_status="unreviewed", adjudication_status="agreement")


class AnnotationIdTests(unittest.TestCase):
    def test_deterministic_given_same_key(self):
        a = make_annotation_id("p2b_0000", "person", "ann1")
        b = make_annotation_id("p2b_0000", "person", "ann1")
        self.assertEqual(a, b)

    def test_distinct_annotators_get_distinct_ids(self):
        a = make_annotation_id("p2b_0000", "person", "ann1")
        b = make_annotation_id("p2b_0000", "person", "ann2")
        self.assertNotEqual(a, b)

    def test_record_annotation_id_matches_helper(self):
        rec = _rec()
        self.assertEqual(rec.annotation_id, make_annotation_id("p2b_0000", "person", "ann1"))


class ImageCompletenessTests(unittest.TestCase):
    def test_missing_classes_reported(self):
        rows = [_rec(class_name="person"), _rec(class_name="face", status="absent", instance_count=0)]
        missing = validate_image_complete(rows, "p2b_0000", "ann1")
        self.assertEqual(missing, sorted(set(CLASS_NAMES) - {"person", "face"}))

    def test_complete_when_all_ten_present(self):
        rows = [
            _rec(class_name=c, status="absent" if c != "person" else "present",
                 instance_count=0 if c != "person" else 1)
            for c in CLASS_NAMES
        ]
        self.assertEqual(validate_image_complete(rows, "p2b_0000", "ann1"), [])

    def test_other_annotators_rows_dont_count(self):
        rows = [_rec(class_name="person", annotator_id="someone_else")]
        missing = validate_image_complete(rows, "p2b_0000", "ann1")
        self.assertEqual(len(missing), len(CLASS_NAMES))


if __name__ == "__main__":
    unittest.main()
