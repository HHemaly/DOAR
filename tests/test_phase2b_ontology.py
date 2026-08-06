from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import (
    ANNOTATION_STATUSES, CLASS_NAMES, CLASSES, POSTPONED_CLASSES, class_by_name,
)


class OntologyValidationTests(unittest.TestCase):
    def test_ten_active_classes(self):
        self.assertEqual(len(CLASSES), 10)
        self.assertEqual(len(CLASS_NAMES), 10)

    def test_no_duplicate_class_names(self):
        self.assertEqual(len(CLASS_NAMES), len(set(CLASS_NAMES)))

    def test_every_class_has_nonempty_minimum_evidence(self):
        for c in CLASSES:
            self.assertTrue(c.minimum_visible_evidence.strip())

    def test_every_class_has_a_valid_label_type(self):
        valid = {"presence", "presence_and_count", "presence_experimental"}
        for c in CLASSES:
            self.assertIn(c.label_type, valid)

    def test_postponed_classes_are_not_in_the_active_list(self):
        self.assertFalse(set(POSTPONED_CLASSES) & set(CLASS_NAMES))

    def test_eye_and_mouth_are_explicitly_postponed_not_dropped(self):
        self.assertIn("eye", POSTPONED_CLASSES)
        self.assertIn("mouth", POSTPONED_CLASSES)

    def test_class_by_name_returns_the_right_class(self):
        self.assertEqual(class_by_name("circle").name, "circle")

    def test_class_by_name_raises_for_unknown_class(self):
        with self.assertRaises(KeyError):
            class_by_name("giraffe")

    def test_no_species_level_animal_classes(self):
        # PHASE2B_AUDIT_AND_PLAN.md Section 3: no species-level animal
        # interpretation in this phase -- only a single general "animal" class.
        animal_like = [c.name for c in CLASSES if "animal" in c.name.lower()]
        self.assertEqual(animal_like, ["animal"])

    def test_annotation_statuses_include_uncertain_and_not_assessable(self):
        self.assertIn("uncertain", ANNOTATION_STATUSES)
        self.assertIn("not_assessable", ANNOTATION_STATUSES)
        self.assertIn("present", ANNOTATION_STATUSES)
        self.assertIn("absent", ANNOTATION_STATUSES)
        self.assertEqual(len(ANNOTATION_STATUSES), 4)


if __name__ == "__main__":
    unittest.main()
