"""Phase 2C.7 detector-policy tests -- synthetic data only, no real
model weights."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7 import detector_policy as pol


class ObjectClassPolicyTests(unittest.TestCase):
    def test_all_ten_object_classes_plus_mouth_and_reference_present(self):
        expected = {"person", "face", "hand", "animal", "house", "tree", "heart", "star",
                    "circle", "vehicle", "mouth", "person_part_reference"}
        self.assertEqual(set(pol.OBJECT_CLASS_POLICY.keys()), expected)

    def test_only_three_statuses_used(self):
        allowed = {pol.VALIDATED_AUTOMATIC, pol.EXPERIMENTAL_AUTOMATIC, pol.DISABLED}
        for entry in pol.OBJECT_CLASS_POLICY.values():
            self.assertIn(entry.status, allowed)

    def test_disabled_classes_have_no_frozen_model(self):
        for target in ("circle", "vehicle"):
            entry = pol.OBJECT_CLASS_POLICY[target]
            self.assertEqual(entry.status, pol.DISABLED)
            self.assertIsNone(entry.best_model)
            self.assertIsNone(entry.threshold)

    def test_validated_classes_have_a_frozen_model_and_threshold(self):
        for target in ("person", "face", "hand", "tree", "house"):
            entry = pol.OBJECT_CLASS_POLICY[target]
            self.assertEqual(entry.status, pol.VALIDATED_AUTOMATIC)
            self.assertIsNotNone(entry.best_model)
            self.assertIsNotNone(entry.threshold)
            self.assertGreaterEqual(entry.balanced_accuracy, pol.MIN_BALANCED_ACCURACY_FOR_VALIDATION)

    def test_no_localization_validated_for_object_classes(self):
        """None of the Phase 2C.4 object classes have real bbox ground
        truth -- only eye does, from this phase's own new evaluation."""
        for target, entry in pol.OBJECT_CLASS_POLICY.items():
            if target == "mouth":
                continue
            self.assertFalse(entry.localization_validated, target)


class BuildEyePolicyEntryTests(unittest.TestCase):
    def test_builds_a_correctly_typed_entry(self):
        entry = pol.build_eye_policy_entry(
            status=pol.VALIDATED_AUTOMATIC, best_model="owlv2", best_model_checkpoint="ckpt",
            prompt="a eye", threshold=0.1, precision=0.8, recall=0.6, balanced_accuracy=0.75,
            n_ground_truth_present=100, localization_validated=True, rationale="test")
        self.assertEqual(entry.target, "eye")
        self.assertEqual(entry.status, pol.VALIDATED_AUTOMATIC)
        self.assertIn("PSY_AR_EYES_CLOSED_003", entry.related_rule_ids)

    def test_usage_text_matches_status(self):
        validated = pol.build_eye_policy_entry(
            status=pol.VALIDATED_AUTOMATIC, best_model="m", best_model_checkpoint="c", prompt="p",
            threshold=0.1, precision=0.8, recall=0.6, balanced_accuracy=0.75,
            n_ground_truth_present=100, localization_validated=True, rationale="r")
        self.assertIn("validated evidence", validated.allowed_downstream_usage)

        experimental = pol.build_eye_policy_entry(
            status=pol.EXPERIMENTAL_AUTOMATIC, best_model="m", best_model_checkpoint="c", prompt="p",
            threshold=0.1, precision=0.3, recall=0.2, balanced_accuracy=0.52,
            n_ground_truth_present=10, localization_validated=False, rationale="r")
        self.assertIn("Technical View", experimental.allowed_downstream_usage)

        disabled = pol.build_eye_policy_entry(
            status=pol.DISABLED, best_model="m", best_model_checkpoint="c", prompt="p",
            threshold=0.1, precision=None, recall=0.0, balanced_accuracy=0.5,
            n_ground_truth_present=2, localization_validated=False, rationale="r")
        self.assertEqual(disabled.allowed_downstream_usage, "not used")


class FullPolicyTests(unittest.TestCase):
    def test_full_policy_includes_eye(self):
        eye_entry = pol.build_eye_policy_entry(
            status=pol.VALIDATED_AUTOMATIC, best_model="m", best_model_checkpoint="c", prompt="p",
            threshold=0.1, precision=0.8, recall=0.6, balanced_accuracy=0.7,
            n_ground_truth_present=170, localization_validated=True, rationale="r")
        full = pol.full_policy(eye_entry)
        self.assertIn("eye", full)
        self.assertEqual(full["eye"].status, pol.VALIDATED_AUTOMATIC)
        self.assertEqual(len(full), len(pol.OBJECT_CLASS_POLICY) + 1)


class ToRowsTests(unittest.TestCase):
    def test_to_rows_flattens_rule_ids(self):
        eye_entry = pol.build_eye_policy_entry(
            status=pol.EXPERIMENTAL_AUTOMATIC, best_model="m", best_model_checkpoint="c", prompt="p",
            threshold=0.1, precision=0.3, recall=0.2, balanced_accuracy=0.5,
            n_ground_truth_present=10, localization_validated=False, rationale="r")
        rows = pol.to_rows(pol.full_policy(eye_entry))
        face_row = next(r for r in rows if r["target"] == "face")
        self.assertIn("EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038", face_row["related_rule_ids"])
        self.assertIsInstance(face_row["related_rule_ids"], str)


if __name__ == "__main__":
    unittest.main()
