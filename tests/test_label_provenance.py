"""Tests for label_provenance.py -- non-destructive, post-hoc dataset-label
auditing (working spec Section 5: dataset-label isolation)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class ExtractOriginalSourceLabelTests(unittest.TestCase):
    def test_recognizes_known_class_folder(self):
        from doar.label_provenance import extract_original_source_label
        self.assertEqual(
            extract_original_source_label(r"C:\data\train\Happy\img001.png"), "Happy")

    def test_returns_none_for_unrecognized_folder(self):
        from doar.label_provenance import extract_original_source_label
        self.assertIsNone(extract_original_source_label(r"C:\uploads\webapp_cases\case_1\drawing.png"))

    def test_case_sensitive_exact_class_match_only(self):
        # CLASSES are exactly Angry/Fear/Happy/Sad -- a folder named "happy"
        # (dataset variant casing) must not silently match.
        from doar.label_provenance import extract_original_source_label
        self.assertIsNone(extract_original_source_label(r"C:\data\train\happy\img001.png"))


class ComputeLabelAuditStatusTests(unittest.TestCase):
    def test_no_original_label_is_uncertain(self):
        from doar.label_provenance import compute_label_audit_status
        result = compute_label_audit_status(None, {"status": "available", "top_class": "Happy"})
        self.assertEqual(result["audit_status"], "UNCERTAIN")
        self.assertIsNone(result["original_source_label"])

    def test_no_emotion_prediction_is_uncertain(self):
        from doar.label_provenance import compute_label_audit_status
        result = compute_label_audit_status("Happy", {"status": "unavailable"})
        self.assertEqual(result["audit_status"], "UNCERTAIN")
        self.assertEqual(result["original_source_label"], "Happy")

    def test_matching_prediction_is_consistent(self):
        from doar.label_provenance import compute_label_audit_status
        result = compute_label_audit_status("Happy", {"status": "available", "top_class": "Happy"})
        self.assertEqual(result["audit_status"], "CONSISTENT")

    def test_mismatched_prediction_is_possible_conflict(self):
        from doar.label_provenance import compute_label_audit_status
        result = compute_label_audit_status("Happy", {"status": "available", "top_class": "Sad"})
        self.assertEqual(result["audit_status"], "POSSIBLE_CONFLICT")
        self.assertEqual(result["model_predicted_label"], "Sad")

    def test_never_returns_a_status_outside_the_allowed_post_hoc_set(self):
        # This function must never itself produce REVIEWED_CONFIRMED,
        # REVIEWED_CORRECTED, or ADJUDICATION_REQUIRED -- those are set only
        # by a later human-review action (not yet implemented).
        from doar.label_provenance import compute_label_audit_status
        allowed = {"CONSISTENT", "POSSIBLE_CONFLICT", "UNCERTAIN"}
        cases = [
            (None, {"status": "unavailable"}),
            ("Happy", {"status": "unavailable"}),
            ("Happy", {"status": "available", "top_class": "Happy"}),
            ("Happy", {"status": "available", "top_class": "Sad"}),
        ]
        for original, emotion in cases:
            result = compute_label_audit_status(original, emotion)
            self.assertIn(result["audit_status"], allowed)


class AnalyzeImageIntegrationTests(unittest.TestCase):
    def test_analyze_image_writes_label_provenance_without_using_it_upstream(self):
        # A synthetic image path outside any dataset split folder must produce
        # label_provenance with original_source_label=None, and must not
        # raise or alter any other analysis field.
        import tempfile
        import numpy as np
        from PIL import Image
        from doar.analysis import analyze_image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "uploaded_drawing.png"
            Image.fromarray(
                (np.random.rand(64, 64, 3) * 255).astype("uint8")
            ).save(image_path)
            result = analyze_image(str(image_path), str(Path(tmp) / "case"))
            self.assertIn("label_provenance", result.to_dict())
            self.assertIsNone(result.label_provenance["original_source_label"])
            self.assertEqual(result.label_provenance["audit_status"], "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
