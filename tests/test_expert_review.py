"""DOAR MVP: expert_review.py -- psychologist review workflow, kept
strictly separate from the AI's own output and from formal ground truth."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.expert_review import load_review, submit_review  # noqa: E402


class LoadReviewTests(unittest.TestCase):
    def test_missing_file_returns_default_not_submitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            review = load_review(tmp)
            self.assertEqual(review["status"], "not_submitted")
            self.assertEqual(review["history"], [])
            self.assertTrue(review["ai_output_preserved"])


class SubmitReviewTests(unittest.TestCase):
    def test_appends_to_history_and_sets_status_submitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_review(tmp, reviewer_name="Dr. Test", action="confirm", target_label="dog")
            review = load_review(tmp)
            self.assertEqual(review["status"], "submitted")
            self.assertEqual(len(review["history"]), 1)
            self.assertEqual(review["history"][0]["action"], "confirm")
            self.assertEqual(review["history"][0]["target_label"], "dog")
            self.assertTrue(review["ai_output_preserved"])

    def test_multiple_reviews_accumulate_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            submit_review(tmp, reviewer_name="A", action="confirm", target_label="dog")
            submit_review(tmp, reviewer_name="B", action="rename", target_label="cat", new_label="dog")
            review = load_review(tmp)
            self.assertEqual(len(review["history"]), 2)
            self.assertEqual(review["history"][1]["reviewer_name"], "B")
            self.assertEqual(review["history"][1]["new_label"], "dog")

    def test_unknown_action_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                submit_review(tmp, reviewer_name="A", action="delete_everything")

    def test_never_touches_detections_or_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            detections_before = {"status": "available", "findings": [{"label": "dog"}]}
            (case_dir / "detections.json").write_text(json.dumps(detections_before), encoding="utf-8")
            analysis_before = {"foo": "bar"}
            (case_dir / "analysis.json").write_text(json.dumps(analysis_before), encoding="utf-8")
            submit_review(case_dir, reviewer_name="A", action="reject", target_label="dog")
            self.assertEqual(json.loads((case_dir / "detections.json").read_text()), detections_before)
            self.assertEqual(json.loads((case_dir / "analysis.json").read_text()), analysis_before)

    def test_refreshes_judges_module_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(
                json.dumps({"module_availability": {"clinician_review": "not_submitted"}}), encoding="utf-8")
            submit_review(case_dir, reviewer_name="A", action="note", note="looks fine")
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["clinician_review"], "submitted")

    def test_no_op_on_judges_when_judges_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Must not raise even though judges.json does not exist.
            submit_review(tmp, reviewer_name="A", action="note", note="x")


if __name__ == "__main__":
    unittest.main()
