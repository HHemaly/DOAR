"""Tests for src/doar/formal_feature_review.py -- Phase G0 Step 7's
append-only expert-review schema for candidate formal/graphic
observations."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.formal_feature_review import (
    load_formal_feature_review, submit_formal_feature_review, export_all_formal_feature_reviews,
)


class SubmitFormalFeatureReviewTests(unittest.TestCase):
    def test_submit_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            submit_formal_feature_review(
                case_dir, reviewer_name="Dr. Test", feature_id="line.orientation_entropy",
                observation_correct="yes", clinically_relevant="uncertain",
                expert_support_level="weak", expert_rule_decision="candidate",
                overall_synthesis_agreement="partially_agree", raw_measurement=0.77,
                doar_observation="Highly multidirectional colouring with low local directional coherence.",
            )
            loaded = load_formal_feature_review(case_dir)
            self.assertEqual(len(loaded["entries"]), 1)
            self.assertEqual(loaded["entries"][0]["feature_id"], "line.orientation_entropy")
            self.assertEqual(loaded["entries"][0]["drawing_id"], "case")

    def test_unknown_observation_correct_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            with self.assertRaises(ValueError):
                submit_formal_feature_review(
                    case_dir, reviewer_name="Dr. Test", feature_id="x",
                    observation_correct="definitely", clinically_relevant="yes",
                    expert_support_level="weak", expert_rule_decision="candidate",
                    overall_synthesis_agreement="agree")

    def test_unknown_expert_rule_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            with self.assertRaises(ValueError):
                submit_formal_feature_review(
                    case_dir, reviewer_name="Dr. Test", feature_id="x",
                    observation_correct="yes", clinically_relevant="yes",
                    expert_support_level="weak", expert_rule_decision="promote_to_production",
                    overall_synthesis_agreement="agree")

    def test_append_only_never_overwrites_a_prior_entry(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            for i in range(3):
                submit_formal_feature_review(
                    case_dir, reviewer_name=f"Dr. {i}", feature_id="stroke.crossing_density",
                    observation_correct="yes", clinically_relevant="yes",
                    expert_support_level="moderate", expert_rule_decision="experimental",
                    overall_synthesis_agreement="agree")
            self.assertEqual(len(load_formal_feature_review(case_dir)["entries"]), 3)

    def test_expert_rule_decision_never_auto_activates_anything(self):
        """Submitting expert_rule_decision='candidate' (or any value) must
        never import or write to any production/master registry module or
        file -- structural guard (the module's own docstring may still
        MENTION the registry filename for documentation)."""
        import inspect
        from doar import formal_feature_review as mod
        source = inspect.getsource(mod)
        for forbidden in ("import master_registry_v3_build", "from .master_registry_v3_build",
                          "rules_registry_v2.json", "RULE_EVIDENCE_MATRIX.csv", "CONCERN_DOMAIN_MAP.json"):
            self.assertNotIn(forbidden, source)


class ExportAllFormalFeatureReviewsTests(unittest.TestCase):
    def test_export_aggregates_across_cases(self):
        with tempfile.TemporaryDirectory() as d:
            cases_dir = Path(d)
            for name in ("case_a", "case_b"):
                case_dir = cases_dir / name
                case_dir.mkdir()
                submit_formal_feature_review(
                    case_dir, reviewer_name="Dr. Test", feature_id="line.mean_width",
                    observation_correct="yes", clinically_relevant="no",
                    expert_support_level="none", expert_rule_decision="reject",
                    overall_synthesis_agreement="disagree")
            all_entries = export_all_formal_feature_reviews(cases_dir)
            self.assertEqual(len(all_entries), 2)

    def test_missing_cases_dir_returns_empty(self):
        self.assertEqual(export_all_formal_feature_reviews(Path("no_such_dir_xyz")), [])


if __name__ == "__main__":
    unittest.main()
