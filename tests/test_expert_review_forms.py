"""Tests for expert_review_forms.py (DOAR-TRACE Phase 1.5, Section 10):
every row must be generated from real registry/construct/case data, never
hand-written, and every form must carry the exact required field set."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.expert_review_forms import (
    CONSTRUCT_FORM_FIELDS, PHASE2A1_PAGE_RULE_FORM_FIELDS, PHASE2A1_RULE_FORM_FIELDS,
    PHASE2A1_THRESHOLD_FORM_FIELDS, RULE_FORM_FIELDS, SENTENCE_FORM_FIELDS,
    build_construct_review_rows, build_phase2a1_page_rule_review_rows,
    build_phase2a1_rule_review_rows, build_phase2a1_threshold_review_rows,
    build_rule_review_rows, build_sentence_review_rows,
)
from doar.registry_v2_build import build_registry_v2
from doar.rule_engine_v2 import ALL_PAGE_GATED_RULE_IDS, V2_RULE_IDS


class RuleReviewFormTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_rule_review_rows()

    def test_one_row_per_registry_rule(self):
        self.assertEqual(len(self.rows), 41)

    def test_every_row_has_the_exact_required_fields(self):
        for row in self.rows:
            self.assertEqual(set(row.keys()), set(RULE_FORM_FIELDS))

    def test_source_text_is_never_empty(self):
        for row in self.rows:
            self.assertTrue(row["source_text"], row["rule_id"])

    def test_reviewer_input_columns_start_blank(self):
        # Ratings/judgments are for a human reviewer -- never pre-filled.
        for row in self.rows:
            self.assertEqual(row["observable_clarity_1_4"], "")
            self.assertEqual(row["interpretation_relevance_1_4"], "")
            self.assertEqual(row["construct_mapping_accepted_y_n"], "")
            self.assertEqual(row["safe_for_parent_y_n"], "")

    def test_source_and_page_cites_a_real_document(self):
        registry_v2 = build_registry_v2()
        real_docs = {r["source_document"] for r in registry_v2["rules"]}
        for row in self.rows:
            self.assertTrue(any(doc in row["source_and_page"] for doc in real_docs), row["rule_id"])


class ConstructReviewFormTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_construct_review_rows()

    def test_one_row_per_construct(self):
        self.assertEqual(len(self.rows), 12)

    def test_every_row_has_the_exact_required_fields(self):
        for row in self.rows:
            self.assertEqual(set(row.keys()), set(CONSTRUCT_FORM_FIELDS))

    def test_comments_column_lists_real_currently_mapped_rules(self):
        row = next(r for r in self.rows if r["construct_id"] == "visual_dominance_or_prominence")
        self.assertIn("PSY_AR_SIZE_FULL_015", row["comments"])

    def test_reviewer_input_columns_start_blank(self):
        for row in self.rows:
            self.assertEqual(row["definition_clarity_1_4"], "")
            self.assertEqual(row["contributing_rules_appropriate_y_n"], "")


class SentenceReviewFormTests(unittest.TestCase):
    def test_built_from_real_generated_claims_document(self):
        generated_claims = {
            "claims": [{
                "claim_id": "claim_001", "claim_type": "individual_rule_suggestion",
                "text": "Example sentence.", "evidence_ids": ["ev_bbox_coverage"],
                "rule_ids": ["PSY_AR_SIZE_FULL_015"], "source_ids": [], "construct_id": None,
                "status": "accepted",
            }],
        }
        rows = build_sentence_review_rows(generated_claims)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0].keys()), set(SENTENCE_FORM_FIELDS))
        self.assertEqual(rows[0]["sentence_text"], "Example sentence.")
        self.assertEqual(rows[0]["verification_status"], "accepted")

    def test_empty_claims_produces_no_rows(self):
        self.assertEqual(build_sentence_review_rows({"claims": []}), [])


class Phase2A1RuleReviewFormTests(unittest.TestCase):
    """DOAR-TRACE Phase 2A.1, Section 7."""

    def setUp(self):
        self.rows = build_phase2a1_rule_review_rows()

    def test_one_row_per_newly_activated_rule(self):
        self.assertEqual(len(self.rows), 4)
        self.assertEqual({r["rule_id"] for r in self.rows}, set(V2_RULE_IDS))

    def test_every_row_has_the_exact_required_fields(self):
        for row in self.rows:
            self.assertEqual(set(row.keys()), set(PHASE2A1_RULE_FORM_FIELDS))

    def test_source_text_and_wording_are_never_empty(self):
        for row in self.rows:
            self.assertTrue(row["source_text"], row["rule_id"])
            self.assertTrue(row["parent_safe_wording_current"], row["rule_id"])

    def test_no_expert_judgment_fields_are_pre_filled(self):
        judgment_fields = [
            "observable_definition_accepted_y_n", "proxy_interpretation_accepted_y_n",
            "threshold_accepted_y_n", "suggested_threshold_or_decision_rule",
            "parent_safe_wording_accepted_y_n", "alternative_explanations_missing",
            "safe_for_individual_display_y_n", "safe_for_combined_aggregation_y_n", "comments",
        ]
        for row in self.rows:
            for field in judgment_fields:
                self.assertEqual(row[field], "", f"{row['rule_id']}.{field} was pre-filled")


class Phase2A1ThresholdReviewFormTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_phase2a1_threshold_review_rows()

    def test_every_row_has_the_exact_required_fields(self):
        for row in self.rows:
            self.assertEqual(set(row.keys()), set(PHASE2A1_THRESHOLD_FORM_FIELDS))

    def test_covers_every_executable_rule_except_directly_sourced_ones(self):
        registry_v2 = build_registry_v2()
        expected = {
            r["rule_id"] for r in registry_v2["rules"]
            if r["allowed_output_level"] == "individual_heuristic_only" and r["threshold_source"] != "directly_sourced"
        }
        self.assertEqual({r["rule_id"] for r in self.rows}, expected)
        # PSY_AR_SIZE_SMALL_016 is directly_sourced -- must be excluded.
        self.assertNotIn("PSY_AR_SIZE_SMALL_016", {r["rule_id"] for r in self.rows})

    def test_no_directly_sourced_threshold_included(self):
        for row in self.rows:
            self.assertNotEqual(row["threshold_source"], "directly_sourced")

    def test_no_expert_judgment_fields_are_pre_filled(self):
        for row in self.rows:
            self.assertEqual(row["threshold_accepted_y_n"], "")
            self.assertEqual(row["suggested_threshold_or_decision_rule"], "")
            self.assertEqual(row["comments"], "")


class Phase2A1PageRuleReviewFormTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_phase2a1_page_rule_review_rows()

    def test_one_row_per_page_gated_rule(self):
        self.assertEqual({r["rule_id"] for r in self.rows}, set(ALL_PAGE_GATED_RULE_IDS))

    def test_every_row_has_the_exact_required_fields(self):
        for row in self.rows:
            self.assertEqual(set(row.keys()), set(PHASE2A1_PAGE_RULE_FORM_FIELDS))

    def test_coverage_full_shows_the_redefined_margin_based_definition(self):
        row = next(r for r in self.rows if r["rule_id"] == "PSY_AR_SIZE_FULL_015")
        self.assertIn("margin", row["current_definition"].lower())

    def test_page_reference_requirement_is_stated_for_every_row(self):
        for row in self.rows:
            self.assertIn("page_relative_features_assessable", row["page_reference_requirement"])

    def test_no_expert_judgment_fields_are_pre_filled(self):
        for row in self.rows:
            self.assertEqual(row["page_reference_definition_accepted_y_n"], "")
            self.assertEqual(row["rule_definition_accepted_y_n"], "")
            self.assertEqual(row["suggested_definition_or_decision_rule"], "")
            self.assertEqual(row["comments"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
