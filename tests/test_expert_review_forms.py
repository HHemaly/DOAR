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
    CONSTRUCT_FORM_FIELDS, RULE_FORM_FIELDS, SENTENCE_FORM_FIELDS,
    build_construct_review_rows, build_rule_review_rows, build_sentence_review_rows,
)
from doar.registry_v2_build import build_registry_v2


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
