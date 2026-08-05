"""Tests for DOAR-TRACE Phase 2A.1, Section 4: redefine_coverage_full's
margin-based override of PSY_AR_SIZE_FULL_015 -- see
docs/PAGE_COVERAGE_DEFINITION_DECISION.md."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.rule_engine_v2 import COVERAGE_FULL_MARGIN_THRESHOLD, redefine_coverage_full


def _rule_eval(status: str, confidence_ceiling: float = 0.15) -> dict:
    return {
        "rule_id": "PSY_AR_SIZE_FULL_015", "tier": "tier_1_prompt_independent",
        "activation_status": "IMPLEMENTED_UNVALIDATED", "status": status,
        "matched_evidence_ids": ["ev_bbox_coverage"] if status == "weak_support" else [],
        "missing_evidence": [], "source_type": "psychologist_supplied_hypothesis",
        "scientific_support": "not_found_for_threshold_claim", "confidence_ceiling": confidence_ceiling,
        "rule_confidence": 0.15 if status == "weak_support" else 0.0, "confidence_ceiling_enforced": True,
        "professional_reasoning": "x" if status == "weak_support" else None,
        "parent_safe_wording": "y" if status == "weak_support" else None,
        "original_arabic": "z", "english_translation": "w", "references": [], "limitations": [],
        "requires_clinician_review": True,
    }


_ASSESSABLE_REF = {"page_relative_features_assessable": True}
_NOT_ASSESSABLE_REF = {"page_relative_features_assessable": False}


class RedefineCoverageFullTests(unittest.TestCase):
    def test_not_assessable_reference_leaves_evaluations_untouched(self):
        original = [_rule_eval("not_assessable")]
        out = redefine_coverage_full(original, {"margins_normalized": [0.01, 0.01, 0.01, 0.01]}, _NOT_ASSESSABLE_REF)
        self.assertEqual(out, original)

    def test_already_not_assessable_rows_are_never_recomputed_even_with_a_valid_reference(self):
        # apply_page_frame_gating already marked this not_assessable for a
        # different (page-frame) reason -- redefine_coverage_full must not
        # second-guess that, even if a page reference happens to look valid.
        original = [_rule_eval("not_assessable")]
        out = redefine_coverage_full(original, {"margins_normalized": [0.01, 0.01, 0.01, 0.01]}, _ASSESSABLE_REF)
        self.assertEqual(out, original)

    def test_all_margins_within_threshold_triggers(self):
        margins = [0.05, 0.05, 0.05, 0.05]
        out = redefine_coverage_full([_rule_eval("not_matched")], {"margins_normalized": margins}, _ASSESSABLE_REF)
        self.assertEqual(out[0]["status"], "weak_support")
        self.assertEqual(out[0]["matched_evidence_ids"], ["ev_bbox_coverage"])
        self.assertIsNotNone(out[0]["parent_safe_wording"])

    def test_one_margin_over_threshold_does_not_trigger(self):
        margins = [0.05, 0.05, 0.05, COVERAGE_FULL_MARGIN_THRESHOLD + 0.01]
        out = redefine_coverage_full([_rule_eval("weak_support")], {"margins_normalized": margins}, _ASSESSABLE_REF)
        self.assertEqual(out[0]["status"], "not_matched")
        self.assertEqual(out[0]["matched_evidence_ids"], [])
        self.assertIsNone(out[0]["parent_safe_wording"])
        self.assertIsNone(out[0]["professional_reasoning"])

    def test_exact_threshold_value_counts_as_within(self):
        margins = [COVERAGE_FULL_MARGIN_THRESHOLD] * 4
        out = redefine_coverage_full([_rule_eval("not_matched")], {"margins_normalized": margins}, _ASSESSABLE_REF)
        self.assertEqual(out[0]["status"], "weak_support")

    def test_old_weak_support_can_flip_to_not_matched(self):
        # Proves this is a genuine override, not merely additive: the OLD
        # (image-relative-area) engine's own weak_support verdict must be
        # replaceable by the new definition's not_matched.
        margins = [0.3, 0.3, 0.3, 0.3]
        out = redefine_coverage_full([_rule_eval("weak_support")], {"margins_normalized": margins}, _ASSESSABLE_REF)
        self.assertEqual(out[0]["status"], "not_matched")

    def test_old_not_matched_can_flip_to_weak_support(self):
        margins = [0.02, 0.02, 0.02, 0.02]
        out = redefine_coverage_full([_rule_eval("not_matched")], {"margins_normalized": margins}, _ASSESSABLE_REF)
        self.assertEqual(out[0]["status"], "weak_support")

    def test_missing_margins_is_not_matched_not_a_crash(self):
        out = redefine_coverage_full([_rule_eval("weak_support")], {"margins_normalized": None}, _ASSESSABLE_REF)
        self.assertEqual(out[0]["status"], "not_matched")
        self.assertEqual(out[0]["missing_evidence"], ["margins_unavailable"])

    def test_other_rules_are_never_touched(self):
        other = {"rule_id": "PSY_AR_SIZE_HALF_014", "status": "weak_support"}
        out = redefine_coverage_full([other], {"margins_normalized": [0.5, 0.5, 0.5, 0.5]}, _ASSESSABLE_REF)
        self.assertEqual(out, [other])

    def test_confidence_ceiling_still_enforced_after_override(self):
        margins = [0.01, 0.01, 0.01, 0.01]
        out = redefine_coverage_full([_rule_eval("not_matched", confidence_ceiling=0.08)], {"margins_normalized": margins}, _ASSESSABLE_REF)
        self.assertLessEqual(out[0]["rule_confidence"], 0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
