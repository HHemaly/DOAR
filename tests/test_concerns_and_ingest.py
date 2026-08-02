"""D6 concern-convergence engine + provenance-preserving PDF ingestion draft."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class ConcernConvergenceTests(unittest.TestCase):
    def test_single_source_type_caps_at_weak_hypothesis(self):
        # Phase 2 (2026-08-02): two clinician-symbolic rules firing together
        # are still one source type. Per the working spec's aggregation
        # policy ("several correlated rules from one family: at most a weak
        # hypothesis"), this must now surface as WEAK_HYPOTHESIS rather than
        # silently vanish -- previously this test asserted `== []`, which
        # was the pre-Phase-2 binary converge-or-nothing behaviour.
        from doar.concerns import derive_concerns
        rules = [
            {"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
             "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.2},
            {"status": "weak_support", "matched_evidence_ids": ["ev_centroid"],
             "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.2},
        ]
        concerns = derive_concerns(rules, enabled=True)
        self.assertEqual(len(concerns), 1)
        self.assertEqual(concerns[0]["aggregation_strength"], "WEAK_HYPOTHESIS")
        self.assertEqual(concerns[0]["source_types"], ["clinician_symbolic"])

    def test_concerns_disabled_by_default(self):
        from doar.concerns import derive_concerns, CONCERNS_ENABLED
        self.assertFalse(CONCERNS_ENABLED)   # Item 11: disabled until a taxonomy exists
        rules = [
            {"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
             "source_type": "objective_feature", "confidence_ceiling": 0.25},
            {"status": "weak_support", "matched_evidence_ids": ["ev_emotion_prediction"],
             "source_type": "model_prediction", "confidence_ceiling": 0.15},
        ]
        # Even converging evidence yields nothing while disabled.
        self.assertEqual(derive_concerns(rules), [])

    def test_converging_independent_sources_raise_capped_concern(self):
        from doar.concerns import derive_concerns
        rules = [
            {"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
             "source_type": "objective_feature", "confidence_ceiling": 0.25},
            {"status": "weak_support", "matched_evidence_ids": ["ev_emotion_prediction"],
             "source_type": "model_prediction", "confidence_ceiling": 0.15},
        ]
        concerns = derive_concerns(rules, enabled=True)
        self.assertEqual(len(concerns), 1)
        c = concerns[0]
        self.assertGreaterEqual(c["source_diversity"], 2)
        self.assertEqual(c["confidence"], 0.15)          # capped by lowest ceiling
        self.assertTrue(c["requires_clinician_review"])
        self.assertEqual(c["approval_status"], "pending")
        self.assertEqual(c["aggregation_strength"], "POSSIBLE_FOR_EXPLORATION")

    def test_current_registry_yields_no_concerns(self):
        # End-to-end: real evaluate_rules over the shipped registry stays empty.
        from doar.rules import evaluate_rules
        composition = {"bounding_box_coverage": 0.5, "placement": "top-left",
                       "foreground_coverage": 0.5}
        _, concerns = evaluate_rules(composition, {}, [])
        self.assertEqual(concerns, [])


class IngestDraftTests(unittest.TestCase):
    def test_draft_is_inert_and_provenance_preserving(self):
        from doar.psychology_ingest import build_draft_registry
        pages = [{"page": 1, "arabic_text": "رسم العيون الواسعة يدل على الانفتاح. رسم الأسد يدل على التفوق."}]
        draft = build_draft_registry(pages, {"source_id": "TEST_PDF"})
        self.assertTrue(draft["activation_blocked"])
        self.assertTrue(draft["draft"])
        self.assertEqual(draft["rule_count"], len(draft["rules"]))
        self.assertGreaterEqual(draft["rule_count"], 2)
        for rule in draft["rules"]:
            self.assertEqual(rule["confidence_ceiling"], 0.0)        # inert
            self.assertEqual(rule["visibility"], "blocked_pending_review")
            self.assertEqual(rule["review_status"], "pending")
            self.assertEqual(rule["page"], 1)                        # provenance
            self.assertTrue(rule["arabic"])                          # literal source kept
            self.assertEqual(rule["english_reviewed"], "")          # awaiting clinician

    def test_empty_pages_produce_no_rules(self):
        from doar.psychology_ingest import build_draft_registry
        draft = build_draft_registry([{"page": 1, "arabic_text": ""}])
        self.assertEqual(draft["rule_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
