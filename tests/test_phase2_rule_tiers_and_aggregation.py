"""Phase 2 tests: rule tiering, missing detectors, unknown context, correlated
evidence, contradictory evidence (documented as NOT implemented), and
prohibited diagnostic output in the new aggregation wording.

None of this makes any rule user-facing: CONCERNS_ENABLED stays False, and
every activation_status in rules_registry.json is unchanged by Phase 2
(DETECTOR_UNAVAILABLE for Tier 2, IMPLEMENTED_UNVALIDATED for Tier 1).
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TierDispatchTests(unittest.TestCase):
    def test_tier2_rule_is_missing_detector(self):
        from doar.rules import _status_for
        rule = {"rule_id": "X", "tier": "tier_2_content_conditional", "observable": "wide_eyes"}
        status, matched, missing = _status_for(rule, {"bounding_box_coverage": 0.5, "placement": "top"})
        self.assertEqual(status, "missing_detector")
        self.assertEqual(matched, [])
        self.assertEqual(missing, ["detector_absent:wide_eyes"])

    def test_tier3_rule_is_not_assessable_context_unknown(self):
        # No Tier 3 rule exists in the shipped registry yet; this is a
        # forward-looking test for the dispatcher branch, using a synthetic
        # rule, exactly like the existing tests exercise concerns.py with
        # synthetic data ahead of real detectors.
        from doar.rules import _status_for
        rule = {"rule_id": "X", "tier": "tier_3_prompt_or_age_dependent", "observable": "omitted_house"}
        status, matched, missing = _status_for(rule, {"bounding_box_coverage": 0.5, "placement": "top"})
        self.assertEqual(status, "not_assessable_context_unknown")
        self.assertEqual(matched, [])
        self.assertEqual(missing, [])

    def test_tier1_coverage_rule_dispatches_correctly(self):
        from doar.rules import _status_for
        rule = {"rule_id": "X", "tier": "tier_1_prompt_independent", "observable": "coverage_full"}
        status, matched, _ = _status_for(rule, {"bounding_box_coverage": 0.95, "placement": "top"})
        self.assertEqual(status, "weak_support")
        self.assertEqual(matched, ["ev_bbox_coverage"])

    def test_tier1_placement_rule_dispatches_correctly(self):
        from doar.rules import _status_for
        rule = {"rule_id": "X", "tier": "tier_1_prompt_independent", "observable": "placement_left"}
        status, matched, _ = _status_for(rule, {"bounding_box_coverage": 0.5, "placement": "middle-left"})
        self.assertEqual(status, "weak_support")
        self.assertEqual(matched, ["ev_centroid"])

    def test_unrecognized_tier1_observable_fails_loudly_not_silently(self):
        # A registry/tier mismatch (a rule claiming tier_1 with an observable
        # this module doesn't know how to evaluate) must raise, not silently
        # degrade to missing_detector -- that would hide a real
        # configuration bug behind a status that looks like "no detector yet".
        from doar.rules import _status_for
        rule = {"rule_id": "X", "tier": "tier_1_prompt_independent", "observable": "nonexistent_observable"}
        with self.assertRaises(ValueError):
            _status_for(rule, {"bounding_box_coverage": 0.5, "placement": "top"})

    def test_real_registry_end_to_end_tier_assignment_matches_matrix(self):
        # Cross-check against RULE_COVERAGE_MATRIX.csv's assignment (6 tier_1,
        # 13 tier_2, 0 tier_3) using the real registry file and the real
        # evaluate_rules() entry point, not a synthetic rule.
        from doar.rules import evaluate_rules
        evaluations, _ = evaluate_rules({"bounding_box_coverage": 0.5, "placement": "top"}, {}, [])
        tiers = [e["tier"] for e in evaluations]
        self.assertEqual(tiers.count("tier_1_prompt_independent"), 6)
        self.assertEqual(tiers.count("tier_2_content_conditional"), 13)
        self.assertEqual(tiers.count("tier_3_prompt_or_age_dependent"), 0)
        for e in evaluations:
            if e["tier"] == "tier_2_content_conditional":
                self.assertEqual(e["activation_status"], "DETECTOR_UNAVAILABLE")
            else:
                self.assertEqual(e["activation_status"], "IMPLEMENTED_UNVALIDATED")


class ModelEvidencePassthroughTests(unittest.TestCase):
    def _model_evidence(self, confidence=0.7):
        from doar.schemas import Evidence
        return Evidence("ev_emotion_prediction", "model_prediction", {"Happy": confidence},
                        "test_model", confidence, [])

    def test_evaluate_rules_no_longer_discards_evidence(self):
        # Regression test for the corrected Phase 2 finding: evaluate_rules()
        # previously did `del colour, evidence`, discarding the only
        # genuinely independent evidence source in the system. It must now
        # reach derive_concerns() (verified indirectly: enabling concerns
        # with a matching rule + model evidence produces a concern with
        # source_type model_prediction present).
        from doar.rules import evaluate_rules
        from doar import concerns as concerns_module
        composition = {"bounding_box_coverage": 0.95, "placement": "top"}  # matches coverage_full
        evidence = [self._model_evidence()]
        original_enabled = concerns_module.CONCERNS_ENABLED
        concerns_module.CONCERNS_ENABLED = True
        try:
            _, produced = evaluate_rules(composition, {}, evidence)
        finally:
            concerns_module.CONCERNS_ENABLED = original_enabled
        self.assertEqual(len(produced), 1)
        self.assertIn("model_prediction", produced[0]["source_types"])
        self.assertEqual(produced[0]["aggregation_strength"], "POSSIBLE_FOR_EXPLORATION")

    def test_rule_plus_model_evidence_reaches_possible_for_exploration(self):
        from doar.concerns import derive_concerns
        rules = [{"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
                  "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.15,
                  "missing_evidence": []}]
        concerns = derive_concerns(rules, [self._model_evidence()], enabled=True)
        self.assertEqual(len(concerns), 1)
        self.assertEqual(concerns[0]["aggregation_strength"], "POSSIBLE_FOR_EXPLORATION")
        # Confidence stays capped by the LOWER of the two contributors (the
        # rule's 0.15 ceiling), never raised by the model's own 0.7 confidence.
        self.assertEqual(concerns[0]["confidence"], 0.15)

    def test_three_distinct_sources_reach_professional_review_suggested(self):
        # Synthetic: no third independent source exists in the shipped
        # pipeline yet (Phase 3 detectors are not built). This documents the
        # top of the vocabulary is reachable by the logic, not that it is
        # reachable by any current real analysis.
        from doar.concerns import derive_concerns
        rules = [
            {"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
             "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.15,
             "missing_evidence": []},
            {"status": "weak_support", "matched_evidence_ids": ["ev_detection_person"],
             "source_type": "objective_feature", "confidence_ceiling": 0.4,
             "missing_evidence": []},
        ]
        concerns = derive_concerns(rules, [self._model_evidence()], enabled=True)
        self.assertEqual(len(concerns), 1)
        self.assertEqual(concerns[0]["aggregation_strength"], "PROFESSIONAL_REVIEW_SUGGESTED")

    def test_no_model_evidence_is_a_safe_no_op(self):
        from doar.rules import evaluate_rules
        composition = {"bounding_box_coverage": 0.5, "placement": "top"}
        # Empty evidence list (no emotion model ran) must not raise.
        evaluations, concerns = evaluate_rules(composition, {}, [])
        self.assertEqual(concerns, [])   # CONCERNS_ENABLED is False in this process by default
        self.assertTrue(evaluations)


class ContradictoryEvidenceBaselineTests(unittest.TestCase):
    def test_contradicting_evidence_is_always_empty_not_fabricated(self):
        # Phase 2 does NOT implement contradiction detection between the
        # emotion model and rule hypotheses -- the registry's 19 rules
        # describe personality/psychological traits with no defined mapping
        # onto the Angry/Fear/Happy/Sad label space, so any such mapping
        # would be invented, not derived. This test documents that
        # `contradicting_evidence` is honestly always [] today, rather than
        # silently fabricating a comparison. See SCIENTIFIC_LIMITATIONS.md.
        from doar.concerns import derive_concerns
        from doar.schemas import Evidence
        rules = [{"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
                  "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.15,
                  "missing_evidence": []}]
        model_evidence = [Evidence("ev_emotion_prediction", "model_prediction", {}, "m", 0.9, [])]
        concerns = derive_concerns(rules, model_evidence, enabled=True)
        self.assertEqual(len(concerns), 1)
        self.assertEqual(concerns[0]["contradicting_evidence"], [])


class ProhibitedDiagnosticOutputTests(unittest.TestCase):
    _CONDITIONS = (r"(anxiety|anxious|depression|depressed|trauma|traumati[sz]ed|aggression|"
                   r"aggressive|autism|autistic|abuse|abused|ptsd|adhd|disorder|mentally ill)")
    _DIAGNOSTIC = re.compile(
        rf"\b(has|had|diagnosed with|suffers? from|proves?|confirms?|exhibits?|shows?|showing|"
        rf"signs? of|evidence of)\s+(a\s+|an\s+|of\s+)?{_CONDITIONS}\b", re.IGNORECASE)

    def test_new_aggregation_wording_contains_no_diagnostic_language(self):
        # Regression guard on the wording introduced/kept in concerns.py this
        # phase, reusing the same diagnostic-pattern shape judges.py's
        # safety_judge scans for (Phase 0 audit confirmed that scanner is
        # real and executed, not decorative).
        from doar.concerns import derive_concerns
        rules = [
            {"status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"],
             "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.2,
             "missing_evidence": []},
            {"status": "weak_support", "matched_evidence_ids": ["ev_centroid"],
             "source_type": "psychologist_supplied_hypothesis", "confidence_ceiling": 0.2,
             "missing_evidence": []},
        ]
        concerns = derive_concerns(rules, enabled=True)
        self.assertEqual(len(concerns), 1)
        text = concerns[0]["professional_wording"] + " " + concerns[0]["parent_safe_wording"]
        self.assertFalse(self._DIAGNOSTIC.search(text), f"diagnostic language found in: {text!r}")
        self.assertIn("not a diagnosis", text.lower())

    def test_concerns_are_disabled_in_production_regardless_of_phase2_fixes(self):
        from doar.concerns import CONCERNS_ENABLED
        self.assertFalse(CONCERNS_ENABLED,
                         "Phase 2 must not flip concern output on in production")


if __name__ == "__main__":
    unittest.main(verbosity=2)
