"""DOAR reasoning chain (V1.6): verified visual observation -> eligible
atomic rule -> evidence family -> concern domain -> candidate hypothesis.

Proves the two load-bearing safety properties from reasoning_chain.py's
own module docstring:
  1. Only case_verification_status=="verified" entities can ever satisfy
     a rule's visual precondition -- uncertain/rejected/unreviewed never do.
  2. VISUALLY VERIFIED != PSYCHOLOGICALLY VALIDATED: multiple rules from
     the same concern domain, all with visually-verified preconditions,
     still cap out at WEAK_HYPOTHESIS (never higher) unless a genuinely
     independent second source (model_evidence) is also supplied -- exact
     same ceiling concerns.py's own _aggregation_strength already enforces
     for ordinary clinician-rule convergence, now proven to hold even when
     the precondition itself was Gemini-verified.
Also proves this module never touches rules.py::evaluate_rules or
concerns.py::derive_concerns (source-level check), and that the parent
package never leaks a rule_id/bbox/internal field.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar import reasoning_chain as rc  # noqa: E402
from doar.visual_entity import VisualEntity  # noqa: E402


def _entity(entity_id, canonical_label, *, status="verified", aliases_en=()):
    return VisualEntity(
        entity_id=entity_id, entity_type="object", canonical_label=canonical_label,
        candidate_labels=((canonical_label, 0.9),), aliases_en=aliases_en, aliases_ar=(),
        broader_categories=(), possible_subtypes=(), visual_similarities=(),
        bbox=(0.1, 0.1, 0.2, 0.2), crop_ref=None, dominant_colors=None, relative_size=None,
        page_position=None, shape_features=None, line_features=None,
        detector="visual_observer:test", checkpoint="", prompt="",
        confidence=0.9, model_validation_status="UNKNOWN", case_verification_status=status,
        evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
        related_rule_ids=(), source="visual_observer", query=None, timestamp="2026-08-14T00:00:00+00:00",
    )


class FrozenFileLoaderTests(unittest.TestCase):
    def test_rule_matrix_has_all_41_rules(self):
        matrix = rc.load_rule_matrix()
        self.assertEqual(len(matrix), 41)
        self.assertIn("PSY_AR_EYES_WIDE_001", matrix)

    def test_relationship_graph_loads(self):
        graph = rc.load_relationship_graph()
        self.assertIn("same_evidence_family_edges", graph)

    def test_concern_domain_map_loads(self):
        domains = rc.load_concern_domain_map()["domains"]
        self.assertIn("depressive_or_low_mood_related", domains)


class VisualPreconditionTests(unittest.TestCase):
    def test_every_rule_gets_exactly_one_check(self):
        checks = rc.check_visual_preconditions([])
        self.assertEqual(len(checks), 41)
        self.assertEqual(len({c.rule_id for c in checks}), 41)

    def test_unverified_entity_never_satisfies_a_precondition(self):
        for status in ("uncertain", "rejected", "unreviewed"):
            entities = [_entity("e1", "lion", status=status)]
            checks = rc.check_visual_preconditions(entities)
            lion_check = next(c for c in checks if c.rule_id == "PSY_AR_ANIMAL_LION_007")
            self.assertEqual(lion_check.status, "not_satisfied", f"status={status} must not satisfy")

    def test_verified_positive_presence_entity_satisfies_precondition(self):
        entities = [_entity("e1", "lion")]
        checks = rc.check_visual_preconditions(entities)
        lion_check = next(c for c in checks if c.rule_id == "PSY_AR_ANIMAL_LION_007")
        self.assertEqual(lion_check.status, "satisfied")
        self.assertEqual(lion_check.matched_entity_ids, ("e1",))

    def test_absence_pattern_satisfied_when_person_present_part_absent(self):
        entities = [_entity("e1", "girl", aliases_en=("person",))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_HANDS_035")
        self.assertEqual(check.status, "satisfied")

    def test_absence_pattern_not_satisfied_when_part_present(self):
        entities = [_entity("e1", "girl", aliases_en=("person",)), _entity("e2", "hand")]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_HANDS_035")
        self.assertEqual(check.status, "not_satisfied")

    def test_absence_pattern_not_satisfied_without_a_person(self):
        checks = rc.check_visual_preconditions([_entity("e1", "hand")])
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_HANDS_035")
        self.assertEqual(check.status, "not_satisfied")

    def test_process_required_rule_always_blocked_structural(self):
        checks = rc.check_visual_preconditions([_entity("e1", "flowers_clouds_sun")])
        check = next(c for c in checks if c.rule_id == "PSY_AR_FLOWERS_CLOUDS_SUN_010")
        self.assertEqual(check.status, "blocked_structural")

    def test_longitudinal_required_rule_always_blocked_structural(self):
        checks = rc.check_visual_preconditions([])
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_REPEATED_ISOLATION_THEMES_028")
        self.assertEqual(check.status, "blocked_structural")

    def test_tier1_composition_rules_marked_not_applicable_not_blocked(self):
        checks = rc.check_visual_preconditions([])
        check = next(c for c in checks if c.rule_id == "PSY_AR_SIZE_FULL_015")
        self.assertEqual(check.status, "not_applicable_to_this_module")

    def test_needs_unbuilt_feature_rules_have_an_explicit_reason(self):
        checks = rc.check_visual_preconditions([])
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_EXCESSIVE_DETAIL_040")
        self.assertEqual(check.status, "blocked_needs_unbuilt_feature")
        self.assertTrue(check.reason)


class EligibleMatchTests(unittest.TestCase):
    def test_only_satisfied_checks_become_matches(self):
        entities = [_entity("e1", "lion")]
        matches = rc.build_eligible_matches(entities)
        rule_ids = {m.rule_id for m in matches}
        self.assertIn("PSY_AR_ANIMAL_LION_007", rule_ids)
        self.assertLessEqual(len(matches), 3)  # lion + animal_choice_general only, roughly

    def test_match_fields_are_copied_verbatim_from_frozen_matrix(self):
        matrix = rc.load_rule_matrix()
        entities = [_entity("e1", "lion")]
        match = next(m for m in rc.build_eligible_matches(entities) if m.rule_id == "PSY_AR_ANIMAL_LION_007")
        row = matrix["PSY_AR_ANIMAL_LION_007"]
        self.assertEqual(match.allowed_output_level, row["allowed_output_level"])
        self.assertEqual(match.evidence_direction, row["evidence_direction"])
        self.assertEqual(match.source_claim, row["source_claim"])

    def test_no_verified_entities_means_no_matches(self):
        self.assertEqual(rc.build_eligible_matches([]), [])


class DeduplicationAndAggregationTests(unittest.TestCase):
    def test_same_family_rules_grouped_together(self):
        entities = [_entity("e1", "eye", aliases_en=("eye", "eyes"))]
        matches = rc.build_eligible_matches(entities)
        families = rc.deduplicate_by_evidence_family(matches)
        self.assertIn("facial_feature_style", families)
        self.assertGreaterEqual(len(families["facial_feature_style"]), 3)  # wide/stern/closed all match "eye"

    def test_domain_aggregation_groups_by_concern_domain(self):
        entities = [_entity("e1", "lion")]
        matches = rc.build_eligible_matches(entities)
        domains = rc.aggregate_by_concern_domain(matches)
        self.assertTrue(all(m.concern_domain == d for d, ms in domains.items() for m in ms))


class CandidateHypothesisSafetyTests(unittest.TestCase):
    """The core 'VISUALLY VERIFIED != PSYCHOLOGICALLY VALIDATED' proof."""

    def test_single_rule_match_is_insufficient_never_a_hypothesis(self):
        # Only PSY_AR_EYES_STERN_002 (aggression_or_threat_related) fires alone here.
        entities = [_entity("e1", "stern eyes", aliases_en=("eye",))]
        matches = [m for m in rc.build_eligible_matches(entities) if m.rule_id == "PSY_AR_EYES_STERN_002"]
        hyps = rc.build_candidate_hypotheses(matches)
        self.assertEqual(hyps, [])

    def test_two_verified_rules_same_domain_cap_at_weak_hypothesis(self):
        # face_expression (facial_feature_style) + missing_mouth (missing_body_part) both
        # map to depressive_or_low_mood_related -- two DIFFERENT evidence families, two
        # DIFFERENT rules, both visually verified -- but still ONE source type
        # (clinician_symbolic). Must stay WEAK_HYPOTHESIS, never higher.
        entities = [_entity("e1", "sad face", aliases_en=("face",)),
                    _entity("e2", "girl", aliases_en=("person",))]
        matches = rc.build_eligible_matches(entities)
        domain_matches = [m for m in matches if m.concern_domain == "depressive_or_low_mood_related"]
        self.assertGreaterEqual(len(domain_matches), 2)
        hyps = rc.build_candidate_hypotheses(domain_matches)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].support_level, "WEAK_HYPOTHESIS")

    def test_independent_model_evidence_can_raise_the_ceiling(self):
        class _FakeEvidence:
            evidence_id = "ev_emotion_test"
            confidence = 0.6

        entities = [_entity("e1", "sad face", aliases_en=("face",)),
                    _entity("e2", "girl", aliases_en=("person",))]
        matches = rc.build_eligible_matches(entities)
        domain_matches = [m for m in matches if m.concern_domain == "depressive_or_low_mood_related"]
        hyps = rc.build_candidate_hypotheses(domain_matches, model_evidence=[_FakeEvidence()])
        self.assertEqual(hyps[0].support_level, "POSSIBLE_FOR_EXPLORATION")

    def test_neutral_and_maltreatment_domains_never_produce_a_hypothesis(self):
        # Many rules land in neutral_descriptive_only_no_construct_proposed --
        # even with strong convergence, this domain must never produce a
        # "hypothesis" (it has no clinical construct at all, per its own definition).
        entities = [_entity("e1", "lion"), _entity("e2", "wide eyes", aliases_en=("eye",))]
        matches = rc.build_eligible_matches(entities)
        neutral_matches = [m for m in matches if m.concern_domain == "neutral_descriptive_only_no_construct_proposed"]
        self.assertGreaterEqual(len(neutral_matches), 2)
        hyps = rc.build_candidate_hypotheses(neutral_matches)
        self.assertEqual(hyps, [])

    def test_disclaimer_present_on_every_hypothesis(self):
        entities = [_entity("e1", "sad face", aliases_en=("face",)),
                    _entity("e2", "girl", aliases_en=("person",))]
        matches = rc.build_eligible_matches(entities)
        hyps = rc.build_candidate_hypotheses(matches)
        for h in hyps:
            self.assertIn("not a final diagnosis", h.disclaimer)


class PackageBuilderTests(unittest.TestCase):
    def _one_hypothesis(self):
        entities = [_entity("e1", "sad face", aliases_en=("face",)),
                    _entity("e2", "girl", aliases_en=("person",))]
        matches = rc.build_eligible_matches(entities)
        return rc.build_candidate_hypotheses(matches)[0]

    def test_clinician_package_has_traceable_evidence(self):
        cp = rc.build_clinician_package(self._one_hypothesis())
        self.assertTrue(cp["why"]["triggered_rule_ids"])
        matrix = rc.load_rule_matrix()
        for rule_id in cp["why"]["triggered_rule_ids"]:
            self.assertIn(rule_id, matrix)  # every cited rule_id is a real, traceable rule
        self.assertIsNone(cp["hypothesis_worth_considering"])
        self.assertIsNone(cp["final_clinical_assessment"])

    def test_parent_package_never_leaks_technical_fields(self):
        hyp = self._one_hypothesis()
        pp = rc.build_parent_package(hyp)
        blob = str(pp).lower()
        for leaked in hyp.supporting_rule_ids:
            self.assertNotIn(leaked.lower(), blob)
        for leaked_field in ("bbox", "evidence_family", "confidence_ceiling", "rule_id"):
            self.assertNotIn(leaked_field, blob)

    def test_llm_writer_payload_matches_policy_schema_fields(self):
        payload = rc.build_llm_writer_payload(self._one_hypothesis(), audience="parent", case_id="case_x")
        for field_name in ("case_id", "verified_observations", "measurements", "eligible_rules",
                           "evidence_strength", "candidate_hypotheses", "contradictions",
                           "alternative_explanations", "missing_information", "references",
                           "allowed_claim_level", "audience"):
            self.assertIn(field_name, payload)
        self.assertEqual(payload["allowed_claim_level"], "LEVEL_2")
        self.assertNotEqual(payload["allowed_claim_level"], "LEVEL_3")

    def test_llm_writer_payload_rejects_invalid_audience(self):
        with self.assertRaises(ValueError):
            rc.build_llm_writer_payload(self._one_hypothesis(), audience="teacher")


class NeverTouchesRealRuleEngineTests(unittest.TestCase):
    """Source-level proof this module cannot reach the psychological rule
    engine or its production concern-activation switch."""

    def test_never_calls_evaluate_rules(self):
        # The module docstring DISCUSSES rules.py::evaluate_rules by name
        # (to document what this module deliberately does not do) -- the
        # real guarantee is that it is never actually CALLED (no "(" call
        # syntax immediately follows the name anywhere in the source).
        source = (ROOT / "src" / "doar" / "reasoning_chain.py").read_text(encoding="utf-8")
        self.assertNotIn("evaluate_rules(", source)

    def test_never_calls_derive_concerns(self):
        source = (ROOT / "src" / "doar" / "reasoning_chain.py").read_text(encoding="utf-8")
        self.assertNotIn("derive_concerns(", source)

    def test_only_reuses_the_pure_aggregation_strength_helper(self):
        source = (ROOT / "src" / "doar" / "reasoning_chain.py").read_text(encoding="utf-8")
        self.assertIn("from .concerns import _aggregation_strength", source)


if __name__ == "__main__":
    unittest.main()
