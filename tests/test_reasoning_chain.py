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

    def test_absence_pattern_abstains_when_part_merely_not_detected(self):
        # Bug 2 fix: a person-like entity being verified while the part was
        # merely never independently verified must NOT satisfy the missing-
        # part rule -- "not detected" is not "confirmed absent". This module
        # has no explicit omission detector, so it abstains (not_satisfied).
        entities = [_entity("e1", "girl", aliases_en=("person",))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_HANDS_035")
        self.assertEqual(check.status, "not_satisfied")
        self.assertIn("not proof of visual absence", check.reason)

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


class Bug1EyeStyleRegressionTests(unittest.TestCase):
    """Bug 1: generic "eye"/"eyes" presence must not satisfy mutually
    incompatible eye-style rules (wide/stern/closed) all at once."""

    def _check(self, rule_id, entities):
        checks = rc.check_visual_preconditions(entities)
        return next(c for c in checks if c.rule_id == rule_id)

    def test_generic_eye_does_not_satisfy_any_style_rule(self):
        for label in ("eye", "eyes", "left eye", "right eye"):
            entities = [_entity("e1", label)]
            for rule_id in ("PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003"):
                check = self._check(rule_id, entities)
                self.assertEqual(check.status, "not_satisfied", f"{label!r} must not satisfy {rule_id}")

    def test_generic_eye_never_satisfies_all_three_styles_at_once(self):
        # The exact reported failure: one generic eye entity must not mean
        # wide AND stern AND closed simultaneously.
        entities = [_entity("e1", "left eye", aliases_en=("eye",)),
                    _entity("e2", "right eye", aliases_en=("eye",))]
        matches = rc.build_eligible_matches(entities)
        style_rule_ids = {m.rule_id for m in matches} & {
            "PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003"}
        self.assertEqual(style_rule_ids, set())

    def test_explicit_closed_eyes_satisfies_closed_predicate(self):
        entities = [_entity("e1", "closed eyes", aliases_en=("eye",))]
        check = self._check("PSY_AR_EYES_CLOSED_003", entities)
        self.assertEqual(check.status, "satisfied")
        self.assertEqual(check.matched_entity_ids, ("e1",))
        # And must NOT also satisfy the other two incompatible styles.
        self.assertEqual(self._check("PSY_AR_EYES_WIDE_001", entities).status, "not_satisfied")
        self.assertEqual(self._check("PSY_AR_EYES_STERN_002", entities).status, "not_satisfied")

    def test_explicit_wide_eyes_satisfies_wide_predicate(self):
        entities = [_entity("e1", "wide eyes", aliases_en=("eye",))]
        check = self._check("PSY_AR_EYES_WIDE_001", entities)
        self.assertEqual(check.status, "satisfied")
        self.assertEqual(check.matched_entity_ids, ("e1",))
        self.assertEqual(self._check("PSY_AR_EYES_STERN_002", entities).status, "not_satisfied")
        self.assertEqual(self._check("PSY_AR_EYES_CLOSED_003", entities).status, "not_satisfied")

    def test_shut_synonym_also_satisfies_closed_predicate(self):
        entities = [_entity("e1", "eyes shut tight", aliases_en=("eye",))]
        self.assertEqual(self._check("PSY_AR_EYES_CLOSED_003", entities).status, "satisfied")

    def test_style_qualifier_on_a_different_entity_does_not_count(self):
        # "wide" appearing on an unrelated entity must not lend its
        # qualifier to a separate, generic eye entity.
        entities = [_entity("e1", "eye"), _entity("e2", "wide smile")]
        self.assertEqual(self._check("PSY_AR_EYES_WIDE_001", entities).status, "not_satisfied")


class Bug2AbsenceInferenceRegressionTests(unittest.TestCase):
    """Bug 2: non-detection/non-verification of a part must never be
    treated as visually confirmed absence."""

    def test_lack_of_mouth_detection_does_not_satisfy_missing_mouth(self):
        entities = [_entity("e1", "girl", aliases_en=("person",))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_MOUTH_036")
        self.assertEqual(check.status, "not_satisfied")

    def test_lack_of_hand_detection_does_not_satisfy_missing_hands(self):
        entities = [_entity("e1", "girl", aliases_en=("person",))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_HANDS_035")
        self.assertEqual(check.status, "not_satisfied")

    def test_an_uncertain_or_rejected_part_candidate_still_does_not_prove_absence(self):
        # A mouth candidate that exists but was left uncertain/rejected by
        # the verifier (a labelling disagreement, not evidence of absence)
        # must not flip the missing-mouth predicate to satisfied.
        for status in ("uncertain", "rejected"):
            entities = [_entity("e1", "girl", aliases_en=("person",)),
                        _entity("e2", "frowning mouth", status=status)]
            checks = rc.check_visual_preconditions(entities)
            check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_MOUTH_036")
            self.assertEqual(check.status, "not_satisfied", f"status={status} must still abstain")

    def test_omission_evidence_extension_point_works_when_evidence_type_exists(self):
        # Proves the extension point is real: IF a rule's own
        # omission_evidence_terms were ever populated with a genuine
        # omission-detector vocabulary, the SAME branch would correctly
        # satisfy the predicate from that explicit evidence -- without
        # needing to touch check_visual_preconditions' control flow at
        # all. Monkeypatches the (currently-empty) vocabulary for one
        # rule only, restores it immediately after.
        original = rc.ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE["EN_COMPILED_MISSING_HANDS_035"]
        person_terms, part_terms, _ = original
        rc.ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE["EN_COMPILED_MISSING_HANDS_035"] = (
            person_terms, part_terms, ("hands not visible", "arms end without hands"))
        try:
            entities = [_entity("e1", "girl", aliases_en=("person",)),
                        _entity("e2", "arms end without hands")]
            checks = rc.check_visual_preconditions(entities)
            check = next(c for c in checks if c.rule_id == "EN_COMPILED_MISSING_HANDS_035")
            self.assertEqual(check.status, "satisfied")
            self.assertEqual(check.matched_entity_ids, ("e2",))
        finally:
            rc.ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE["EN_COMPILED_MISSING_HANDS_035"] = original

    def test_no_rule_has_real_omission_vocabulary_today(self):
        # The extension point exists but is INERT by default -- neither
        # rule can be satisfied by the current pipeline, matching the
        # task's explicit "prefer conservative abstention" requirement.
        for rule_id, (_, _, omission_terms) in rc.ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE.items():
            self.assertEqual(omission_terms, (), f"{rule_id} must have no live omission vocabulary yet")


class Bug3ContextSensitiveShapeRegressionTests(unittest.TestCase):
    """Bug 3: an object's own geometric descriptor (e.g. a circular face)
    must not satisfy a standalone shape-symbolism predicate."""

    def test_circular_face_does_not_satisfy_circle_symbolism(self):
        entities = [_entity("e1", "green face circle", aliases_en=("head", "face"))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "PSY_AR_CIRCLES_011")
        self.assertEqual(check.status, "not_satisfied")

    def test_round_head_does_not_satisfy_circle_symbolism(self):
        entities = [_entity("e1", "round head", aliases_en=("head", "circle"))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "PSY_AR_CIRCLES_011")
        self.assertEqual(check.status, "not_satisfied")

    def test_standalone_circle_still_satisfies_circle_symbolism(self):
        entities = [_entity("e1", "circle", aliases_en=("shape",))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "PSY_AR_CIRCLES_011")
        self.assertEqual(check.status, "satisfied")
        self.assertEqual(check.matched_entity_ids, ("e1",))


class Bug4FaceExpressionRegressionTests(unittest.TestCase):
    """Bug 4: generic face/eyes/hair presence must not automatically
    satisfy the face-expression rule -- explicit expression evidence
    (per the rule's own source_claim vocabulary) is required."""

    def test_generic_face_alone_does_not_satisfy_face_expression(self):
        entities = [_entity("e1", "green face circle", aliases_en=("head", "face"))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_FACE_EXPRESSION_021")
        self.assertEqual(check.status, "not_satisfied")

    def test_generic_eyes_alone_does_not_satisfy_face_expression(self):
        entities = [_entity("e1", "eyes")]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_FACE_EXPRESSION_021")
        self.assertEqual(check.status, "not_satisfied")

    def test_generic_hair_alone_does_not_satisfy_face_expression(self):
        entities = [_entity("e1", "hair", aliases_en=("head hair",))]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "EN_COMPILED_FACE_EXPRESSION_021")
        self.assertEqual(check.status, "not_satisfied")

    def test_explicit_expression_evidence_satisfies_face_expression(self):
        for label in ("sad face", "smiling face", "frowning mouth", "crying face"):
            entities = [_entity("e1", label)]
            checks = rc.check_visual_preconditions(entities)
            check = next(c for c in checks if c.rule_id == "EN_COMPILED_FACE_EXPRESSION_021")
            self.assertEqual(check.status, "satisfied", f"{label!r} should satisfy face_expression")


class RealCachedEvidenceRegressionTests(unittest.TestCase):
    """Uses the REAL cached p2b_0003 Observer+Verifier data (the exact
    fixture the bug report was found against) to prove the previous
    contradictory activation -- one generic eye entity satisfying wide
    AND stern AND closed simultaneously -- cannot recur."""

    def _load_real_p2b_0003_entities(self):
        verification_path = (
            ROOT / "outputs" / "prototype_cases" / "gemini_verifier_dev_check_1786580377"
            / "p2b_0003_verification.json")
        if not verification_path.exists():
            self.skipTest(f"real cached fixture not present in this checkout: {verification_path}")
        import json
        rows = json.loads(verification_path.read_text(encoding="utf-8"))
        entities = []
        for row in rows:
            oc = row["observer_candidate"]
            aliases = tuple(oc.get("alternative_labels") or ()) + (row.get("verifier_independent_label") or "",) + \
                tuple(row.get("verifier_independent_alternative_labels") or ())
            aliases = tuple(a for a in aliases if a)
            entities.append(VisualEntity(
                entity_id=row["observation_id"], entity_type=oc.get("entity_type", "unknown"),
                canonical_label=oc["label"], candidate_labels=((oc["label"], oc.get("confidence") or 0.0),),
                aliases_en=aliases, aliases_ar=(),
                broader_categories=(), possible_subtypes=(), visual_similarities=(),
                bbox=tuple(oc["bbox"]) if oc.get("bbox") else None, crop_ref=None, dominant_colors=None,
                relative_size=None, page_position=None, shape_features=None, line_features=None,
                detector=f"visual_observer:{row.get('observer_model', '')}", checkpoint="", prompt="",
                confidence=oc.get("confidence") or 0.0, model_validation_status="UNKNOWN",
                case_verification_status=row["verification_status"],
                evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
                related_rule_ids=(), source="visual_observer", query=None,
                timestamp="2026-08-13T00:00:00+00:00",
            ))
        return entities

    def test_real_p2b_0003_eye_entities_no_longer_trigger_all_three_styles(self):
        entities = self._load_real_p2b_0003_entities()
        matches = rc.build_eligible_matches(entities)
        rule_ids = {m.rule_id for m in matches}
        style_rule_ids = rule_ids & {"PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003"}
        self.assertEqual(style_rule_ids, set(),
                          "real p2b_0003 'left eye'/'right eye' entities must not satisfy any eye-style rule")

    def test_real_p2b_0003_missing_mouth_no_longer_activates(self):
        # p2b_0003's own observer candidate "frowning mouth" exists but was
        # left `uncertain` by the verifier -- the exact scenario this bug
        # report names explicitly.
        entities = self._load_real_p2b_0003_entities()
        matches = rc.build_eligible_matches(entities)
        rule_ids = {m.rule_id for m in matches}
        self.assertNotIn("EN_COMPILED_MISSING_MOUTH_036", rule_ids)

    def test_real_p2b_0003_face_circle_no_longer_satisfies_standalone_circle_symbolism(self):
        entities = self._load_real_p2b_0003_entities()
        matches = rc.build_eligible_matches(entities)
        rule_ids = {m.rule_id for m in matches}
        self.assertNotIn("PSY_AR_CIRCLES_011", rule_ids)


class EligibilityGateRegressionTests(unittest.TestCase):
    """E6-EXPANDED registry feasibility audit found that a rule's own
    allowed_output_level=="disabled" did NOT stop it from becoming an
    EligibleAtomicRuleMatch: `build_eligible_matches` promoted every
    `satisfied` precondition check regardless of the matrix row's own
    allowed_output_level. Concrete measured impact: 13 of E6-PILOT's own
    24 matched associations (54%) were `disabled` rules; a fresh check of
    this exact fixture set shows 7/7 semantic-path matches disabled, 0
    enabled (see eligibility_fix/BEFORE_STATE_defect_demonstration.txt).
    These tests pin the CORRECT behavior -- they fail against the
    unfixed code and must pass once the eligibility gate is added."""

    def test_a_disabled_rule_whose_precondition_is_satisfied_never_becomes_an_eligible_match(self):
        matrix = rc.load_rule_matrix()
        self.assertEqual(matrix["PSY_AR_ANIMAL_LION_007"]["allowed_output_level"], "disabled")
        entities = [_entity("e1", "lion")]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "PSY_AR_ANIMAL_LION_007")
        self.assertEqual(check.status, "satisfied", "fixture must actually satisfy the precondition")
        matches = rc.build_eligible_matches(entities)
        self.assertNotIn("PSY_AR_ANIMAL_LION_007", {m.rule_id for m in matches},
                          "a disabled rule must never be promoted to an EligibleAtomicRuleMatch")

    def test_no_eligible_match_ever_carries_a_disabled_allowed_output_level(self):
        matrix = rc.load_rule_matrix()
        entities = [_entity("e1", "lion"), _entity("e2", "wolf"),
                    _entity("e3", "stern eyes", aliases_en=("eye",)),
                    _entity("e4", "wide eyes", aliases_en=("eye",)),
                    _entity("e5", "sad face"), _entity("e6", "house"), _entity("e7", "tree")]
        matches = rc.build_eligible_matches(entities)
        for m in matches:
            self.assertEqual(matrix[m.rule_id]["allowed_output_level"], "individual_heuristic_only",
                              f"{m.rule_id} is disabled and must never appear in build_eligible_matches output")

    def test_disabled_rule_precondition_checks_remain_traceable_for_technical_view(self):
        # Preserve-traceability requirement: check_visual_preconditions
        # (the function clinician_review_app.py's Technical/debug view
        # reads directly via bundle["checks"], independent of
        # build_eligible_matches) must still report "satisfied" for a
        # disabled rule whose precondition genuinely matched -- only
        # PROMOTION into an EligibleAtomicRuleMatch is gated, never the
        # underlying precondition check itself.
        entities = [_entity("e1", "lion")]
        checks = rc.check_visual_preconditions(entities)
        check = next(c for c in checks if c.rule_id == "PSY_AR_ANIMAL_LION_007")
        self.assertEqual(check.status, "satisfied")


class EligibleMatchTests(unittest.TestCase):
    def test_only_satisfied_enabled_checks_become_matches(self):
        # PSY_AR_ANIMAL_LION_007 is allowed_output_level=="disabled" in the
        # frozen registry -- its precondition is satisfied by "lion" but it
        # must not be promoted (EligibilityGateRegressionTests covers this
        # directly); this test now uses hand-constructed matches for the
        # enabled-rule construction-fidelity check below instead.
        entities = [_entity("e1", "lion")]
        matches = rc.build_eligible_matches(entities)
        rule_ids = {m.rule_id for m in matches}
        self.assertNotIn("PSY_AR_ANIMAL_LION_007", rule_ids)

    def test_match_fields_are_copied_verbatim_from_frozen_matrix(self):
        # No semantic (VisualEntity-presence) rule in the current registry
        # is allowed_output_level=="individual_heuristic_only" -- all 10
        # enabled rules are Tier-1 composition/line rules evaluated via
        # drawing_synthesis.check_deterministic_preconditions instead (see
        # reasoning_chain.check_visual_preconditions' own final branch), so
        # this exercises build_eligible_matches' real construction path
        # end-to-end by monkeypatching load_rule_matrix's returned dict to
        # temporarily mark PSY_AR_ANIMAL_LION_007 enabled -- RULE_EVIDENCE_
        # MATRIX.csv on disk is never touched.
        real_matrix = rc.load_rule_matrix()
        patched_matrix = dict(real_matrix)
        patched_matrix["PSY_AR_ANIMAL_LION_007"] = dict(
            real_matrix["PSY_AR_ANIMAL_LION_007"], allowed_output_level="individual_heuristic_only")
        original_loader = rc.load_rule_matrix
        rc.load_rule_matrix = lambda: patched_matrix
        try:
            entities = [_entity("e1", "lion")]
            match = next(m for m in rc.build_eligible_matches(entities) if m.rule_id == "PSY_AR_ANIMAL_LION_007")
        finally:
            rc.load_rule_matrix = original_loader
        row = real_matrix["PSY_AR_ANIMAL_LION_007"]
        self.assertEqual(match.allowed_output_level, "individual_heuristic_only")
        self.assertEqual(match.evidence_direction, row["evidence_direction"])
        self.assertEqual(match.source_claim, row["source_claim"])

    def test_no_verified_entities_means_no_matches(self):
        self.assertEqual(rc.build_eligible_matches([]), [])


def _match(rule_id, concern_domain, evidence_family, *, matched_entity_ids=("e1",)):
    """Hand-constructs an EligibleAtomicRuleMatch directly, bypassing
    build_eligible_matches' own precondition-satisfaction + (now gated)
    eligibility machinery -- used by tests below that exercise DOWNSTREAM
    dedup/aggregation/hypothesis/package logic, which operates on any list
    of EligibleAtomicRuleMatch objects regardless of how they were built.
    No semantic (VisualEntity-presence) rule in the current registry is
    allowed_output_level=="individual_heuristic_only" (the 10 enabled
    rules are all Tier-1 composition/line rules evaluated via a different
    module -- see reasoning_chain.check_visual_preconditions' own final
    branch), so building real matches for THESE downstream tests via
    build_eligible_matches is no longer possible post-fix; EligibilityGate
    RegressionTests above covers the gate itself directly."""
    return rc.EligibleAtomicRuleMatch(
        rule_id=rule_id, evidence_family=evidence_family, concern_domain=concern_domain,
        allowed_output_level="individual_heuristic_only", source_claim="test source claim",
        possible_interpretation="test interpretation", alternative_explanations=(),
        matched_entity_ids=matched_entity_ids, evidence_direction="concern",
        context_transfer_justification="test")


class DeduplicationAndAggregationTests(unittest.TestCase):
    def test_same_family_rules_grouped_together(self):
        matches = [_match("PSY_AR_EYES_WIDE_001", "neutral_descriptive_only_no_construct_proposed", "facial_feature_style"),
                   _match("PSY_AR_EYES_STERN_002", "aggression_or_threat_related", "facial_feature_style")]
        families = rc.deduplicate_by_evidence_family(matches)
        self.assertIn("facial_feature_style", families)
        self.assertGreaterEqual(len(families["facial_feature_style"]), 2)

    def test_domain_aggregation_groups_by_concern_domain(self):
        matches = [_match("PSY_AR_ANIMAL_LION_007", "neutral_descriptive_only_no_construct_proposed", "animal_symbolism")]
        domains = rc.aggregate_by_concern_domain(matches)
        self.assertTrue(all(m.concern_domain == d for d, ms in domains.items() for m in ms))


class CandidateHypothesisSafetyTests(unittest.TestCase):
    """The core 'VISUALLY VERIFIED != PSYCHOLOGICALLY VALIDATED' proof."""

    def test_single_rule_match_is_insufficient_never_a_hypothesis(self):
        matches = [_match("PSY_AR_EYES_STERN_002", "aggression_or_threat_related", "facial_feature_style")]
        hyps = rc.build_candidate_hypotheses(matches)
        self.assertEqual(hyps, [])

    def test_two_verified_rules_same_domain_cap_at_weak_hypothesis(self):
        # stern_eyes (facial_feature_style) + tiger_wolf (animal_symbolism) both
        # map to aggression_or_threat_related -- two DIFFERENT evidence families, two
        # DIFFERENT rules -- but still ONE source type (clinician_symbolic).
        # Must stay WEAK_HYPOTHESIS, never higher.
        matches = [_match("PSY_AR_EYES_STERN_002", "aggression_or_threat_related", "facial_feature_style"),
                   _match("PSY_AR_ANIMAL_TIGER_WOLF_004", "aggression_or_threat_related", "animal_symbolism")]
        hyps = rc.build_candidate_hypotheses(matches)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].support_level, "WEAK_HYPOTHESIS")

    def test_independent_model_evidence_can_raise_the_ceiling(self):
        class _FakeEvidence:
            evidence_id = "ev_emotion_test"
            confidence = 0.6

        matches = [_match("EN_COMPILED_FACE_EXPRESSION_021", "depressive_or_low_mood_related", "facial_feature_style")]
        hyps = rc.build_candidate_hypotheses(matches, model_evidence=[_FakeEvidence()])
        self.assertEqual(hyps[0].support_level, "POSSIBLE_FOR_EXPLORATION")

    def test_neutral_and_maltreatment_domains_never_produce_a_hypothesis(self):
        # Many rules land in neutral_descriptive_only_no_construct_proposed --
        # even with strong convergence, this domain must never produce a
        # "hypothesis" (it has no clinical construct at all, per its own definition).
        matches = [_match("PSY_AR_ANIMAL_LION_007", "neutral_descriptive_only_no_construct_proposed", "animal_symbolism"),
                   _match("PSY_AR_EYES_WIDE_001", "neutral_descriptive_only_no_construct_proposed", "facial_feature_style")]
        hyps = rc.build_candidate_hypotheses(matches)
        self.assertEqual(hyps, [])

    def test_disclaimer_present_on_every_hypothesis(self):
        matches = [_match("PSY_AR_EYES_STERN_002", "aggression_or_threat_related", "facial_feature_style"),
                   _match("PSY_AR_ANIMAL_TIGER_WOLF_004", "aggression_or_threat_related", "animal_symbolism")]
        hyps = rc.build_candidate_hypotheses(matches)
        self.assertTrue(hyps, "fixture must actually produce a hypothesis for this test to check anything")
        for h in hyps:
            self.assertIn("not a final diagnosis", h.disclaimer)


class PackageBuilderTests(unittest.TestCase):
    def _one_hypothesis(self):
        matches = [_match("PSY_AR_EYES_STERN_002", "aggression_or_threat_related", "facial_feature_style"),
                   _match("PSY_AR_ANIMAL_TIGER_WOLF_004", "aggression_or_threat_related", "animal_symbolism")]
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
