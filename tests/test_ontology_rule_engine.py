"""Tests for the declarative Rule schema and the three-valued rule
engine.

Covers, per the task's required test list: TRUE/FALSE/UNKNOWN premise
composition, prerequisite handling (a rule requiring age with age=None
must not evaluate normally), omission safety (a missing detector must
never silently become a confirmed absence), and trace generation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.ontology.context import CaseContext, EMPTY_CONTEXT
from doar.ontology.engine import RuleOutcome, evaluate_premise, evaluate_rule
from doar.ontology.feature_schema import FeatureDefinition, FeatureOntology, FeatureType
from doar.ontology.observation import FeatureObservation, not_assessable
from doar.ontology.rule import (
    AllOf,
    AnyOf,
    Citation,
    ContextRequirements,
    EvidenceGrade,
    FeatureCondition,
    NotOf,
    Rule,
    RuleFramework,
)
from doar.ontology.store import FeatureStore
from doar.ontology.tristate import TriState


def _rule(**overrides) -> Rule:
    defaults = dict(
        rule_id="R1", name="test rule", framework=RuleFramework.CLASSICAL_PROJECTIVE,
        premise=FeatureCondition("f", "equals", True), domain="test_domain",
        citation=Citation(document="TODO", status="TODO"),
        evidence_grade=EvidenceGrade.TODO, confidence_ceiling=0.2,
        professional_reasoning="reasoning", parent_safe_wording="wording",
    )
    defaults.update(overrides)
    return Rule(**defaults)


def _store_with(feature_id="f", value=True, source="existing_cv") -> FeatureStore:
    store = FeatureStore()
    store.add_observation(FeatureObservation(
        feature_id=feature_id, value=value, value_type=FeatureType.BOOLEAN, assessable=True, source=source,
    ))
    return store


class RuleSchemaValidationTests(unittest.TestCase):
    def test_empirical_expressive_without_real_citation_raises(self):
        with self.assertRaises(ValueError):
            _rule(framework=RuleFramework.EMPIRICAL_EXPRESSIVE, citation=Citation(document="TODO", status="TODO"))

    def test_empirical_expressive_with_real_citation_is_allowed(self):
        rule = _rule(
            framework=RuleFramework.EMPIRICAL_EXPRESSIVE,
            citation=Citation(document="Some Journal Article", status="documented", doi="10.1/x"),
        )
        self.assertEqual(rule.framework, RuleFramework.EMPIRICAL_EXPRESSIVE)

    def test_invalid_framework_raises(self):
        with self.assertRaises(ValueError):
            _rule(framework="NOT_A_REAL_FRAMEWORK")

    def test_invalid_evidence_grade_raises(self):
        with self.assertRaises(ValueError):
            _rule(evidence_grade="NOT_A_REAL_GRADE")

    def test_confidence_ceiling_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _rule(confidence_ceiling=1.5)

    def test_required_feature_ids_deduplicates_across_premise_tree(self):
        rule = _rule(premise=AllOf((FeatureCondition("f"), AnyOf((FeatureCondition("f"), FeatureCondition("g"))))))
        self.assertEqual(rule.required_feature_ids(), ("f", "g"))


class ThreeValuedPremiseCompositionTests(unittest.TestCase):
    def test_all_of_true_true_matches(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="a", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="x"))
        store.add_observation(FeatureObservation(feature_id="b", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="x"))
        premise = AllOf((FeatureCondition("a"), FeatureCondition("b")))
        result, _trace = evaluate_premise(premise, store, None)
        self.assertIs(result, TriState.TRUE)

    def test_all_of_true_and_unassessable_is_unknown_not_false(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="a", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="x"))
        store.add_observation(not_assessable("b", value_type=FeatureType.BOOLEAN, source="x", reason="no detector"))
        premise = AllOf((FeatureCondition("a"), FeatureCondition("b")))
        result, _trace = evaluate_premise(premise, store, None)
        self.assertIs(result, TriState.UNKNOWN)

    def test_all_of_false_and_unassessable_is_false(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="a", value=False, value_type=FeatureType.BOOLEAN, assessable=True, source="x"))
        store.add_observation(not_assessable("b", value_type=FeatureType.BOOLEAN, source="x", reason="no detector"))
        premise = AllOf((FeatureCondition("a"), FeatureCondition("b")))
        result, _trace = evaluate_premise(premise, store, None)
        self.assertIs(result, TriState.FALSE)

    def test_any_of_true_and_unassessable_is_true(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="a", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="x"))
        store.add_observation(not_assessable("b", value_type=FeatureType.BOOLEAN, source="x", reason="no detector"))
        premise = AnyOf((FeatureCondition("a"), FeatureCondition("b")))
        result, _trace = evaluate_premise(premise, store, None)
        self.assertIs(result, TriState.TRUE)

    def test_any_of_false_and_unassessable_is_unknown(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="a", value=False, value_type=FeatureType.BOOLEAN, assessable=True, source="x"))
        store.add_observation(not_assessable("b", value_type=FeatureType.BOOLEAN, source="x", reason="no detector"))
        premise = AnyOf((FeatureCondition("a"), FeatureCondition("b")))
        result, _trace = evaluate_premise(premise, store, None)
        self.assertIs(result, TriState.UNKNOWN)

    def test_not_of_unassessable_is_unknown(self):
        store = FeatureStore()
        store.add_observation(not_assessable("a", value_type=FeatureType.BOOLEAN, source="x", reason="no detector"))
        result, _trace = evaluate_premise(NotOf(FeatureCondition("a")), store, None)
        self.assertIs(result, TriState.UNKNOWN)


class OmissionSafetyTests(unittest.TestCase):
    """The task's worked example: human_figure_present=TRUE,
    hands_visible_region=UNKNOWN, hands_present reported FALSE by a
    detector => hands_missing must NOT automatically become TRUE."""

    def _ontology(self) -> FeatureOntology:
        return FeatureOntology([
            FeatureDefinition(feature_id="human_figure_present", name="x", category="c", value_type=FeatureType.BOOLEAN, definition="d"),
            FeatureDefinition(feature_id="hands_region_visible", name="x", category="c", value_type=FeatureType.BOOLEAN, definition="d", requirements=("human_figure_present",)),
            FeatureDefinition(feature_id="hands_present", name="x", category="c", value_type=FeatureType.BOOLEAN, definition="d", requirements=("human_figure_present", "hands_region_visible")),
        ])

    def test_hands_present_false_without_visibility_prerequisite_is_unknown(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="human_figure_present", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="detector"))
        # hands_region_visible was never established (no observation at all
        # -- store.effective() returns NOT_ASSESSABLE for it).
        store.add_observation(FeatureObservation(feature_id="hands_present", value=False, value_type=FeatureType.BOOLEAN, assessable=True, source="detector"))

        rule = _rule(premise=FeatureCondition("hands_present", "equals", False))
        result = evaluate_rule(rule, store, EMPTY_CONTEXT, ontology=self._ontology())

        # The naive reading of the raw detector output ("not detected") is
        # FALSE for hands_present, which superficially looks like it should
        # trigger a "hands_missing" style rule. Because hands_region_visible
        # was never confirmed, the engine must refuse to use it.
        self.assertEqual(result.outcome, RuleOutcome.MISSING_PREREQUISITE)
        self.assertIn("hands_region_visible", result.missing_prerequisites)

    def test_hands_present_false_with_all_prerequisites_satisfied_is_not_matched_when_premise_is_true_present(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="human_figure_present", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="detector"))
        store.add_observation(FeatureObservation(feature_id="hands_region_visible", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="detector"))
        store.add_observation(FeatureObservation(feature_id="hands_present", value=False, value_type=FeatureType.BOOLEAN, assessable=True, source="detector"))

        rule = _rule(premise=FeatureCondition("hands_present", "equals", False))
        result = evaluate_rule(rule, store, EMPTY_CONTEXT, ontology=self._ontology())

        # Now every prerequisite genuinely holds, so a real omission claim
        # is permitted to resolve to a real outcome.
        self.assertIn(result.outcome, (RuleOutcome.MATCHED, RuleOutcome.MATCHED_PROVISIONAL))


class ContextPrerequisiteTests(unittest.TestCase):
    def test_age_required_with_age_none_is_missing_prerequisite(self):
        rule = _rule(requirements=ContextRequirements(age=True))
        store = _store_with()
        result = evaluate_rule(rule, store, CaseContext(age=None))
        self.assertEqual(result.outcome, RuleOutcome.MISSING_PREREQUISITE)

    def test_age_required_out_of_range_is_not_applicable_to_age(self):
        rule = _rule(requirements=ContextRequirements(age=True, age_range=(9.0, 12.0)))
        store = _store_with()
        result = evaluate_rule(rule, store, CaseContext(age=5))
        self.assertEqual(result.outcome, RuleOutcome.NOT_APPLICABLE_TO_AGE)

    def test_age_in_range_proceeds_to_premise_evaluation(self):
        rule = _rule(requirements=ContextRequirements(age=True, age_range=(9.0, 12.0)))
        store = _store_with(value=True)
        result = evaluate_rule(rule, store, CaseContext(age=10))
        self.assertIn(result.outcome, (RuleOutcome.MATCHED, RuleOutcome.MATCHED_PROVISIONAL))

    def test_task_required_with_task_none_is_missing_prerequisite(self):
        rule = _rule(requirements=ContextRequirements(task=True))
        store = _store_with()
        result = evaluate_rule(rule, store, CaseContext(drawing_task=None))
        self.assertEqual(result.outcome, RuleOutcome.MISSING_PREREQUISITE)

    def test_task_required_wrong_task_is_not_applicable_to_task(self):
        rule = _rule(requirements=ContextRequirements(task=True, allowed_tasks=("draw_a_person",)))
        store = _store_with()
        result = evaluate_rule(rule, store, CaseContext(drawing_task="free_drawing"))
        self.assertEqual(result.outcome, RuleOutcome.NOT_APPLICABLE_TO_TASK)

    def test_narrative_required_with_narrative_none_is_missing_prerequisite(self):
        rule = _rule(requirements=ContextRequirements(narrative=True))
        store = _store_with()
        result = evaluate_rule(rule, store, CaseContext(narrative=None))
        self.assertEqual(result.outcome, RuleOutcome.MISSING_PREREQUISITE)

    def test_re_running_with_context_later_supplied_changes_outcome_no_code_change(self):
        rule = _rule(requirements=ContextRequirements(age=True))
        store = _store_with(value=True)
        first = evaluate_rule(rule, store, CaseContext(age=None))
        self.assertEqual(first.outcome, RuleOutcome.MISSING_PREREQUISITE)
        second = evaluate_rule(rule, store, CaseContext(age=9))
        self.assertIn(second.outcome, (RuleOutcome.MATCHED, RuleOutcome.MATCHED_PROVISIONAL))


class OutcomeVocabularyTests(unittest.TestCase):
    def test_matched_provisional_for_single_uncorroborated_source(self):
        store = _store_with(value=True, source="existing_cv")
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertEqual(result.outcome, RuleOutcome.MATCHED_PROVISIONAL)
        self.assertGreater(result.rule_confidence, 0.0)

    def test_matched_for_corroborated_source(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="f", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="vlm_a"))
        store.add_observation(FeatureObservation(feature_id="f", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="vlm_b"))
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertEqual(result.outcome, RuleOutcome.MATCHED)

    def test_not_matched_when_premise_false(self):
        store = _store_with(value=False)
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertEqual(result.outcome, RuleOutcome.NOT_MATCHED)
        self.assertEqual(result.rule_confidence, 0.0)

    def test_not_assessable_when_feature_never_observed(self):
        store = FeatureStore()
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertEqual(result.outcome, RuleOutcome.NOT_ASSESSABLE)

    def test_conflicting_evidence_outcome(self):
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="f", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="vlm_a"))
        store.add_observation(FeatureObservation(feature_id="f", value=False, value_type=FeatureType.BOOLEAN, assessable=True, source="vlm_b"))
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertEqual(result.outcome, RuleOutcome.CONFLICTING_EVIDENCE)

    def test_professional_reasoning_only_populated_on_match(self):
        store = _store_with(value=False)
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertIsNone(result.professional_reasoning)
        self.assertIsNone(result.parent_safe_wording)


class TraceGenerationTests(unittest.TestCase):
    def test_trace_reflects_actual_stored_values(self):
        store = _store_with(value=True, source="existing_cv")
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        self.assertEqual(result.requested_feature_ids, ("f",))
        self.assertEqual(result.feature_snapshots["f"]["value"], True)
        self.assertEqual(result.feature_snapshots["f"]["status"], "AI_ONLY")
        self.assertEqual(result.premise_trace.result, "true")
        leaf = result.premise_trace.leaf_conditions()[0]
        self.assertEqual(leaf.feature_id, "f")
        self.assertEqual(leaf.effective_value, True)

    def test_trace_records_missing_prerequisites(self):
        ontology = FeatureOntology([
            FeatureDefinition(feature_id="parent", name="x", category="c", value_type=FeatureType.BOOLEAN, definition="d"),
            FeatureDefinition(feature_id="child", name="x", category="c", value_type=FeatureType.BOOLEAN, definition="d", requirements=("parent",)),
        ])
        store = FeatureStore()
        store.add_observation(FeatureObservation(feature_id="child", value=True, value_type=FeatureType.BOOLEAN, assessable=True, source="detector"))
        rule = _rule(premise=FeatureCondition("child"))
        result = evaluate_rule(rule, store, EMPTY_CONTEXT, ontology=ontology)
        self.assertEqual(result.outcome, RuleOutcome.MISSING_PREREQUISITE)
        self.assertEqual(result.missing_prerequisites, ("parent",))

    def test_trace_is_json_serializable(self):
        import json
        store = _store_with(value=True)
        rule = _rule()
        result = evaluate_rule(rule, store, EMPTY_CONTEXT)
        json.dumps(result.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
