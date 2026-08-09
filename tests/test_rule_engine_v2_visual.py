"""DOAR V1 rule integration: rule_engine_v2.py::evaluate_visual_object_presence_rules
-- the REAL rule engine's new extension for static_detector rules. Proves
both halves of the registry gate independently: (1) a synthetic rule with
the gate OPEN really does fire from validated visual evidence with correct
provenance, and (2) the REAL, unmodified rules_registry_v2.json -- where
every static_detector rule's allowed_output_level is currently 'disabled'
-- correctly abstains today, never fabricating a trigger."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.rule_engine_v2 import evaluate_visual_object_presence_rules  # noqa: E402
from doar.schemas import Evidence  # noqa: E402


def _rule(rule_id, *, observable, allowed_output_level, observability_class="static_detector"):
    return {
        "rule_id": rule_id, "observable": observable, "observability_class": observability_class,
        "allowed_output_level": allowed_output_level, "confidence_ceiling": 0.2,
        "scientific_support": "exploratory", "professional_wording": "professional text",
        "parent_safe_wording": "parent safe text", "possible_interpretation": "interpretation",
        "reference_ids": [], "limitations": ["test limitation"],
    }


def _visual_evidence(label, *, rule_eligible, confidence=0.8, evidence_id=None):
    return Evidence(
        evidence_id=evidence_id or f"ev_visual_vf_{label}_{confidence}", kind="visual_detection",
        value={"label": label, "rule_eligible": rule_eligible}, method="grounding_dino_object_classes",
        confidence=confidence, limitations=[])


class SyntheticEnabledGateTests(unittest.TestCase):
    """Proves the mechanism itself is real and correct -- using a
    controlled, clearly-synthetic registry, NOT a claim about production
    behavior (see RealRegistryTodayTests below for that)."""

    def test_enabled_rule_with_eligible_evidence_fires_weak_support(self):
        rules = {"TEST_HOUSE_RULE": _rule("TEST_HOUSE_RULE", observable="house",
                                           allowed_output_level="individual_heuristic_only")}
        ev = _visual_evidence("house", rule_eligible=True, evidence_id="ev_visual_vf_house_abc")
        result = evaluate_visual_object_presence_rules(rules, [ev])
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["rule_id"], "TEST_HOUSE_RULE")
        self.assertEqual(row["status"], "weak_support")
        self.assertEqual(row["matched_evidence_ids"], ["ev_visual_vf_house_abc"])
        self.assertTrue(row["visual_evidence_sourced"])
        self.assertEqual(row["professional_reasoning"], "professional text")
        self.assertEqual(row["parent_safe_wording"], "parent safe text")

    def test_picks_highest_confidence_match_when_multiple_exist(self):
        rules = {"TEST_HOUSE_RULE": _rule("TEST_HOUSE_RULE", observable="house",
                                           allowed_output_level="individual_heuristic_only")}
        low = _visual_evidence("house", rule_eligible=True, confidence=0.3, evidence_id="ev_low")
        high = _visual_evidence("house", rule_eligible=True, confidence=0.9, evidence_id="ev_high")
        result = evaluate_visual_object_presence_rules(rules, [low, high])
        self.assertEqual(result[0]["matched_evidence_ids"], ["ev_high"])

    def test_non_static_detector_rule_is_never_touched(self):
        rules = {"TEST_STATIC_DIRECT": _rule("TEST_STATIC_DIRECT", observable="house",
                                              allowed_output_level="individual_heuristic_only",
                                              observability_class="static_direct")}
        ev = _visual_evidence("house", rule_eligible=True)
        self.assertEqual(evaluate_visual_object_presence_rules(rules, [ev]), [])

    def test_no_matching_evidence_produces_no_row(self):
        rules = {"TEST_HOUSE_RULE": _rule("TEST_HOUSE_RULE", observable="house",
                                           allowed_output_level="individual_heuristic_only")}
        ev = _visual_evidence("tree", rule_eligible=True)
        self.assertEqual(evaluate_visual_object_presence_rules(rules, [ev]), [])


class RegistryGateRespectedTests(unittest.TestCase):
    def test_disabled_rule_never_fires_even_with_perfect_matching_evidence(self):
        rules = {"TEST_HOUSE_RULE": _rule("TEST_HOUSE_RULE", observable="house",
                                           allowed_output_level="disabled")}
        ev = _visual_evidence("house", rule_eligible=True, confidence=0.99)
        result = evaluate_visual_object_presence_rules(rules, [ev])
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["status"], "missing_detector")
        self.assertNotEqual(row["status"], "weak_support")
        self.assertEqual(row["matched_evidence_ids"], [])
        self.assertIn("rule_disabled_pending_registry_review", row["missing_evidence"][0])
        self.assertTrue(row["visual_evidence_sourced"])

    def test_experimental_evidence_cannot_activate_even_an_enabled_rule(self):
        rules = {"TEST_HEART_RULE": _rule("TEST_HEART_RULE", observable="heart",
                                           allowed_output_level="individual_heuristic_only")}
        ev = _visual_evidence("heart", rule_eligible=False, confidence=0.99)
        # rule_eligible=False (experimental/unknown/disabled) -- structurally
        # invisible to this function, regardless of how the rule's own gate
        # is configured.
        self.assertEqual(evaluate_visual_object_presence_rules(rules, [ev]), [])

    def test_unmapped_or_unknown_evidence_cannot_activate_a_rule(self):
        rules = {"TEST_SUN_RULE": _rule("TEST_SUN_RULE", observable="sun",
                                         allowed_output_level="individual_heuristic_only")}
        ev = _visual_evidence("sun", rule_eligible=False)
        self.assertEqual(evaluate_visual_object_presence_rules(rules, [ev]), [])

    def test_disabled_for_rules_evidence_never_reaches_here(self):
        rules = {"TEST_CIRCLE_RULE": _rule("TEST_CIRCLE_RULE", observable="circle",
                                            allowed_output_level="individual_heuristic_only")}
        ev = _visual_evidence("circle", rule_eligible=False)
        self.assertEqual(evaluate_visual_object_presence_rules(rules, [ev]), [])

    def test_never_produces_not_matched_absence_claim(self):
        # Even when a rule exists and NOTHING was detected for its
        # observable, this function must never claim a scientifically
        # unjustified "not_matched" absence -- it should simply produce no
        # row (nothing evaluable to say), not a fabricated negative.
        rules = {"TEST_HOUSE_RULE": _rule("TEST_HOUSE_RULE", observable="house",
                                           allowed_output_level="individual_heuristic_only")}
        statuses = {row["status"] for row in evaluate_visual_object_presence_rules(rules, [])}
        self.assertNotIn("not_matched", statuses)


class RealRegistryTodayTests(unittest.TestCase):
    """The honest, real, production answer to "can a validated visual
    finding trigger a rule today?" -- uses the REAL, unmodified
    rules_registry_v2.json (no synthetic data), proving Stage 9's outcome
    B (correct abstention) from the actual current registry state, not an
    assumption."""

    def setUp(self):
        self.rules_v2_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}

    def test_every_static_detector_rule_is_currently_disabled(self):
        static_detector_rules = [r for r in self.rules_v2_by_id.values()
                                  if r.get("observability_class") == "static_detector"]
        self.assertGreater(len(static_detector_rules), 0)
        for r in static_detector_rules:
            self.assertNotEqual(r["allowed_output_level"], "individual_heuristic_only",
                                 f"{r['rule_id']} is enabled -- if a registry curator flipped this, "
                                 f"update this test and the DOAR_V1_RULE_INTEGRATION_REPORT.md claim "
                                 f"about outcome B accordingly.")

    def test_validated_house_and_tree_evidence_correctly_abstain_today(self):
        house_ev = _visual_evidence("house", rule_eligible=True, confidence=0.9)
        tree_ev = _visual_evidence("tree", rule_eligible=True, confidence=0.9)
        result = evaluate_visual_object_presence_rules(self.rules_v2_by_id, [house_ev, tree_ev])
        statuses = {row["status"] for row in result}
        self.assertEqual(statuses, {"missing_detector"})
        rule_ids = {row["rule_id"] for row in result}
        self.assertEqual(rule_ids, {"EN_COMPILED_HOUSE_023", "EN_COMPILED_TREE_024"})


if __name__ == "__main__":
    unittest.main()
