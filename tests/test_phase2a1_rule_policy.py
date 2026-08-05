"""Tests for DOAR-TRACE Phase 2A.1, Section 8: the rule-policy record
(page_reference_requirement, feature_version, known_robustness_limitations,
expert_review_status) every registry-v2 rule now carries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.registry_v2_build import build_registry_v2
from doar.rule_engine_v2 import ALL_PAGE_GATED_RULE_IDS, V2_RULE_IDS


class RulePolicyFieldsTests(unittest.TestCase):
    def setUp(self):
        self.rules_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}

    def test_page_gated_rules_require_a_page_reference(self):
        for rule_id in ALL_PAGE_GATED_RULE_IDS:
            self.assertEqual(self.rules_by_id[rule_id]["page_reference_requirement"], "required", rule_id)

    def test_stroke_proxy_rules_do_not_require_a_page_reference(self):
        stroke_rule_ids = V2_RULE_IDS - ALL_PAGE_GATED_RULE_IDS
        self.assertEqual(stroke_rule_ids, {
            "EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031",
            "EN_COMPILED_LINE_SHAKY_BROKEN_032",
        })
        for rule_id in stroke_rule_ids:
            self.assertEqual(self.rules_by_id[rule_id]["page_reference_requirement"], "not_required", rule_id)

    def test_non_page_relative_disabled_rules_do_not_require_a_page_reference(self):
        rule = self.rules_by_id["EN_COMPILED_EXCESSIVE_DETAIL_040"]
        self.assertEqual(rule["page_reference_requirement"], "not_required")

    def test_every_executable_rule_has_expert_review_status_pending(self):
        executable_ids = {rid for rid, r in self.rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"}
        self.assertEqual(len(executable_ids), 10)
        for rule_id in executable_ids:
            self.assertEqual(self.rules_by_id[rule_id]["expert_review_status"], "pending_review", rule_id)

    def test_every_disabled_rule_has_not_applicable_expert_review_status(self):
        for rule_id, rule in self.rules_by_id.items():
            if rule["allowed_output_level"] != "individual_heuristic_only":
                self.assertEqual(rule["expert_review_status"], "not_applicable_rule_disabled", rule_id)

    def test_executable_rules_carry_a_real_feature_version(self):
        for rule_id in V2_RULE_IDS | {"PSY_AR_SIZE_FULL_015", "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_SMALL_016"}:
            self.assertIsNotNone(self.rules_by_id[rule_id]["feature_version"], rule_id)

    def test_disabled_rules_have_no_feature_version(self):
        rule = self.rules_by_id["EN_COMPILED_EXCESSIVE_DETAIL_040"]
        self.assertIsNone(rule["feature_version"])

    def test_stroke_rules_cite_their_real_phase2a_robustness_finding(self):
        for rule_id in ("EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031"):
            limitation = self.rules_by_id[rule_id]["known_robustness_limitations"]
            self.assertIn("1 of 80", limitation)
            self.assertIn("brightness_down", limitation)

    def test_placement_rules_cite_zero_recorded_failures(self):
        for rule_id in ("PSY_AR_PLACE_TOP_017", "EN_COMPILED_PLACEMENT_CENTER_029"):
            limitation = self.rules_by_id[rule_id]["known_robustness_limitations"]
            self.assertIn("0 of 80", limitation)

    def test_disabled_rules_show_not_applicable_robustness_limitations(self):
        rule = self.rules_by_id["EN_COMPILED_EXCESSIVE_DETAIL_040"]
        self.assertEqual(rule["known_robustness_limitations"], "not_applicable_rule_disabled")

    def test_no_rule_promoted_beyond_individual_heuristic_only_by_this_section(self):
        # Section 8's explicit constraint: recording these new fields must
        # never itself change allowed_output_level.
        executable_ids = {rid for rid, r in self.rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"}
        self.assertEqual(executable_ids, {
            "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_FULL_015", "PSY_AR_SIZE_SMALL_016",
            "PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019",
            "EN_COMPILED_PLACEMENT_CENTER_029", "EN_COMPILED_LINE_HEAVY_PRESSURE_030",
            "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "EN_COMPILED_LINE_SHAKY_BROKEN_032",
        })


class NoSoleProxySeriousWarningTests(unittest.TestCase):
    """Section 8's explicit constraint: "Do not promote any new serious
    low-mood/fear/anger warning solely from these proxy rules." Verifies
    the structural precondition that makes this true: no serious
    construct (tension_or_anger_pattern/fear_or_insecurity_pattern/
    low_mood_or_emotional_distress_pattern) has more than one of the 4
    new rules mapped to it, so none of them can ever supply the required
    >=2 independent families on their own."""

    _SERIOUS_CONSTRUCTS = {
        "low_mood_or_emotional_distress_pattern", "fear_or_insecurity_pattern", "tension_or_anger_pattern",
    }

    def test_at_most_one_new_rule_per_serious_construct(self):
        rules_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}
        by_construct: dict[str, list[str]] = {}
        for rule_id in V2_RULE_IDS:
            construct = rules_by_id[rule_id]["target_construct"]
            if construct in self._SERIOUS_CONSTRUCTS:
                by_construct.setdefault(construct, []).append(rule_id)
        for construct, rule_ids in by_construct.items():
            self.assertEqual(len(rule_ids), 1, f"{construct} has {rule_ids} -- could combine among themselves")


if __name__ == "__main__":
    unittest.main(verbosity=2)
