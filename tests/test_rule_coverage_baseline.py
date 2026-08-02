"""Baseline-documentation tests for the rule/concern engine's current real
coverage, added after the 2026-08-02 audit (CURRENT_STATE_AUDIT.md Section 5).

These tests do not assert anything is "wrong" -- missing_detector is the
textually correct status for a rule with no detector, and concerns staying
empty is the intended safe-by-default behaviour. Their purpose is to pin down
the *current* baseline in an executable, reviewable form, so that Phase 2/3
work (RULE_COVERAGE_MATRIX.csv, IMPLEMENTATION_PLAN.md) has a measurable
"before" to diff against, instead of relying on a one-off manual audit.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

REGISTRY = ROOT / "resources" / "psychology_sources" / "rules_registry.json"

# Wired per RULE_COVERAGE_MATRIX.csv (repo root) as of 2026-08-02 -- the only
# observables rules.py::_status_for() has a real branch for.
WIRED_OBSERVABLES = {
    "coverage_about_half", "coverage_full", "coverage_small",
    "placement_top", "placement_left", "placement_right",
}


class RuleRegistryBaselineTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    def test_registry_has_nineteen_rules(self):
        self.assertEqual(len(self.registry["rules"]), 19)

    def test_exactly_six_observables_are_wired_to_real_features(self):
        observables = {rule["observable"] for rule in self.registry["rules"]}
        self.assertEqual(observables & WIRED_OBSERVABLES, WIRED_OBSERVABLES)
        self.assertEqual(len(observables - WIRED_OBSERVABLES), 13)

    def test_unwired_rules_always_return_missing_detector(self):
        from doar.rules import evaluate_rules
        # A composition dict that would make every Tier-1 (wired) rule match,
        # so any rule NOT returning weak_support here has no detector, not a
        # threshold mismatch.
        composition = {"bounding_box_coverage": 0.5, "placement": "top-left"}
        evaluations, _ = evaluate_rules(composition, {}, [])
        by_observable = {}
        for rule_def, evaluation in zip(self.registry["rules"], evaluations):
            by_observable[rule_def["observable"]] = evaluation["status"]
        unwired = {obs: status for obs, status in by_observable.items()
                   if obs not in WIRED_OBSERVABLES}
        self.assertEqual(len(unwired), 13)
        self.assertTrue(all(status == "missing_detector" for status in unwired.values()),
                        f"Expected all 13 unwired rules to be missing_detector, got: {unwired}")

    def test_wired_rules_can_produce_weak_support(self):
        from doar.rules import evaluate_rules
        composition = {"bounding_box_coverage": 0.5, "placement": "top-left"}
        evaluations, _ = evaluate_rules(composition, {}, [])
        wired_statuses = {
            rule_def["observable"]: evaluation["status"]
            for rule_def, evaluation in zip(self.registry["rules"], evaluations)
            if rule_def["observable"] in WIRED_OBSERVABLES
        }
        self.assertTrue(any(status == "weak_support" for status in wired_statuses.values()))


class ConcernEngineUnreachabilityBaselineTests(unittest.TestCase):
    def test_real_evaluate_rules_output_is_always_tagged_clinician_symbolic(self):
        # rules.py::evaluate_rules() stamps source_type="psychologist_supplied_hypothesis"
        # on EVERY evaluation it emits, including the 6 wired to real
        # composition measurements -- so concerns.py's diversity check can
        # never see a second, non-clinical source type from this engine's own
        # output, regardless of the CONCERNS_ENABLED flag. This is the
        # structural reason CURRENT_STATE_AUDIT.md Section 5.3 gives for why
        # concerns cannot converge even if re-enabled.
        from doar.rules import evaluate_rules
        from doar.concerns import _source_type
        composition = {"bounding_box_coverage": 0.5, "placement": "top-left"}
        evaluations, _ = evaluate_rules(composition, {}, [])
        weak_support = [e for e in evaluations if e["status"] == "weak_support"]
        self.assertTrue(weak_support, "expected at least one wired rule to match")
        source_types = {
            _source_type(evidence_id, rule)
            for rule in weak_support
            for evidence_id in rule["matched_evidence_ids"]
        }
        self.assertEqual(source_types, {"clinician_symbolic"})

    def test_real_rule_output_caps_at_weak_hypothesis_not_a_full_concern(self):
        # UPDATED for Phase 2 (2026-08-02): the original Phase-0 finding that
        # real rule output "can never converge even if enabled" turned out to
        # be based on an inaccurate read of the code (see DECISION_LOG.md
        # 2026-08-02 correction) -- clinician-symbolic tagging for these
        # rules is intentional, not a bug, and was left unchanged. What
        # Phase 2 actually fixed is that evaluate_rules() no longer discards
        # the emotion model's independent evidence (test_phase2_rule_tiers_
        # and_aggregation.py::ModelEvidencePassthroughTests). Two Tier-1
        # rules matching together with NO model evidence are still exactly
        # ONE source type, so per the working spec's own policy ("several
        # correlated rules from one family: at most a weak hypothesis") this
        # now correctly produces a WEAK_HYPOTHESIS-level concern -- not [],
        # and not a stronger POSSIBLE_FOR_EXPLORATION/PROFESSIONAL_REVIEW
        # level either, since no second independent source is present.
        from doar.rules import evaluate_rules
        from doar.concerns import derive_concerns
        # A composition designed to match BOTH a coverage rule and a
        # placement rule -- two different evidence IDs, from two genuinely
        # different objective measurements (bounding-box coverage vs.
        # centroid placement) -- but the SAME (clinician_symbolic) source type.
        composition = {"bounding_box_coverage": 0.5, "placement": "top-left"}
        evaluations, _ = evaluate_rules(composition, {}, [])
        weak_support = [e for e in evaluations if e["status"] == "weak_support"]
        self.assertGreaterEqual(len(weak_support), 2,
                                "expected >=2 matched rules from 2 different evidence IDs")
        concerns = derive_concerns(evaluations, enabled=True)
        self.assertEqual(len(concerns), 1)
        self.assertEqual(concerns[0]["aggregation_strength"], "WEAK_HYPOTHESIS")
        self.assertEqual(concerns[0]["source_types"], ["clinician_symbolic"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
