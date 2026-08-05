"""Tests for DOAR-TRACE Phase 2A, Section 7+8: rule_engine_v2.py's 4
newly-activated rules, page-frame gating of the historical engine's
output, and the aggregation-safety requirements Section 8 mandates
(mutual exclusivity by construction, evidence-family independence,
proxy-safe wording, no double-counting)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.registry_v2_build import build_registry_v2
from doar.rule_engine_v2 import (
    ALL_PAGE_GATED_RULE_IDS,
    FRAGMENTATION_SHAKY_THRESHOLD,
    INTENSITY_PROXY_HEAVY_THRESHOLD,
    INTENSITY_PROXY_LIGHT_THRESHOLD,
    PAGE_GATED_HISTORICAL_RULE_IDS,
    PLACEMENT_CENTER_BAND,
    V2_RULE_IDS,
    apply_page_frame_gating,
    evaluate_v2_rules,
)


def _feature(value: float, missing: bool = False) -> dict:
    return {"value": value, "missing": missing, "confidence": 0.9, "method": "test", "evidence_id": "ev_x"}


def _objective_features(**overrides: float) -> dict:
    base = {
        "composition.centroid_x": _feature(0.5),
        "composition.centroid_y": _feature(0.5),
        "stroke.intensity_proxy": _feature(0.5423),  # real p50
        "stroke.fragmentation": _feature(0.1694),  # real p50
    }
    for key, value in overrides.items():
        base[key] = _feature(value)
    return base


# Phase 2A.1 migration: apply_page_frame_gating/evaluate_v2_rules now
# gate on the RESOLVED page_reference (page_relative_features_assessable),
# not the raw automatic page_frame status directly -- see
# docs/PAGE_REFERENCE_MODEL.md and rule_engine_v2.py's docstrings.
_ASSESSABLE_PAGE = {"page_reference_mode": "auto_detected_page", "page_relative_features_assessable": True}
_NOT_ASSESSABLE_PAGE = {"page_reference_mode": "cropped_or_content_only", "page_relative_features_assessable": False}


class MutualExclusivityByConstructionTests(unittest.TestCase):
    """Section 8: heavy/light line-pressure must be mutually exclusive not
    by convention but by construction -- no value of stroke.intensity_proxy
    can ever satisfy both thresholds simultaneously."""

    def test_thresholds_do_not_overlap(self):
        self.assertLess(INTENSITY_PROXY_LIGHT_THRESHOLD, INTENSITY_PROXY_HEAVY_THRESHOLD)

    def test_no_value_in_a_dense_sweep_triggers_both(self):
        rules_v2_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}
        for i in range(0, 1001):
            value = i / 1000.0
            objective_features = _objective_features(**{"stroke.intensity_proxy": value})
            evaluations = evaluate_v2_rules(rules_v2_by_id, objective_features, _ASSESSABLE_PAGE)
            statuses = {e["rule_id"]: e["status"] for e in evaluations}
            both_matched = (
                statuses["EN_COMPILED_LINE_HEAVY_PRESSURE_030"] == "weak_support"
                and statuses["EN_COMPILED_LINE_LIGHT_PRESSURE_031"] == "weak_support"
            )
            self.assertFalse(both_matched, f"value={value} triggered both heavy and light pressure")

    def test_fragmentation_threshold_is_independent_of_intensity_thresholds(self):
        # Distinct feature (stroke.fragmentation), so its own threshold is
        # not required to relate numerically to the intensity thresholds --
        # it must simply exist and be a real, documented value in [0, 1].
        self.assertTrue(0.0 < FRAGMENTATION_SHAKY_THRESHOLD < 1.0)

    def test_heavy_and_light_do_trigger_at_their_respective_extremes(self):
        rules_v2_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}
        heavy = evaluate_v2_rules(rules_v2_by_id, _objective_features(**{"stroke.intensity_proxy": 0.99}), _ASSESSABLE_PAGE)
        light = evaluate_v2_rules(rules_v2_by_id, _objective_features(**{"stroke.intensity_proxy": 0.01}), _ASSESSABLE_PAGE)
        heavy_status = {e["rule_id"]: e["status"] for e in heavy}
        light_status = {e["rule_id"]: e["status"] for e in light}
        self.assertEqual(heavy_status["EN_COMPILED_LINE_HEAVY_PRESSURE_030"], "weak_support")
        self.assertEqual(heavy_status["EN_COMPILED_LINE_LIGHT_PRESSURE_031"], "not_matched")
        self.assertEqual(light_status["EN_COMPILED_LINE_LIGHT_PRESSURE_031"], "weak_support")
        self.assertEqual(light_status["EN_COMPILED_LINE_HEAVY_PRESSURE_030"], "not_matched")


class PlacementCenterPageFrameGatingTests(unittest.TestCase):
    def setUp(self):
        self.rules_v2_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}

    def test_not_assessable_when_page_not_visible(self):
        evaluations = evaluate_v2_rules(self.rules_v2_by_id, _objective_features(), _NOT_ASSESSABLE_PAGE)
        by_id = {e["rule_id"]: e for e in evaluations}
        self.assertEqual(by_id["EN_COMPILED_PLACEMENT_CENTER_029"]["status"], "not_assessable")
        self.assertEqual(by_id["EN_COMPILED_PLACEMENT_CENTER_029"]["missing_evidence"], ["page_reference_not_assessable"])

    def test_evaluated_normally_when_page_visible(self):
        lo, hi = PLACEMENT_CENTER_BAND
        centered = (lo + hi) / 2
        evaluations = evaluate_v2_rules(
            self.rules_v2_by_id,
            _objective_features(**{"composition.centroid_x": centered, "composition.centroid_y": centered}),
            _ASSESSABLE_PAGE,
        )
        by_id = {e["rule_id"]: e for e in evaluations}
        self.assertEqual(by_id["EN_COMPILED_PLACEMENT_CENTER_029"]["status"], "weak_support")

    def test_off_center_does_not_match_even_when_assessable(self):
        evaluations = evaluate_v2_rules(
            self.rules_v2_by_id,
            _objective_features(**{"composition.centroid_x": 0.05, "composition.centroid_y": 0.05}),
            _ASSESSABLE_PAGE,
        )
        by_id = {e["rule_id"]: e for e in evaluations}
        self.assertEqual(by_id["EN_COMPILED_PLACEMENT_CENTER_029"]["status"], "not_matched")


class HistoricalEngineGatingTests(unittest.TestCase):
    """apply_page_frame_gating post-processes rules.py's output without
    ever touching rules.py itself."""

    def test_assessable_page_leaves_evaluations_unchanged(self):
        original = [{"rule_id": "PSY_AR_SIZE_FULL_015", "status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"], "missing_evidence": []}]
        gated = apply_page_frame_gating(original, _ASSESSABLE_PAGE)
        self.assertEqual(gated, original)

    def test_not_assessable_page_gates_weak_support_and_not_matched_only(self):
        original = [
            {"rule_id": "PSY_AR_SIZE_FULL_015", "status": "weak_support", "matched_evidence_ids": ["ev_bbox_coverage"], "missing_evidence": [], "rule_confidence": 0.15, "professional_reasoning": "x", "parent_safe_wording": "y"},
            {"rule_id": "PSY_AR_SIZE_SMALL_016", "status": "not_matched", "matched_evidence_ids": [], "missing_evidence": []},
            {"rule_id": "PSY_AR_EYES_WIDE_001", "status": "missing_detector", "matched_evidence_ids": [], "missing_evidence": ["detector_absent:eyes_wide"]},
        ]
        gated = apply_page_frame_gating(original, _NOT_ASSESSABLE_PAGE)
        by_id = {g["rule_id"]: g for g in gated}
        self.assertEqual(by_id["PSY_AR_SIZE_FULL_015"]["status"], "not_assessable")
        self.assertEqual(by_id["PSY_AR_SIZE_FULL_015"]["matched_evidence_ids"], [])
        self.assertEqual(by_id["PSY_AR_SIZE_FULL_015"]["rule_confidence"], 0.0)
        self.assertIsNone(by_id["PSY_AR_SIZE_FULL_015"]["parent_safe_wording"])
        self.assertEqual(by_id["PSY_AR_SIZE_SMALL_016"]["status"], "not_assessable")
        # A rule with no page-frame relevance (missing_detector) must be left alone.
        self.assertEqual(by_id["PSY_AR_EYES_WIDE_001"]["status"], "missing_detector")

    def test_all_page_gated_rule_ids_is_superset_of_historical_ones(self):
        self.assertTrue(PAGE_GATED_HISTORICAL_RULE_IDS <= ALL_PAGE_GATED_RULE_IDS)
        self.assertIn("EN_COMPILED_PLACEMENT_CENTER_029", ALL_PAGE_GATED_RULE_IDS)


class ProxySafeWordingTests(unittest.TestCase):
    """Section 8/task-wide mandatory wording: line-quality proxy rules must
    never claim actual physical pencil pressure or an internal emotional
    state as directly observed."""

    _FORBIDDEN = ("pressed the pencil", "drew anxiously", "the child is", "the child pressed")

    def setUp(self):
        rules = build_registry_v2()["rules"]
        self.by_id = {r["rule_id"]: r for r in rules}

    def test_no_forbidden_phrasing_in_any_v2_rule_wording(self):
        for rule_id in V2_RULE_IDS:
            rule = self.by_id[rule_id]
            for field in ("parent_safe_wording", "professional_wording"):
                text = rule[field].lower()
                for forbidden in self._FORBIDDEN:
                    self.assertNotIn(forbidden, text, f"{rule_id}.{field} contains forbidden phrase {forbidden!r}")

    def test_stroke_proxy_rules_use_required_appearance_wording(self):
        heavy = self.by_id["EN_COMPILED_LINE_HEAVY_PRESSURE_030"]
        self.assertIn("appear relatively dark or thick", heavy["parent_safe_wording"])
        light = self.by_id["EN_COMPILED_LINE_LIGHT_PRESSURE_031"]
        self.assertIn("appear relatively light or thin", light["parent_safe_wording"])
        shaky = self.by_id["EN_COMPILED_LINE_SHAKY_BROKEN_032"]
        self.assertIn("fragmentation proxy is elevated", shaky["parent_safe_wording"])

    def test_evidence_family_independence_intensity_vs_fragmentation(self):
        # Section 8: line-intensity and line-fragmentation may only count as
        # separate families because their dependency groups are genuinely
        # distinct (different underlying feature).
        heavy = self.by_id["EN_COMPILED_LINE_HEAVY_PRESSURE_030"]
        light = self.by_id["EN_COMPILED_LINE_LIGHT_PRESSURE_031"]
        shaky = self.by_id["EN_COMPILED_LINE_SHAKY_BROKEN_032"]
        self.assertEqual(heavy["evidence_family"], light["evidence_family"])
        self.assertNotEqual(heavy["evidence_family"], shaky["evidence_family"])
        self.assertNotEqual(heavy["dependency_group"], shaky["dependency_group"])
        self.assertEqual(heavy["dependency_group"], light["dependency_group"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
