"""Tests for src/doar/drawing_synthesis.py -- the unified multimodal
evidence + always-on drawing synthesis module (DOAR realignment
milestone).

Proves the load-bearing guarantees from the module's own docstring:
  1. Deterministic (colour/line/composition) features round-trip through
     the cache byte-for-byte reproducibly, including the resolved
     page_reference.
  2. The deterministic rule bridge produces exactly one precondition
     check per rule in DETERMINISTIC_RULE_IDS (10), reusing the frozen
     thresholds from rule_engine_v2.py/rules.py verbatim, and the 7
     page-relative rules among them report `not_assessable` -- never a
     guess from the raw image frame -- when the page is not confirmed.
  3. Unified evidence normalization never fabricates a value for
     unavailable/non-conclusive evidence (EvidenceItem's own invariant),
     and every semantic/deterministic/relationship/emotion item is
     represented.
  4. build_overall_synthesis requires >=2 independent evidence families
     (MIN_EVIDENCE, reused from concerns.py) before calling a direction
     "convergent" -- a single matching family is `limited_association`,
     never promoted to a whole-drawing positive/concern pattern.
  5. End to end, every one of the 15 real cached development images
     produces: a non-empty objective profile, an always-on overall
     synthesis, and literature_linked_associations that are a strict
     subset of all_matches (i.e. features without rules stay descriptive
     only, never silently promoted).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar import drawing_synthesis as ds  # noqa: E402
from doar import reasoning_chain as rc  # noqa: E402
from doar.evidence_schema import CONCLUSIVE_STATUSES  # noqa: E402
from doar.visual_entity import VisualEntity  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_development_benchmark", ROOT / "scripts" / "run_development_benchmark.py")
rdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdb)


def _entity(entity_id, canonical_label, *, status="verified", bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.9):
    return VisualEntity(
        entity_id=entity_id, entity_type="object", canonical_label=canonical_label,
        candidate_labels=((canonical_label, confidence),), aliases_en=(), aliases_ar=(),
        broader_categories=(), possible_subtypes=(), visual_similarities=(),
        bbox=bbox, crop_ref=None, dominant_colors=None, relative_size=None,
        page_position=None, shape_features=None, line_features=None,
        detector="visual_observer:test", checkpoint="", prompt="",
        confidence=confidence, model_validation_status="UNKNOWN", case_verification_status=status,
        evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
        related_rule_ids=(), source="visual_observer", query=None, timestamp="2026-08-14T00:00:00+00:00",
    )


class DeterministicCacheTests(unittest.TestCase):
    def test_compute_deterministic_features_returns_60_objective_features(self):
        image_path = ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0003.jpeg"
        if not image_path.exists():
            self.skipTest("dev-set image not present on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            result = ds.compute_deterministic_features(image_path, Path(tmp) / "artifacts")
        self.assertEqual(len(result["objective_features"]), 60)
        self.assertIn("segmentation.bounding_box_coverage", result["objective_features"])

    def test_compute_deterministic_features_resolves_a_page_reference(self):
        image_path = ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0003.jpeg"
        if not image_path.exists():
            self.skipTest("dev-set image not present on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            result = ds.compute_deterministic_features(image_path, Path(tmp) / "artifacts")
        self.assertIn("page_reference", result)
        self.assertIn("page_reference_mode", result["page_reference"])
        self.assertIn("page_relative_features_assessable", result["page_reference"])

    def test_cache_round_trips_byte_reproducibly(self):
        image_path = ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0003.jpeg"
        if not image_path.exists():
            self.skipTest("dev-set image not present on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            first = ds.load_or_compute_deterministic_features("p2b_0003", image_path, cache_dir=cache_dir)
            self.assertTrue(ds.deterministic_cache_path("p2b_0003", cache_dir).exists())
            second = ds.load_or_compute_deterministic_features("p2b_0003", image_path, cache_dir=cache_dir)
            self.assertEqual(first["composition"], second["composition"])
            self.assertEqual(first["page_reference"], second["page_reference"])
            for name, fv in first["objective_features"].items():
                other = second["objective_features"][name].value
                if isinstance(fv.value, float) and math.isnan(fv.value):
                    self.assertTrue(math.isnan(other), f"{name}: NaN did not round-trip as NaN")
                else:
                    self.assertEqual(fv.value, other, f"{name} did not round-trip identically")


_ASSESSABLE_PAGE_REFERENCE = {"page_reference_mode": "auto_detected_page", "page_relative_features_assessable": True}
_NOT_ASSESSABLE_PAGE_REFERENCE = {"page_reference_mode": "cropped_or_content_only", "page_relative_features_assessable": False}


class DeterministicPreconditionTests(unittest.TestCase):
    def _features(self, bbox_coverage, cx, cy, placement, intensity, fragmentation, margins=(0.05, 0.05, 0.05, 0.05)):
        from doar.features import FeatureValue
        def fv(value, missing=False, evidence_id="ev_x"):
            return FeatureValue(value=value, valid_min=None, valid_max=None, confidence=1.0,
                                 method="test", evidence_id=evidence_id, missing=missing)
        return {
            "segmentation.bounding_box_coverage": fv(bbox_coverage, evidence_id="ev_bbox_coverage"),
            "composition.centroid_x": fv(cx, evidence_id="ev_cx"),
            "composition.centroid_y": fv(cy, evidence_id="ev_cy"),
            "stroke.intensity_proxy": fv(intensity, evidence_id="ev_intensity"),
            "stroke.fragmentation": fv(fragmentation, evidence_id="ev_frag"),
        }, {"placement": placement, "margins_normalized": list(margins)}

    def test_exactly_ten_checks_matching_deterministic_rule_ids(self):
        objective_features, composition = self._features(0.5, 0.5, 0.5, "middle_center", 0.5, 0.1)
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        self.assertEqual({c.rule_id for c in checks}, ds.DETERMINISTIC_RULE_IDS)
        self.assertEqual(len(checks), 10)

    def test_coverage_about_half_satisfied_in_band(self):
        objective_features, composition = self._features(0.5, 0.5, 0.5, "middle_center", 0.5, 0.1)
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        half = next(c for c in checks if c.rule_id == "PSY_AR_SIZE_HALF_014")
        self.assertEqual(half.status, "satisfied")

    def test_coverage_full_uses_margin_based_redefinition_not_raw_area(self):
        # rule_engine_v2.redefine_coverage_full's own formula: content must
        # approach ALL FOUR page margins, reused verbatim -- a high raw
        # bounding_box_coverage alone (0.95) is NOT sufficient if the
        # margins themselves are wide.
        objective_features, composition = self._features(
            0.95, 0.5, 0.5, "middle_center", 0.5, 0.1, margins=(0.3, 0.3, 0.3, 0.3))
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        full = next(c for c in checks if c.rule_id == "PSY_AR_SIZE_FULL_015")
        self.assertEqual(full.status, "not_satisfied")

    def test_coverage_full_satisfied_when_margins_are_tight(self):
        objective_features, composition = self._features(
            0.95, 0.5, 0.5, "middle_center", 0.5, 0.1,
            margins=(ds.COVERAGE_FULL_MARGIN_THRESHOLD, ds.COVERAGE_FULL_MARGIN_THRESHOLD,
                     ds.COVERAGE_FULL_MARGIN_THRESHOLD, ds.COVERAGE_FULL_MARGIN_THRESHOLD))
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        full = next(c for c in checks if c.rule_id == "PSY_AR_SIZE_FULL_015")
        self.assertEqual(full.status, "satisfied")

    def test_missing_bbox_coverage_leaves_size_rules_not_satisfied(self):
        objective_features, composition = self._features(0.5, 0.5, 0.5, "middle_center", 0.5, 0.1)
        objective_features["segmentation.bounding_box_coverage"] = objective_features["segmentation.bounding_box_coverage"].__class__(
            value=0.0, valid_min=None, valid_max=None, confidence=0.0, method="x", evidence_id="ev_bbox_coverage", missing=True)
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        for rule_id in ("PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_SMALL_016"):
            self.assertEqual(next(c for c in checks if c.rule_id == rule_id).status, "not_satisfied")

    def test_heavy_and_light_pressure_are_mutually_exclusive_thresholds(self):
        objective_features, composition = self._features(0.5, 0.5, 0.5, "middle_center",
                                                           ds.INTENSITY_PROXY_HEAVY_THRESHOLD + 0.05, 0.1)
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        heavy = next(c for c in checks if c.rule_id == "EN_COMPILED_LINE_HEAVY_PRESSURE_030")
        light = next(c for c in checks if c.rule_id == "EN_COMPILED_LINE_LIGHT_PRESSURE_031")
        self.assertEqual(heavy.status, "satisfied")
        self.assertEqual(light.status, "not_satisfied")

    def test_build_deterministic_eligible_matches_only_includes_satisfied(self):
        objective_features, composition = self._features(
            0.95, 0.5, 0.5, "top_center", 0.9, 0.9,
            margins=(ds.COVERAGE_FULL_MARGIN_THRESHOLD, ds.COVERAGE_FULL_MARGIN_THRESHOLD,
                     ds.COVERAGE_FULL_MARGIN_THRESHOLD, ds.COVERAGE_FULL_MARGIN_THRESHOLD))
        matches = ds.build_deterministic_eligible_matches(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        matched_ids = {m.rule_id for m in matches}
        self.assertIn("PSY_AR_SIZE_FULL_015", matched_ids)
        self.assertIn("EN_COMPILED_LINE_HEAVY_PRESSURE_030", matched_ids)
        self.assertNotIn("PSY_AR_SIZE_HALF_014", matched_ids)
        self.assertNotIn("PSY_AR_SIZE_SMALL_016", matched_ids)


class PageFrameSafetyTests(unittest.TestCase):
    """Section 2 fix: page-relative rules (size/coverage/placement) must
    report `not_assessable`, never be computed from the raw image frame,
    when the page itself is not confirmed/detected -- reuses rule_engine_
    v2.ALL_PAGE_GATED_RULE_IDS as the authoritative gated set."""

    _ALL_SATISFIABLE_FEATURES_KWARGS = dict(
        bbox_coverage=0.5, cx=0.5, cy=0.5, placement="top_center", intensity=0.5, fragmentation=0.1,
        margins=(0.05, 0.05, 0.05, 0.05))

    def _features(self, **kwargs):
        return DeterministicPreconditionTests._features(self, **{**self._ALL_SATISFIABLE_FEATURES_KWARGS, **kwargs})

    def test_page_gated_rules_are_not_assessable_when_page_reference_is_none(self):
        objective_features, composition = self._features()
        checks = ds.check_deterministic_preconditions(composition, objective_features, None)
        for c in checks:
            if c.rule_id in ds._PAGE_GATED_DETERMINISTIC_RULE_IDS:
                self.assertEqual(c.status, "not_assessable", f"{c.rule_id} should be not_assessable with no page_reference")

    def test_page_gated_rules_are_not_assessable_when_page_not_confirmed(self):
        objective_features, composition = self._features()
        checks = ds.check_deterministic_preconditions(composition, objective_features, _NOT_ASSESSABLE_PAGE_REFERENCE)
        for c in checks:
            if c.rule_id in ds._PAGE_GATED_DETERMINISTIC_RULE_IDS:
                self.assertEqual(c.status, "not_assessable")

    def test_line_pressure_rules_are_never_page_gated(self):
        # Line-darkness/fragmentation proxies are not page-relative -- they
        # must still evaluate normally even when the page is not assessable.
        objective_features, composition = self._features(intensity=ds.INTENSITY_PROXY_HEAVY_THRESHOLD + 0.05)
        checks = ds.check_deterministic_preconditions(composition, objective_features, _NOT_ASSESSABLE_PAGE_REFERENCE)
        heavy = next(c for c in checks if c.rule_id == "EN_COMPILED_LINE_HEAVY_PRESSURE_030")
        self.assertEqual(heavy.status, "satisfied")

    def test_not_assessable_matches_are_excluded_from_eligible_matches(self):
        objective_features, composition = self._features()
        matches = ds.build_deterministic_eligible_matches(composition, objective_features, _NOT_ASSESSABLE_PAGE_REFERENCE)
        matched_ids = {m.rule_id for m in matches}
        self.assertFalse(matched_ids & ds._PAGE_GATED_DETERMINISTIC_RULE_IDS)

    def test_page_gated_rule_set_matches_rule_engine_v2_authoritative_set(self):
        from doar.rule_engine_v2 import ALL_PAGE_GATED_RULE_IDS
        self.assertEqual(ds._PAGE_GATED_DETERMINISTIC_RULE_IDS, ALL_PAGE_GATED_RULE_IDS)

    def test_page_assessable_rules_still_evaluate_normally(self):
        objective_features, composition = self._features(
            margins=(ds.COVERAGE_FULL_MARGIN_THRESHOLD, ds.COVERAGE_FULL_MARGIN_THRESHOLD,
                     ds.COVERAGE_FULL_MARGIN_THRESHOLD, ds.COVERAGE_FULL_MARGIN_THRESHOLD))
        checks = ds.check_deterministic_preconditions(composition, objective_features, _ASSESSABLE_PAGE_REFERENCE)
        for c in checks:
            self.assertNotEqual(c.status, "not_assessable")


class EvidenceInvariantTests(unittest.TestCase):
    """EvidenceItem itself raises if a non-conclusive item carries a
    value -- these tests prove normalize_* never trips that invariant,
    i.e. never fabricates a value for unavailable/rejected evidence."""

    def test_normalize_semantic_evidence_never_fabricates_value_for_rejected(self):
        entities = [_entity("e1", "lion", status="rejected")]
        items = ds.normalize_semantic_evidence(entities, {})
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].item.value)
        self.assertEqual(items[0].item.status, "requires_manual_review")
        self.assertIn("lion", items[0].item.reason)

    def test_normalize_semantic_evidence_verified_carries_real_value(self):
        entities = [_entity("e1", "lion", status="verified")]
        items = ds.normalize_semantic_evidence(entities, {})
        self.assertEqual(items[0].item.value, "lion")
        self.assertIn(items[0].item.status, CONCLUSIVE_STATUSES)

    def test_normalize_deterministic_evidence_missing_feature_has_no_value(self):
        from doar.features import FeatureValue
        objective_features = {
            "shape.enclosed_shape_count": FeatureValue(
                value=float("nan"), valid_min=None, valid_max=None, confidence=0.0,
                method="not_evaluated_no_detector", evidence_id="ev_shape", missing=True),
        }
        items = ds.normalize_deterministic_evidence(objective_features, {})
        self.assertIsNone(items[0].item.value)
        self.assertEqual(items[0].item.status, "unavailable")

    def test_normalize_relationship_evidence_always_emits_entity_count(self):
        items = ds.normalize_relationship_evidence({"entity_count": 0, "repeated_labels": {}})
        feature_ids = {i.item.feature_id for i in items}
        self.assertIn("relationships.entity_count", feature_ids)
        self.assertFalse(any(i.rule_eligible for i in items))

    def test_normalize_emotion_evidence_is_unavailable_placeholder_when_no_model(self):
        items = ds.normalize_emotion_evidence(None)
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].item.value)
        self.assertEqual(items[0].item.status, "unavailable")
        self.assertFalse(items[0].rule_eligible)

    def test_normalize_emotion_evidence_skips_placeholder_when_model_evidence_given(self):
        items = ds.normalize_emotion_evidence([{"some": "real evidence"}])
        self.assertEqual(items, [])


class OverallSynthesisTests(unittest.TestCase):
    """Section 1 fix: a single individual_heuristic_only rule must never
    by itself produce a whole-drawing positive/concern reading --
    `convergent_*` requires >=2 INDEPENDENT evidence families (MIN_EVIDENCE,
    reused verbatim from concerns.py) WITHIN ONE CONCERN DOMAIN (acceptance-
    audit fix -- reuses `reasoning_chain.aggregate_by_concern_domain`, the
    SAME strict per-domain partition `build_candidate_hypotheses` itself
    uses, so evidence from different, unrelated domains is never pooled to
    manufacture a false "convergent" reading). Exactly one family, or
    multiple families scattered across different domains with no single
    domain reaching convergence, is `limited_association` -- not silently
    dropped, not overclaimed."""

    # _DISTRESS_DOMAINS has 5 members; used here to construct same-domain
    # vs. cross-domain fixtures precisely.
    _DOMAIN_A = "anxiety_or_stress_related"
    _DOMAIN_B = "aggression_or_threat_related"

    def _match(self, rule_id, concern_domain, evidence_family="test_family"):
        return rc.EligibleAtomicRuleMatch(
            rule_id=rule_id, evidence_family=evidence_family, concern_domain=concern_domain,
            allowed_output_level="clinician_symbolic", source_claim="test", possible_interpretation="test",
            alternative_explanations=(), matched_entity_ids=("e1",), evidence_direction="test",
            context_transfer_justification="test")

    def test_no_matches_is_insufficient_interpretable_evidence(self):
        result = ds.build_overall_synthesis([])
        self.assertEqual(result["level"], "insufficient_interpretable_evidence")

    def test_single_positive_family_is_limited_association_not_convergent(self):
        # Two RULES but the SAME evidence family -- still just one
        # independent unit of evidence per deduplicate_by_evidence_family.
        matches = [self._match("R1", "positive_affect_or_social_engagement", "family_a"),
                   self._match("R2", "positive_affect_or_social_engagement", "family_a")]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "limited_association")
        self.assertEqual(result["positive_evidence_families"], ["family_a"])

    def test_single_concern_family_is_limited_association_not_convergent(self):
        # This is the exact bug scenario reported: a single weak rule
        # (e.g. EN_COMPILED_LINE_LIGHT_PRESSURE_031) must not alone yield
        # a "distress-associated drawing" reading.
        matches = [self._match("EN_COMPILED_LINE_LIGHT_PRESSURE_031", self._DOMAIN_A, "line_intensity_quality")]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "limited_association")
        self.assertNotIn("convergent", result["level"])
        self.assertIn("EN_COMPILED_LINE_LIGHT_PRESSURE_031", result["concern_rule_ids"])
        self.assertIsNone(result["convergent_concern_domain"])

    def test_two_independent_positive_families_is_convergent(self):
        # Only one positive domain exists in the frozen domain map today,
        # so two distinct families here are automatically same-domain.
        matches = [self._match("R1", "positive_affect_or_social_engagement", "family_a"),
                   self._match("R2", "positive_affect_or_social_engagement", "family_b")]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "convergent_positive_pattern")
        self.assertEqual(result["convergent_positive_domain"], "positive_affect_or_social_engagement")
        self.assertEqual(set(result["positive_evidence_families"]), {"family_a", "family_b"})

    def test_two_independent_families_in_different_concern_domains_do_NOT_converge(self):
        """Acceptance-audit regression: p2b_0001/p2b_0005/p2b_0068's exact
        real-world shape -- two independent families, but each in a
        DIFFERENT concern domain -- must NOT produce convergent_concern_
        pattern. This was the confirmed bug (all 3 real convergent cases
        were cross-domain)."""
        matches = [self._match("R1", self._DOMAIN_A, "family_a"), self._match("R2", self._DOMAIN_B, "family_b")]
        result = ds.build_overall_synthesis(matches)
        self.assertNotEqual(result["level"], "convergent_concern_pattern")
        self.assertEqual(result["level"], "limited_association")
        self.assertIsNone(result["convergent_concern_domain"])
        # Explicit cross-domain wording, per the required behavior.
        self.assertIn("different", result["summary"])
        self.assertIn("not one coherent convergent pattern", result["summary"])

    def test_two_independent_families_in_the_SAME_concern_domain_DO_converge(self):
        """The positive control for the fix above: when both independent
        families sit inside the SAME concern domain, convergence is
        legitimate and convergent_concern_pattern is correct."""
        matches = [self._match("R1", self._DOMAIN_A, "family_a"), self._match("R2", self._DOMAIN_A, "family_b")]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "convergent_concern_pattern")
        self.assertEqual(result["convergent_concern_domain"], self._DOMAIN_A)
        self.assertEqual(result["concern_domain_family_counts"][self._DOMAIN_A], 2)

    def test_three_domains_one_reaching_convergence_is_still_convergent(self):
        # A third, single-family domain must not block a genuinely
        # convergent domain elsewhere from being recognized.
        matches = [self._match("R1", self._DOMAIN_A, "family_a"), self._match("R2", self._DOMAIN_A, "family_b"),
                   self._match("R3", "social_withdrawal_related", "family_c")]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "convergent_concern_pattern")
        self.assertEqual(result["convergent_concern_domain"], self._DOMAIN_A)

    def test_scattered_individual_associations_are_limited_not_mixed(self):
        matches = [self._match("R1", "positive_affect_or_social_engagement", "family_a"),
                   self._match("R2", self._DOMAIN_A, "family_b")]
        result = ds.build_overall_synthesis(matches)
        # Requirement 6: individual associations of both directions
        # existing is NOT sufficient for mixed_evidence -- neither side
        # is domain-convergent here.
        self.assertEqual(result["level"], "limited_association")
        self.assertIsNone(result["convergent_positive_domain"])
        self.assertIsNone(result["convergent_concern_domain"])

    def test_mixed_evidence_requires_genuine_convergence_on_both_sides(self):
        matches = [
            self._match("P1", "positive_affect_or_social_engagement", "pos_family_a"),
            self._match("P2", "positive_affect_or_social_engagement", "pos_family_b"),
            self._match("C1", self._DOMAIN_A, "con_family_a"),
            self._match("C2", self._DOMAIN_A, "con_family_b"),
        ]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "mixed_evidence")
        self.assertEqual(result["convergent_positive_domain"], "positive_affect_or_social_engagement")
        self.assertEqual(result["convergent_concern_domain"], self._DOMAIN_A)

    def test_neutral_only_domain_is_descriptive_only(self):
        matches = [self._match("R1", "neutral_descriptive_only", "family_a")]
        result = ds.build_overall_synthesis(matches)
        self.assertEqual(result["level"], "descriptive_only")

    def test_supporting_families_and_rule_ids_are_always_present(self):
        matches = [self._match("R1", "anxiety_or_stress_related", "family_a")]
        result = ds.build_overall_synthesis(matches)
        for key in ("positive_evidence_families", "positive_rule_ids", "concern_evidence_families", "concern_rule_ids"):
            self.assertIn(key, result)

    def test_min_evidence_convergence_bar_is_reused_from_concerns_py_not_reinvented(self):
        from doar.concerns import MIN_EVIDENCE
        self.assertEqual(MIN_EVIDENCE, ds.MIN_EVIDENCE)


class CandidateHypothesisUnchangedTests(unittest.TestCase):
    """Requirement 7 (domain-scoped convergence fix): candidate-hypothesis
    logic must remain completely unchanged -- `synthesize_drawing` still
    calls `reasoning_chain.build_candidate_hypotheses` directly and
    unmodified, and its own domain-siloed, MIN_EVIDENCE-gated,
    WEAK_HYPOTHESIS-capped behavior is unaffected by whatever
    `build_overall_synthesis` now concludes."""

    def _match(self, rule_id, concern_domain, evidence_family):
        return rc.EligibleAtomicRuleMatch(
            rule_id=rule_id, evidence_family=evidence_family, concern_domain=concern_domain,
            allowed_output_level="clinician_symbolic", source_claim="test", possible_interpretation="test",
            alternative_explanations=(), matched_entity_ids=("e1",), evidence_direction="test",
            context_transfer_justification="test")

    def test_synthesize_drawing_still_calls_the_real_unmodified_build_candidate_hypotheses(self):
        source = Path(ds.__file__).read_text(encoding="utf-8")
        self.assertIn("rc.build_candidate_hypotheses(all_matches, model_evidence=model_evidence)", source)

    def test_cross_domain_limited_association_still_yields_zero_hypotheses(self):
        # Same shape as the cross-domain non-convergence test above:
        # build_overall_synthesis correctly reports limited_association,
        # and build_candidate_hypotheses independently (and correctly)
        # still raises nothing -- MIN_EVIDENCE=2 WITHIN one domain is
        # exactly what it already required, unaffected by this fix.
        matches = [self._match("R1", "anxiety_or_stress_related", "family_a"),
                   self._match("R2", "aggression_or_threat_related", "family_b")]
        self.assertEqual(ds.build_overall_synthesis(matches)["level"], "limited_association")
        self.assertEqual(rc.build_candidate_hypotheses(matches), [])

    def test_same_domain_convergent_evidence_still_reaches_weak_hypothesis_via_unmodified_logic(self):
        matches = [self._match("R1", "anxiety_or_stress_related", "family_a"),
                   self._match("R2", "anxiety_or_stress_related", "family_b")]
        self.assertEqual(ds.build_overall_synthesis(matches)["level"], "convergent_concern_pattern")
        hypotheses = rc.build_candidate_hypotheses(matches)
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].support_level, "WEAK_HYPOTHESIS")

    def test_all_15_real_cases_hypothesis_count_matches_pre_fix_baseline(self):
        """The domain-scoping fix changes overall_synthesis LEVELS on the
        real dev set (p2b_0001/p2b_0005/p2b_0068 move from convergent_
        concern_pattern to limited_association) but must not change
        candidate_hypotheses at all -- baseline was 0/15 both before and
        after this fix.

        This is a real-data integration check over the cached development
        set's Observer/Verifier output -- that cache is optional/generated
        (never committed; absent on a clean checkout, e.g. CI) and is not
        meaningfully faked by a small fixture without duplicating the
        cache's own 15-case richness, so it self-skips when genuinely
        absent rather than failing a clean runner -- same pattern already
        used by the single-image tests earlier in this file and by
        scripts/verifier_error_analysis.py's own real-cached-data tests."""
        seen_any = False
        for image in rdb.load_development_set():
            image_id = image["image_id"]
            rows, _ = rdb.find_saved_verification_rows(image_id)
            if rows is None:
                continue
            seen_any = True
            entities = rdb.entities_from_verification_rows(rows)
            det = ds.load_or_compute_deterministic_features(image_id, ROOT / image["relative_path"])
            result = ds.synthesize_drawing(image_id, entities, det)
            self.assertEqual(len(result.candidate_hypotheses), 0, f"{image_id}: hypothesis count changed")
        if not seen_any:
            self.skipTest("no cached development-set Observer/Verifier data present in this checkout")


class RealDevelopmentSetEndToEndTests(unittest.TestCase):
    """Sweeps all 15 real cached development images -- proves the
    integration actually works end to end, not just against synthetic
    fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.images = rdb.load_development_set()

    def test_every_case_produces_objective_features_and_overall_synthesis(self):
        seen_any = False
        with tempfile.TemporaryDirectory() as tmp:
            for image in self.images:
                image_id = image["image_id"]
                rows, _ = rdb.find_saved_verification_rows(image_id)
                if rows is None:
                    continue
                seen_any = True
                entities = rdb.entities_from_verification_rows(rows)
                image_path = ROOT / image["relative_path"]
                det = ds.compute_deterministic_features(image_path, Path(tmp) / image_id / "artifacts")
                result = ds.synthesize_drawing(image_id, entities, det)

                self.assertGreater(len(result.unified_evidence), 0, f"{image_id}: no unified evidence")
                self.assertGreater(len(result.objective_profile), 0, f"{image_id}: empty objective profile")
                self.assertIn(result.overall_synthesis["level"], (
                    "insufficient_interpretable_evidence", "descriptive_only", "limited_association",
                    "convergent_positive_pattern", "convergent_concern_pattern", "mixed_evidence"))
                self.assertTrue(result.overall_synthesis["summary"], f"{image_id}: empty synthesis summary")

                # A single-family match must never reach a convergent_*
                # level -- proves the anti-overclaiming fix holds on real data.
                if result.overall_synthesis["level"] in ("convergent_positive_pattern", "convergent_concern_pattern"):
                    families = (result.overall_synthesis["positive_evidence_families"]
                                if result.overall_synthesis["level"] == "convergent_positive_pattern"
                                else result.overall_synthesis["concern_evidence_families"])
                    self.assertGreaterEqual(len(families), 2, f"{image_id}: convergent level with <2 evidence families")

                # Every literature-linked association must correspond to a
                # real eligible match -- descriptive-only evidence never
                # silently promoted into an association.
                association_rule_ids = {a["rule_id"] for a in result.literature_linked_associations}
                all_rule_ids = {m.rule_id for m in rc.build_eligible_matches(entities)} | {
                    m.rule_id for m in ds.build_deterministic_eligible_matches(
                        det["composition"], det["objective_features"], det.get("page_reference"))}
                self.assertTrue(association_rule_ids.issubset(all_rule_ids), f"{image_id}: association not backed by an eligible match")

                # Zero candidate hypotheses is an accepted, valid outcome.
                self.assertIsInstance(result.candidate_hypotheses, list)
        if not seen_any:
            # Real-data integration sweep over the optional, generated
            # development-set cache (never committed) -- self-skips on a
            # clean checkout (e.g. CI) rather than failing; still runs the
            # full sweep whenever the cache exists locally.
            self.skipTest("no cached development-set Observer/Verifier data present in this checkout")

    def test_page_reference_is_present_and_resolved_for_every_real_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            for image in self.images:
                image_id = image["image_id"]
                rows, _ = rdb.find_saved_verification_rows(image_id)
                if rows is None:
                    continue
                image_path = ROOT / image["relative_path"]
                det = ds.compute_deterministic_features(image_path, Path(tmp) / image_id / "artifacts")
                self.assertIn("page_reference", det)
                self.assertIn("page_relative_features_assessable", det["page_reference"])


class ValidationInvariantRegressionTests(unittest.TestCase):
    """Regression test for a real validation-SCRIPT bug caught during the
    post-correction acceptance audit (not a pipeline bug): an ad-hoc
    revalidation script recomputed `all_matches` via
    `build_deterministic_eligible_matches(composition, objective_features)`
    WITHOUT threading the resolved `page_reference` through (silently
    falling back to the default `page_reference=None`), while
    `synthesize_drawing` itself always correctly passes the real one
    (see `synthesize_drawing`'s own call). For 14/15 development images
    (page not assessable) this mismatch was invisible -- both the None
    default and the real not-assessable page_reference gate away the
    same 7 page-relative rules, so the two computations happened to
    agree. Only p2b_0061 -- the ONE development image where the page IS
    confirmed assessable -- exposed it: with the real page_reference, a
    page-relative rule legitimately matches; with the buggy None default,
    that same rule is incorrectly withheld, so the script's own
    `association_rule_ids.issubset(all_rule_ids)` check failed on a
    completely correct pipeline result. The PIPELINE was right; the
    VALIDATOR was wrong -- fixed by threading `det.get("page_reference")`
    through every recomputation, exactly as `synthesize_drawing` already
    does internally. These tests pin the correct behaviour (and, in the
    last test, literally reproduce the old bug) so this exact class of
    validation mistake cannot silently regress or be "fixed" by quietly
    dropping the invariant instead of fixing the check."""

    @classmethod
    def setUpClass(cls):
        cls.images = {im["image_id"]: im for im in rdb.load_development_set()}

    def _deterministic_features_and_entities(self, image_id, tmp):
        rows, _ = rdb.find_saved_verification_rows(image_id)
        if rows is None:
            self.skipTest(f"no cached data for {image_id}")
        image = self.images[image_id]
        image_path = ROOT / image["relative_path"]
        det = ds.compute_deterministic_features(image_path, Path(tmp) / image_id / "artifacts")
        entities = rdb.entities_from_verification_rows(rows)
        return det, entities

    def test_p2b_0061_is_the_page_assessable_case_this_regression_depends_on(self):
        # This test's whole premise depends on p2b_0061 being the
        # page-assessable dev case -- pinned explicitly so a future cache/
        # threshold change that flips this is caught here, not silently.
        with tempfile.TemporaryDirectory() as tmp:
            det, _ = self._deterministic_features_and_entities("p2b_0061", tmp)
        self.assertTrue(det["page_reference"]["page_relative_features_assessable"])

    def test_omitting_page_reference_silently_changes_the_deterministic_match_set(self):
        """The exact failure mode: recomputing matches with the default
        `page_reference=None` yields a DIFFERENT (smaller) rule_id set
        than passing the real, resolved page_reference, for the one case
        where it matters."""
        with tempfile.TemporaryDirectory() as tmp:
            det, _ = self._deterministic_features_and_entities("p2b_0061", tmp)
        with_real_page_reference = {m.rule_id for m in ds.build_deterministic_eligible_matches(
            det["composition"], det["objective_features"], det["page_reference"])}
        with_omitted_page_reference = {m.rule_id for m in ds.build_deterministic_eligible_matches(
            det["composition"], det["objective_features"])}
        self.assertTrue(
            with_real_page_reference - with_omitted_page_reference,
            "expected the real page_reference to unlock at least one page-relative match that the "
            "page_reference=None default incorrectly withholds")

    def test_association_subset_invariant_holds_when_page_reference_is_threaded_correctly(self):
        """The CORRECT invariant, proven on the real, assessable case:
        every literature-linked association is backed by a real eligible
        match when the SAME page_reference `synthesize_drawing` used
        internally is also used to recompute the comparison set."""
        with tempfile.TemporaryDirectory() as tmp:
            det, entities = self._deterministic_features_and_entities("p2b_0061", tmp)
        result = ds.synthesize_drawing("p2b_0061", entities, det)
        association_rule_ids = {a["rule_id"] for a in result.literature_linked_associations}
        all_rule_ids = {m.rule_id for m in rc.build_eligible_matches(entities)} | {
            m.rule_id for m in ds.build_deterministic_eligible_matches(
                det["composition"], det["objective_features"], det["page_reference"])}
        self.assertTrue(association_rule_ids.issubset(all_rule_ids))

    def test_the_old_buggy_validator_check_would_have_failed_on_this_correct_result(self):
        """Literally reproduces the old (buggy) validation script's check
        -- page_reference omitted from the recomputation -- against the
        SAME correct `result` used above, and shows it incorrectly
        rejects it. If this assertion ever stops holding, the historical
        bug this test documents no longer reproduces; update or remove
        this test rather than deleting the invariant above."""
        with tempfile.TemporaryDirectory() as tmp:
            det, entities = self._deterministic_features_and_entities("p2b_0061", tmp)
        result = ds.synthesize_drawing("p2b_0061", entities, det)
        association_rule_ids = {a["rule_id"] for a in result.literature_linked_associations}
        buggy_all_rule_ids = {m.rule_id for m in rc.build_eligible_matches(entities)} | {
            m.rule_id for m in ds.build_deterministic_eligible_matches(
                det["composition"], det["objective_features"])}  # page_reference omitted -- the historical bug
        self.assertFalse(association_rule_ids.issubset(buggy_all_rule_ids))


if __name__ == "__main__":
    unittest.main()
