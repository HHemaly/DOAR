"""Focused tests for the new interpretable case-synthesis layer
(src/doar/case_interpretation.py) -- Milestone 3.

No live API/network access and no real emotion checkpoint are required:
the emotion-availability tests use a real case with no checkpoint (the
honest "unavailable" path), and the model_evidence-aware tests inject a
synthetic `bundle["emotion"]` dict (the exact shape live_case_bundle.py
already produces from analysis.json's real "emotion" field) rather than
loading actual model weights.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import case_interpretation as ci
from doar.live_case_bundle import build_live_case_bundle
from doar.timed_analysis import analyze_image_with_timing

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False


def _entity_dict(entity_id: str, canonical_label: str, status: str) -> dict:
    return {
        "entity_id": entity_id, "entity_type": "object", "canonical_label": canonical_label,
        "candidate_labels": [[canonical_label, 0.9]], "aliases_en": [], "aliases_ar": [],
        "broader_categories": [], "possible_subtypes": [], "visual_similarities": [],
        "bbox": None, "crop_ref": None, "dominant_colors": None, "relative_size": None,
        "page_position": None, "shape_features": None, "line_features": None,
        "detector": "test-detector", "checkpoint": "n/a", "prompt": canonical_label,
        "confidence": 0.9, "model_validation_status": "EXPERIMENTAL",
        "case_verification_status": status, "evidence_status": "present",
        "rule_mapping_status": "unmapped", "related_rule_ids": [], "source": "initial_scan",
        "query": None, "timestamp": "2026-08-19T00:00:00",
    }


def _heavy_pressure_case(tmp_dir: str, entities: list[dict] | None = None) -> dict:
    """A solid-fill square reliably satisfies EN_COMPILED_LINE_HEAVY_PRESSURE_030
    (aggression_or_threat_related, evidence_family=line_intensity_quality) --
    the same fixture already used elsewhere in this suite."""
    case_dir = Path(tmp_dir) / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "drawing.png"
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    image.save(path)
    analyze_image_with_timing(str(path), str(case_dir), None)
    (case_dir / "detections.json").write_text(
        json.dumps({"entities": entities or []}), encoding="utf-8")
    return build_live_case_bundle(case_dir)


def _with_emotion(bundle: dict, top_class: str, probabilities: dict) -> dict:
    return {**bundle, "emotion": {
        "status": "available", "probabilities": probabilities, "top_class": top_class,
        "confidence": max(probabilities.values()), "calibration_status": "calibrated",
        "model_name": "efficientnet_b0", "model_version": "seed42_calibrated",
        "checkpoint_sha256": "test-sha",
    }}


HAPPY_PROBS = {"Happy": 0.7, "Sad": 0.1, "Fear": 0.1, "Angry": 0.1}
SAD_PROBS = {"Happy": 0.1, "Sad": 0.7, "Fear": 0.1, "Angry": 0.1}


class EmotionAvailabilityTests(unittest.TestCase):
    def test_no_checkpoint_produces_no_fabricated_probabilities(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            self.assertEqual(bundle["emotion"]["status"], "unavailable")
            interp = ci.build_case_interpretation(bundle)
            self.assertEqual(interp.expressive_profile.availability, "unavailable")
            self.assertIsNone(interp.expressive_profile.base_emotion_probabilities)
            self.assertIsNone(interp.expressive_profile.top_emotion)

    def test_available_checkpoint_propagates_real_probabilities(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            interp = ci.build_case_interpretation(bundle)
            self.assertEqual(interp.expressive_profile.availability, "available")
            self.assertEqual(interp.expressive_profile.base_emotion_probabilities, HAPPY_PROBS)
            self.assertEqual(interp.expressive_profile.top_emotion, "Happy")

    def test_happy_classification_is_never_worded_as_the_child_is_happy(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            interp = ci.build_case_interpretation(bundle)
            text = interp.final_synthesis.lower()
            self.assertNotIn("the child is happy", text)
            self.assertNotIn("the child feels happy", text)

    def test_sad_classification_is_never_worded_as_depression(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Sad", SAD_PROBS)
            interp = ci.build_case_interpretation(bundle)
            text = interp.final_synthesis.lower()
            self.assertNotIn("depressed", text)
            self.assertNotIn("the child is sad", text)
            self.assertNotIn("has depression", text)


class VisualConceptTests(unittest.TestCase):
    def test_every_concept_has_traceable_supporting_evidence_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d, [_entity_dict("e-sun", "sun", "verified")])
            interp = ci.build_case_interpretation(bundle)
            self.assertTrue(interp.visual_concepts)
            for c in interp.visual_concepts:
                self.assertTrue(c.provenance, f"{c.concept_name} has no provenance")
                self.assertTrue(c.source, f"{c.concept_name} has no source")

    def test_entity_concept_traces_to_the_real_entity_id(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d, [_entity_dict("e-sun", "sun", "verified")])
            interp = ci.build_case_interpretation(bundle)
            sun_concepts = [c for c in interp.visual_concepts if c.concept_name == "sun"]
            self.assertEqual(len(sun_concepts), 1)
            self.assertEqual(sun_concepts[0].supporting_evidence, ("e-sun",))
            self.assertEqual(sun_concepts[0].status, "verified")

    def test_unsupported_concept_is_never_invented(self):
        """No facial-expression detector exists in DOAR today (its rules
        are allowed_output_level=disabled) -- a bare rectangle drawing with
        no entities must never produce a "smiling"/"frowning" concept."""
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            interp = ci.build_case_interpretation(bundle)
            names = {c.concept_name for c in interp.visual_concepts}
            self.assertFalse(any("smil" in n or "frown" in n for n in names))


class RicherExpressiveDescriptorTests(unittest.TestCase):
    def test_descriptor_can_exist_without_a_same_named_psychological_rule(self):
        """'positive'/'negative' are not rule_ids or concern domains --
        they may still appear, backed only by the emotion classifier."""
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Sad", SAD_PROBS)
            interp = ci.build_case_interpretation(bundle)
            labels = {d.label for d in interp.expressive_profile.richer_descriptors}
            self.assertIn("negative", labels)

    def test_every_descriptor_has_a_non_empty_reason(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            interp = ci.build_case_interpretation(bundle)
            for desc in interp.expressive_profile.richer_descriptors:
                self.assertTrue(desc.reason.strip())

    def test_cheerful_requires_real_positive_affect_evidence_not_just_happy(self):
        with tempfile.TemporaryDirectory() as d:
            # No sun/flower/heart entity, no positive_affect rule match --
            # "cheerful" must not fire even though the classifier is Happy.
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            interp = ci.build_case_interpretation(bundle)
            labels = {d.label for d in interp.expressive_profile.richer_descriptors}
            self.assertNotIn("cheerful", labels)
            self.assertIn("positive", labels)


class ConcernDomainTests(unittest.TestCase):
    def test_maltreatment_domain_is_always_not_currently_assessed(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            interp = ci.build_case_interpretation(bundle)
            by_domain = {c.domain: c for c in interp.concern_domains}
            self.assertEqual(by_domain["maltreatment_or_safety_concern"].support_level,
                              "NOT_CURRENTLY_ASSESSED")

    def test_unmapped_domains_are_not_currently_assessed(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            interp = ci.build_case_interpretation(bundle)
            by_domain = {c.domain: c for c in interp.concern_domains}
            for domain in ("trauma_related", "bullying_related", "autism_related"):
                self.assertEqual(by_domain[domain].support_level, "NOT_CURRENTLY_ASSESSED")

    def test_single_supporting_family_is_weak_not_insufficient(self):
        """One assessable, satisfied evidence family (out of one assessable
        family total) -- INDEPENDENT_FAMILY_RATIO = 1/1 -> WEAK.
        INSUFFICIENT is reserved for zero assessable families at all
        (nothing DOAR could even check), a distinct case."""
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)  # no emotion, no entities -- one rule only
            interp = ci.build_case_interpretation(bundle)
            by_domain = {c.domain: c for c in interp.concern_domains}
            aggression = by_domain["aggression_or_threat_related"]
            self.assertEqual(aggression.support_level, "WEAK")
            self.assertEqual(aggression.independent_evidence_families, ("line_intensity_quality",))

    def test_emotion_classifier_never_strengthens_an_unrelated_concern_domain(self):
        """IMPORTANT (explicit requirement): the general Happy/Sad/Fear/Angry
        classifier must NOT automatically strengthen a concern domain --
        concern-domain support comes ONLY from domain-relevant governed
        rule evidence. A single-family rule match stays WEAK regardless of
        whether an (irrelevant) emotion classification is also available."""
        with tempfile.TemporaryDirectory() as d:
            without_emotion = ci.build_case_interpretation(_heavy_pressure_case(d))
        with tempfile.TemporaryDirectory() as d:
            with_emotion = ci.build_case_interpretation(_with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS))
        by_domain_a = {c.domain: c for c in without_emotion.concern_domains}
        by_domain_b = {c.domain: c for c in with_emotion.concern_domains}
        self.assertEqual(by_domain_a["aggression_or_threat_related"].support_level,
                          by_domain_b["aggression_or_threat_related"].support_level)
        self.assertEqual(by_domain_a["aggression_or_threat_related"].independent_evidence_families,
                          by_domain_b["aggression_or_threat_related"].independent_evidence_families)

    def test_concern_result_exposes_why(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            interp = ci.build_case_interpretation(bundle)
            for c in interp.concern_domains:
                self.assertTrue(c.plain_language_reason.strip())

    def test_disabled_rules_are_missing_evidence_not_negative_evidence(self):
        """depressive_or_low_mood_related's rules are ALL allowed_output_
        level=disabled today (no built detector) -- that must surface as
        INSUFFICIENT (0 assessable families) with the reasons recorded in
        missing_expected_evidence, never as NONE/evidence-of-absence."""
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            interp = ci.build_case_interpretation(bundle)
            by_domain = {c.domain: c for c in interp.concern_domains}
            depression = by_domain["depressive_or_low_mood_related"]
            self.assertEqual(depression.support_level, "INSUFFICIENT")
            self.assertTrue(depression.missing_expected_evidence)


class EvidenceAggregatorTests(unittest.TestCase):
    """Requirement: a small, pluggable EvidenceAggregator interface: only
    INDEPENDENT_FAMILY_RATIO is implemented today; no arbitrary point
    weights anywhere."""

    def test_independent_family_ratio_basic_math(self):
        agg = ci.IndependentFamilyRatioAggregator()
        result = agg.aggregate(assessable_families={"a", "b", "c", "d"}, supporting_families={"a", "b"})
        self.assertEqual(result.support_ratio, 0.5)
        self.assertEqual(result.supporting_family_count, 2)
        self.assertEqual(result.assessable_family_count, 4)
        self.assertEqual(result.support_level, "MODERATE")

    def test_zero_assessable_families_is_insufficient_with_no_ratio(self):
        agg = ci.IndependentFamilyRatioAggregator()
        result = agg.aggregate(assessable_families=set(), supporting_families=set())
        self.assertEqual(result.support_level, "INSUFFICIENT")
        self.assertIsNone(result.support_ratio)

    def test_zero_supporting_of_assessable_is_none_not_negative(self):
        agg = ci.IndependentFamilyRatioAggregator()
        result = agg.aggregate(assessable_families={"a", "b"}, supporting_families=set())
        self.assertEqual(result.support_level, "NONE")
        self.assertEqual(result.support_ratio, 0.0)

    def test_one_of_one_is_weak(self):
        agg = ci.IndependentFamilyRatioAggregator()
        result = agg.aggregate(assessable_families={"a"}, supporting_families={"a"})
        self.assertEqual(result.support_level, "WEAK")
        self.assertEqual(result.support_ratio, 1.0)

    def test_build_concern_domains_accepts_a_custom_aggregator(self):
        class _AlwaysModerate(ci.EvidenceAggregator):
            def aggregate(self, *, assessable_families, supporting_families):
                if not assessable_families:
                    return ci.AggregationResult("INSUFFICIENT", None, 0, 0, (), ())
                return ci.AggregationResult("MODERATE", 1.0, len(assessable_families), len(assessable_families),
                                            tuple(assessable_families), tuple(assessable_families))

        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            domains = ci.build_concern_domains(bundle, aggregator=_AlwaysModerate())
            by_domain = {c.domain: c for c in domains}
            self.assertEqual(by_domain["aggression_or_threat_related"].support_level, "MODERATE")


class ConsistencyStateMachineTests(unittest.TestCase):
    """Unit tests against `_consistency_status` directly, with hand-crafted
    domain-support inputs -- isolates the CONSISTENT/MIXED/CONFLICT/
    INSUFFICIENT_EVIDENCE decision logic from the real rule-matching
    pipeline's own incidental match count for a given drawing."""

    def test_sad_with_zero_corroborating_concern_is_mixed(self):
        self.assertEqual(ci._consistency_status("Sad", {}), "MIXED")

    def test_happy_with_only_weak_negative_concern_is_mixed(self):
        support = {"anxiety_or_stress_related": "WEAK"}
        self.assertEqual(ci._consistency_status("Happy", support), "MIXED")

    def test_happy_with_moderate_negative_concern_is_conflict(self):
        support = {"depressive_or_low_mood_related": "MODERATE"}
        self.assertEqual(ci._consistency_status("Happy", support), "CONFLICT")

    def test_sad_with_corroborating_concern_is_consistent(self):
        support = {"depressive_or_low_mood_related": "MODERATE"}
        self.assertEqual(ci._consistency_status("Sad", support), "CONSISTENT")

    def test_no_emotion_and_no_concern_is_insufficient_evidence(self):
        self.assertEqual(ci._consistency_status(None, {}), "INSUFFICIENT_EVIDENCE")


class ConsistencyRealPipelineTests(unittest.TestCase):

    def test_single_family_local_match_is_mixed_not_conflict_with_positive_emotion(self):
        """A lone single-family match (the heavy-pressure fixture) must
        never alone reach CONFLICT with a Happy classification -- CONFLICT
        requires >=2 independent families (see
        ConsistencyStateMachineTests for the CONFLICT-reaching case,
        constructed directly against `_consistency_status` since real
        drawings that satisfy two specific deterministic thresholds at
        once are fragile to construct from raw pixels)."""
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            interp = ci.build_case_interpretation(bundle)
            self.assertEqual(interp.consistency_status, "MIXED")
            self.assertIn("some disagreement", interp.final_synthesis)

    def test_consistent_when_nothing_contradicts_the_classifier(self):
        """A blank canvas still yields one real (WEAK) deterministic
        finding in this codebase's own line-pressure proxy -- with no
        emotion classifier to disagree with (neutral valence), that is
        CONSISTENT (nothing contradicts), not INSUFFICIENT_EVIDENCE
        (which is reserved for genuinely zero evidence on both sides --
        see ConsistencyStateMachineTests.test_no_emotion_and_no_concern_
        is_insufficient_evidence for that exact case, unit-tested
        directly rather than via a fragile real-image fixture)."""
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            path = case_dir / "drawing.png"
            Image.new("RGB", (300, 300), "white").save(path)
            analyze_image_with_timing(str(path), str(case_dir), None)
            (case_dir / "detections.json").write_text(json.dumps({"entities": []}), encoding="utf-8")
            bundle = build_live_case_bundle(case_dir)
            interp = ci.build_case_interpretation(bundle)
            self.assertEqual(interp.consistency_status, "CONSISTENT")

    def test_two_independent_families_with_disagreeing_emotion_is_conflict(self):
        """Real end-to-end CONFLICT case: fabricate a domain support dict
        with 2 independent families (as INDEPENDENT_FAMILY_RATIO would
        report for a genuinely multi-family-supported domain) and confirm
        the synthesis text reflects it -- exercises `_final_synthesis`
        directly since crafting a real image that satisfies two distinct
        deterministic thresholds simultaneously is fragile/threshold-
        dependent, while the aggregation and wording logic themselves are
        what this test is actually verifying."""
        profile = ci.ExpressiveProfile(
            availability="available", unavailable_reason=None, base_emotion_probabilities=HAPPY_PROBS,
            top_emotion="Happy", calibration_status="calibrated", model_identity={}, richer_descriptors=(),
            supporting_visual_evidence=())
        concerns = (ci.ConcernDomainResult(
            domain="depressive_or_low_mood_related", domain_label="Possible depressive/low-mood presentation",
            support_level="MODERATE", support_ratio=1.0, assessable_family_count=2,
            supporting_observations=("face_expression", "dark_colors_sad_isolation"),
            supporting_rule_ids=("EN_COMPILED_FACE_EXPRESSION_021", "EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038"),
            independent_evidence_families=("facial_feature_style", "colour_mood_flags"),
            contradictory_evidence=(), missing_expected_evidence=(), alternative_explanations=(), sources=(),
            plain_language_reason="2 of 2 assessable independent evidence families support this domain."),)
        global_impression = ci.GlobalImpression(
            overall_scene="x", expressive_tone="Happy", supporting_global_evidence=(),
            supporting_local_concepts=(), contradictions=(), availability="available", provenance={})
        consistency = ci._consistency_status("Happy", {"depressive_or_low_mood_related": "MODERATE"})
        global_local = ci._global_local_consistency(global_impression, "Happy",
                                                     {"depressive_or_low_mood_related": "MODERATE"})
        synthesis = ci._final_synthesis(profile, concerns, global_impression, consistency, global_local)
        self.assertEqual(consistency, "CONFLICT")
        self.assertIn("needs closer review", synthesis)


class GlobalImpressionTests(unittest.TestCase):
    """Requirement: a whole-image `global_impression` branch, built from
    composition + relational entity layout + the expressive classifier --
    never from one isolated local feature."""

    def test_global_impression_reflects_whole_drawing_not_one_feature(self):
        with tempfile.TemporaryDirectory() as d:
            entities = [_entity_dict("e-sun", "sun", "verified"), _entity_dict("e-person", "person", "verified")]
            for e in entities:
                e["bbox"] = [0.1, 0.1, 0.2, 0.2]
            bundle = _heavy_pressure_case(d, entities)
            interp = ci.build_case_interpretation(bundle)
            g = interp.global_impression
            self.assertEqual(g.availability, "available")
            # Built from >1 kind of whole-image evidence, not a single rule/entity.
            self.assertGreaterEqual(len(g.supporting_global_evidence), 2)
            self.assertIn("composition.bounding_box_coverage", g.supporting_global_evidence)
            self.assertIn("relationships.entity_count", g.supporting_global_evidence)
            self.assertIn("2", g.overall_scene)  # entity_count=2 mentioned in the whole-scene sentence

    def test_global_impression_is_pluggable(self):
        class _StubGlobalProvider(ci.GlobalImageRepresentationProvider):
            def describe(self, bundle):
                return {"overall_scene": "STUB whole-image description.", "evidence": ["stub_model_output"]}

        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            interp = ci.build_case_interpretation(bundle, global_provider=_StubGlobalProvider())
            self.assertEqual(interp.global_impression.overall_scene, "STUB whole-image description.")
            self.assertEqual(interp.global_impression.provenance["provider"], "_StubGlobalProvider")
            self.assertIn("STUB whole-image description.", interp.final_synthesis)

    def test_visual_judge_audit_is_audit_only_never_changes_scene_or_tone(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            baseline = ci.build_case_interpretation(bundle)
            audited = ci.build_case_interpretation(bundle, visual_judge_audit={
                "status": "ok",
                "audit": {"supported_observations": [], "likely_missed_observations": ["a tree"],
                          "disputed_observations": ["the sun"], "unavailable_checks": [],
                          "overall_agreement": "partial"},
            })
            self.assertEqual(baseline.global_impression.overall_scene, audited.global_impression.overall_scene)
            self.assertEqual(baseline.global_impression.expressive_tone, audited.global_impression.expressive_tone)
            self.assertEqual(baseline.consistency_status, audited.consistency_status)
            # ...but the audit note IS surfaced, clearly labeled, for human review.
            self.assertTrue(any("AUDIT_ONLY" in c for c in audited.global_impression.contradictions))
            self.assertIn("visual_consistency_judge(AUDIT_ONLY)", audited.global_impression.supporting_global_evidence)


class GlobalLocalConsistencyStateMachineTests(unittest.TestCase):
    def _global(self, availability="available", contradictions=()):
        return ci.GlobalImpression(
            overall_scene="x", expressive_tone="Happy", supporting_global_evidence=("x",),
            supporting_local_concepts=(), contradictions=contradictions, availability=availability,
            provenance={"provider": "test"})

    def test_agreeing_global_and_local_is_global_local_consistent(self):
        g = self._global()
        result = ci._global_local_consistency(g, "Sad", {"depressive_or_low_mood_related": "MODERATE"})
        self.assertEqual(result, "GLOBAL_LOCAL_CONSISTENT")

    def test_weak_disagreement_is_global_local_mixed_not_conflict(self):
        g = self._global()
        result = ci._global_local_consistency(g, "Happy", {"anxiety_or_stress_related": "WEAK"})
        self.assertEqual(result, "GLOBAL_LOCAL_MIXED")

    def test_strong_independent_disagreement_is_global_local_conflict(self):
        g = self._global()
        result = ci._global_local_consistency(g, "Happy", {"depressive_or_low_mood_related": "STRONG"})
        self.assertEqual(result, "GLOBAL_LOCAL_CONFLICT")

    def test_unavailable_global_impression_is_insufficient_evidence(self):
        g = self._global(availability="unavailable")
        result = ci._global_local_consistency(g, "Happy", {})
        self.assertEqual(result, "INSUFFICIENT_EVIDENCE")

    def test_flagged_contradiction_without_strong_concern_support_is_mixed_not_conflict(self):
        """A single AUDIT_ONLY/global note alone -- with no >=2-family
        concern-domain corroboration -- must never escalate to CONFLICT."""
        g = self._global(contradictions=("AUDIT_ONLY: something looked off.",))
        result = ci._global_local_consistency(g, "Happy", {"anxiety_or_stress_related": "WEAK"})
        self.assertEqual(result, "GLOBAL_LOCAL_MIXED")


class GroundingSummaryTests(unittest.TestCase):
    def test_summarize_for_grounding_only_includes_notable_domains(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _with_emotion(_heavy_pressure_case(d), "Happy", HAPPY_PROBS)
            interp = ci.build_case_interpretation(bundle)
            summary = ci.summarize_for_grounding(interp)
            domains = {d["domain_label"] for d in summary["notable_concern_domains"]}
            self.assertTrue(domains)
            self.assertEqual(summary["consistency_status"], interp.consistency_status)
            self.assertIn("global_local_consistency", summary)
            self.assertIn("global_overall_scene", summary)

    def test_case_interpretation_reaches_ask_doar_evidence_package(self):
        from doar import human_interaction as hi
        with tempfile.TemporaryDirectory() as d:
            bundle = _heavy_pressure_case(d)
            sa = hi.answer_question("What did you find in this drawing?", bundle)
            self.assertIsInstance(sa.answer, str)  # smoke: build_evidence_package didn't crash


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ParentAndPsychologistUiTests(unittest.TestCase):
    def _build_case(self, tmp_dir: str) -> Path:
        case_dir = Path(tmp_dir) / "case"
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / "drawing.png"
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
        image.save(path)
        analyze_image_with_timing(str(path), str(case_dir), None)
        (case_dir / "detections.json").write_text(json.dumps({"entities": []}), encoding="utf-8")
        return case_dir

    def test_parent_view_states_overall_feeling_and_why_with_no_raw_ids(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            parent_tab = at.tabs[0]
            headers = " ".join(s.value for s in parent_tab.subheader)
            # Pre-doctor stabilization pass: Parent View sections were
            # renamed/reordered (item I) -- "Overall feeling" -> "Overall
            # expressive impression", "Why DOAR thinks this" -> "What DOAR
            # noticed", "What else DOAR found" -> "Concern indicators",
            # "Conclusion" -> "Overall conclusion".
            self.assertIn("Overall expressive impression", headers)
            self.assertIn("What DOAR noticed", headers)
            self.assertIn("Concern indicators", headers)
            self.assertIn("Overall conclusion", headers)
            parent_text = " ".join(m.value for m in parent_tab.markdown)
            self.assertNotIn("EN_COMPILED_", parent_text)
            self.assertNotIn("weak_support", parent_text)
            self.assertNotIn("evidence_family", parent_text)

    def test_psychologist_view_shows_concern_domains_and_reasons(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            psych_tab = at.tabs[1]
            headers = " ".join(s.value for s in psych_tab.subheader)
            self.assertIn("Expressive profile", headers)
            self.assertIn("Concern-domain classifications", headers)
            self.assertIn("consistency", headers.lower())
            expander_labels = [e.label for e in psych_tab.expander]
            self.assertTrue(any("aggression" in lbl.lower() or "threat" in lbl.lower()
                                or "depress" in lbl.lower() or "anxiety" in lbl.lower()
                                for lbl in expander_labels))
            self.assertIn("Global (whole-drawing) impression", headers)


if __name__ == "__main__":
    unittest.main()
