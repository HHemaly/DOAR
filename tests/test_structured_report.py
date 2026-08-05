"""Tests for structured_report.py (DOAR-TRACE Phase 1.5, Section 6):
Level A/B/C separation, the Phase-1 single-rule-as-theme bug fix,
dependency-aware aggregation, ordinal levels, the expressive-content
model evidence family and its mandatory wording, contradiction
preservation, and real end-to-end structured_analysis.json generation
without any LLM.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.construct_registry_build import build_construct_registry
from doar.registry_v2_build import build_registry_v2
from doar.structured_report import (
    MODEL_MIN_CONFIDENCE_FOR_AGGREGATION, build_combined_hypotheses, build_individual_rule_suggestions,
    detect_cross_theme_contradictions,
)

REGISTRY_V2 = build_registry_v2()
RULES_V2_BY_ID = {r["rule_id"]: r for r in REGISTRY_V2["rules"]}
CONSTRUCT_REGISTRY = build_construct_registry()
CONSTRUCTS_BY_ID = {c["construct_id"]: c for c in CONSTRUCT_REGISTRY["constructs"]}


def _rule_eval(rule_id, status="weak_support", evidence_ids=None, missing=None):
    return {
        "rule_id": rule_id, "status": status,
        "matched_evidence_ids": evidence_ids or [], "missing_evidence": missing or [],
        "references": [], "limitations": [],
    }


def _emotion(status="unavailable", top_class=None, confidence=None):
    return {"status": status, "top_class": top_class, "confidence": confidence, "calibration_status": "temperature_scaled"}


class IndividualRuleSuggestionsTests(unittest.TestCase):
    def test_triggered_rule_produces_one_suggestion(self):
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", evidence_ids=["ev_bbox_coverage"])]
        suggestions = build_individual_rule_suggestions(evals, RULES_V2_BY_ID)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["rule_id"], "PSY_AR_SIZE_FULL_015")

    def test_not_triggered_rule_produces_no_suggestion(self):
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", status="not_matched")]
        self.assertEqual(build_individual_rule_suggestions(evals, RULES_V2_BY_ID), [])

    def test_missing_detector_rule_produces_no_suggestion(self):
        evals = [_rule_eval("PSY_AR_EYES_WIDE_001", status="missing_detector")]
        self.assertEqual(build_individual_rule_suggestions(evals, RULES_V2_BY_ID), [])

    def test_suggestion_carries_all_required_fields(self):
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", evidence_ids=["ev_bbox_coverage"])]
        suggestion = build_individual_rule_suggestions(evals, RULES_V2_BY_ID)[0]
        required = {
            "rule_id", "observable", "possible_interpretation", "evidence_level_as_written",
            "evidence_family", "target_construct", "evidence_ids", "source_citation", "limitations",
            "alternative_explanations", "parent_safe_wording", "professional_wording",
            "contextual_question", "reference_ids", "requires_clinician_review",
        }
        self.assertEqual(set(suggestion.keys()), required)


class CombinedHypothesesTests(unittest.TestCase):
    def test_single_rule_alone_never_produces_a_combined_hypothesis(self):
        # THE Phase-1 bug fix: a full-page rule alone must not produce a
        # combined "visual_dominance_or_prominence" (formerly "self_esteem") theme.
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", evidence_ids=["ev_bbox_coverage"])]
        hyps = build_combined_hypotheses(evals, RULES_V2_BY_ID, _emotion(), CONSTRUCTS_BY_ID)
        self.assertEqual(hyps, [])

    def test_two_rules_from_the_same_dependency_group_do_not_combine(self):
        # Both rules cite the SAME evidence_id (ev_bbox_coverage) -- even if
        # they mapped to the same construct, this must not count as 2
        # independent evidence IDs. (In the real registry these two rules
        # map to different constructs anyway; this proves the mechanism
        # directly using two same-construct-forced synthetic rule evals.)
        evals = [
            _rule_eval("PSY_AR_SIZE_SMALL_016", evidence_ids=["ev_bbox_coverage"]),
            _rule_eval("PSY_AR_SIZE_FULL_015", evidence_ids=["ev_bbox_coverage"]),
        ]
        # Force both onto the same construct+family for this test only.
        rules = dict(RULES_V2_BY_ID)
        rules["PSY_AR_SIZE_FULL_015"] = {**rules["PSY_AR_SIZE_FULL_015"], "target_construct": "fear_or_insecurity_pattern"}
        hyps = build_combined_hypotheses(evals, rules, _emotion(), CONSTRUCTS_BY_ID)
        self.assertEqual(hyps, [])  # only 1 distinct evidence_id -> below the >=2 threshold

    def test_size_rule_plus_confident_matching_model_output_produces_level_2(self):
        evals = [_rule_eval("PSY_AR_SIZE_SMALL_016", evidence_ids=["ev_bbox_coverage"])]
        emotion = _emotion("available", "Fear", 0.85)
        hyps = build_combined_hypotheses(evals, RULES_V2_BY_ID, emotion, CONSTRUCTS_BY_ID)
        self.assertEqual(len(hyps), 1)
        hyp = hyps[0]
        self.assertEqual(hyp["target_construct"], "fear_or_insecurity_pattern")
        self.assertEqual(hyp["ordinal_level"], 2)
        self.assertEqual(hyp["level_label"], "possible_pattern")
        self.assertTrue(hyp["uses_expressive_model"])
        self.assertEqual(set(hyp["contributing_evidence_families"]), {"size_composition", "global_expressive_content_model"})

    def test_low_confidence_model_output_is_suppressed_from_aggregation(self):
        evals = [_rule_eval("PSY_AR_SIZE_SMALL_016", evidence_ids=["ev_bbox_coverage"])]
        emotion = _emotion("available", "Fear", MODEL_MIN_CONFIDENCE_FOR_AGGREGATION - 0.01)
        hyps = build_combined_hypotheses(evals, RULES_V2_BY_ID, emotion, CONSTRUCTS_BY_ID)
        self.assertEqual(hyps, [])

    def test_model_output_mapping_to_a_different_construct_does_not_combine(self):
        evals = [_rule_eval("PSY_AR_SIZE_SMALL_016", evidence_ids=["ev_bbox_coverage"])]  # fear_or_insecurity_pattern
        emotion = _emotion("available", "Happy", 0.9)  # positive_affective_tone
        hyps = build_combined_hypotheses(evals, RULES_V2_BY_ID, emotion, CONSTRUCTS_BY_ID)
        self.assertEqual(hyps, [])

    def test_expressive_model_wording_never_says_the_child_is(self):
        evals = [_rule_eval("PSY_AR_SIZE_SMALL_016", evidence_ids=["ev_bbox_coverage"])]
        emotion = _emotion("available", "Fear", 0.85)
        hyp = build_combined_hypotheses(evals, RULES_V2_BY_ID, emotion, CONSTRUCTS_BY_ID)[0]
        combined_text = " ".join(hyp["supporting_texts"])
        self.assertNotIn("the child is", combined_text.lower())
        self.assertIn("visual model found the drawing most similar", combined_text)

    def test_three_independent_families_reach_level_3(self):
        # Synthetic: 3 distinct families mapped to the same construct via a
        # patched registry, to prove the ordinal-level computation itself
        # (today's real rule set cannot reach 3 families on any real image).
        rules = dict(RULES_V2_BY_ID)
        rules["PSY_AR_PLACE_TOP_017"] = {**rules["PSY_AR_PLACE_TOP_017"], "target_construct": "fear_or_insecurity_pattern",
                                          "evidence_family": "spatial_placement"}
        evals = [
            _rule_eval("PSY_AR_SIZE_SMALL_016", evidence_ids=["ev_bbox_coverage"]),
            _rule_eval("PSY_AR_PLACE_TOP_017", evidence_ids=["ev_centroid"]),
        ]
        emotion = _emotion("available", "Fear", 0.9)
        hyps = build_combined_hypotheses(evals, rules, emotion, CONSTRUCTS_BY_ID)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0]["ordinal_level"], 3)
        self.assertEqual(hyps[0]["level_label"], "converging_drawing_pattern")

    def test_speculative_only_evidence_cannot_trigger_a_serious_construct_alone(self):
        # Force two Speculative-graded contributions onto a serious
        # construct (tension_or_anger_pattern) with no ungraded (Arabic) or
        # stronger-graded contributor -- must be suppressed.
        rules = dict(RULES_V2_BY_ID)
        rules["EN_COMPILED_ANIMAL_CHOICE_GENERAL_022"] = {
            **rules["EN_COMPILED_ANIMAL_CHOICE_GENERAL_022"],
            "target_construct": "tension_or_anger_pattern", "evidence_level_as_written": ["Speculative"],
        }
        rules["PSY_AR_STARS_009"] = {
            **rules["PSY_AR_STARS_009"],
            "target_construct": "tension_or_anger_pattern", "evidence_level_as_written": ["Speculative"],
        }
        evals = [
            _rule_eval("EN_COMPILED_ANIMAL_CHOICE_GENERAL_022", evidence_ids=["ev_a"]),
            _rule_eval("PSY_AR_STARS_009", evidence_ids=["ev_b"]),
        ]
        hyps = build_combined_hypotheses(evals, rules, _emotion(), CONSTRUCTS_BY_ID)
        self.assertEqual(hyps, [])


class ContradictionDetectionTests(unittest.TestCase):
    def test_opposing_constructs_both_present_are_flagged_not_dropped(self):
        hyps = [
            {"target_construct": "low_mood_or_emotional_distress_pattern", "contributing_rule_ids": ["r1"]},
            {"target_construct": "positive_affective_tone", "contributing_rule_ids": ["r2"]},
        ]
        contradictions = detect_cross_theme_contradictions(hyps)
        self.assertEqual(len(contradictions), 1)
        self.assertIn("not_resolved", contradictions[0]["resolution"])

    def test_no_contradiction_when_only_one_side_present(self):
        hyps = [{"target_construct": "low_mood_or_emotional_distress_pattern", "contributing_rule_ids": ["r1"]}]
        self.assertEqual(detect_cross_theme_contradictions(hyps), [])


class StructuredAnalysisEndToEndTests(unittest.TestCase):
    """Real analyze_image -> structured_analysis.json, no LLM, no mocking."""

    def _analyze(self, image: Image.Image, checkpoint=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        out = Path(temp.name) / "out"
        analyze_image(path, out, emotion_checkpoint=checkpoint)
        return out

    def test_full_page_drawing_alone_stays_individual_never_combined(self):
        # Phase 2A migration note: the original synthetic image (a 200x200
        # canvas with only a 2px white margin) satisfied coverage_full
        # (>=0.90 bounding-box coverage) but is now correctly gated
        # `not_assessable` by page_frame.py -- a 2px margin cannot pass the
        # border-uniformity check, and a real finding from this phase is
        # that coverage_full (>=0.90) and a genuinely detectable full-page
        # margin are close to mutually exclusive under this heuristic's ~3%
        # border band (see docs/PAGE_FRAME_ASSESSABILITY.md).
        #
        # Phase 2A.1 migration note: a centered 300x300/40px-margin
        # rectangle (this test's first migration) now ALSO satisfies
        # coverage_full's redefined margin-based condition (Section 4,
        # docs/PAGE_COVERAGE_DEFINITION_DECISION.md) as well as
        # placement_center -- 2 independent rules/families genuinely
        # converging on `visual_dominance_or_prominence`, correctly
        # promoted to a real Level-C hypothesis. That is CORRECT new
        # behavior, not a regression -- see
        # test_two_independent_rules_do_combine_under_the_redefined_coverage_full
        # below, which asserts exactly that. This test's own intent (a
        # single triggered rule stays individual, never combined) is
        # migrated again to an OFF-CENTER rectangle, which triggers
        # coverage_about_half + 2 unmapped placement rules
        # (target_construct=None, so neither can ever combine) instead.
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 209, 209), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        self.assertTrue(any(s["rule_id"] == "PSY_AR_SIZE_HALF_014" for s in doc["individual_rule_suggestions"]))
        self.assertEqual(doc["combined_drawing_level_hypotheses"], [])

    def test_two_independent_rules_do_combine_under_the_redefined_coverage_full(self):
        # Real, correct new behavior introduced by Phase 2A.1 Section 4:
        # a centered, page-frame-assessable rectangle whose margins are
        # all <=15% now satisfies BOTH the redefined coverage_full
        # (approaches all 4 margins) and placement_center -- 2
        # independent rules, 2 independent evidence families
        # (size_composition + spatial_placement), both mapping to
        # visual_dominance_or_prominence -- a genuine Level-C
        # convergence, the first ever reachable via 2 registry rules
        # alone (previously only the expressive model could supply a
        # second family; see docs/AGGREGATION_POLICY.md).
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        hyps = doc["combined_drawing_level_hypotheses"]
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0]["target_construct"], "visual_dominance_or_prominence")
        self.assertEqual(
            set(hyps[0]["contributing_rule_ids"]),
            {"PSY_AR_SIZE_FULL_015", "EN_COMPILED_PLACEMENT_CENTER_029"},
        )

    def test_unavailable_detectors_are_explicit_not_omitted(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((60, 60, 140, 140), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["detections"]["status"], "unavailable")
        self.assertIn("reason", doc["detections"])
        eyes_rule = next(r for r in doc["rule_evaluations"] if r["rule_id"] == "PSY_AR_EYES_WIDE_001")
        self.assertEqual(eyes_rule["status"], "missing_detector")
        self.assertEqual(eyes_rule["allowed_output_level"], "disabled")

    def test_all_required_top_level_keys_present(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((60, 60, 140, 140), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        required = {
            "quality", "segmentation", "objective_features", "detections", "model_output",
            "expressive_model_observation", "individual_rule_suggestions", "rule_evaluations",
            "combined_drawing_level_hypotheses", "cross_theme_contradictions", "references",
            "limitations", "suggested_parent_questions", "missing_evidence",
        }
        self.assertTrue(required <= set(doc.keys()), required - set(doc.keys()))

    def test_no_clinical_probability_is_invented(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        text = json.dumps(doc)
        for forbidden in ("probability of diagnosis", "% likely to have", "clinical_probability"):
            self.assertNotIn(forbidden, text)

    def test_no_checkpoint_run_has_no_expressive_model_observation(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((60, 60, 140, 140), fill="black")
        out = self._analyze(image, checkpoint=None)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        self.assertIsNone(doc["expressive_model_observation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
