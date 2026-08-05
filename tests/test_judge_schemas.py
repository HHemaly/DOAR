"""Tests for judge_schemas.py (DOAR-TRACE 4G): shared JudgeVerdict schema,
real judges wrapping judges.py's tested logic, and the 4 interface-only
judges honestly reporting not_implemented (never a fabricated pass)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.judges import run_judges
from doar.judge_schemas import (
    JUDGE_IDS, JudgeVerdict, aggregation_judge_v2, detection_judge_v2, feature_judge_v2,
    language_judge_v2, model_judge_v2, quality_judge_v2, relation_judge_v2, rule_judge_v2,
    run_all_judges_v2,
)


def _real_case():
    temp = tempfile.TemporaryDirectory()
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
    path = Path(temp.name) / "image.png"
    image.save(path)
    out = Path(temp.name) / "out"
    result = analyze_image(path, out)
    analysis = result.to_dict()
    judges_output = run_judges(analysis)
    return temp, analysis, judges_output


class JudgeVerdictSchemaTests(unittest.TestCase):
    def test_valid_verdict_constructs(self):
        verdict = JudgeVerdict(
            judge_id="quality_judge", target="x", status="pass", confidence=None,
            reasons=["ok"], limitations=[], version="v1",
        )
        self.assertEqual(verdict.status, "pass")

    def test_unknown_judge_id_raises(self):
        with self.assertRaises(ValueError):
            JudgeVerdict(judge_id="not_a_real_judge", target="x", status="pass",
                         confidence=None, reasons=["ok"], limitations=[], version="v1")

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            JudgeVerdict(judge_id="quality_judge", target="x", status="maybe",
                         confidence=None, reasons=["ok"], limitations=[], version="v1")

    def test_not_implemented_without_reason_raises(self):
        with self.assertRaises(ValueError):
            JudgeVerdict(judge_id="detection_judge", target="x", status="not_implemented",
                         confidence=None, reasons=[], limitations=[], version="v1")

    def test_to_dict_round_trips(self):
        verdict = JudgeVerdict(judge_id="rule_judge", target="x", status="pass",
                               confidence=0.5, reasons=["a"], limitations=["b"], version="v1")
        data = verdict.to_dict()
        self.assertEqual(data["judge_id"], "rule_judge")
        self.assertEqual(data["confidence"], 0.5)


class RealJudgeWrapperTests(unittest.TestCase):
    """These 4 judges wrap judges.py's already-tested real logic -- their
    status must match the real underlying judge, not be independently
    invented."""

    def setUp(self):
        self.temp, self.analysis, self.judges_output = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_quality_judge_v2_matches_real_status(self):
        verdict = quality_judge_v2(self.analysis, self.judges_output)
        self.assertEqual(verdict.status, self.judges_output["quality_judge"]["status"])
        self.assertTrue(verdict.reasons)

    def test_feature_judge_v2_matches_real_status(self):
        verdict = feature_judge_v2(self.analysis, self.judges_output)
        self.assertEqual(verdict.status, self.judges_output["feature_judge"]["status"])

    def test_rule_judge_v2_matches_real_status(self):
        verdict = rule_judge_v2(self.analysis, self.judges_output)
        self.assertEqual(verdict.status, self.judges_output["rule_judge"]["status"])

    def test_model_judge_v2_matches_real_status_and_carries_confidence(self):
        verdict = model_judge_v2(self.analysis, self.judges_output)
        # No emotion checkpoint was supplied in _real_case -> real emotion
        # judge status is "unavailable", mapped to "requires_review".
        self.assertEqual(self.judges_output["emotion_judge"]["status"], "unavailable")
        self.assertEqual(verdict.status, "requires_review")
        # No fabricated confidence when the model never ran.
        self.assertIsNone(verdict.confidence)


class InterfaceOnlyJudgeTests(unittest.TestCase):
    """These 4 judges must NEVER report pass/fail -- only not_implemented,
    with an honest reason, since no underlying capability exists."""

    def setUp(self):
        self.temp, self.analysis, self.judges_output = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_detection_judge_is_not_implemented(self):
        verdict = detection_judge_v2(self.analysis)
        self.assertEqual(verdict.status, "not_implemented")
        self.assertTrue(verdict.reasons)

    def test_relation_judge_is_not_implemented(self):
        verdict = relation_judge_v2(self.analysis)
        self.assertEqual(verdict.status, "not_implemented")

    def test_aggregation_judge_is_not_implemented_with_no_structured_analysis(self):
        verdict = aggregation_judge_v2(None)
        self.assertEqual(verdict.status, "not_implemented")

    def test_language_judge_is_not_implemented(self):
        verdict = language_judge_v2("some claim text")
        self.assertEqual(verdict.status, "not_implemented")

    def test_none_of_the_three_interface_only_judges_ever_report_pass(self):
        for factory, arg in (
            (detection_judge_v2, self.analysis), (relation_judge_v2, self.analysis), (language_judge_v2, "x"),
        ):
            verdict = factory(arg)
            self.assertNotEqual(verdict.status, "pass", factory.__name__)
            self.assertNotEqual(verdict.status, "fail", factory.__name__)


class AggregationJudgeOperationalTests(unittest.TestCase):
    """DOAR-TRACE Phase 1.5 Section 8: aggregation_judge is now real."""

    def test_no_combined_hypotheses_passes_trivially(self):
        verdict = aggregation_judge_v2({"combined_drawing_level_hypotheses": [], "cross_theme_contradictions": []})
        self.assertEqual(verdict.status, "pass")

    def test_a_valid_combined_hypothesis_passes(self):
        structured = {
            "combined_drawing_level_hypotheses": [{
                "target_construct": "fear_or_insecurity_pattern",
                "contributing_evidence_families": ["size_composition", "global_expressive_content_model"],
                "contributing_evidence_ids": ["ev_bbox_coverage", "ev_emotion_prediction"],
                "contributing_rule_ids": ["PSY_AR_SIZE_SMALL_016"],
                "uses_expressive_model": True,
                "ordinal_level": 2,
            }],
            "cross_theme_contradictions": [],
        }
        verdict = aggregation_judge_v2(structured)
        self.assertEqual(verdict.status, "pass", verdict.reasons)

    def test_seeded_single_contributor_hypothesis_is_caught(self):
        # A hypothesis whose only real contributor is one rule (no model,
        # no second rule) should never have been promoted -- the judge
        # must catch it even if it slipped past the aggregator itself.
        structured = {
            "combined_drawing_level_hypotheses": [{
                "target_construct": "fear_or_insecurity_pattern",
                "contributing_evidence_families": ["size_composition", "spatial_placement"],
                "contributing_evidence_ids": ["ev_bbox_coverage", "ev_centroid"],
                "contributing_rule_ids": ["PSY_AR_SIZE_SMALL_016"],  # only 1 real rule
                "uses_expressive_model": False,
                "ordinal_level": 2,
            }],
            "cross_theme_contradictions": [],
        }
        verdict = aggregation_judge_v2(structured)
        self.assertEqual(verdict.status, "fail")
        self.assertTrue(any("single contributor" in r for r in verdict.reasons))

    def test_seeded_repeated_evidence_id_is_caught(self):
        structured = {
            "combined_drawing_level_hypotheses": [{
                "target_construct": "fear_or_insecurity_pattern",
                "contributing_evidence_families": ["size_composition", "global_expressive_content_model"],
                "contributing_evidence_ids": ["ev_bbox_coverage", "ev_bbox_coverage"],  # duplicated
                "contributing_rule_ids": ["PSY_AR_SIZE_SMALL_016"],
                "uses_expressive_model": True,
                "ordinal_level": 2,
            }],
            "cross_theme_contradictions": [],
        }
        verdict = aggregation_judge_v2(structured)
        self.assertEqual(verdict.status, "fail")
        self.assertTrue(any("double-counting" in r for r in verdict.reasons))

    def test_seeded_below_threshold_family_count_is_caught(self):
        structured = {
            "combined_drawing_level_hypotheses": [{
                "target_construct": "fear_or_insecurity_pattern",
                "contributing_evidence_families": ["size_composition"],  # only 1 family
                "contributing_evidence_ids": ["ev_bbox_coverage", "ev_x"],
                "contributing_rule_ids": ["PSY_AR_SIZE_SMALL_016"],
                "uses_expressive_model": False,
                "ordinal_level": 2,
            }],
            "cross_theme_contradictions": [],
        }
        verdict = aggregation_judge_v2(structured)
        self.assertEqual(verdict.status, "fail")

    def test_seeded_omitted_contradiction_is_caught(self):
        structured = {
            "combined_drawing_level_hypotheses": [
                {"target_construct": "low_mood_or_emotional_distress_pattern",
                 "contributing_evidence_families": ["facial_feature_style", "global_expressive_content_model"],
                 "contributing_evidence_ids": ["ev_a", "ev_b"], "contributing_rule_ids": ["r1"],
                 "uses_expressive_model": True, "ordinal_level": 2},
                {"target_construct": "positive_affective_tone",
                 "contributing_evidence_families": ["shape_symbolism", "global_expressive_content_model"],
                 "contributing_evidence_ids": ["ev_c", "ev_d"], "contributing_rule_ids": ["r2"],
                 "uses_expressive_model": True, "ordinal_level": 2},
            ],
            "cross_theme_contradictions": [],  # omitted despite the two constructs being opposites
        }
        verdict = aggregation_judge_v2(structured)
        self.assertEqual(verdict.status, "fail")
        self.assertTrue(any("contradiction" in r for r in verdict.reasons))


class RunAllJudgesV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp, self.analysis, self.judges_output = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_returns_exactly_the_eight_required_judges(self):
        verdicts = run_all_judges_v2(self.analysis, self.judges_output)
        self.assertEqual(set(verdicts.keys()), set(JUDGE_IDS))
        self.assertEqual(len(verdicts), 8)

    def test_every_verdict_is_a_valid_judgeverdict(self):
        verdicts = run_all_judges_v2(self.analysis, self.judges_output)
        for judge_id, verdict in verdicts.items():
            self.assertIsInstance(verdict, JudgeVerdict)
            self.assertEqual(verdict.judge_id, judge_id)

    def test_exactly_three_are_not_implemented_without_structured_analysis(self):
        # aggregation_judge is real but reports not_implemented when no
        # structured_analysis is supplied (as here) -- so 3 not_implemented
        # (detection/relation/language) + aggregation_judge (not_implemented
        # here specifically because of missing input, not missing capability).
        verdicts = run_all_judges_v2(self.analysis, self.judges_output)
        not_implemented = {j for j, v in verdicts.items() if v.status == "not_implemented"}
        self.assertEqual(not_implemented, {"detection_judge", "relation_judge", "aggregation_judge", "language_judge"})

    def test_aggregation_judge_becomes_operational_when_structured_analysis_supplied(self):
        structured = {"combined_drawing_level_hypotheses": [], "cross_theme_contradictions": []}
        verdicts = run_all_judges_v2(self.analysis, self.judges_output, structured_analysis=structured)
        self.assertEqual(verdicts["aggregation_judge"].status, "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
