"""Tests for structured_report.py (DOAR-TRACE 4D): dependency-aware
aggregation, no double counting, contradictory-evidence preservation,
unavailable evidence never rendered as absent, and real end-to-end
structured_analysis.json generation without any LLM.
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
from doar.registry_v2_build import build_registry_v2
from doar.structured_report import (
    aggregate_candidate_themes, detect_cross_theme_contradictions,
)

REGISTRY_V2 = build_registry_v2()
RULES_V2_BY_ID = {r["rule_id"]: r for r in REGISTRY_V2["rules"]}


def _rule_eval(rule_id, status="weak_support", evidence_ids=None, missing=None):
    return {
        "rule_id": rule_id, "status": status,
        "matched_evidence_ids": evidence_ids or [], "missing_evidence": missing or [],
        "references": [], "limitations": [],
    }


class AggregateThemesTests(unittest.TestCase):
    def test_single_triggered_rule_produces_question_generating_theme(self):
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", evidence_ids=["ev_bbox_coverage"])]
        themes = aggregate_candidate_themes(evals, RULES_V2_BY_ID)
        self.assertEqual(len(themes), 1)
        self.assertEqual(themes[0]["target_construct"], "self_esteem")
        self.assertEqual(themes[0]["allowed_output_level"], "question_generating")

    def test_not_triggered_rules_produce_no_theme(self):
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", status="not_matched")]
        themes = aggregate_candidate_themes(evals, RULES_V2_BY_ID)
        self.assertEqual(themes, [])

    def test_missing_detector_rules_produce_no_theme(self):
        evals = [_rule_eval("PSY_AR_EYES_WIDE_001", status="missing_detector")]
        themes = aggregate_candidate_themes(evals, RULES_V2_BY_ID)
        self.assertEqual(themes, [])

    def test_two_evidence_ids_two_families_same_construct_escalates_to_combined_hypothesis(self):
        # Synthetic: two rules mapped to the SAME target_construct via a
        # patched registry, from two different evidence families, to prove
        # the escalation threshold works (the real 19-rule registry never
        # has two rules sharing a construct today -- see
        # docs/RULE_PDF_COVERAGE_AUDIT.md).
        synthetic_registry = {
            "rule_a": {**RULES_V2_BY_ID["PSY_AR_SIZE_FULL_015"], "target_construct": "shared_theme",
                       "evidence_family": "size_composition"},
            "rule_b": {**RULES_V2_BY_ID["PSY_AR_PLACE_TOP_017"], "target_construct": "shared_theme",
                       "evidence_family": "spatial_placement"},
        }
        evals = [
            _rule_eval("rule_a", evidence_ids=["ev_bbox_coverage"]),
            _rule_eval("rule_b", evidence_ids=["ev_centroid"]),
        ]
        themes = aggregate_candidate_themes(evals, synthetic_registry)
        self.assertEqual(len(themes), 1)
        self.assertEqual(themes[0]["allowed_output_level"], "combined_hypothesis_only")
        self.assertEqual(set(themes[0]["supporting_evidence_ids"]), {"ev_bbox_coverage", "ev_centroid"})

    def test_same_evidence_id_repeated_does_not_double_count_toward_escalation(self):
        # Two rules citing the SAME evidence_id and the SAME family must not
        # be treated as two independent sources.
        synthetic_registry = {
            "rule_a": {**RULES_V2_BY_ID["PSY_AR_SIZE_FULL_015"], "target_construct": "shared_theme",
                       "evidence_family": "size_composition"},
            "rule_b": {**RULES_V2_BY_ID["PSY_AR_SIZE_SMALL_016"], "target_construct": "shared_theme",
                       "evidence_family": "size_composition"},
        }
        evals = [
            _rule_eval("rule_a", evidence_ids=["ev_bbox_coverage"]),
            _rule_eval("rule_b", evidence_ids=["ev_bbox_coverage"]),
        ]
        themes = aggregate_candidate_themes(evals, synthetic_registry)
        self.assertEqual(len(themes), 1)
        # Only 1 distinct evidence_id and 1 distinct family -> must NOT escalate.
        self.assertEqual(themes[0]["allowed_output_level"], "question_generating")
        self.assertEqual(themes[0]["supporting_evidence_ids"], ["ev_bbox_coverage"])

    def test_missing_evidence_from_blocked_rules_is_preserved_not_dropped(self):
        evals = [_rule_eval("PSY_AR_SIZE_FULL_015", evidence_ids=["ev_bbox_coverage"],
                             missing=["detector_absent:wide_eyes"])]
        themes = aggregate_candidate_themes(evals, RULES_V2_BY_ID)
        self.assertIn("detector_absent:wide_eyes", themes[0]["missing_evidence"])


class ContradictionDetectionTests(unittest.TestCase):
    def test_opposing_constructs_both_present_are_flagged_not_dropped(self):
        themes = [
            {"target_construct": "introversion", "supporting_rule_ids": ["PSY_AR_PLACE_LEFT_018"]},
            {"target_construct": "extraversion", "supporting_rule_ids": ["PSY_AR_PLACE_RIGHT_019"]},
        ]
        contradictions = detect_cross_theme_contradictions(themes)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(set((contradictions[0]["construct_a"], contradictions[0]["construct_b"])),
                          {"introversion", "extraversion"})
        self.assertIn("not_resolved", contradictions[0]["resolution"])

    def test_no_contradiction_when_only_one_side_present(self):
        themes = [{"target_construct": "introversion", "supporting_rule_ids": ["PSY_AR_PLACE_LEFT_018"]}]
        self.assertEqual(detect_cross_theme_contradictions(themes), [])

    def test_no_contradiction_for_unrelated_constructs(self):
        themes = [
            {"target_construct": "self_esteem", "supporting_rule_ids": ["a"]},
            {"target_construct": "loneliness", "supporting_rule_ids": ["b"]},
        ]
        self.assertEqual(detect_cross_theme_contradictions(themes), [])


class StructuredAnalysisEndToEndTests(unittest.TestCase):
    """Real analyze_image -> structured_analysis.json, no LLM, no mocking."""

    def _analyze(self, image: Image.Image):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        out = Path(temp.name) / "out"
        analyze_image(path, out)
        return out

    def test_full_page_drawing_produces_a_self_esteem_theme(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        constructs = {t["target_construct"] for t in doc["candidate_drawing_level_themes"]}
        self.assertIn("self_esteem", constructs)

    def test_unavailable_detectors_are_explicit_not_omitted(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((60, 60, 140, 140), fill="black")
        out = self._analyze(image)
        doc = json.loads((out / "structured_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["detections"]["status"], "unavailable")
        self.assertIn("reason", doc["detections"])
        # tier-2 rules must still appear in rule_evaluations, tagged disabled
        # -- never silently missing from the list.
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
            "rule_evaluations", "candidate_drawing_level_themes", "references", "limitations",
            "suggested_parent_questions", "missing_evidence",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
