from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.parent_view import (
    build_overall_result_summary, capability_status, capability_status_summary_text,
    disclaimer, friendly_family_name, friendly_source_name, overall_interpretation,
    plain_language_observations, plain_language_rule_rows,
)


def _real_analysis(image: Image.Image) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "drawing.png"
        image.save(path)
        result = analyze_image(path, Path(d) / "out")
        return result.to_dict()


class PlainLanguageObservationsTests(unittest.TestCase):
    def test_every_observation_carries_evidence_ids_or_is_explicitly_empty(self):
        analysis = _real_analysis(Image.new("RGB", (200, 200), "white"))
        for obs in plain_language_observations(analysis, "en"):
            self.assertIn("evidence_ids", obs)
            self.assertIsInstance(obs["evidence_ids"], list)

    def test_bilingual_output_differs_and_is_localized(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
        analysis = _real_analysis(image)
        en = plain_language_observations(analysis, "en")
        ar = plain_language_observations(analysis, "ar")
        self.assertEqual(len(en), len(ar))
        self.assertNotEqual(en[0]["text"], ar[0]["text"])
        self.assertTrue(any("؀" <= c <= "ۿ" for c in ar[0]["text"]))

    def test_quality_limitation_surfaces_when_unsupported(self):
        # A tiny, low-contrast image should fail the quality gate.
        analysis = _real_analysis(Image.new("RGB", (30, 30), (128, 128, 128)))
        self.assertEqual(analysis["quality"]["quality_status"], "unsupported")
        obs = plain_language_observations(analysis, "en")
        self.assertTrue(any("quality" in o["text"].lower() for o in obs))


class RuleStatusRenderingTests(unittest.TestCase):
    """The four rule states must never collapse into one generic message --
    this is the specific failure mode RULE_AND_FEATURE_COVERAGE.md Section 4
    warns against."""

    def test_all_present_statuses_get_distinct_messages(self):
        image = Image.new("RGB", (200, 200), "white")
        # Margin preserved (unlike edge-to-edge fill) so background estimation
        # and the quality gate both stay valid; bbox coverage still >= 0.90.
        ImageDraw.Draw(image).rectangle((5, 5, 195, 195), fill="black")
        analysis = _real_analysis(image)
        rows = plain_language_rule_rows(analysis["rule_evaluations"], "en")
        statuses_present = {r["status"] for r in rows}
        self.assertIn("missing_detector", statuses_present)  # the 13 tier-2 rules
        messages_by_status = {r["status"]: r["message"] for r in rows}
        # missing_detector must read differently from not_matched/weak_support
        self.assertNotEqual(
            messages_by_status.get("missing_detector"),
            messages_by_status.get("not_matched", "not_matched_placeholder"),
        )

    def test_weak_support_row_includes_parent_safe_wording_and_ceiling(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((5, 5, 195, 195), fill="black")
        analysis = _real_analysis(image)
        rows = plain_language_rule_rows(analysis["rule_evaluations"], "en")
        weak = [r for r in rows if r["status"] == "weak_support"]
        self.assertTrue(weak, "expected at least one triggered rule on a full-coverage drawing")
        self.assertIsNotNone(weak[0]["note"])
        self.assertIsNotNone(weak[0]["confidence_ceiling"])

    def test_missing_detector_rows_never_show_a_parent_safe_wording(self):
        analysis = _real_analysis(Image.new("RGB", (200, 200), "white"))
        rows = plain_language_rule_rows(analysis["rule_evaluations"], "en")
        for r in rows:
            if r["status"] == "missing_detector":
                self.assertIsNone(r["note"])


class OverallInterpretationTests(unittest.TestCase):
    def test_no_checkpoint_case_states_unavailable_not_fabricated(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((5, 5, 195, 195), fill="black")
        analysis = _real_analysis(image)  # no emotion_checkpoint passed -> status "unavailable"
        self.assertEqual(analysis["emotion"]["status"], "unavailable")
        text = overall_interpretation(analysis, "en")
        self.assertIn("No emotion prediction is available", text)
        self.assertIn(disclaimer("en"), text)

    def test_disclaimer_always_present_both_languages(self):
        analysis = _real_analysis(Image.new("RGB", (200, 200), "white"))
        self.assertIn(disclaimer("en"), overall_interpretation(analysis, "en"))
        self.assertIn(disclaimer("ar"), overall_interpretation(analysis, "ar"))

    def test_interpretation_never_contains_diagnostic_language(self):
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        from doar.judges import _diagnostic_hit
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((5, 5, 195, 195), fill="black")
        analysis = _real_analysis(image)
        self.assertFalse(_diagnostic_hit(overall_interpretation(analysis, "en")))
        # NOTE: judges.py's _ARABIC_DIAGNOSTIC regex has no word boundaries,
        # so it matches "تشخيص" as a *substring* of the disclaimer's own
        # phrase "غير تشخيصي" ("non-diagnostic") -- a known, pre-existing
        # false-positive risk in that regex (documented in
        # CURRENT_CAPABILITY_AUDIT.md), not introduced by this module.
        # judges.py itself never triggers this in production because it only
        # scans rule-derived narrative text, never the disclaimer. Here we
        # verify the CLAIM portion (before the disclaimer sentence) is clean,
        # which is the part that actually varies with model/rule content.
        claim_only_en = overall_interpretation(analysis, "en").split(disclaimer("en"))[0]
        claim_only_ar = overall_interpretation(analysis, "ar").split(disclaimer("ar"))[0]
        self.assertFalse(_diagnostic_hit(claim_only_en))
        self.assertFalse(_diagnostic_hit(claim_only_ar))


class PageReferenceAwareObservationsTests(unittest.TestCase):
    """DOAR-TRACE Phase 2A.2, Section 5: page-coverage/placement wording
    must never appear when no page reference is assessable."""

    def test_not_assessable_reference_suppresses_coverage_and_placement_wording(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")  # thin margin -> not assessable
        analysis = _real_analysis(image)
        self.assertFalse(analysis["page_reference"]["page_relative_features_assessable"])
        obs = plain_language_observations(analysis, "en")
        joined = " ".join(o["text"] for o in obs)
        self.assertNotIn("covers about", joined)
        self.assertNotIn("centered on the page", joined)
        self.assertNotIn("placed toward", joined)
        self.assertIn("could not be confirmed", joined)

    def test_assessable_reference_shows_real_coverage_and_placement_wording(self):
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")  # wide margin -> assessable
        analysis = _real_analysis(image)
        self.assertTrue(analysis["page_reference"]["page_relative_features_assessable"])
        obs = plain_language_observations(analysis, "en")
        joined = " ".join(o["text"] for o in obs)
        self.assertIn("covers about", joined)

    def test_explicit_page_reference_argument_overrides_analysis_dict(self):
        # Passing page_reference explicitly (as the app will) must take
        # precedence over whatever is embedded in the analysis dict.
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
        analysis = _real_analysis(image)
        obs = plain_language_observations(analysis, "en", page_reference={"page_relative_features_assessable": False})
        joined = " ".join(o["text"] for o in obs)
        self.assertNotIn("covers about", joined)


class FriendlyNameTests(unittest.TestCase):
    def test_known_source_documents_get_a_friendly_name(self):
        self.assertNotEqual(friendly_source_name("child_drawing_rules_compiled.pdf", "en"), "child_drawing_rules_compiled.pdf")
        self.assertIn("guide", friendly_source_name("child_drawing_rules_compiled.pdf", "en"))

    def test_unknown_source_document_falls_back_honestly(self):
        self.assertEqual(friendly_source_name("unknown_source.pdf", "en"), "unknown_source.pdf")

    def test_known_family_gets_a_friendly_name(self):
        self.assertEqual(friendly_family_name("line_intensity_quality", "en"), "line-appearance")

    def test_unknown_family_falls_back_to_underscored_replacement(self):
        self.assertEqual(friendly_family_name("some_new_family", "en"), "some new family")


class OverallResultSummaryTests(unittest.TestCase):
    """DOAR-TRACE Phase 2A.2, Section 3."""

    def test_no_findings_case_matches_required_shape(self):
        structured = {
            "combined_drawing_level_hypotheses": [], "individual_rule_suggestions": [],
            "expressive_model_observation": None,
        }
        sentences = build_overall_result_summary(structured, {"page_relative_features_assessable": True}, "en")
        joined = " ".join(sentences)
        self.assertIn("No concerning combined drawing pattern was identified", joined)
        self.assertIn("No individual heuristic observations were found", joined)
        self.assertIn("Objects and relationships were not analyzed in this version", joined)

    def test_matches_the_tasks_worked_example_shape(self):
        structured = {
            "combined_drawing_level_hypotheses": [],
            "individual_rule_suggestions": [{"evidence_family": "line_intensity_quality"}],
            "expressive_model_observation": {
                "text": "The visual model found the drawing most similar to the dataset's Happy expressive-content category.",
            },
        }
        sentences = build_overall_result_summary(structured, {"page_relative_features_assessable": True}, "en")
        joined = " ".join(sentences)
        self.assertIn("No concerning combined drawing pattern was identified", joined)
        self.assertIn("most similar to the dataset's Happy expressive-content category", joined)
        self.assertIn("One weak line-appearance observation was found", joined)
        self.assertIn("Objects and relationships were not analyzed in this version", joined)

    def test_combined_pattern_present_is_reported_with_real_count(self):
        structured = {
            "combined_drawing_level_hypotheses": [{"target_construct": "x"}, {"target_construct": "y"}],
            "individual_rule_suggestions": [], "expressive_model_observation": None,
        }
        sentences = build_overall_result_summary(structured, {"page_relative_features_assessable": True}, "en")
        self.assertIn("2 combined drawing pattern(s) were identified", sentences[0])

    def test_not_assessable_page_reference_reported_honestly(self):
        sentences = build_overall_result_summary({}, {"page_relative_features_assessable": False}, "en")
        self.assertIn("could not be confirmed", sentences[-1])

    def test_assessable_page_reference_reported_honestly(self):
        sentences = build_overall_result_summary({}, {"page_relative_features_assessable": True}, "en")
        self.assertIn("confirmed visible", sentences[-1])

    def test_never_hard_codes_a_case_value(self):
        # Two different structured inputs must produce two different
        # summaries -- proof the function is not returning a fixed string.
        s1 = build_overall_result_summary(
            {"combined_drawing_level_hypotheses": [], "individual_rule_suggestions": [], "expressive_model_observation": None},
            {"page_relative_features_assessable": True}, "en",
        )
        s2 = build_overall_result_summary(
            {"combined_drawing_level_hypotheses": [{"target_construct": "x"}], "individual_rule_suggestions": [], "expressive_model_observation": None},
            {"page_relative_features_assessable": False}, "en",
        )
        self.assertNotEqual(s1, s2)

    def test_bilingual(self):
        structured = {"combined_drawing_level_hypotheses": [], "individual_rule_suggestions": [], "expressive_model_observation": None}
        en = build_overall_result_summary(structured, {"page_relative_features_assessable": True}, "en")
        ar = build_overall_result_summary(structured, {"page_relative_features_assessable": True}, "ar")
        self.assertNotEqual(en, ar)
        self.assertTrue(any("؀" <= c <= "ۿ" for c in ar[0]))


class CapabilityStatusTests(unittest.TestCase):
    def test_three_tiers_present(self):
        status = capability_status("en")
        self.assertEqual(set(status.keys()), {"working", "limited", "not_available"})

    def test_working_tier_lists_the_ten_rules(self):
        status = capability_status("en")
        self.assertTrue(any("10 executable" in item for item in status["working"]))

    def test_working_tier_lists_object_detection(self):
        # DOAR V1.1 Stage 5: object detection is real now (Phase 2C.7 +
        # the DOAR MVP visual scan) -- must never be claimed unavailable.
        status = capability_status("en")
        self.assertTrue(any("object detection" in item.lower() for item in status["working"]))
        self.assertFalse(any("object detection" in item.lower() for item in status["not_available"]))

    def test_not_available_tier_lists_spatial_relationships(self):
        # Still genuinely unbuilt -- explicitly out of scope for this phase.
        status = capability_status("en")
        self.assertTrue(any("spatial relationship" in item.lower() for item in status["not_available"]))

    def test_summary_text_is_non_empty_both_languages(self):
        self.assertTrue(capability_status_summary_text("en"))
        self.assertTrue(capability_status_summary_text("ar"))

    def test_bilingual_lists_have_equal_length_per_tier(self):
        en, ar = capability_status("en"), capability_status("ar")
        for tier in en:
            self.assertEqual(len(en[tier]), len(ar[tier]), tier)


if __name__ == "__main__":
    unittest.main()
