"""Focused tests for the Human Interaction Layer v1 (src/doar/human_interaction.py
+ its wiring into scripts/clinician_review_app.py's Parent/Clinician views and
Ask DOAR chat). Covers the acceptance criteria from Part I of the task spec:
Parent View, Clinician View, and the 6 required Ask DOAR scenarios. Uses only
real, already-cached development-set data (h38 primarily) -- no live Gemini
calls, no mutation of any cached case file.

This module NEVER asserts anything about rule eligibility, convergence, or
hypothesis logic itself -- those are exercised by the existing drawing_synthesis/
reasoning_chain test suites. Everything here is about the presentation/Q&A
layer built on top of that already-frozen output.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from doar import human_interaction as hi  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

try:
    import clinician_review_app as app
    _APP_AVAILABLE = _STREAMLIT_TESTING_AVAILABLE
except ImportError:  # pragma: no cover - environment-dependent
    _APP_AVAILABLE = False


def _h38_bundle():
    return app.load_case_bundle("h38")


# ---------------------------------------------------------------------------
# Part A -- Parent View builder-function tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available in this environment")
class ObservationBulletsTests(unittest.TestCase):
    def setUp(self):
        self.bundle = _h38_bundle()

    def test_confirmed_observations_are_case_specific(self):
        obs = hi.observation_bullets(self.bundle)
        joined = " | ".join(obs["confirmed"])
        self.assertIn("Sun", joined)
        self.assertIn("Traffic light", joined)
        for line in obs["confirmed"]:
            self.assertIn("confirmed", line)

    def test_unreviewed_car_is_not_silently_erased(self):
        """h38's 'car' candidate stayed unreviewed (see
        h38_perception_diagnostic.json) -- it must still be visible in a
        compact note, never hidden entirely."""
        obs = hi.observation_bullets(self.bundle)
        self.assertIsNotNone(obs["not_confirmed_note"])
        self.assertIn("car", obs["not_confirmed_note"])

    def test_no_person_object_invented_for_h38(self):
        """People are visible to a human looking at h38, but were never
        proposed by the Observer -- the bullets must not invent them."""
        obs = hi.observation_bullets(self.bundle)
        joined = " ".join(obs["confirmed"] + obs["possible"])
        self.assertNotIn("person", joined.lower())
        self.assertNotIn("people", joined.lower())


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available in this environment")
class EvidenceChainsTests(unittest.TestCase):
    def setUp(self):
        self.bundle = _h38_bundle()
        self.chains = hi.build_evidence_chains(self.bundle)

    def test_at_least_one_chain_for_h38(self):
        self.assertTrue(self.chains)

    def test_rule_shown_in_human_wording_not_raw_id(self):
        for c in self.chains:
            self.assertNotEqual(c["rule_display_name"], c["rule_id"])
            self.assertNotIn("EN_COMPILED", c["rule_display_name"])
            self.assertNotIn("PSY_AR", c["rule_display_name"])

    def test_reference_is_accessible_for_h38s_matched_rule(self):
        for c in self.chains:
            src = c["source"]
            self.assertTrue(src["citation_title"] or src["source_pdf_page_section"],
                             "matched rule must expose SOME real source field, not silently empty")

    def test_rule_id_still_present_for_technical_details_but_not_required_up_front(self):
        for c in self.chains:
            self.assertTrue(c["rule_id"])  # available for the "Technical details" expander
            self.assertTrue(c["rule_suggests"])  # human explanation stands on its own

    def test_unmatched_verified_observation_shown_with_honest_reason_not_hidden(self):
        unmatched = hi.unmatched_verified_observations(self.bundle)
        for u in unmatched:
            self.assertIn("No approved DOAR psychological interpretation", u["reason"])


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available in this environment")
class OverallInterpretationTests(unittest.TestCase):
    def test_names_the_specific_finding_not_a_generic_template_only(self):
        """h38 is a single-chain limited_association case -- the summary
        must name the actual matched feature, not just a bland 'nothing
        conclusive' boilerplate that would be true of any case."""
        bundle = _h38_bundle()
        text = hi.overall_interpretation_text(bundle)
        chains = hi.build_evidence_chains(bundle)
        self.assertEqual(len(chains), 1)
        self.assertIn(chains[0]["rule_display_name"].casefold(), text.casefold())
        self.assertIn("stress", text.lower())


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available in this environment")
class CaseSpecificQuestionsTests(unittest.TestCase):
    def test_questions_are_grounded_in_this_drawings_own_content(self):
        bundle = _h38_bundle()
        questions = hi.case_specific_questions(bundle)
        joined = " ".join(questions).lower()
        self.assertIn("tell me about the sun", joined)

    def test_questions_are_not_leading(self):
        bundle = _h38_bundle()
        questions = hi.case_specific_questions(bundle)
        for q in questions:
            self.assertNotIn("why", q.lower())
            self.assertNotIn("sad", q.lower())
            self.assertNotIn("upset", q.lower())


class FooterDisclaimerTests(unittest.TestCase):
    def test_footer_text_matches_spec_and_is_non_diagnostic(self):
        self.assertIn("not definitive diagnoses", hi.FOOTER_DISCLAIMER)
        self.assertFalse(hi.is_unsupported_diagnostic_claim(hi.FOOTER_DISCLAIMER))


# ---------------------------------------------------------------------------
# Part B -- Clinician View / Ask DOAR rendering tests (real AppTest run)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class AppRenderTests(unittest.TestCase):
    def _run_app(self):
        at = AppTest.from_file(str(ROOT / "scripts" / "clinician_review_app.py"))
        at.run(timeout=90)
        self.assertFalse(at.exception, at.exception)
        return at

    def test_app_renders_without_exception(self):
        self._run_app()

    def test_footer_disclaimer_appears_exactly_once(self):
        """One footer only (Part A) -- the disclaimer is a single
        st.caption call in the whole app, so it must appear exactly once
        no matter how many feature cards are rendered."""
        at = self._run_app()
        captions = [c.value for c in at.caption]
        self.assertEqual(captions.count(hi.FOOTER_DISCLAIMER), 1)

    def test_ask_doar_present_in_both_parent_and_clinician_views(self):
        at = self._run_app()
        subheaders = [s.value for s in at.subheader]
        self.assertEqual(subheaders.count("Ask DOAR"), 2)

    def test_evidence_rule_association_chain_visible_in_clinician_view(self):
        at = self._run_app()
        subheaders = [s.value for s in at.subheader]
        self.assertIn("3. Evidence -> rule -> association", subheaders)

    def test_parent_view_has_all_six_sections(self):
        at = self._run_app()
        subheaders = [s.value for s in at.subheader]
        for expected in ("What we noticed", "What these features may suggest",
                          "Overall interpretation", "Sources and rules",
                          "Suggested questions to ask the child"):
            self.assertIn(expected, subheaders)


# ---------------------------------------------------------------------------
# Part C/F -- Ask DOAR: the 6 required Q&A scenarios
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available in this environment")
class AskDoarScenarioTests(unittest.TestCase):
    """Numbered exactly per the task spec's Part I list of 6 scenarios."""

    def setUp(self):
        self.bundle = _h38_bundle()

    def test_1_case_question_uses_actual_matched_evidence_and_rule(self):
        answer = hi.answer_question("Why did you mention stress?", self.bundle)
        self.assertEqual(answer.category, "case")
        self.assertEqual(answer.judge_verdict, "PASS")
        claim = answer.claims[0]
        self.assertIn("EN_COMPILED_LINE_LIGHT_PRESSURE_031", claim["rule_ids"])
        self.assertIn("ev_feature_stroke_intensity_proxy", claim["evidence_ids"])
        self.assertIn("stress", answer.answer.lower())
        self.assertNotIn("Your child has", answer.answer)

    def test_2_rule_source_question_returns_a_real_reference(self):
        answer = hi.answer_question("What is the source for the light-line rule?", self.bundle)
        self.assertEqual(answer.category, "rule_source")
        self.assertEqual(answer.judge_verdict, "PASS")
        self.assertIn("child_drawing_rules_compiled.pdf", answer.answer)

    def test_3_general_question_without_internal_or_external_support_is_honest(self):
        answer = hi.answer_question("What does red mean?", self.bundle)
        self.assertEqual(answer.category, "general")
        self.assertEqual(answer.judge_verdict, "PASS")
        self.assertIn("couldn't find reliable evidence", answer.answer)
        self.assertFalse(answer.used_external_research)

    def test_4_is_my_child_depressed_answers_directly_without_diagnosing(self):
        answer = hi.answer_question("Is my child depressed?", self.bundle)
        self.assertEqual(answer.judge_verdict, "PASS")
        lower = answer.answer.lower()
        # Must directly engage with the question (not a blanket refusal
        # just because "depressed" appears in it)...
        self.assertNotEqual(answer.answer, hi._FALLBACK_ANSWER)
        self.assertIn("depress", lower)
        # ...but must never assert a diagnosis.
        self.assertFalse(hi.is_unsupported_diagnostic_claim(answer.answer))
        self.assertNotIn("your child has depression", lower)
        self.assertNotIn("is depressed.", lower)

    def test_5_visual_question_with_missing_saved_object_invokes_on_demand_path(self):
        def _stub_predict_fn(image_path, query):
            return []  # honest "found nothing" -- never tuned toward any h38 outcome

        answer = hi.answer_question("Are there people in the car?", self.bundle,
                                     open_vocab_predict_fn=_stub_predict_fn)
        self.assertEqual(answer.category, "visual")
        self.assertTrue(answer.used_visual_recheck)
        self.assertIn(hi._Q_AND_A_VISUAL_RECHECK_TAG, answer.provenance)

    def test_6a_judge_rejects_fabricated_evidence_rule_and_source_ids(self):
        fabricated = {
            "answer": "This is fully supported by the evidence.", "category": "case",
            "evidence_ids": ["FAKE_EVIDENCE_999"], "rule_ids": ["FAKE_RULE_999"],
            "source_ids": ["FAKE_RULE_999"], "used_external_research": False,
            "used_visual_recheck": False,
        }
        verification = hi.deterministic_verify(fabricated, self.bundle)
        self.assertFalse(verification["all_passed"])
        resolved = hi.build_structured_answer("Why?", "case", fabricated, self.bundle)
        # Never leaks the fabricated claim to the caller -- always resolves
        # to PASS via the documented fallback, never loops indefinitely.
        self.assertEqual(resolved.judge_verdict, "PASS")
        self.assertEqual(resolved.answer, hi._FALLBACK_ANSWER)
        self.assertTrue(resolved.judge_reasons)

    def test_6b_judge_rejects_external_research_mislabeled_as_original(self):
        mislabeled = {
            "answer": "Red is strongly linked to anger in children's drawings.",
            "category": "general", "evidence_ids": [], "rule_ids": [], "source_ids": [],
            "used_external_research": True, "used_visual_recheck": False,
        }
        verification = hi.deterministic_verify(mislabeled, self.bundle)
        self.assertFalse(verification["all_passed"])
        self.assertIn("external research used but not labeled as separate from the original DOAR analysis",
                       verification["extra_checks_failed"])

    def test_6c_judge_rejects_unsupported_diagnostic_claim(self):
        diagnostic = {
            "answer": "The child is depressed.", "category": "case",
            "evidence_ids": [], "rule_ids": [], "source_ids": [],
            "used_external_research": False, "used_visual_recheck": False,
        }
        verification = hi.deterministic_verify(diagnostic, self.bundle)
        self.assertFalse(verification["all_passed"])
        self.assertIn("unsupported diagnostic language in answer text", verification["extra_checks_failed"])
        resolved = hi.build_structured_answer("Is my child depressed?", "case", diagnostic, self.bundle)
        self.assertEqual(resolved.judge_verdict, "PASS")
        self.assertFalse(hi.is_unsupported_diagnostic_claim(resolved.answer))

    def test_6d_non_answering_answer_is_resolved_not_leaked(self):
        off_topic = {
            "answer": "Bananas are a good source of potassium.", "category": "case",
            "evidence_ids": [], "rule_ids": [], "source_ids": [],
            "used_external_research": False, "used_visual_recheck": False,
        }
        resolved = hi.build_structured_answer("Why did you mention stress?", "case", off_topic, self.bundle)
        self.assertEqual(resolved.judge_verdict, "PASS")
        self.assertNotEqual(resolved.answer, off_topic["answer"])


# ---------------------------------------------------------------------------
# Negation-aware diagnostic-language safety wrapper -- explicit tests per
# this task's own instruction ("Add explicit tests for this behavior").
# ---------------------------------------------------------------------------


class NegationAwareSafetyTests(unittest.TestCase):
    def test_bare_diagnostic_claim_is_blocked(self):
        self.assertTrue(hi.is_unsupported_diagnostic_claim("The child is depressed."))

    def test_negated_explanatory_statement_is_allowed(self):
        self.assertFalse(hi.is_unsupported_diagnostic_claim(
            "This drawing alone is not enough to say that the child is depressed."))

    def test_association_wording_is_allowed(self):
        self.assertFalse(hi.is_unsupported_diagnostic_claim(
            "This rule is associated with depression-related indicators."))

    def test_unrelated_sentence_before_a_negated_one_does_not_leak_a_block(self):
        text = "The drawing uses bright colours. This is not enough to say the child is anxious."
        self.assertFalse(hi.is_unsupported_diagnostic_claim(text))

    def test_arabic_bare_diagnostic_claim_is_blocked(self):
        self.assertTrue(hi.is_unsupported_diagnostic_claim("الطفل يعاني من اكتئاب."))

    def test_arabic_negated_statement_is_allowed(self):
        self.assertFalse(hi.is_unsupported_diagnostic_claim(
            "هذا لا يكفي لإثبات تشخيص الاكتئاب لدى الطفل."))


if __name__ == "__main__":
    unittest.main()
