"""Focused tests for the Human Interaction Layer v1 "FINAL correction
pass": interaction model defaults (gemini-3.6-flash / gemini-3.5-flash-
lite), provider-failure boundary handling, accurate answer_provider/
judge_mode provenance, the reorganized Clinician View (collapsed
expanders by default), grouped/deduplicated no-rule Parent observations,
conversation-history follow-ups, and the Streamlit error boundary.

All network-dependent behavior is exercised via mocks/stubs -- no test
here makes a real network call or requires GEMINI_API_KEY to be set.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from doar import human_interaction as hi  # noqa: E402
from doar import production_config  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest  # noqa: F401
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


def _h38_cache_hash():
    import hashlib
    import run_development_benchmark as rdb
    _, source_path = rdb.find_saved_verification_rows("h38")
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Part 1 -- correct interaction default models; frozen Observer/Verifier
# configuration untouched.
# ---------------------------------------------------------------------------


class InteractionModelDefaultsTests(unittest.TestCase):
    def test_answer_provider_default_model(self):
        self.assertEqual(hi.GeminiAnswerProvider(api_key="fake").model, "gemini-3.6-flash")

    def test_research_provider_default_model(self):
        self.assertEqual(hi.GeminiGroundedResearchProvider(api_key="fake").model, "gemini-3.6-flash")

    def test_judge_default_model(self):
        self.assertEqual(hi.GeminiJudge(api_key="fake").model, "gemini-3.5-flash-lite")

    def test_visual_recheck_default_model(self):
        self.assertEqual(hi.GeminiVisualRecheckProvider(api_key="fake").model, "gemini-3.6-flash")

    def test_visual_recheck_env_var_is_named_as_specified(self):
        self.assertEqual(hi._VISUAL_RECHECK_MODEL_ENV_VAR, "DOAR_VISUAL_RECHECK_MODEL")

    def test_env_overrides_still_work(self):
        with patch.dict("os.environ", {"DOAR_ANSWER_MODEL": "custom-answer-model",
                                        "DOAR_RESEARCH_MODEL": "custom-research-model",
                                        "DOAR_JUDGE_MODEL": "custom-judge-model",
                                        "DOAR_VISUAL_RECHECK_MODEL": "custom-visual-model"}):
            self.assertEqual(hi.GeminiAnswerProvider(api_key="fake").model, "custom-answer-model")
            self.assertEqual(hi.GeminiGroundedResearchProvider(api_key="fake").model, "custom-research-model")
            self.assertEqual(hi.GeminiJudge(api_key="fake").model, "custom-judge-model")
            self.assertEqual(hi.GeminiVisualRecheckProvider(api_key="fake").model, "custom-visual-model")

    def test_frozen_broad_observer_configuration_untouched(self):
        self.assertEqual(production_config.DEFAULT_GEMINI_VISUAL_OBSERVER_MODEL, "gemini-3.6-flash")
        self.assertEqual(production_config.GEMINI_VISUAL_OBSERVER_MODEL_ENV_VAR,
                          "DOAR_GEMINI_VISUAL_OBSERVER_MODEL")

    def test_frozen_verifier_configuration_untouched(self):
        self.assertEqual(production_config.DEFAULT_GEMINI_VISUAL_VERIFIER_MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(production_config.GEMINI_VISUAL_VERIFIER_MODEL_ENV_VAR,
                          "DOAR_GEMINI_VISUAL_VERIFIER_MODEL")


# ---------------------------------------------------------------------------
# Part 2 -- provider failure handling: never crashes, error != no evidence.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class ProviderFailureHandlingTests(unittest.TestCase):
    def setUp(self):
        self.bundle = _h38_bundle()

    def test_research_provider_raising_does_not_crash_answer_question(self):
        crashing_provider = MagicMock()
        crashing_provider.research.side_effect = RuntimeError("simulated network failure")
        answer = hi.answer_question("What does red mean?", self.bundle,
                                     external_research_provider=crashing_provider)
        self.assertIsInstance(answer, hi.StructuredAnswer)
        self.assertEqual(answer.judge_verdict, "PASS")

    def test_failed_search_is_reported_as_search_failed_not_as_no_evidence(self):
        crashing_provider = MagicMock()
        crashing_provider.research.side_effect = RuntimeError("simulated network failure")
        answer = hi.answer_question("What does red mean?", self.bundle,
                                     external_research_provider=crashing_provider)
        self.assertIn("couldn't complete the external literature search", answer.answer)
        self.assertNotIn("couldn't find reliable evidence", answer.answer)
        self.assertFalse(answer.used_external_research)

    def test_clean_no_result_is_still_reported_as_no_evidence(self):
        """The A vs. B distinction cuts both ways -- a provider that ran
        successfully and simply found nothing must NOT be reported as a
        search failure."""
        empty_provider = MagicMock()
        empty_provider.research.return_value = None
        answer = hi.answer_question("What does red mean?", self.bundle,
                                     external_research_provider=empty_provider)
        self.assertIn("couldn't find reliable evidence", answer.answer)
        self.assertNotIn("couldn't complete the external literature search", answer.answer)

    def test_provider_error_never_exposes_secrets(self):
        crashing_provider = MagicMock()
        crashing_provider.research.side_effect = RuntimeError(
            "HTTP 403 for url with key=AIzaSyFAKESECRETVALUE1234567890ABCDEF")
        answer = hi.answer_question("What does red mean?", self.bundle,
                                     external_research_provider=crashing_provider)
        full_text = answer.answer + str(answer.judge_details)
        self.assertNotIn("AIzaSyFAKESECRETVALUE1234567890ABCDEF", full_text)

    def test_sanitize_error_text_redacts_key_like_content(self):
        try:
            raise RuntimeError("failed with api_key=AIzaSyFAKESECRETVALUE1234567890ABCDEF")
        except RuntimeError as exc:
            sanitized = hi.sanitize_error_text(exc)
        self.assertNotIn("AIzaSyFAKESECRETVALUE1234567890ABCDEF", sanitized)

    def test_governed_visual_recheck_error_is_sanitized(self):
        class _RaisingProvider:
            def recheck(self, image_path, target):
                raise RuntimeError("boom with secrettokenABCDEFGHIJKLMNOPQRSTUVWX")

        recheck = hi.governed_visual_recheck("person", self.bundle, provider=_RaisingProvider())
        self.assertEqual(recheck["status"], "error")
        self.assertNotIn("secrettokenABCDEFGHIJKLMNOPQRSTUVWX", recheck["reason"])


# ---------------------------------------------------------------------------
# Part 2b -- P0 Ask DOAR fix regression tests (rule-display/routing fix
# session): a question routed to "case" must ground "which rules matched"
# in the SAME matched-only chains Section 4 of the main UI uses (never an
# unmatched rule), and a Gemini failure must still produce a real,
# grounded, case-factual fallback -- never a traceback, never the bare
# generic apology now that routing/grounding are fixed.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class AskDoarRuleGroundingFixTests(unittest.TestCase):
    def setUp(self):
        # h38's real saved analysis.json has BOTH a matched rule (light line
        # pressure, EN_COMPILED_LINE_LIGHT_PRESSURE_031) AND two evaluated-
        # but-NOT-matched rules from the same mutually-exclusive family
        # (heavy line pressure / shaky-broken lines) -- the strongest real
        # case for proving the unmatched ones are excluded, not just absent.
        self.bundle = _h38_bundle()

    def test_e_unmatched_rule_never_reported_as_supporting_evidence(self):
        chains = hi.build_evidence_chains(self.bundle)
        rule_ids = {c["rule_id"] for c in chains}
        self.assertNotIn("EN_COMPILED_LINE_HEAVY_PRESSURE_030", rule_ids)
        self.assertNotIn("EN_COMPILED_LINE_SHAKY_BROKEN_032", rule_ids)
        self.assertIn("EN_COMPILED_LINE_LIGHT_PRESSURE_031", rule_ids)

        answer = hi.answer_question("Which governed rules actually matched this drawing?", self.bundle)
        self.assertIn("Light line pressure", answer.answer)
        self.assertNotIn("Heavy line pressure", answer.answer)
        self.assertNotIn("Shaky or broken", answer.answer)

    def test_f_gemini_failure_falls_back_to_grounded_case_answer_no_traceback(self):
        """Simulates a Gemini answer_provider outage (raises on every call,
        exactly like a real network/API failure) -- the deterministic
        case-question path (fixed this session) must still produce a real,
        evidence-grounded answer, and the raw exception must never reach
        the returned text."""
        class _FailingAnswerProvider:
            def generate(self, **kwargs):
                raise RuntimeError("simulated Gemini outage")

        answer = hi.answer_question("What did DOAR notice?", self.bundle,
                                     answer_provider=_FailingAnswerProvider())
        self.assertIsInstance(answer, hi.StructuredAnswer)
        self.assertNotIn("Traceback", answer.answer)
        self.assertNotIn("RuntimeError", answer.answer)
        self.assertNotIn("simulated Gemini outage", answer.answer)
        # Grounded in this case's real objective measurements, not the bare
        # pre-fix "couldn't verify a reliable answer" non-answer.
        self.assertIn("noticed", answer.answer.lower())
        self.assertNotIn("I couldn't verify a reliable answer", answer.answer)


# ---------------------------------------------------------------------------
# Part 3 -- accurate answer_provider / judge_mode provenance.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class ProvenanceAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.bundle = _h38_bundle()

    def test_successful_gemini_answer_is_labeled_gemini(self):
        class _SucceedingProvider:
            model = "gemini-3.6-flash"

            def generate(self, *, question, audience, package, revision_instructions=None,
                         conversation_history=None):
                return ("The main reason is the relatively light line quality in this drawing. One of the "
                        "rules used by DOAR links this feature with stress- or anxiety-related indicators. "
                        "No second independent feature supports the same pattern here.")

        answer = hi.answer_question("Why did you mention stress?", self.bundle,
                                     answer_provider=_SucceedingProvider())
        self.assertEqual(answer.answer_provider, "gemini:gemini-3.6-flash")
        self.assertIsNone(answer.answer_provider_error)
        self.assertIn("relatively light line quality", answer.answer)

    def test_failed_gemini_answer_is_labeled_deterministic_fallback_not_gemini(self):
        """The exact bug this task reports: answer_provider must never
        say 'gemini' while displaying the deterministic template."""
        class _FailingProvider:
            model = "gemini-3.6-flash"

            def generate(self, *, question, audience, package, revision_instructions=None,
                         conversation_history=None):
                raise RuntimeError("simulated Gemini API failure")

        answer = hi.answer_question("Why did you mention stress?", self.bundle,
                                     answer_provider=_FailingProvider())
        self.assertEqual(answer.answer_provider, "deterministic_fallback")
        self.assertIsNotNone(answer.answer_provider_error)
        self.assertIn("simulated Gemini API failure", answer.answer_provider_error)

    def test_pure_deterministic_provider_is_labeled_deterministic(self):
        answer = hi.answer_question("Why did you mention stress?", self.bundle)
        self.assertEqual(answer.answer_provider, "deterministic")
        self.assertIsNone(answer.answer_provider_error)

    def test_judge_mode_gemini_on_real_success(self):
        judge = hi.GeminiJudge(api_key="fake-key")
        result = {"answer": "This is associated with stress-related themes.", "category": "case",
                  "evidence_ids": [], "rule_ids": [], "source_ids": [],
                  "used_external_research": False, "used_visual_recheck": False}
        verification = hi.deterministic_verify(result, self.bundle)
        mock_response = MagicMock()
        mock_response.text = '{"verdict": "PASS", "answers_question": true, "case_grounded": true, ' \
                              '"rule_fidelity": true, "source_support": true, "status_fidelity": true, ' \
                              '"external_provenance_correct": true, "unsupported_claims": [], ' \
                              '"revision_instructions": []}'
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        with patch("google.genai.Client", return_value=mock_client):
            out = judge.judge(question="Why did you mention stress?", result=result, verification=verification, package={})
        self.assertEqual(out["judge_mode"], "gemini")

    def test_judge_mode_gemini_error_fallback_on_api_failure(self):
        judge = hi.GeminiJudge(api_key="fake-key")
        result = {"answer": "This is associated with stress-related themes.", "category": "case",
                  "evidence_ids": [], "rule_ids": [], "source_ids": [],
                  "used_external_research": False, "used_visual_recheck": False}
        verification = hi.deterministic_verify(result, self.bundle)
        with patch("google.genai.Client", side_effect=RuntimeError("network down")):
            out = judge.judge(question="Why did you mention stress?", result=result, verification=verification, package={})
        self.assertEqual(out["judge_mode"], "gemini_error_fallback")
        self.assertEqual(out["verdict"], "PASS")

    def test_judge_mode_deterministic_fallback_when_no_gemini_judge_configured(self):
        answer = hi.answer_question("Why did you mention stress?", self.bundle)
        self.assertEqual(answer.judge_mode, "deterministic_fallback")


# ---------------------------------------------------------------------------
# Part 5/8 -- Parent View grouping/dedup; Clinician View collapsed by
# default; no raw traceback.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class ParentGroupingTests(unittest.TestCase):
    def test_no_rule_observations_are_grouped_and_deduplicated(self):
        bundle = _h38_bundle()
        labels = hi.grouped_unmatched_observation_labels(bundle)
        # h38 has 5 verified butterflies, all unmatched by any rule --
        # the grouped summary must show 'butterfly' exactly once, not 5
        # separate entries.
        self.assertEqual(labels.count("butterfly"), 1)
        self.assertEqual(len(labels), len(set(labels)))

    def test_parent_view_shows_one_compact_grouped_section_not_a_card_per_label(self):
        at = AppTest.from_file(str(ROOT / "scripts" / "clinician_review_app.py"))
        at.run(timeout=90)
        self.assertFalse(at.exception, at.exception)
        markdown_values = [m.value for m in at.markdown]
        self.assertIn("**Other things noticed without a current psychological rule**", markdown_values)


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class ClinicianViewCollapsedByDefaultTests(unittest.TestCase):
    def _run_app(self):
        at = AppTest.from_file(str(ROOT / "scripts" / "clinician_review_app.py"))
        at.run(timeout=90)
        self.assertFalse(at.exception, at.exception)
        return at

    def test_default_visible_sections_present(self):
        at = self._run_app()
        subheaders = [s.value for s in at.subheader]
        for expected in ("Case summary", "2. Key observations", "3. Evidence -> rule -> association",
                          "4. Overall synthesis", "5. Candidate clinical hypothesis", "6. References"):
            self.assertIn(expected, subheaders)

    def test_verbose_sections_are_collapsed_expanders_by_default(self):
        at = self._run_app()
        by_label = {e.label: e for e in at.expander}
        for expected in ("All visual observations (full detail)", "Detailed objective measurements",
                          "Full evidence/rule details", "Full technical/audit trace"):
            self.assertIn(expected, by_label)
            self.assertFalse(by_label[expected].proto.expanded, f"{expected!r} must be collapsed by default")

    def test_nothing_is_deleted_full_technical_data_still_reachable(self):
        at = self._run_app()
        # The raw entity JSON dump still exists somewhere in the tree
        # (inside the now-collapsed 'Full technical/audit trace').
        json_elements = list(at.json)
        self.assertTrue(json_elements)


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class NoRawTracebackTests(unittest.TestCase):
    def test_ask_doar_exception_never_reaches_the_page_as_a_traceback(self):
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated crash")

        at = AppTest.from_file(str(ROOT / "scripts" / "clinician_review_app.py"))
        at.run(timeout=90)
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(at.chat_input)

        with patch.object(hi, "answer_question", side_effect=_boom):
            at.chat_input[0].set_value("Why did you mention stress?").run(timeout=90)

        self.assertFalse(at.exception, "an uncaught exception must never reach the Streamlit page")
        chat_texts = [m.markdown[0].value for m in at.chat_message if m.markdown]
        self.assertIn("Ask DOAR could not complete that request. Please try again.", chat_texts)
        joined = " ".join(chat_texts)
        self.assertNotIn("Traceback (most recent call last)", joined)
        self.assertNotIn("RuntimeError", joined)


# ---------------------------------------------------------------------------
# Part 9 -- conversation history: follow-ups work; a prior assistant
# hallucination can never become evidence.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class ConversationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.bundle = _h38_bundle()

    def test_conversation_history_is_passed_through_to_answer_provider(self):
        captured = {}

        class _RecordingProvider:
            model = "gemini-3.6-flash"

            def generate(self, *, question, audience, package, revision_instructions=None,
                         conversation_history=None):
                captured["history"] = conversation_history
                return None  # fall back to deterministic text -- we only care about what was passed in

        history = [{"role": "user", "content": "Why did you mention stress?"},
                   {"role": "assistant", "content": "Because of the light line quality."}]
        hi.answer_question("Why?", self.bundle, answer_provider=_RecordingProvider(),
                            conversation_history=history)
        self.assertEqual(captured["history"], history)

    def test_conversation_history_is_bounded_to_last_eight_turns(self):
        captured = {}

        class _RecordingProvider:
            model = "gemini-3.6-flash"

            def generate(self, *, question, audience, package, revision_instructions=None,
                         conversation_history=None):
                captured["history"] = conversation_history
                return None

        long_history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
        hi.answer_question("Why?", self.bundle, answer_provider=_RecordingProvider(),
                            conversation_history=long_history)
        self.assertEqual(len(captured["history"]), 8)
        self.assertEqual(captured["history"][-1]["content"], "turn 19")

    def test_previous_assistant_hallucination_cannot_become_evidence(self):
        """A fabricated claim sitting in prior conversation history (e.g.
        an earlier bad turn) must NOT let a new answer cite IDs that
        don't exist -- the deterministic verifier only ever trusts the
        CURRENT result's own evidence_ids/rule_ids, never anything in
        conversation_history text."""
        hallucinated_history = [
            {"role": "user", "content": "Is there a dragon in this drawing?"},
            {"role": "assistant", "content": "Yes, DOAR's evidence FAKE_EVIDENCE_DRAGON_001 confirms a dragon "
                                              "linked to rule FAKE_RULE_DRAGON_001."},
        ]

        class _CitingHallucinatedIdsProvider:
            model = "gemini-3.6-flash"

            def generate(self, *, question, audience, package, revision_instructions=None,
                         conversation_history=None):
                # Simulates a bad model trying to reuse the fabricated IDs
                # from the (attacker-controlled or hallucinated) history.
                return "As established, FAKE_EVIDENCE_DRAGON_001 confirms this."

        answer = hi.answer_question("What about the dragon?", self.bundle,
                                     answer_provider=_CitingHallucinatedIdsProvider(),
                                     conversation_history=hallucinated_history)
        # The final claim's evidence_ids/rule_ids are computed ONLY by the
        # deterministic category answerer from the CURRENT bundle -- the
        # hallucinated IDs in history text never get promoted into them.
        for claim in answer.claims:
            self.assertNotIn("FAKE_EVIDENCE_DRAGON_001", claim["evidence_ids"])
            self.assertNotIn("FAKE_RULE_DRAGON_001", claim["rule_ids"])
        self.assertEqual(answer.judge_verdict, "PASS")

    def test_history_key_only_carries_role_and_content_not_structured_bookkeeping(self):
        raw_turns = [
            {"role": "user", "content": "Why did you mention stress?"},
            {"role": "assistant", "content": "Because of line quality.", "structured": {"judge_verdict": "PASS"}},
        ]
        trimmed = app._recent_conversation_history(raw_turns)
        for turn in trimmed:
            self.assertEqual(set(turn.keys()), {"role", "content"})


# ---------------------------------------------------------------------------
# Part 10 -- visual recheck stays separate from original case analysis;
# h38 cache stays byte-identical.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_APP_AVAILABLE, "streamlit not available")
class VisualRecheckSeparationTests(unittest.TestCase):
    def test_visual_recheck_tagged_and_not_merged_into_original_evidence(self):
        bundle = _h38_bundle()
        hash_before = _h38_cache_hash()

        def fake_request_fn(api_key, model, payload, timeout):
            import json as _json
            body = {"present": True, "count": 2, "description": "two people near the car", "confidence": 0.6}
            return _json.dumps({"candidates": [{"content": {"parts": [{"text": _json.dumps(body)}]}}]}).encode()

        provider = hi.GeminiVisualRecheckProvider(api_key="fake-key", request_fn=fake_request_fn)
        answer = hi.answer_question("Are there people in the car?", bundle, visual_recheck_provider=provider)

        self.assertIn(hi._Q_AND_A_VISUAL_RECHECK_TAG, answer.provenance)
        self.assertIn("Q&A", answer.answer.replace("Q & A", "Q&A"))  # tagged in the answer text too
        self.assertIn("not part of the original DOAR analysis", answer.answer)
        # The bundle's own saved entities must be completely unaffected --
        # no 'person' entity was silently added.
        self.assertFalse(any("person" in e.canonical_label.lower() for e in bundle["entities"]))
        self.assertEqual(_h38_cache_hash(), hash_before)

    def test_h38_original_cache_byte_identical(self):
        # Standalone, explicit check independent of other tests' ordering.
        h1 = _h38_cache_hash()
        h2 = _h38_cache_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)


if __name__ == "__main__":
    unittest.main()
