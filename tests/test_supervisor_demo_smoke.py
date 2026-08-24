"""End-to-end smoke test for the Supervisor Demo view added in
doar_prototype_app.py (feature/supervisor-demo-v2), using Streamlit's
AppTest harness (bare-mode script execution, no browser/server -- same
pattern as test_prototype_app_smoke.py).

Covers the exact walkthrough path a supervisor would click through: Home ->
each prepared case's Drawing Analysis -> Technical Trace -> Research
Progress -> back to Home, and confirms opening a prepared case never
requires GEMINI_API_KEY (the env var is forced unset for the main path
tests)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

DEMO_CASE_KEYS = ["case1", "case2", "case3"]


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class SupervisorDemoSmokeTests(unittest.TestCase):
    def setUp(self):
        # Confirm the prepared-case path never depends on Gemini: unset the
        # real key for the duration of the main walkthrough tests, restore
        # afterwards so other tests in the same process are unaffected.
        self._old_key = os.environ.pop("GEMINI_API_KEY", None)
        self.addCleanup(self._restore_key)

    def _restore_key(self):
        if self._old_key is not None:
            os.environ["GEMINI_API_KEY"] = self._old_key

    def test_open_analysis_button_click_case1_no_crash(self):
        """P0 regression test for the real crash:
        streamlit.errors.StreamlitAPIException: st.session_state.supervisor_section
        cannot be modified after the widget with key supervisor_section is
        instantiated. The prior test suite missed this because it wrote to
        at.session_state["supervisor_section"] directly instead of clicking
        the real rendered "Open Analysis" button, which is the only path
        that actually exercises the widget-instantiation-then-write ordering
        that Streamlit forbids. This test clicks the REAL button located by
        its actual rendered key (key=f"sd_open_{case_key}" in _sd_home)."""
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        at.button(key="sd_open_case1").click()
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertEqual(at.session_state["supervisor_section"], "Drawing Analysis")
        self.assertEqual(at.session_state["supervisor_active_case"], "case1")
        self.assertTrue(any("Example Drawing 1" in m.value for m in at.markdown if m.value.startswith("##")))

    def test_open_analysis_button_click_all_three_cases_no_crash(self):
        """Same real-click path as above, exercised for all three prepared
        cases with a real click on the sidebar navigation radio (the widget
        bound to key="supervisor_section") back to Home in between -- the
        exact click sequence a supervisor clicking through the UI would
        produce: Home -> Open Analysis -> Home -> Open Analysis -> ..."""
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        for case_key in DEMO_CASE_KEYS:
            # Real click on the sidebar "Navigation" radio widget itself
            # (not a direct session_state write) to return Home before each
            # case is opened.
            at.sidebar.radio(key="supervisor_section").set_value("Home")
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0,
                              [str(e) for e in at.exception] + [f"nav-home before {case_key}"])

            at.button(key=f"sd_open_{case_key}").click()
            at.run(timeout=120)
            self.assertEqual(len(at.exception), 0,
                              [str(e) for e in at.exception] + [f"open {case_key}"])
            self.assertEqual(at.session_state["supervisor_section"], "Drawing Analysis")
            self.assertEqual(at.session_state["supervisor_active_case"], case_key)

    def test_fresh_launch_defaults_to_home(self):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertEqual(at.session_state["supervisor_view_mode"], "Supervisor Demo")
        self.assertEqual(at.session_state["supervisor_section"], "Home")
        self.assertTrue(any("Example Drawings" in h.value for h in at.subheader))

    def test_each_prepared_case_renders_drawing_analysis_and_technical_trace(self):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        for case_key in DEMO_CASE_KEYS:
            at.session_state["supervisor_active_case"] = case_key
            at.session_state["supervisor_section"] = "Drawing Analysis"
            at.run(timeout=120)
            self.assertEqual(len(at.exception), 0,
                              [str(e) for e in at.exception] + [f"case={case_key} (Drawing Analysis)"])

            at.session_state["supervisor_section"] = "Technical Trace"
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0,
                              [str(e) for e in at.exception] + [f"case={case_key} (Technical Trace)"])

    def test_research_progress_renders(self):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["supervisor_section"] = "Research Progress"
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(len(at.dataframe) > 0)
        self.assertTrue(any("Research Progress" in h.value for h in at.markdown if h.value.startswith("##")))

    def test_returning_home_from_a_case_does_not_crash(self):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["supervisor_active_case"] = "case1"
        at.session_state["supervisor_section"] = "Drawing Analysis"
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        at.session_state["supervisor_section"] = "Home"
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

    def test_drawing_analysis_and_technical_trace_require_a_case_first(self):
        """No case picked yet -> a friendly quick-pick prompt, never a crash
        and never a bare KeyError from an unset active_case."""
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["supervisor_section"] = "Drawing Analysis"
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(len(at.info) > 0)

    def test_ask_doar_degrades_gracefully_without_gemini_key(self):
        """P0: Ask DOAR must never crash/traceback when GEMINI_API_KEY is
        unset -- it should still answer from the deterministic fallback."""
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["supervisor_active_case"] = "case1"
        at.session_state["supervisor_section"] = "Drawing Analysis"
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        at.text_input(key="sd_case1_chat_q").input("What did DOAR notice?")
        at.button(key="sd_case1_chat_send").click()
        at.run(timeout=90)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        # No raw traceback/exception class name leaked into a chat bubble.
        chat_markdown = " ".join(m.value for m in at.markdown if m.value.startswith(("**Answer", "**الرد")))
        self.assertNotIn("Traceback", chat_markdown)
        self.assertTrue(len(chat_markdown) > 0)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class RuleDisplayFixTests(unittest.TestCase):
    """Targeted regression tests for the P0 rule-display fix (Section 4,
    "MATCHED GOVERNED EVIDENCE"): an evaluated-but-unmatched rule must
    never render as if it were supporting evidence, the summary counter
    must count matches only, and the empty state must use the exact
    required wording. case1 (h38) and case2 (e2e_check) each have >=1
    real matched rule; case3 (a111) has zero -- both real outcomes are
    exercised against the live case bundles, not synthetic fixtures."""

    def setUp(self):
        self._old_key = os.environ.pop("GEMINI_API_KEY", None)
        self.addCleanup(self._restore_key)

    def _restore_key(self):
        if self._old_key is not None:
            os.environ["GEMINI_API_KEY"] = self._old_key

    def _open_case(self, case_key: str):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["supervisor_active_case"] = case_key
        at.session_state["supervisor_section"] = "Drawing Analysis"
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        return at

    def test_a_unmatched_rule_not_rendered_in_main_matched_evidence_section(self):
        """case1 (h38) has both a matched rule (light line pressure) AND at
        least one evaluated-but-not-matched rule (heavy line pressure /
        shaky-broken lines, mutually exclusive with the match) in its real
        saved analysis.json -- confirm the NOT-MATCHED ones never render as
        a badge/card anywhere on the Drawing Analysis page."""
        at = self._open_case("case1")
        page_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        self.assertNotIn("NOT MATCHED", page_text)
        self.assertNotIn("غير مطابق", page_text)

    def test_b_matched_rule_counter_equals_real_matched_count(self):
        """case1 (h38) has exactly 1 real weak_support rule
        (EN_COMPILED_LINE_LIGHT_PRESSURE_031) in outputs/prototype_cases/
        h38_1786305027/analysis.json; case2 (e2e_check) has exactly 2."""
        at1 = self._open_case("case1")
        summary1 = " ".join(m.value for m in at1.markdown if "Matched governed rules" in m.value)
        self.assertIn("Matched governed rules: 1", summary1)

        at2 = self._open_case("case2")
        summary2 = " ".join(m.value for m in at2.markdown if "Matched governed rules" in m.value)
        self.assertIn("Matched governed rules: 2", summary2)

    def test_c_no_match_shows_exact_required_message(self):
        """case3 (a111) has zero weak_support rows in its real saved
        analysis.json -- the main UI must show the exact required
        sentence, never a silently empty section."""
        at = self._open_case("case3")
        summary = " ".join(m.value for m in at.markdown if "Matched governed rules" in m.value)
        self.assertIn("Matched governed rules: 0", summary)
        info_text = " ".join(i.value for i in at.info)
        self.assertIn("No currently enabled governed rules matched this drawing.", info_text)

    def test_d_ask_doar_answers_what_did_doar_notice_from_case_evidence(self):
        """Grounded-fallback path, exercised deterministically (no
        GEMINI_API_KEY in this process) -- must cite this case's own real
        objective measurements, never the old generic non-answer."""
        at = self._open_case("case1")
        at.text_input(key="sd_case1_chat_q").input("What did DOAR notice?")
        at.button(key="sd_case1_chat_send").click()
        at.run(timeout=90)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        answer = " ".join(m.value for m in at.markdown if m.value.startswith("**Answer"))
        self.assertNotIn("I couldn't verify a reliable answer", answer)
        self.assertIn("noticed", answer.lower())


if __name__ == "__main__":
    unittest.main()
