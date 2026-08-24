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

    def test_fresh_launch_defaults_to_home(self):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertEqual(at.session_state["supervisor_view_mode"], "Supervisor Demo")
        self.assertEqual(at.session_state["supervisor_section"], "Home")
        self.assertTrue(any("TRY A PREPARED DEMONSTRATION" in h.value for h in at.subheader))

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


if __name__ == "__main__":
    unittest.main()
