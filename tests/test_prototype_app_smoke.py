"""End-to-end smoke test for doar_prototype_app.py using Streamlit's
AppTest harness (bare-mode script execution, no browser/server). Builds one
real case via the real pipeline (no mocking) and confirms both the Parent
and Technical views render without raising any exception."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

from doar.profile import ChildProfile, save_profile
from doar.timed_analysis import analyze_image_with_timing


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class PrototypeAppSmokeTests(unittest.TestCase):
    def _build_case(self, tmp_dir: str, with_checkpoint: bool = False) -> Path:
        # Phase 2A migration note: widened from a 200x200/5px-margin canvas.
        # A margin that thin cannot pass page_frame.py's border-uniformity
        # check, so PSY_AR_SIZE_FULL_015/HALF_014 would now be correctly
        # gated `not_assessable` instead of triggering -- see
        # docs/PAGE_FRAME_ASSESSABILITY.md. This margin IS wide enough to
        # be detected as a real full page.
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
        path = Path(tmp_dir) / "drawing.png"
        image.save(path)
        case_dir = Path(tmp_dir) / "case"
        checkpoint = None
        if with_checkpoint:
            candidate = ROOT / "outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt"
            checkpoint = str(candidate) if candidate.exists() else None
        analyze_image_with_timing(path, case_dir, checkpoint)
        save_profile(case_dir, ChildProfile(age_range="6-7", drawing_instruction="draw anything",
                                             date="2026-08-03", parent_concern="checking in"))
        return case_dir

    def test_renders_both_views_without_exception_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d, with_checkpoint=False)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            self.assertEqual(len(at.tabs), 3)

    def test_renders_both_views_without_exception_with_real_checkpoint(self):
        candidate = ROOT / "outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt"
        if not candidate.exists():
            self.skipTest("Reference checkpoint not present in this checkout")
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d, with_checkpoint=True)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

    def test_individual_suggestion_renders_and_never_becomes_a_combined_theme(self):
        """DOAR-TRACE Phase 1.5: a page-coverage drawing reliably triggers
        PSY_AR_SIZE_HALF_014 alone -- confirms it renders as an individual
        rule suggestion (Section 3 of the Parent view) and, per the
        Phase-1.5 bug fix, is never promoted to a combined drawing-level
        hypothesis on its own. (Phase 2A migration: was PSY_AR_SIZE_FULL_015
        on a thinner-margin canvas; that margin is no longer page-frame-
        assessable. Phase 2A.1 migration: the centered 40px-margin square
        `_build_case` uses now ALSO satisfies the redefined coverage_full
        + placement_center together -- a real, correct Level-C combination
        (Section 4) -- so this test uses its own OFF-CENTER image instead,
        which triggers coverage_about_half + 2 unmapped placement rules
        that can never combine.

        Phase 2A.2 migration: Section 4 now explicitly requires internal
        rule IDs to be hidden from the Parent view -- the individual-
        suggestion expander title is now "Observation N: <friendly
        family name>" (e.g. "page-space use"), never the raw rule_id.
        This test now checks for that friendly label instead."""
        with tempfile.TemporaryDirectory() as d:
            image = Image.new("RGB", (300, 300), "white")
            ImageDraw.Draw(image).rectangle((20, 20, 209, 209), fill="black")
            path = Path(d) / "drawing.png"
            image.save(path)
            case_dir = Path(d) / "case"
            analyze_image_with_timing(str(path), str(case_dir), None)
            save_profile(case_dir, ChildProfile(age_range="6-7", drawing_instruction="draw anything",
                                                 date="2026-08-05", parent_concern="checking in"))
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            # Milestone 2: Parent View shows individual observations as plain,
            # parent-safe bullets (no per-observation expander, no raw rule_id);
            # that professional detail now lives only in the Psychologist tab.
            parent_tab, psychologist_tab = at.tabs[0], at.tabs[1]
            parent_text = " ".join(m.value for m in parent_tab.markdown)
            self.assertNotIn("PSY_AR_SIZE_HALF_014", parent_text,
                              "raw rule_id must never appear in the Parent view")
            # Milestone 3: professional rule-ID detail now lives inside the
            # Psychologist view's collapsed "Technical evidence" expander
            # (a dataframe of all rule evaluations), not in an expander title.
            tech_expanders = [e for e in psychologist_tab.expander if "Technical evidence" in e.label]
            self.assertEqual(len(tech_expanders), 1)
            rules_df = tech_expanders[0].dataframe[0].value
            self.assertIn("PSY_AR_SIZE_HALF_014", rules_df.to_string())
            structured = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(structured["combined_drawing_level_hypotheses"], [])

    def test_page_not_assessable_case_renders_warning_not_a_crash(self):
        # DOAR-TRACE Phase 2A, Section 9: a thin-margin image (page not
        # confirmed visible) must render an explicit Parent-view warning
        # explaining why page-relative observations are suppressed --
        # never a silent absence and never an exception.
        with tempfile.TemporaryDirectory() as d:
            image = Image.new("RGB", (200, 200), "white")
            ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
            path = Path(d) / "drawing.png"
            image.save(path)
            case_dir = Path(d) / "case"
            analyze_image_with_timing(str(path), str(case_dir), None)
            structured = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
            self.assertNotIn(structured["page_frame_assessment"]["page_frame_status"], ("full_page_detected", "likely_full_page"))
            self.assertTrue(structured["page_relative_rules_not_assessable"])

            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            warning_texts = " ".join(w.value for w in at.warning)
            self.assertTrue(
                "page" in warning_texts.lower() or "الصفحة" in warning_texts,
                f"expected a page-frame warning, got warnings: {[w.value for w in at.warning]}",
            )

    def test_no_case_selected_shows_info_not_a_crash(self):
        # feature/supervisor-demo-v2: with no case_dir pre-seeded, the app
        # now defaults to the new Supervisor Demo Home page (not the legacy
        # upload prompt) -- see doar_prototype_app.py's "Smart default"
        # comment next to `supervisor_view_mode`. This test now explicitly
        # selects the legacy view to confirm ITS OWN "no case selected"
        # behaviour is unchanged and still doesn't crash.
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["supervisor_view_mode"] = "Full Research App (legacy)"
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(any("Upload a drawing" in i.value for i in at.info))

    def test_no_case_selected_defaults_to_supervisor_demo_home(self):
        """New default (feature/supervisor-demo-v2): a completely fresh
        launch, with nothing pre-seeded, lands on the Supervisor Demo Home
        page -- never a crash, never the bare legacy upload prompt."""
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertEqual(at.session_state["supervisor_view_mode"], "Supervisor Demo")
        self.assertTrue(any("TRY A PREPARED DEMONSTRATION" in h.value for h in at.subheader))


if __name__ == "__main__":
    unittest.main()
