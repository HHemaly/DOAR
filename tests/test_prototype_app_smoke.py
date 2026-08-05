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
            self.assertEqual(len(at.tabs), 2)

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
        assessable, see _build_case.)"""
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d, with_checkpoint=False)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            expander_labels = [e.label for e in at.expander]
            self.assertTrue(
                any("PSY_AR_SIZE_HALF_014" in label for label in expander_labels),
                f"expected an individual-suggestion expander for PSY_AR_SIZE_HALF_014, got: {expander_labels}",
            )
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
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(any("Upload a drawing" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main()
