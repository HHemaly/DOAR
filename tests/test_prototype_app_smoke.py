"""End-to-end smoke test for doar_prototype_app.py using Streamlit's
AppTest harness (bare-mode script execution, no browser/server). Builds one
real case via the real pipeline (no mocking) and confirms both the Parent
and Technical views render without raising any exception."""
from __future__ import annotations

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
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((5, 5, 195, 195), fill="black")
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

    def test_candidate_themes_and_dependency_grouping_render_doar_trace_4e(self):
        """DOAR-TRACE 4E: a full-page-coverage drawing reliably triggers the
        coverage_full rule -> a self_esteem candidate theme -- confirms the
        new Parent-view theme section and Technical-view dependency/
        aggregation table both render real content, not just avoid crashing."""
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d, with_checkpoint=False)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            expander_labels = [e.label for e in at.expander]
            self.assertTrue(
                any("self_esteem" in label for label in expander_labels),
                f"expected a self_esteem theme expander, got: {expander_labels}",
            )

    def test_no_case_selected_shows_info_not_a_crash(self):
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(any("Upload a drawing" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main()
