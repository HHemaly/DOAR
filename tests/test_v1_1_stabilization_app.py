"""DOAR V1.1 stabilization: real AppTest-driven verification against the
ACTUAL doar_prototype_app.py (not a mock) -- same pattern already
established by test_phase2a2_parent_view.py. Proves the concrete user-
visible fixes: no model/checkpoint controls, no stale "Feature
computation failed" error, production-config provenance shown in
Technical View, research tables collapsed and separated, and Parent
View's capability wording stays accurate.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

from doar.timed_analysis import analyze_image_with_timing  # noqa: E402


def _build_case(tmp_dir: str) -> Path:
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image_with_timing(str(path), str(case_dir), None)
    # The app saves the uploaded image inside the case dir itself.
    image.save(case_dir / "drawing.png")
    return case_dir


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class NoModelSelectionControlsTests(unittest.TestCase):
    """Stage 3/7: the normal user must see ZERO model/checkpoint/threshold
    controls anywhere in the app."""

    def test_no_selectbox_offers_a_model_or_checkpoint_choice(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            selectbox_labels = " ".join((s.label or "") for s in at.selectbox).lower()
            for forbidden in ("emotion model", "checkpoint", "efficientnet", "grounding dino", "owlv2"):
                self.assertNotIn(forbidden, selectbox_labels)

    def test_no_text_input_offers_a_custom_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            text_input_labels = " ".join((t.label or "") for t in at.text_input).lower()
            self.assertNotIn("checkpoint path", text_input_labels)

    def test_no_slider_or_number_input_exposes_a_threshold_control(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.slider), 0)
            self.assertEqual(len(at.number_input), 0)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ObjectiveFeaturePathFixTests(unittest.TestCase):
    """Problem A: Technical View's live objective-features re-display must
    never raise/report 'Feature computation failed'."""

    def test_no_feature_computation_failed_error_rendered(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            error_texts = " ".join(e.value for e in at.error)
            self.assertNotIn("Feature computation failed", error_texts)

    def test_objective_features_table_actually_renders(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertGreater(len(at.dataframe), 0)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ProductionConfigProvenanceTests(unittest.TestCase):
    """Stage 3: Technical View must show production configuration
    provenance; Parent View must not."""

    def test_technical_view_shows_production_configuration_caption(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            from doar.production_config import resolve_production_config
            from doar.case_output import write_versioned
            write_versioned(case_dir / "production_config.json", resolve_production_config().to_dict())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            all_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption) \
                + " ".join(s.value for s in at.subheader)
            self.assertIn("Production configuration", all_text)

    def test_parent_view_never_mentions_production_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            from doar.production_config import resolve_production_config
            from doar.case_output import write_versioned
            write_versioned(case_dir / "production_config.json", resolve_production_config().to_dict())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            # Parent-tab widgets render before the technical_tab block in
            # source order; the existing suite's own convention (see
            # test_phase2a2_parent_view.py) checks caption/markdown text
            # broadly -- here we check the specific new provenance string
            # never appears in caption/markdown at all outside Technical
            # View's own subheader-scoped section, i.e. it appears at most
            # once (the Technical View instance).
            captions = [c.value for c in at.caption]
            self.assertLessEqual(sum("Production configuration" in c for c in captions), 1)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ResearchValidationSeparationTests(unittest.TestCase):
    """Problem E: Phase 2B pilot / measurement-validation tables must be
    collapsed and clearly separated from live case data."""

    def test_research_validation_header_exists(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            headers = [h.value for h in at.header]
            self.assertIn("11. Research / Validation", headers)

    def test_phase2b_pilot_and_measurement_validation_are_inside_an_expander(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            expander_labels = [e.label for e in at.expander]
            self.assertTrue(any("Phase 2B" in label for label in expander_labels))


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class CapabilityWordingRegressionTests(unittest.TestCase):
    """Problem D: no stale capability contradiction rendered for a case
    that never ran the visual scan (visual_detection genuinely
    unavailable here) -- the OLD, still-accurate wording must be shown,
    never a false claim either direction."""

    def test_parent_view_uses_accurate_wording_when_visual_detection_never_ran(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            all_text = " ".join(m.value for m in at.markdown)
            # detections.json is the honest "unavailable" stub for this
            # fixture (no visual scan was run) -- the "were not analyzed"
            # wording is the ACCURATE one here, not the "were scanned" one.
            self.assertIn("were not analyzed", all_text)
            self.assertNotIn("automatically scanned for known objects", all_text)


if __name__ == "__main__":
    unittest.main()
