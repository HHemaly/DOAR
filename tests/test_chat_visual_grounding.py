"""DOAR MVP: chat.py's new optional registry_v2/open_vocab_predict_fn
wiring -- respond_to_chat must route visual-object questions through
visual_qa.py (on-demand search when injected, honest 'unavailable' when
not), while every non-visual question keeps behaving exactly as before
(regression coverage for the existing, already-tested qa.py path)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image  # noqa: E402
from doar.chat import respond_to_chat  # noqa: E402

FAKE_REGISTRY_V2 = {"rules": []}


def _real_case(tmp_dir: str) -> Path:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image(path, case_dir)
    # The real app (doar_prototype_app.py) saves the uploaded image INSIDE
    # the case directory -- search_visual globs for it there.
    image.save(case_dir / "drawing.png")
    return case_dir


class RespondToChatVisualGroundingTests(unittest.TestCase):
    def test_non_visual_question_unaffected_by_new_optional_params(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "What is the emotion prediction?", "en")
            self.assertFalse(response.escalated)
            self.assertIn(response.availability, ("unavailable", "available"))

    def test_visual_question_without_injected_search_is_honest_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "Is there a dog in the drawing?", "en")
            self.assertEqual(response.availability, "missing_detector")
            self.assertFalse(response.escalated)

    def test_visual_question_with_injected_on_demand_search_finds_it(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)

            def open_vocab_predict_fn(image_path, target):
                return [(True, 0.7, None, "fake_model", "ckpt", target)]

            response = respond_to_chat(
                case_dir, "Is there a dog in the drawing?", "en",
                registry_v2=FAKE_REGISTRY_V2, open_vocab_predict_fn=open_vocab_predict_fn)
            self.assertEqual(response.availability, "available")
            self.assertIsNotNone(response.non_diagnostic_warning)
            self.assertFalse(response.escalated)

    def test_safeguarding_escalation_still_short_circuits_before_visual_grounding(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "I want to hurt myself", "en")
            self.assertTrue(response.escalated)


if __name__ == "__main__":
    unittest.main()
