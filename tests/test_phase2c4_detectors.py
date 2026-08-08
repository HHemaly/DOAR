"""Phase 2C.4 detector-wrapper tests -- synthetic RawDetection lists only,
no real model weights or network access (mirrors
tests/test_phase2b_inference.py's injected-backend discipline exactly)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c4.detectors import (
    OpenVocabDetector, RawDetection, build_class_presence,
    florence2_prompt_for_class, grounding_dino_text_prompt, make_default_label_to_classes,
    owlv2_text_queries,
)

CLASSES = ("person", "face", "hand", "animal", "house", "tree", "heart", "star", "circle", "vehicle")


class LabelToClassesTests(unittest.TestCase):
    def test_bare_class_name_matches(self):
        f = make_default_label_to_classes(CLASSES)
        self.assertEqual(f("circle"), ["circle"])

    def test_article_prefixed_label_matches(self):
        f = make_default_label_to_classes(CLASSES)
        self.assertEqual(f("a circle"), ["circle"])

    def test_case_insensitive(self):
        f = make_default_label_to_classes(CLASSES)
        self.assertEqual(f("A CIRCLE"), ["circle"])

    def test_merged_label_matches_both_classes(self):
        """The real Grounding DINO quirk observed this session: two
        detections sharing a box can merge into one label string."""
        f = make_default_label_to_classes(CLASSES)
        result = f("a person a face")
        self.assertEqual(set(result), {"person", "face"})

    def test_unmatched_label_returns_empty_list(self):
        f = make_default_label_to_classes(CLASSES)
        self.assertEqual(f("a spaceship"), [])

    def test_no_partial_word_match(self):
        """'hand' must not match inside 'handle' or similar -- whole-word
        boundaries only."""
        f = make_default_label_to_classes(CLASSES)
        self.assertEqual(f("a handlebar"), [])


class BuildClassPresenceTests(unittest.TestCase):
    def test_every_class_present_in_output_even_with_zero_detections(self):
        result = build_class_presence([], CLASSES)
        self.assertEqual(set(result.keys()), set(CLASSES))
        self.assertTrue(all(detected is False and score == 0.0 for detected, score in result.values()))

    def test_single_detection_marks_that_class_present(self):
        result = build_class_presence([RawDetection(label="a circle", score=0.42)], CLASSES)
        self.assertEqual(result["circle"], (True, 0.42))
        self.assertEqual(result["person"], (False, 0.0))

    def test_best_score_is_the_max_across_multiple_detections(self):
        detections = [RawDetection(label="a circle", score=0.3), RawDetection(label="a circle", score=0.7)]
        result = build_class_presence(detections, CLASSES)
        self.assertEqual(result["circle"], (True, 0.7))

    def test_merged_label_marks_both_classes_present(self):
        result = build_class_presence([RawDetection(label="a person a face", score=0.4)], CLASSES)
        self.assertTrue(result["person"][0])
        self.assertTrue(result["face"][0])

    def test_unmatched_label_affects_nothing(self):
        result = build_class_presence([RawDetection(label="a spaceship", score=0.9)], CLASSES)
        self.assertTrue(all(not detected for detected, _ in result.values()))


class OpenVocabDetectorTests(unittest.TestCase):
    def test_wraps_an_injected_predict_fn(self):
        def fake_predict_fn(image_path):
            return [RawDetection(label="a house", score=0.55)]

        detector = OpenVocabDetector(fake_predict_fn, model_name="fake_model", class_names=CLASSES)
        result = detector.predict_image("irrelevant/path.jpg")
        self.assertEqual(result["house"], (True, 0.55))
        self.assertEqual(detector.model_name, "fake_model")

    def test_no_real_model_weights_touched_by_this_test(self):
        """Structural guard: OpenVocabDetector itself never imports torch/
        transformers/ultralytics -- only the load_real_* functions do."""
        import inspect
        source = inspect.getsource(OpenVocabDetector)
        for forbidden in ("import torch", "import transformers", "from transformers", "from ultralytics"):
            self.assertNotIn(forbidden, source)


class PromptDefinitionTests(unittest.TestCase):
    """Pins the exact prompts every model receives -- required to be
    documented and auditable, per instruction."""

    def test_owlv2_queries_one_per_class(self):
        queries = owlv2_text_queries(CLASSES)
        self.assertEqual(len(queries), len(CLASSES))
        self.assertEqual(queries[0], "a person")

    def test_grounding_dino_prompt_format(self):
        prompt = grounding_dino_text_prompt(CLASSES)
        self.assertTrue(prompt.endswith("."))
        self.assertIn("person. face.", prompt)

    def test_florence2_prompt_is_bare_noun(self):
        self.assertEqual(florence2_prompt_for_class("circle"), "circle")


if __name__ == "__main__":
    unittest.main()
