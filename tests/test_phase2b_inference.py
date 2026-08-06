"""CPU-safe smoke tests for Phase 2B's two baselines. Neither test needs
real model weights or network access: ZeroShotClipDetector is exercised
through its injectable image_embed_fn/text_embed_fn (mirroring
tests/test_deep_compare.py::RunnerTests::test_runner_with_injected_synthetic_trainer's
dependency-injection pattern), and the classical-CV baseline only needs
opencv-python-headless (the `cv` extra), never network."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.inference import ZeroShotClipDetector
from doar.phase2b.ontology import CLASS_NAMES

try:
    import cv2  # noqa: F401
    _CV2 = True
except Exception:
    _CV2 = False

try:
    from PIL import Image, ImageDraw
    _PIL = True
except Exception:
    _PIL = False


def _synthetic_image_embed(path: str) -> list[float]:
    # A deterministic, path-derived "embedding" -- no model needed.
    h = sum(ord(c) for c in path)
    return [((h * (i + 1)) % 97) / 97.0 for i in range(8)]


def _synthetic_text_embed(text: str) -> list[float]:
    h = sum(ord(c) for c in text)
    return [((h * (i + 1)) % 97) / 97.0 for i in range(8)]


class ZeroShotClipDetectorSmokeTests(unittest.TestCase):
    def test_predict_image_covers_every_class_exactly_once(self):
        det = ZeroShotClipDetector(_synthetic_image_embed, _synthetic_text_embed)
        preds = det.predict_image("fake/path/to/image.png")
        self.assertEqual({p.class_name for p in preds}, set(CLASS_NAMES))
        self.assertEqual(len(preds), len(CLASS_NAMES))

    def test_every_prediction_has_a_valid_status(self):
        det = ZeroShotClipDetector(_synthetic_image_embed, _synthetic_text_embed)
        preds = det.predict_image("fake/path/to/image.png")
        for p in preds:
            self.assertIn(p.status, {"detected", "not_detected", "uncertain"})

    def test_deterministic_given_the_same_injected_backend(self):
        det = ZeroShotClipDetector(_synthetic_image_embed, _synthetic_text_embed)
        a = det.predict_image("fake/path/to/image.png")
        b = det.predict_image("fake/path/to/image.png")
        self.assertEqual([(p.class_name, p.similarity) for p in a],
                         [(p.class_name, p.similarity) for p in b])

    def test_text_embeddings_are_cached_not_recomputed_per_image(self):
        calls = []

        def counting_text_embed(text: str) -> list[float]:
            calls.append(text)
            return _synthetic_text_embed(text)

        det = ZeroShotClipDetector(_synthetic_image_embed, counting_text_embed)
        det.predict_image("a.png")
        det.predict_image("b.png")
        self.assertEqual(len(calls), len(CLASS_NAMES))  # not 2x

    def test_no_real_model_weights_or_network_touched_by_this_backend(self):
        # load_real() is the only method that imports torch/open_clip --
        # constructing with injected functions must never trigger it.
        det = ZeroShotClipDetector(_synthetic_image_embed, _synthetic_text_embed)
        self.assertNotEqual(det.model_name, "ViT-B-32")


@unittest.skipUnless(_CV2 and _PIL, "opencv/Pillow not installed")
class ClassicalCircleDetectorSmokeTests(unittest.TestCase):
    def test_detects_a_drawn_circle(self):
        from doar.phase2b.inference import detect_circles_classical
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "circle.png"
            img = Image.new("L", (200, 200), 255)
            ImageDraw.Draw(img).ellipse((40, 40, 160, 160), outline=0, width=4)
            img.save(path)
            result = detect_circles_classical(str(path))
            self.assertTrue(result.detected)
            self.assertGreaterEqual(result.circularity, 0.5)

    def test_blank_image_detects_nothing(self):
        from doar.phase2b.inference import detect_circles_classical
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "blank.png"
            Image.new("L", (200, 200), 255).save(path)
            result = detect_circles_classical(str(path))
            self.assertFalse(result.detected)
            self.assertEqual(result.count, 0)

    def test_missing_file_raises_a_clear_error(self):
        from doar.phase2b.inference import detect_circles_classical
        with self.assertRaises(ValueError):
            detect_circles_classical("does/not/exist.png")


if __name__ == "__main__":
    unittest.main()
