"""Phase 2C.7 / DOAR MVP: runtime.py's pure, injectable routing logic --
object_class_predict_fn / part_predict_fn / combined_fallback_predict_fn /
the box-normalization helper. Never touches real model weights; the
load_real_*/load_open_vocab_query_fn loaders themselves are excluded here
by design (mirrors every other load_real_* module in this project)."""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7.runtime import (  # noqa: E402
    _xyxy_pixels_to_xywh_normalized, combined_fallback_predict_fn, object_class_predict_fn,
    part_predict_fn,
)


@dataclass(frozen=True)
class _FakeRawBoxDetection:
    label: str
    score: float
    bbox_xywh_normalized: tuple


class _FakeOpenVocabDetector:
    def __init__(self, presence: dict):
        self._presence = presence

    def predict_image(self, image_path: str) -> dict:
        return self._presence


class ObjectClassPredictFnTests(unittest.TestCase):
    def test_wraps_presence_dict_with_none_bbox(self):
        detector = _FakeOpenVocabDetector({"person": (True, 0.9), "tree": (False, 0.0)})
        predict = object_class_predict_fn(detector)
        result = predict("fake.png")
        self.assertEqual(result["person"], (True, 0.9, None))
        self.assertEqual(result["tree"], (False, 0.0, None))


class PartPredictFnTests(unittest.TestCase):
    def test_picks_highest_scoring_detection_per_target(self):
        raw = [
            _FakeRawBoxDetection(label="eye", score=0.3, bbox_xywh_normalized=(0.1, 0.1, 0.1, 0.1)),
            _FakeRawBoxDetection(label="eye", score=0.7, bbox_xywh_normalized=(0.2, 0.2, 0.1, 0.1)),
            _FakeRawBoxDetection(label="unrelated thing", score=0.9, bbox_xywh_normalized=(0.5, 0.5, 0.1, 0.1)),
        ]
        predict = part_predict_fn(lambda path: raw)
        result = predict("fake.png")
        detected, score, bbox = result["eye"]
        self.assertTrue(detected)
        self.assertEqual(score, 0.7)
        self.assertEqual(bbox, (0.2, 0.2, 0.1, 0.1))
        self.assertFalse(result["mouth"][0])

    def test_no_raw_detections_returns_all_absent(self):
        predict = part_predict_fn(lambda path: [])
        result = predict("fake.png")
        self.assertTrue(all(not detected for detected, _score, _bbox in result.values()))


class CombinedFallbackPredictFnTests(unittest.TestCase):
    def test_fallback_only_fills_primary_misses(self):
        primary = lambda path: {"eye": (False, 0.0, None), "mouth": (True, 0.8, None)}  # noqa: E731
        fallback = lambda path: {"eye": (True, 0.4, (0.1, 0.1, 0.1, 0.1)), "mouth": (True, 0.2, None)}  # noqa: E731
        predict = combined_fallback_predict_fn(primary, fallback)
        result = predict("fake.png")
        self.assertEqual(result["eye"], (True, 0.4, (0.1, 0.1, 0.1, 0.1)))  # recovered by fallback
        self.assertEqual(result["mouth"], (True, 0.8, None))  # primary already found it -- fallback ignored

    def test_primary_and_fallback_both_miss_stays_absent(self):
        primary = lambda path: {"eye": (False, 0.0, None)}  # noqa: E731
        fallback = lambda path: {"eye": (False, 0.0, None)}  # noqa: E731
        predict = combined_fallback_predict_fn(primary, fallback)
        self.assertEqual(predict("fake.png")["eye"], (False, 0.0, None))


class XyxyToXywhNormalizedTests(unittest.TestCase):
    def test_converts_and_normalizes_pixel_box(self):
        result = _xyxy_pixels_to_xywh_normalized((10, 20, 60, 70), image_width=100, image_height=100)
        self.assertEqual(result, (0.1, 0.2, 0.5, 0.5))

    def test_clamps_out_of_bounds_coordinates(self):
        x, y, w, h = _xyxy_pixels_to_xywh_normalized((-10, -10, 200, 200), image_width=100, image_height=100)
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertEqual((w, h), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
