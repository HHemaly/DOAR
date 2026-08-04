from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.timed_analysis import analyze_image_with_timing


class TimedAnalysisTests(unittest.TestCase):
    def test_real_execution_produces_positive_timing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(path)
            out = Path(d) / "out"
            result, timing = analyze_image_with_timing(path, out)
            self.assertGreater(timing["processing_seconds"], 0.0)
            self.assertEqual(timing["stage"], "full_analyze_image_pipeline")
            self.assertIsNotNone(result.schema_version)

    def test_timing_json_written_and_reloadable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(path)
            out = Path(d) / "out"
            analyze_image_with_timing(path, out)
            timing_path = out / "timing.json"
            self.assertTrue(timing_path.exists())
            reloaded = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertIn("processing_seconds", reloaded)

    def test_does_not_alter_analysis_json_contents_vs_plain_analyze_image(self):
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(path)
            plain = analyze_image(path, Path(d) / "plain").to_dict()
            timed, _ = analyze_image_with_timing(path, Path(d) / "timed")
            timed = timed.to_dict()
            plain.pop("image_path"), timed.pop("image_path")  # differs only by output dir path
            self.assertEqual(plain["composition"], timed["composition"])
            self.assertEqual(plain["quality"], timed["quality"])


if __name__ == "__main__":
    unittest.main()
