"""D9 — real image-quality gating (resolution / blur / contrast)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class QualityGateTests(unittest.TestCase):
    def test_tiny_image_is_unsupported(self):
        from PIL import Image
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.png"
            img = Image.new("RGB", (40, 40), "white")
            img.paste((0, 0, 0), (5, 5, 20, 20))
            img.save(p)
            a = analyze_image(p, Path(d) / "out")
            self.assertFalse(a.quality["supported"])          # 40px < 100px min
            self.assertFalse(a.quality["resolution_ok"])
            self.assertTrue(any("resolution" in r for r in a.quality["unsupported_reasons"]))

    def test_flat_low_contrast_is_unsupported(self):
        from PIL import Image
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "flat.png"
            Image.new("RGB", (256, 256), (200, 200, 200)).save(p)  # uniform grey
            a = analyze_image(p, Path(d) / "out")
            self.assertFalse(a.quality["supported"])            # no contrast / blur signal
            self.assertFalse(a.quality["contrast_ok"])

    def test_detailed_image_is_supported(self):
        from PIL import Image
        import numpy as np
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sharp.png"
            # high-frequency checkerboard -> sharp, high contrast, big enough
            arr = np.indices((256, 256)).sum(0) % 2 * 255
            Image.fromarray(arr.astype("uint8")).convert("RGB").save(p)
            a = analyze_image(p, Path(d) / "out")
            self.assertTrue(a.quality["resolution_ok"])
            self.assertTrue(a.quality["blur_ok"])
            self.assertTrue(a.quality["contrast_ok"])
            self.assertTrue(a.quality["supported"])

    def test_quality_judge_reflects_gate(self):
        from PIL import Image
        from doar.analysis import analyze_image
        from doar.judges import run_judges
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.png"
            Image.new("RGB", (40, 40), "white").save(p)
            a = analyze_image(p, Path(d) / "out")
            judges = run_judges(a.to_dict())
            self.assertEqual(judges["quality_judge"]["status"], "requires_review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
