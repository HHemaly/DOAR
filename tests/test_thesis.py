"""Item 18 — thesis outputs: figures only from existing data, each with source.

Fixtures below are SYNTHETIC experiment outputs, used only to verify the
collation behaviour (that figures pair with matching source data and missing
sources are reported, not fabricated).
"""

from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _write(root, rel, obj):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


class ThesisTests(unittest.TestCase):
    def test_no_sources_generates_nothing_and_reports_missing(self):
        from doar.thesis import generate_thesis_outputs
        with tempfile.TemporaryDirectory() as d:
            res = generate_thesis_outputs(d)
            self.assertEqual(res["generated"], 0)
            self.assertTrue(res["missing"])       # sources reported missing, not faked
            manifest = json.loads((Path(d) / "thesis" / "thesis_manifest.json").read_text())
            self.assertEqual(manifest["generated_figures"], [])

    def test_each_figure_has_matching_source_data(self):
        try:
            import matplotlib  # noqa
        except ImportError:
            self.skipTest("matplotlib not installed")
        from doar.thesis import generate_thesis_outputs
        with tempfile.TemporaryDirectory() as d:
            # SYNTHETIC deep-comparison + ablation outputs.
            _write(d, "deep_comparison.json", {"leaderboard": [
                {"model": "resnet18", "mean_valid_macro_f1": 0.6, "std_valid_macro_f1": 0.02},
                {"model": "small_cnn", "mean_valid_macro_f1": 0.4, "std_valid_macro_f1": 0.03}]})
            _write(d, "ablation.json", {"results": [
                {"configuration": "all_features", "mean_macro_f1": 0.6, "std_macro_f1": 0.01},
                {"configuration": "without_colour", "mean_macro_f1": 0.55, "std_macro_f1": 0.02}]})
            res = generate_thesis_outputs(d)
            self.assertGreaterEqual(res["generated"], 2)
            for fig in res["figures"]:
                # every figure references a source_data file that exists
                self.assertTrue((Path(d) / "thesis" / fig["source_data"]).exists())
                self.assertTrue((Path(d) / "thesis" / fig["figure"]).exists())

    def test_agreement_figure_requires_real_reviews(self):
        try:
            import matplotlib  # noqa
        except ImportError:
            self.skipTest("matplotlib not installed")
        from doar.thesis import generate_thesis_outputs
        with tempfile.TemporaryDirectory() as d:
            _write(d, "agreement.json", {"has_reviews": False, "note": "unavailable"})
            res = generate_thesis_outputs(d)
            names = [f["figure"] for f in res["figures"]]
            self.assertFalse(any("agreement" in n for n in names))  # not generated


if __name__ == "__main__":
    unittest.main(verbosity=2)
