"""Item 17 — ablation column-selection logic (CPU). Training path needs sklearn."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class AblationSelectionTests(unittest.TestCase):
    NAMES = ["colour.dark", "colour.warm", "composition.empty", "stroke.edge",
             "segmentation.coverage", "shape.count", "quality.blur"]

    def test_all_features_keeps_everything(self):
        from doar.ablation import select_columns
        keep = select_columns(self.NAMES, [])
        self.assertEqual(len(keep), len(self.NAMES))

    def test_dropping_colour_removes_colour_columns(self):
        from doar.ablation import select_columns
        keep = select_columns(self.NAMES, ["colour"])
        kept = [self.NAMES[i] for i in keep]
        self.assertFalse(any(n.startswith("colour.") for n in kept))
        self.assertIn("composition.empty", kept)

    def test_dropping_multiple_families(self):
        from doar.ablation import select_columns
        keep = select_columns(self.NAMES, ["colour", "stroke"])
        kept = [self.NAMES[i] for i in keep]
        self.assertNotIn("colour.dark", kept)
        self.assertNotIn("stroke.edge", kept)
        self.assertIn("segmentation.coverage", kept)

    def test_configs_cover_required_ablations(self):
        from doar.ablation import ablation_configs
        names = {c["name"] for c in ablation_configs()}
        for required in ("all_features", "without_colour", "without_composition",
                         "without_stroke", "without_segmentation"):
            self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
