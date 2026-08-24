"""Phase G0.1, Step 3 -- tests for FORMAL_FEATURE_DEFINITION_TABLE.csv.

Verifies the table's feature_id set is EXACTLY the set of keys
compute_formal_features() really returns (no undocumented feature, no
stale/renamed entry left behind), every row is filled in for every
required column, and every row's psychological_interpretation_status is
OBJECTIVE_ONLY (Step 3's explicit requirement for this phase)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.case_artifacts import resolve_analysis_artifacts
from doar.formal_feature_definition_table_build import (
    FIELDNAMES, FORMAL_FEATURE_DEFINITION_TABLE_PATH, build_formal_feature_definition_table,
)
from doar.formal_features import compute_formal_features
from doar.timed_analysis import analyze_image_with_timing


def _real_feature_ids() -> set[str]:
    with tempfile.TemporaryDirectory() as d:
        case_dir = Path(d) / "case"
        case_dir.mkdir()
        path = case_dir / "drawing.png"
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).line((50, 150, 250, 150), fill="black", width=5)
        image.save(path)
        analyze_image_with_timing(str(path), str(case_dir), None)
        analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
        resolved = resolve_analysis_artifacts(analysis, case_dir)
        return set(compute_formal_features(str(path), resolved).keys())


class DefinitionTableCoversExactlyTheRealImplementationTests(unittest.TestCase):
    def test_table_feature_ids_exactly_match_real_implementation_output(self):
        table_ids = {r["feature_id"] for r in build_formal_feature_definition_table()}
        self.assertEqual(table_ids, _real_feature_ids())

    def test_no_duplicate_feature_ids(self):
        ids = [r["feature_id"] for r in build_formal_feature_definition_table()]
        self.assertEqual(len(ids), len(set(ids)))


class DefinitionTableCompletenessTests(unittest.TestCase):
    def test_every_row_has_every_required_column_filled(self):
        rows = build_formal_feature_definition_table()
        # `notes` is the only column allowed to be genuinely empty (not
        # every feature needs an extra note).
        required = [f for f in FIELDNAMES if f != "notes"]
        for row in rows:
            for field in required:
                self.assertTrue(str(row[field]).strip(), f"{row['feature_id']} has an empty '{field}'")

    def test_every_row_is_objective_only(self):
        for row in build_formal_feature_definition_table():
            self.assertEqual(row["psychological_interpretation_status"], "OBJECTIVE_ONLY",
                              f"{row['feature_id']} is not OBJECTIVE_ONLY -- no new interpretation allowed this phase")

    def test_renamed_features_are_documented_with_their_old_name(self):
        by_id = {r["feature_id"]: r for r in build_formal_feature_definition_table()}
        self.assertIn("crossing_density", by_id["stroke.junction_corner_density_proxy"]["notes"].lower())
        self.assertIn("scribble_density", by_id["stroke.scribble_candidate_score"]["notes"])
        self.assertIn("foreground_coverage", by_id["colour.chromatic_coverage"]["what_it_does_NOT_measure"])
        self.assertIn("fill.uniformity", by_id["fill.global_spatial_density_uniformity"]["notes"])


class DefinitionTableFileOnDiskTests(unittest.TestCase):
    def test_written_file_matches_build_output_when_present(self):
        if not FORMAL_FEATURE_DEFINITION_TABLE_PATH.exists():
            self.skipTest("FORMAL_FEATURE_DEFINITION_TABLE.csv not yet generated in this checkout")
        import csv
        with open(FORMAL_FEATURE_DEFINITION_TABLE_PATH, encoding="utf-8") as f:
            on_disk = list(csv.DictReader(f))
        fresh = build_formal_feature_definition_table()
        self.assertEqual(on_disk, fresh)


if __name__ == "__main__":
    unittest.main()
