"""Tests for this session's thesis.py additions: the 3 new auto-discovered
leaderboard figure types, and build_cross_family_model_comparison()'s
mandatory, validated data_provenance labeling."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.thesis import DATA_PROVENANCE_LABELS, build_cross_family_model_comparison


def _write_leaderboard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"leaderboard": rows}), encoding="utf-8")


class DataProvenanceLabelTests(unittest.TestCase):
    def test_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_cross_family_model_comparison(d, data_provenance="not_a_real_label")

    def test_accepts_all_four_documented_labels(self):
        with tempfile.TemporaryDirectory() as d:
            for label in DATA_PROVENANCE_LABELS:
                result = build_cross_family_model_comparison(d, data_provenance=label)
                self.assertEqual(result["data_provenance"], label)

    def test_non_clean_labels_carry_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            result = build_cross_family_model_comparison(d, data_provenance="preliminary_contaminated_split")
            self.assertIn("NOT a clean-split", result["warning"])


class CrossFamilyComparisonTests(unittest.TestCase):
    def test_pulls_best_row_from_each_discovered_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_leaderboard(root / "A" / "validation_leaderboard.json", [
                {"model": "logistic_regression", "mean_macro_f1": 0.55, "std_macro_f1": 0.02, "test_used": False},
                {"model": "random_forest", "mean_macro_f1": 0.50, "std_macro_f1": 0.01, "test_used": False},
            ])
            _write_leaderboard(root / "D" / "deep_comparison.json", [
                {"model": "efficientnet_b0", "mean_valid_macro_f1": 0.73, "std_valid_macro_f1": 0.01},
            ])
            result = build_cross_family_model_comparison(root, data_provenance="smoke_test_not_a_result")
            families = {r["family"]: r for r in result["rows"]}
            self.assertEqual(families["A_objective_features"]["best_configuration"], "logistic_regression")
            self.assertEqual(families["A_objective_features"]["mean_macro_f1"], 0.55)
            self.assertEqual(families["D_E_deep_image_models"]["mean_macro_f1"], 0.73)
            self.assertEqual(len(result["missing_families"]), 3)  # B, C, F not present

    def test_every_row_stamped_with_the_same_provenance_label(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_leaderboard(root / "validation_leaderboard.json", [
                {"model": "logistic_regression", "mean_macro_f1": 0.5, "std_macro_f1": 0.0},
            ])
            result = build_cross_family_model_comparison(root, data_provenance="clean_development_split")
            self.assertTrue(all(r["data_provenance"] == "clean_development_split" for r in result["rows"]))

    def test_empty_root_reports_all_families_missing_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            result = build_cross_family_model_comparison(d, data_provenance="smoke_test_not_a_result")
            self.assertEqual(result["rows"], [])
            self.assertEqual(len(result["missing_families"]), 5)

    def test_writes_output_file_when_path_given(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_leaderboard(root / "validation_leaderboard.json", [
                {"model": "logistic_regression", "mean_macro_f1": 0.5, "std_macro_f1": 0.0},
            ])
            out_path = root / "comparison.json"
            build_cross_family_model_comparison(root, data_provenance="smoke_test_not_a_result",
                                                output_path=out_path)
            self.assertTrue(out_path.exists())
            reloaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["data_provenance"], "smoke_test_not_a_result")


class GenerateThesisOutputsNewFigureTypesTests(unittest.TestCase):
    def test_all_three_new_leaderboard_types_generate_figures_when_present(self):
        from doar.thesis import generate_thesis_outputs
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_leaderboard(root / "A" / "validation_leaderboard.json", [
                {"model": "logistic_regression", "mean_macro_f1": 0.5, "std_macro_f1": 0.01}])
            (root / "B").mkdir()
            (root / "B" / "handcrafted_group_leaderboard.json").write_text(json.dumps({
                "leaderboard": [{"configuration": "hog_only", "mean_macro_f1": 0.4, "std_macro_f1": 0.02}]
            }), encoding="utf-8")
            (root / "C").mkdir()
            (root / "C" / "embedding_classifier_leaderboard.json").write_text(json.dumps({
                "leaderboard": [{"model": "logistic_regression", "mean_macro_f1": 0.6, "std_macro_f1": 0.03}]
            }), encoding="utf-8")
            result = generate_thesis_outputs(root)
            figure_names = {Path(f["figure"]).stem for f in result["figures"]}
            self.assertIn("objective_feature_classifiers", figure_names)
            self.assertIn("handcrafted_group_comparison", figure_names)
            self.assertIn("embedding_classifier_comparison", figure_names)


if __name__ == "__main__":
    unittest.main()
