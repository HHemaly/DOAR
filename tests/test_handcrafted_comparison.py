from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.handcrafted_comparison import (
    GROUP_PREFIXES, handcrafted_group_configs, run_handcrafted_group_comparison,
    select_group_columns,
)


class SelectGroupColumnsTests(unittest.TestCase):
    def test_hog_only_selects_hog_columns(self):
        names = ["colour.red", "composition.x", "hog.bin_0000", "hog.bin_0001"]
        keep = select_group_columns(names, ["hog"])
        self.assertEqual(keep, [2, 3])

    def test_geometry_spans_three_prefixes(self):
        names = ["composition.x", "segmentation.y", "shape.z", "colour.red", "quality.w"]
        keep = select_group_columns(names, ["geometry"])
        self.assertEqual(keep, [0, 1, 2])

    def test_combined_group_unions_prefixes(self):
        names = ["hog.bin_0000", "colour.red", "composition.x", "quality.w"]
        keep = select_group_columns(names, ["hog", "colour", "geometry"])
        self.assertEqual(keep, [0, 1, 2])  # quality excluded

    def test_quality_and_stroke_are_never_in_any_group(self):
        all_prefixes = {p for prefixes in GROUP_PREFIXES.values() for p in prefixes}
        self.assertNotIn("quality.", all_prefixes)
        self.assertNotIn("stroke.", all_prefixes)

    def test_four_configs_match_the_spec(self):
        names = {c["name"] for c in handcrafted_group_configs()}
        self.assertEqual(names, {"hog_only", "colour_only", "geometry_only", "hog_colour_geometry"})


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class RunHandcraftedGroupComparisonTests(unittest.TestCase):
    def _fixture(self, root: Path):
        classes = ["Angry", "Fear", "Happy", "Sad"]
        obj_rows, hog_rows = [], []
        for i in range(24):
            cls = classes[i % 4]
            split = "train" if i < 16 else "valid"
            obj_rows.append({
                "image_id": f"id{i}", "split": split, "class": cls,
                "colour.red_ratio": float(classes.index(cls)) / 4, "colour.diversity": 1.0,
                "composition.centroid_x": 0.5, "segmentation.confidence": 0.9, "shape.contour_proxy_count": 2.0,
            })
            hog_rows.append({"image_id": f"id{i}", "split": split, "class": cls,
                             "hog.bin_0000": float(i % 3), "hog.bin_0001": 0.1})
        obj_csv, hog_csv = root / "objective.csv", root / "hog.csv"
        _write_csv(obj_csv, obj_rows)
        _write_csv(hog_csv, hog_rows)
        return obj_csv, hog_csv

    def test_end_to_end_produces_four_configurations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            obj_csv, hog_csv = self._fixture(root)
            result = run_handcrafted_group_comparison(
                obj_csv, hog_csv, root / "out", models=["logistic_regression"], seeds=(42,))
            configs = {r["configuration"] for r in result["leaderboard"]}
            self.assertEqual(configs, {"hog_only", "colour_only", "geometry_only", "hog_colour_geometry"})
            for row in result["leaderboard"]:
                self.assertFalse(row.get("test_used", False))
                self.assertIn("mean_macro_f1", row)

    def test_mismatched_split_between_sources_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            obj_csv, hog_csv = self._fixture(root)
            # Corrupt one hog row's split so it disagrees with the objective CSV.
            rows = list(csv.DictReader(open(hog_csv, encoding="utf-8")))
            rows[0]["split"] = "valid" if rows[0]["split"] == "train" else "train"
            _write_csv(hog_csv, rows)
            with self.assertRaises(ValueError):
                run_handcrafted_group_comparison(obj_csv, hog_csv, root / "out",
                                                 models=["logistic_regression"], seeds=(42,))

    def test_no_common_image_ids_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            obj_csv, hog_csv = self._fixture(root)
            rows = list(csv.DictReader(open(hog_csv, encoding="utf-8")))
            for r in rows:
                r["image_id"] = "different_" + r["image_id"]
            _write_csv(hog_csv, rows)
            with self.assertRaises(ValueError):
                run_handcrafted_group_comparison(obj_csv, hog_csv, root / "out",
                                                 models=["logistic_regression"], seeds=(42,))

    def test_result_artifacts_written_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            obj_csv, hog_csv = self._fixture(root)
            out = root / "out"
            run_handcrafted_group_comparison(obj_csv, hog_csv, out,
                                             models=["logistic_regression"], seeds=(42,))
            self.assertTrue((out / "handcrafted_group_leaderboard.json").exists())
            self.assertTrue((out / "handcrafted_group_leaderboard.csv").exists())
            run_dirs = list((out / "runs").iterdir())
            self.assertEqual(len(run_dirs), 4)  # 4 configs x 1 model x 1 seed


if __name__ == "__main__":
    unittest.main()
