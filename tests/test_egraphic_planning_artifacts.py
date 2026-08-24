"""Phase G0.2, Steps 8-11 -- tests for E_GRAPHIC_EXPERIMENT_PLAN.json and
E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv. Both are RESEARCH-ONLY planning
files: no experiment has been run, no method chosen as a winner, no new
dependency installed."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.egraphic_annotation_spec_build import (
    E_GRAPHIC_ANNOTATION_SPEC_PATH, FIELDNAMES as ANNOTATION_FIELDNAMES, build_egraphic_annotation_spec,
)
from doar.egraphic_plan_build import E_GRAPHIC_EXPERIMENT_PLAN_PATH, build_egraphic_experiment_plan


class ExperimentPlanTests(unittest.TestCase):
    def test_all_six_groups_present(self):
        doc = build_egraphic_experiment_plan()
        expected_groups = {
            "A_LINE_WIDTH_QUALITY", "B_LINE_ORIENTATION_ORGANIZATION", "C_SCRIBBLE_CROSSING_COMPLEXITY",
            "D_COLOUR_SHADING", "E_SPATIAL_ORGANIZATION", "F_REGION_SPECIFIC_FUTURE",
        }
        self.assertEqual(set(doc["task_count_by_group"]), expected_groups)
        for group, count in doc["task_count_by_group"].items():
            self.assertGreater(count, 0, group)

    def test_no_task_declares_a_winner(self):
        doc = build_egraphic_experiment_plan()
        for task in doc["tasks"]:
            self.assertIn(task["status"], ("PLANNED_NOT_STARTED",), task["experiment_id"])

    def test_every_task_has_required_fields(self):
        doc = build_egraphic_experiment_plan()
        required = ("experiment_id", "feature_group", "target_observable", "current_method",
                    "ground_truth_type", "suggested_metric")
        for task in doc["tasks"]:
            for field in required:
                self.assertTrue(str(task[field]).strip(), f"{task.get('experiment_id')} missing {field}")

    def test_no_task_id_duplicated(self):
        doc = build_egraphic_experiment_plan()
        ids = [t["experiment_id"] for t in doc["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_region_specific_group_all_require_segmentation(self):
        doc = build_egraphic_experiment_plan()
        for task in doc["tasks"]:
            if task["feature_group"] == "F_REGION_SPECIFIC_FUTURE":
                self.assertTrue(task["requires_segmentation"], task["experiment_id"])
                self.assertTrue(task["current_method"].startswith("NOT_YET_IMPLEMENTED"), task["experiment_id"])

    def test_region_colouring_direction_experiment_specified(self):
        doc = build_egraphic_experiment_plan()
        spec = doc["region_colouring_direction_experiment"]
        self.assertEqual(spec["experiment_name"], "REGION_COLOURING_DIRECTION")
        self.assertEqual(spec["current_status"], "NEEDS_SEGMENTATION_MASK")
        self.assertGreaterEqual(len(spec["measurements_planned"]), 9)
        self.assertIn("mask_quality_required", spec)
        self.assertIn("ground_truth_annotation_the_psychologist_would_provide", spec)
        # Must not silently approximate with a bounding box.
        self.assertIn("bounding-box", " ".join(spec["explicitly_not_implemented_this_phase"]).lower())

    def test_line_width_limitation_preserved_and_registered_as_baseline(self):
        doc = build_egraphic_experiment_plan()
        limitation = doc["line_width_limitation"]
        self.assertEqual(limitation["feature_id"], "line.mean_width")
        self.assertEqual(limitation["current_method_status"], "BASELINE_METHOD")
        self.assertTrue(limitation["limitation"].strip())
        self.assertTrue(limitation["no_new_dependency_installed_this_phase"])

    def test_no_new_dependency_actually_imported_by_this_module(self):
        """Mentioning a future candidate dependency in a documentation
        string (e.g. 'requires_new_dependency': 'scikit-image...') is
        expected and fine -- the guard is that no ACTUAL import statement
        for one exists, so this checks import LINES specifically, not
        substring mentions anywhere in the module's text."""
        import inspect
        from doar import egraphic_plan_build as mod
        source = inspect.getsource(mod)
        import_lines = [ln.strip() for ln in source.splitlines()
                        if ln.strip().startswith(("import ", "from ")) and "__future__" not in ln]
        for line in import_lines:
            self.assertNotIn("cv2", line)
            for forbidden in ("skimage", "sam2", "deeplsd", "sold2"):
                self.assertNotIn(forbidden, line.lower())
        # And confirm the module only actually imports the plain stdlib it needs.
        self.assertEqual(set(import_lines), {"import json", "from pathlib import Path", "from typing import Any"})

    def test_file_on_disk_matches_build_output_when_present(self):
        if not E_GRAPHIC_EXPERIMENT_PLAN_PATH.exists():
            self.skipTest("E_GRAPHIC_EXPERIMENT_PLAN.json not yet generated in this checkout")
        on_disk = json.loads(E_GRAPHIC_EXPERIMENT_PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, build_egraphic_experiment_plan())


class AnnotationSpecTests(unittest.TestCase):
    def test_no_row_asks_for_a_psychological_judgment(self):
        forbidden_phrases = ("unstable", "dysregulat", "trauma", "abuse", "diagnos", "disorder", "ocd", "adhd")
        rows = build_egraphic_annotation_spec()
        for row in rows:
            haystack = " ".join(str(v) for v in row.values()).lower()
            for phrase in forbidden_phrases:
                self.assertNotIn(phrase, haystack, f"{row['annotation_id']} contains forbidden phrase {phrase!r}")

    def test_every_row_has_allowed_values_and_a_definition(self):
        for row in build_egraphic_annotation_spec():
            self.assertTrue(row["plain_definition"].strip(), row["annotation_id"])
            self.assertTrue(row["allowed_values"].strip(), row["annotation_id"])

    def test_no_duplicate_annotation_ids(self):
        ids = [r["annotation_id"] for r in build_egraphic_annotation_spec()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_region_specific_rows_flag_region_annotation_required(self):
        for row in build_egraphic_annotation_spec():
            if "region" in row["canonical_observable_id"] or "REGION" in row["human_label"]:
                self.assertEqual(row["region_annotation_required"], "yes", row["annotation_id"])

    def test_orientation_consistency_worked_example_matches_task_instructions(self):
        by_id = {r["annotation_id"]: r for r in build_egraphic_annotation_spec()}
        row = by_id["ANNOT_07_ORIENTATION_CONSISTENCY"]
        self.assertEqual(row["allowed_values"], "LOW|MEDIUM|HIGH|CANNOT_ASSESS")

    def test_scribble_appearance_worked_example_matches_task_instructions(self):
        by_id = {r["annotation_id"]: r for r in build_egraphic_annotation_spec()}
        row = by_id["ANNOT_09_SCRIBBLE_APPEARANCE"]
        self.assertEqual(row["allowed_values"], "PRESENT|NOT_PRESENT|UNCERTAIN")

    def test_file_on_disk_matches_build_output_when_present(self):
        if not E_GRAPHIC_ANNOTATION_SPEC_PATH.exists():
            self.skipTest("E_GRAPHIC_EXPERT_ANNOTATION_SPEC.csv not yet generated in this checkout")
        with open(E_GRAPHIC_ANNOTATION_SPEC_PATH, encoding="utf-8") as f:
            on_disk = list(csv.DictReader(f))
        self.assertEqual(on_disk, build_egraphic_annotation_spec())


if __name__ == "__main__":
    unittest.main()
