"""Focused test for scripts/run_development_benchmark.py: loading and
computing salience/agreement metrics from the two annotators' saved
records must never write back to those files -- human annotation is the
reference; nothing downstream may mutate it, not even by accident.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("run_development_benchmark", ROOT / "scripts" / "run_development_benchmark.py")
rdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdb)


class OriginalAnnotationFilesUntouchedTests(unittest.TestCase):
    def test_load_and_salience_computation_leaves_files_byte_identical(self):
        import tempfile

        a1_record = {"annotator_id": "A1", "image_id": "fixture", "items": [
            {"label": "sun", "location": "top", "salient": True, "confidence": "clear", "note": ""},
            {"label": "tree", "location": "left", "salient": False, "confidence": "clear", "note": ""},
        ], "complete": True}
        a2_record = {"annotator_id": "A2", "image_id": "fixture", "items": [
            {"label": "sun", "location": "top-right", "salient": True, "confidence": "clear", "note": ""},
            {"label": "car", "location": "bottom", "salient": True, "confidence": "ambiguous", "note": "small"},
        ], "complete": True}

        with tempfile.TemporaryDirectory() as tmp:
            original_dir = rdb.ANNOTATIONS_DIR
            rdb.ANNOTATIONS_DIR = Path(tmp)
            try:
                (Path(tmp) / "A1").mkdir(parents=True)
                (Path(tmp) / "A2").mkdir(parents=True)
                a1_path = Path(tmp) / "A1" / "fixture.json"
                a2_path = Path(tmp) / "A2" / "fixture.json"
                a1_path.write_text(json.dumps(a1_record, indent=2), encoding="utf-8")
                a2_path.write_text(json.dumps(a2_record, indent=2), encoding="utf-8")
                before_a1, before_a2 = a1_path.read_bytes(), a2_path.read_bytes()

                human = rdb.load_human_annotations("fixture")
                self.assertIsNotNone(human["A1"])
                self.assertIsNotNone(human["A2"])
                self.assertEqual(len(human["A1"]), 2)
                self.assertEqual(len(human["A2"]), 2)

                from doar import benchmark_metrics as bm
                salience = bm.normalized_salience(human["A1"], human["A2"])
                # only salient=True items count: A1's "tree" is excluded.
                self.assertEqual({i.label for i in salience["salient"]}, {"sun", "car"})
                self.assertEqual({i.label for i in salience["core_salient"]}, {"sun"})

                after_a1, after_a2 = a1_path.read_bytes(), a2_path.read_bytes()
                self.assertEqual(before_a1, after_a1)
                self.assertEqual(before_a2, after_a2)
            finally:
                rdb.ANNOTATIONS_DIR = original_dir

    def test_load_human_annotations_returns_none_for_missing_annotator_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            original_dir = rdb.ANNOTATIONS_DIR
            rdb.ANNOTATIONS_DIR = Path(tmp)
            try:
                human = rdb.load_human_annotations("never_annotated")
                self.assertIsNone(human["A1"])
                self.assertIsNone(human["A2"])
            finally:
                rdb.ANNOTATIONS_DIR = original_dir

    def test_compute_visual_metrics_uses_new_schema_end_to_end(self):
        from doar import benchmark_metrics as bm

        human = {
            "A1": [bm.HumanAnnotationItem(label="sun", location="top", salient=True, confidence="clear")],
            "A2": [bm.HumanAnnotationItem(label="sun", location="top-right", salient=True, confidence="clear")],
        }
        candidates_full = [{"entity_type": "sun", "bbox": (0, 0, 1, 1), "confidence": 0.9,
                             "case_verification_status": "verified"}]
        metrics = rdb.compute_visual_metrics(["sun"], candidates_full, human)
        self.assertEqual(metrics["per_annotator"]["A1"]["precision_recall_f1"]["precision"], 1.0)
        self.assertEqual(metrics["salience"]["primary_salient_recall"]["salient_recall"], 1.0)
        self.assertEqual(metrics["salience"]["sensitivity_salient_recall"]["salient_recall"], 1.0)
        self.assertIsNotNone(metrics["inter_annotator_agreement"])


if __name__ == "__main__":
    unittest.main()
