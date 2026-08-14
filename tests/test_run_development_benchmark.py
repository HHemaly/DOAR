"""Focused test for scripts/run_development_benchmark.py: loading and
computing salience/agreement metrics from the two annotators' saved
Pass-1 records must never write back to those files -- human annotation
is the reference; nothing downstream may mutate it, not even by
accident.
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

        a1_pass1 = {"annotator_id": "A1", "image_id": "fixture", "pass": 1,
                    "items": [{"label": "sun", "location": "top"}, {"label": "tree", "location": "left"}],
                    "elapsed_seconds": 42.0, "timed_out": True}
        a2_pass1 = {"annotator_id": "A2", "image_id": "fixture", "pass": 1,
                    "items": [{"label": "sun", "location": "top-right"}, {"label": "car", "location": "bottom"}],
                    "elapsed_seconds": 55.0, "timed_out": False}

        with tempfile.TemporaryDirectory() as tmp:
            original_dir = rdb.ANNOTATIONS_DIR
            rdb.ANNOTATIONS_DIR = Path(tmp)
            try:
                (Path(tmp) / "A1").mkdir(parents=True)
                (Path(tmp) / "A2").mkdir(parents=True)
                a1_path = Path(tmp) / "A1" / "fixture_pass1.json"
                a2_path = Path(tmp) / "A2" / "fixture_pass1.json"
                a1_path.write_text(json.dumps(a1_pass1, indent=2), encoding="utf-8")
                a2_path.write_text(json.dumps(a2_pass1, indent=2), encoding="utf-8")
                before_a1, before_a2 = a1_path.read_bytes(), a2_path.read_bytes()

                human = rdb.load_human_annotations("fixture")
                self.assertIsNotNone(human["A1"]["pass1"])
                self.assertIsNotNone(human["A2"]["pass1"])

                from doar import benchmark_metrics as bm
                salience = bm.normalized_pass1_salience(human["A1"]["pass1"], human["A2"]["pass1"])
                self.assertEqual({i.label for i in salience["salient"]}, {"sun", "tree", "car"})
                self.assertEqual({i.label for i in salience["core_salient"]}, {"sun"})

                after_a1, after_a2 = a1_path.read_bytes(), a2_path.read_bytes()
                self.assertEqual(before_a1, after_a1)
                self.assertEqual(before_a2, after_a2)
            finally:
                rdb.ANNOTATIONS_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
