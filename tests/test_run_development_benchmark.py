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


class _patched:
    """Temporarily overrides module-level attributes on `rdb` (or any
    object), restoring the originals on exit -- used throughout this
    file so each test's tempdir substitution can never leak into other
    tests or into the real annotations/outputs trees."""

    def __init__(self, obj, **attrs):
        self.obj = obj
        self.attrs = attrs
        self._originals = {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self._originals[name] = getattr(self.obj, name)
            setattr(self.obj, name, value)
        return self

    def __exit__(self, *exc_info):
        for name, value in self._originals.items():
            setattr(self.obj, name, value)


class LiveCacheTests(unittest.TestCase):
    def test_cached_live_image_reused_with_zero_api_calls(self):
        import tempfile
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as live_cache_tmp:
            live_cache_dir = Path(live_cache_tmp)
            fixture_rows = [{"drawing_id": "fixture", "observation_id": "fixture_c00",
                              "observer_candidate": {"label": "sun", "alternative_labels": [], "bbox": None,
                                                      "confidence": 0.9, "entity_type": "sun"},
                              "verifier_independent_label": "sun", "verifier_independent_alternative_labels": [],
                              "verifier_confidence": 0.9, "verification_status": "verified",
                              "verifier_notes": "", "observer_model": "m", "verifier_model": "m",
                              "verifier_source_note": "", "runtime_seconds": 1.0}]
            (live_cache_dir / "fixture_verification.json").write_text(json.dumps(fixture_rows), encoding="utf-8")

            with _patched(rdb, LIVE_CACHE_DIR=live_cache_dir, SAVED_VERIFICATION_DIRS=[]):
                fake_live_call = Mock(side_effect=AssertionError("live Observer/Verifier must not be called"))
                with _patched(rdb, run_live_observer_and_verifier=fake_live_call):
                    rows, source_path = rdb.find_saved_verification_rows("fixture")
                    self.assertIsNotNone(rows)
                    self.assertEqual(rows, fixture_rows)
                    self.assertEqual(source_path, live_cache_dir / "fixture_verification.json")

                    # Mirrors main()'s own gating condition -- rows is
                    # already non-None, so the live path is never reached.
                    if rows is None:
                        rdb.run_live_observer_and_verifier("fixture", Path("unused.jpg"))
                    fake_live_call.assert_not_called()

    def test_missing_image_returns_none_triggering_live_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as live_cache_tmp:
            with _patched(rdb, LIVE_CACHE_DIR=Path(live_cache_tmp), SAVED_VERIFICATION_DIRS=[]):
                rows, source_path = rdb.find_saved_verification_rows("never_seen_image")
                self.assertIsNone(rows)
                self.assertIsNone(source_path)
                # main()'s gate `if rows is None and args.run_live:` would
                # now be reached for this image.
                self.assertTrue(rows is None)

    def test_legacy_saved_gemini_data_still_works_when_cache_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as empty_live_cache:
            # Real, already-saved dev-check data (from prior phases) --
            # SAVED_VERIFICATION_DIRS left at its real, module-computed
            # value; only the (empty) live cache is substituted.
            with _patched(rdb, LIVE_CACHE_DIR=Path(empty_live_cache)):
                self.assertTrue(rdb.SAVED_VERIFICATION_DIRS, "expected at least one real gemini_verifier_dev_check_* dir")
                rows, source_path = rdb.find_saved_verification_rows("h38")
                self.assertIsNotNone(rows)
                self.assertGreater(len(rows), 0)
                self.assertIn("gemini_verifier_dev_check_", str(source_path))

    def test_live_cache_takes_priority_over_legacy_when_both_exist(self):
        import tempfile

        with tempfile.TemporaryDirectory() as live_cache_tmp:
            live_cache_dir = Path(live_cache_tmp)
            (live_cache_dir / "h38_verification.json").write_text(json.dumps([{"cache": "hit"}]), encoding="utf-8")
            with _patched(rdb, LIVE_CACHE_DIR=live_cache_dir):
                # SAVED_VERIFICATION_DIRS also has real h38 data -- the
                # live cache must win.
                rows, source_path = rdb.find_saved_verification_rows("h38")
                self.assertEqual(rows, [{"cache": "hit"}])
                self.assertEqual(source_path, live_cache_dir / "h38_verification.json")

    def test_atomic_write_produces_valid_json_and_no_leftover_tmp_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "fixture_verification.json"
            rdb._atomic_write_json(target, [{"a": 1}])
            self.assertTrue(target.exists())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), [{"a": 1}])
            leftover_tmp_files = list(target.parent.glob("*.tmp-*"))
            self.assertEqual(leftover_tmp_files, [])

    def test_live_cache_path_naming(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with _patched(rdb, LIVE_CACHE_DIR=Path(tmp)):
                path = rdb.live_cache_path("p2b_0005")
                self.assertEqual(path, Path(tmp) / "p2b_0005_verification.json")


class AnnotationCountIndependentOfObserverDataTests(unittest.TestCase):
    """Bug: images_with_annotations previously only counted images that
    ALSO had Observer/Verifier data (the "no_observer_data" branch never
    set has_human_annotations at all). Runs the REAL main() end to end
    (no --run-live, no network) against fixture annotations for all 15
    images but Observer/Verifier data cached for only 5, proving the
    fixed count is 15, not 5."""

    def test_15_annotated_images_report_15_even_with_only_5_observer_backed(self):
        import shutil
        import tempfile

        images = rdb.load_development_set()
        self.assertEqual(len(images), 15)
        observer_backed_ids = {im["image_id"] for im in images[:5]}

        with tempfile.TemporaryDirectory() as annotations_tmp, tempfile.TemporaryDirectory() as live_cache_tmp:
            annotations_dir = Path(annotations_tmp)
            live_cache_dir = Path(live_cache_tmp)
            for annotator_id in ("A1", "A2"):
                (annotations_dir / annotator_id).mkdir(parents=True)
            for image in images:
                record = {"annotator_id": "A1", "image_id": image["image_id"],
                          "items": [{"label": "sun", "location": "top", "salient": True,
                                     "confidence": "clear", "note": ""}], "complete": True}
                for annotator_id in ("A1", "A2"):
                    record["annotator_id"] = annotator_id
                    (annotations_dir / annotator_id / f"{image['image_id']}.json").write_text(
                        json.dumps(record), encoding="utf-8")
                if image["image_id"] in observer_backed_ids:
                    (live_cache_dir / f"{image['image_id']}_verification.json").write_text(
                        json.dumps([]), encoding="utf-8")

            output_root = ROOT / "outputs" / "prototype_cases"
            before_dirs = {p.name for p in output_root.glob("development_benchmark_*")}
            old_argv = sys.argv

            with _patched(rdb, ANNOTATIONS_DIR=annotations_dir, LIVE_CACHE_DIR=live_cache_dir,
                          SAVED_VERIFICATION_DIRS=[]):
                sys.argv = ["run_development_benchmark.py"]
                try:
                    rdb.main()
                finally:
                    sys.argv = old_argv

            after_dirs = {p.name for p in output_root.glob("development_benchmark_*")}
            new_dirs = after_dirs - before_dirs
            self.assertEqual(len(new_dirs), 1, "expected exactly one new development_benchmark_* run directory")
            run_dir = output_root / next(iter(new_dirs))
            try:
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["images_with_human_annotations"], 15)
                self.assertEqual(summary["images_with_saved_or_live_observer_data"], 5)
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
