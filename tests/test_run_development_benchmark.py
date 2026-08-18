"""Focused test for scripts/run_development_benchmark.py: loading and
computing salience/agreement metrics from the two annotators' saved
records must never write back to those files -- human annotation is the
reference; nothing downstream may mutate it, not even by accident.
"""
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

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
        """Portable fallback-SEARCH-ORDER logic (live cache miss -> fall
        through to the legacy `gemini_verifier_dev_check_*` dirs) --
        exercised against a small synthetic fixture rather than the real,
        generated `outputs/prototype_cases/gemini_verifier_dev_check_*`
        data (optional, never committed, absent on a clean checkout e.g.
        CI). This tests the SAME branch in `find_saved_verification_rows`
        the original real-data version did; it is not a weaker test, just
        a portable one -- `test_live_cache_takes_priority_over_legacy_
        when_both_exist` below is the fixture-based sibling proving the
        opposite priority order the same way."""
        import tempfile

        with tempfile.TemporaryDirectory() as empty_live_cache, tempfile.TemporaryDirectory() as legacy_root:
            legacy_dir = Path(legacy_root) / "gemini_verifier_dev_check_19990101_000000"
            legacy_dir.mkdir()
            fixture_rows = [{"drawing_id": "h38", "observation_id": "h38_c00",
                              "observer_candidate": {"label": "sun", "alternative_labels": [], "bbox": None,
                                                      "confidence": 0.9, "entity_type": "sun"},
                              "verifier_independent_label": "sun", "verifier_independent_alternative_labels": [],
                              "verifier_confidence": 0.9, "verification_status": "verified",
                              "verifier_notes": "", "observer_model": "m", "verifier_model": "m",
                              "verifier_source_note": "", "runtime_seconds": 1.0}]
            (legacy_dir / "h38_verification.json").write_text(json.dumps(fixture_rows), encoding="utf-8")

            with _patched(rdb, LIVE_CACHE_DIR=Path(empty_live_cache), SAVED_VERIFICATION_DIRS=[legacy_dir]):
                rows, source_path = rdb.find_saved_verification_rows("h38")
                self.assertIsNotNone(rows)
                self.assertEqual(rows, fixture_rows)
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


from doar.visual_observer import GeminiVisualObserver, GeminiVisualVerifier  # noqa: E402


def _fake_gemini_observer_response(candidates_payload, *, response_id="gemini_resp_test_1",
                                    model_version="gemini-3.6-flash"):
    """Same wire shape `tests/test_visual_observer.py::_fake_gemini_response`
    builds -- a real Gemini generateContent body wrapping the structured
    candidate JSON as `.analyze()`'s own parser expects."""
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps({"candidates": candidates_payload})}],
                                     "role": "model"}, "finishReason": "STOP", "index": 0}],
        "modelVersion": model_version, "responseId": response_id,
    }
    return json.dumps(body).encode("utf-8")


def _fake_gemini_verifier_response(label, alternative_labels=(), confidence=None,
                                    *, response_id="gemini_verifier_resp_1", model_version="gemini-3.5-flash-lite"):
    inner = {"label": label, "alternative_labels": list(alternative_labels), "confidence": confidence}
    body = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(inner)}], "role": "model"},
                         "finishReason": "STOP", "index": 0}],
        "modelVersion": model_version, "responseId": response_id,
    }
    return json.dumps(body).encode("utf-8")


def _tmp_real_image_path(size=(64, 64), color=(255, 220, 0)):
    """The verifier's `.verify()` crops the image via PIL, so unlike the
    observer's fixture image (arbitrary bytes are fine -- the mocked
    transport never actually decodes them), this needs real, decodable
    pixel data."""
    import tempfile

    from PIL import Image
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.new("RGB", size, color).save(path)
    return path


class LiveObserverVerifierInterfaceTests(unittest.TestCase):
    """Regression test for the exact bug this phase fixes: a prior version
    called `observer.observe(image_path).candidates`, an interface that
    never existed on the frozen `GeminiVisualObserver` (the real method is
    `.analyze(image_path) -> list[VisualObserverCandidate]`), raising
    `AttributeError` on the very first `--run-live` invocation.

    Uses REAL `GeminiVisualObserver`/`GeminiVisualVerifier` instances
    (not mocks/fakes of the classes) with their own `request_fn` swapped
    for a canned response -- the officially-supported test seam these
    classes already document and `tests/test_visual_observer.py` already
    uses throughout -- so `run_live_observer_and_verifier` exercises the
    REAL `.analyze()`/`.verify()` method resolution. Had the bug still
    been present, this test would fail with the same AttributeError the
    real `--run-live` run hit, not a mock-shaped false pass."""

    def test_live_observer_and_verifier_uses_the_real_analyze_and_verify_interface(self):
        observer_key_var = "DOAR_TEST_RDB_OBSERVER_KEY_UNUSED"
        verifier_key_var = "DOAR_TEST_RDB_VERIFIER_KEY_UNUSED"

        candidates_payload = [
            {"label": "car", "alternative_labels": ["vehicle"], "entity_type": "object",
             "bbox": [0.1, 0.1, 0.4, 0.4], "count": 1, "confidence": 0.8},
        ]
        observer_raw = _fake_gemini_observer_response(candidates_payload, response_id="gemini_resp_car")
        fake_observer = GeminiVisualObserver(
            api_key_env_var=observer_key_var, model="gemini-test-observer-model",
            request_fn=lambda key, model, body, timeout: observer_raw)

        verifier_raw = _fake_gemini_verifier_response("car", ["vehicle"], confidence=0.85)
        fake_verifier = GeminiVisualVerifier(
            api_key_env_var=verifier_key_var, model="gemini-test-verifier-model",
            request_fn=lambda key, model, body, timeout: verifier_raw)

        image_path = _tmp_real_image_path()
        try:
            with mock.patch.dict(os.environ, {observer_key_var: "fake-observer-key",
                                               verifier_key_var: "fake-verifier-key"}):
                rows = rdb.run_live_observer_and_verifier(
                    "fixture_image", Path(image_path),
                    observer=fake_observer, verifier=fake_verifier, sleep_seconds=0)
        finally:
            os.unlink(image_path)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["drawing_id"], "fixture_image")
        self.assertEqual(row["observation_id"], "fixture_image_c00")

        # Step 5: candidate fields/bbox translated correctly into the
        # existing benchmark row schema.
        oc = row["observer_candidate"]
        self.assertEqual(oc["label"], "car")
        self.assertEqual(oc["alternative_labels"], ["vehicle"])
        self.assertEqual(oc["bbox"], (0.1, 0.1, 0.4, 0.4))
        self.assertEqual(oc["confidence"], 0.8)
        self.assertEqual(oc["entity_type"], "object")

        self.assertEqual(row["verifier_independent_label"], "car")
        self.assertEqual(row["verifier_independent_alternative_labels"], ["vehicle"])
        self.assertEqual(row["verifier_confidence"], 0.85)
        self.assertEqual(row["verification_status"], "verified")
        self.assertEqual(row["observer_model"], "gemini-test-observer-model")
        self.assertEqual(row["verifier_model"], "gemini-test-verifier-model")

        # The row schema this benchmark script's OWN reader expects --
        # proves the fix's output is consumable by the rest of the script,
        # not just structurally similar.
        entities = rdb.entities_from_verification_rows(rows)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].canonical_label, "car")
        self.assertEqual(entities[0].case_verification_status, "verified")

    def test_no_candidates_returns_empty_rows_without_calling_verifier(self):
        observer_key_var = "DOAR_TEST_RDB_OBSERVER_KEY_UNUSED_2"
        observer_raw = _fake_gemini_observer_response([])
        fake_observer = GeminiVisualObserver(api_key_env_var=observer_key_var, request_fn=lambda *a: observer_raw)

        def verifier_must_not_be_called(*_a):
            raise AssertionError("verifier.verify must not be called when there are no candidates")
        fake_verifier = GeminiVisualVerifier(request_fn=verifier_must_not_be_called)

        image_path = _tmp_real_image_path()
        try:
            with mock.patch.dict(os.environ, {observer_key_var: "fake-observer-key"}):
                rows = rdb.run_live_observer_and_verifier(
                    "fixture_image", Path(image_path),
                    observer=fake_observer, verifier=fake_verifier, sleep_seconds=0)
        finally:
            os.unlink(image_path)

        self.assertEqual(rows, [])

    def test_default_observer_verifier_are_the_real_frozen_classes(self):
        # Confirms the DI seam's defaults are the actual production
        # classes (not a test double silently substituted), matching
        # main()'s own no-kwargs call site.
        import inspect
        sig = inspect.signature(rdb.run_live_observer_and_verifier)
        source = inspect.getsource(rdb.run_live_observer_and_verifier)
        self.assertIn("observer or GeminiVisualObserver()", source)
        self.assertIn("verifier or GeminiVisualVerifier()", source)
        self.assertIn("observer", sig.parameters)
        self.assertIn("verifier", sig.parameters)


class RawVsVerifiedIntegrationTests(unittest.TestCase):
    """Bug 5, regression item 10: the benchmark runner computes RAW
    Observer and VERIFIED-only metrics SEPARATELY (not just one blended
    number), per annotator, independently."""

    def test_compute_raw_vs_verified_metrics_scores_both_conditions_per_annotator(self):
        from doar import benchmark_metrics as bm

        human = {
            "A1": [bm.HumanAnnotationItem(label="sun", location="top", salient=True, confidence="clear"),
                   bm.HumanAnnotationItem(label="person", location="center", salient=True, confidence="clear")],
            "A2": None,
        }
        # RAW finds sun + a hallucinated "dinosaur"; VERIFIED drops the
        # dinosaur but ALSO wrongly drops the real "person".
        result = rdb.compute_raw_vs_verified_metrics(["sun", "dinosaur", "person"], ["sun"], human)

        self.assertIsNone(result["A2"])
        a1 = result["A1"]
        self.assertEqual(a1["raw_observer"]["precision_recall_f1"]["recall"], 1.0)
        self.assertEqual(a1["verified_only"]["precision_recall_f1"]["recall"], 0.5)
        # Precision improves (dinosaur dropped) but recall worsens (person dropped) --
        # both directions visible, not collapsed into one ambiguous "correction rate".
        self.assertGreater(a1["delta_verified_minus_raw"]["precision"], 0)
        self.assertLess(a1["delta_verified_minus_raw"]["recall"], 0)

    def test_missing_annotator_reports_none_not_a_crash(self):
        result = rdb.compute_raw_vs_verified_metrics(["sun"], ["sun"], {"A1": None, "A2": None})
        self.assertIsNone(result["A1"])
        self.assertIsNone(result["A2"])

    def test_summarize_raw_vs_verified_deltas_averages_across_images_and_keeps_per_image_detail(self):
        deltas_by_annotator = {
            "A1": [
                {"image_id": "img1", "precision": 0.5, "recall": -0.5, "f1": 0.0, "salient_recall": -0.5},
                {"image_id": "img2", "precision": 0.3, "recall": -0.1, "f1": 0.1, "salient_recall": None},
            ],
            "A2": [],
        }
        summary = rdb.summarize_raw_vs_verified_deltas(deltas_by_annotator)
        self.assertAlmostEqual(summary["A1"]["mean_delta"]["precision"], 0.4)
        self.assertAlmostEqual(summary["A1"]["mean_delta"]["recall"], -0.3)
        # salient_recall mean ignores the None entry rather than treating it as 0.
        self.assertAlmostEqual(summary["A1"]["mean_delta"]["salient_recall"], -0.5)
        self.assertEqual(len(summary["A1"]["per_image_deltas"]), 2)
        self.assertIsNone(summary["A2"])

    def test_real_cached_p2b_0033_data_shows_verifier_dropping_a_real_mouth_candidate(self):
        # p2b_0033's own cached data: the Observer's "tears" candidate was
        # independently re-read by the Verifier as "mouth" but REJECTED
        # (a labelling mismatch, not proof the mouth wasn't drawn) -- so
        # VERIFIED-only loses that candidate entirely while RAW still has
        # it under its own (wrong) "tears" label. This is exactly the
        # asymmetric-correction scenario Bug 5 exists to make visible.
        from doar import benchmark_metrics as bm

        live_cache_path = ROOT / "outputs" / "prototype_cases" / "development_live_cache" / "p2b_0033_verification.json"
        if not live_cache_path.exists():
            self.skipTest(f"real cached fixture not present in this checkout: {live_cache_path}")
        rows = json.loads(live_cache_path.read_text(encoding="utf-8"))
        raw_labels = [row["observer_candidate"]["label"] for row in rows]
        verified_labels = [row["observer_candidate"]["label"] for row in rows if row["verification_status"] == "verified"]
        self.assertIn("tears", raw_labels)
        self.assertNotIn("tears", verified_labels)

        items = [bm.HumanAnnotationItem(label="crying face", location="center", salient=True, confidence="clear")]
        result = bm.raw_vs_verified_comparison(raw_labels, verified_labels, items)
        self.assertEqual(result["raw_observer"]["precision_recall_f1"]["total_candidates"], len(raw_labels))
        self.assertEqual(result["verified_only"]["precision_recall_f1"]["total_candidates"], len(verified_labels))
        self.assertLess(len(verified_labels), len(raw_labels))


class ImageIdAndForceLiveTests(unittest.TestCase):
    """`--image-id` restricts a run to one image; `--force-live` refetches
    even when cached data already exists -- the mechanism used to refresh
    a single stale image (e.g. p2b_0004) without touching the other 14."""

    def test_image_id_restricts_to_one_image_and_force_live_overwrites_cache(self):
        import shutil
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as live_cache_tmp, tempfile.TemporaryDirectory() as annotations_tmp:
            live_cache_dir = Path(live_cache_tmp)
            annotations_dir = Path(annotations_tmp)
            for aid in ("A1", "A2"):
                (annotations_dir / aid).mkdir(parents=True)

            stale_rows = [{"drawing_id": "p2b_0004", "observation_id": "p2b_0004_c00",
                           "observer_candidate": {"label": "stale", "alternative_labels": [], "bbox": [0.0, 0.0, 0.1, 5.0],
                                                   "confidence": 0.9, "entity_type": "object"},
                           "verifier_independent_label": None, "verifier_independent_alternative_labels": [],
                           "verifier_confidence": None, "verification_status": "unreviewed",
                           "verifier_notes": "stale fixture", "observer_model": "m", "verifier_model": "m",
                           "verifier_source_note": "", "runtime_seconds": None}]
            (live_cache_dir / "p2b_0004_verification.json").write_text(json.dumps(stale_rows), encoding="utf-8")

            fresh_rows = [{"drawing_id": "p2b_0004", "observation_id": "p2b_0004_c00",
                           "observer_candidate": {"label": "fresh sun", "alternative_labels": [], "bbox": [0.1, 0.1, 0.2, 0.2],
                                                   "confidence": 0.9, "entity_type": "object"},
                           "verifier_independent_label": "sun", "verifier_independent_alternative_labels": [],
                           "verifier_confidence": 0.9, "verification_status": "verified",
                           "verifier_notes": "", "observer_model": "m", "verifier_model": "m",
                           "verifier_source_note": "", "runtime_seconds": 1.0}]

            def fake_live_call(image_id, image_path, **kwargs):
                self.assertEqual(image_id, "p2b_0004")
                return fresh_rows

            output_root = ROOT / "outputs" / "prototype_cases"
            before_dirs = {p.name for p in output_root.glob("development_benchmark_*")}
            old_argv = sys.argv

            with _patched(rdb, ANNOTATIONS_DIR=annotations_dir, LIVE_CACHE_DIR=live_cache_dir,
                          SAVED_VERIFICATION_DIRS=[]):
                with patch.object(rdb, "run_live_observer_and_verifier", side_effect=fake_live_call) as mock_live:
                    sys.argv = ["run_development_benchmark.py", "--image-id", "p2b_0004", "--run-live", "--force-live"]
                    try:
                        rdb.main()
                    finally:
                        sys.argv = old_argv
                    mock_live.assert_called_once()

            after_dirs = {p.name for p in output_root.glob("development_benchmark_*")}
            new_dirs = after_dirs - before_dirs
            self.assertEqual(len(new_dirs), 1)
            run_dir = output_root / next(iter(new_dirs))
            try:
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                # Only the one requested image was processed.
                self.assertEqual(summary["total_images"], 1)
                self.assertEqual(summary["per_image"][0]["image_id"], "p2b_0004")
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)

            # The live cache file was overwritten with the fresh rows, not the stale ones.
            saved = json.loads((live_cache_dir / "p2b_0004_verification.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["observer_candidate"]["label"], "fresh sun")

    def test_unknown_image_id_exits_without_crashing(self):
        old_argv = sys.argv
        sys.argv = ["run_development_benchmark.py", "--image-id", "not_a_real_image"]
        try:
            rdb.main()  # must return cleanly, not raise
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
