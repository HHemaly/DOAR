"""End-to-end Streamlit smoke tests for phase2c_annotation_app.py, using
streamlit.testing.v1.AppTest (bare-mode script execution, same tool used
for phase7b_review_app.py's own smoke tests). Every test points the app at
a throwaway temp directory via DOAR_PHASE2C1_* environment variables --
never the real outputs/phase2c1/ tree, which may hold real annotator
progress on real (though blinded) images.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

_ENV_VARS = ("DOAR_PHASE2C1_IMAGES_DIR", "DOAR_PHASE2C1_STORE_PATH",
             "DOAR_PHASE2C1_EXPORT_DIR", "DOAR_PHASE2C1_MAPPING_PATH")


class _IsolatedAppTestCase(unittest.TestCase):
    def setUp(self):
        if not _STREAMLIT_TESTING_AVAILABLE:
            self.skipTest("streamlit.testing.v1.AppTest not available")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.images_dir = self.tmp_dir / "images"
        self.images_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (40, 40), "white").save(self.images_dir / f"p2b_{i:04d}.jpg")
        self.store_path = self.tmp_dir / "store.csv"
        self.export_dir = self.tmp_dir / "exports"
        self.mapping_path = self.tmp_dir / "mapping.csv"

        self._env_backup = {k: os.environ.get(k) for k in _ENV_VARS}
        os.environ["DOAR_PHASE2C1_IMAGES_DIR"] = str(self.images_dir)
        os.environ["DOAR_PHASE2C1_STORE_PATH"] = str(self.store_path)
        os.environ["DOAR_PHASE2C1_EXPORT_DIR"] = str(self.export_dir)
        os.environ["DOAR_PHASE2C1_MAPPING_PATH"] = str(self.mapping_path)

        real_private_dir = (ROOT / "outputs/phase2c1").resolve()
        for p in (self.images_dir, self.store_path, self.export_dir, self.mapping_path):
            self.assertFalse(
                str(p.resolve()).startswith(str(real_private_dir)),
                f"{p} must never resolve inside the real {real_private_dir}")

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def _app(self) -> "AppTest":
        at = AppTest.from_file(str(ROOT / "phase2c_annotation_app.py"))
        at.run(timeout=60)
        return at

    def _set_annotator(self, at, annotator_id="tester1"):
        at.sidebar.text_input(key="annotator_id").set_value(annotator_id)
        at.run(timeout=60)
        return at


class RendersWithoutExceptionTests(_IsolatedAppTestCase):
    def test_renders_before_annotator_id_set(self):
        at = self._app()
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

    def test_renders_after_annotator_id_set(self):
        at = self._app()
        at = self._set_annotator(at)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

    def test_missing_images_dir_shows_error_not_crash(self):
        import shutil
        shutil.rmtree(self.images_dir)
        at = self._app()
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(len(at.error) > 0)


class NoLeakedInformationTests(_IsolatedAppTestCase):
    def test_no_emotion_word_appears_in_rendered_output(self):
        at = self._app()
        at = self._set_annotator(at)
        all_text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.text)
        for label in ("Angry", "Fear", "Happy", "Sad"):
            self.assertNotIn(label, all_text)


class PrimaryAnnotationSaveTests(_IsolatedAppTestCase):
    def test_save_creates_ten_rows_for_one_image(self):
        at = self._app()
        at = self._set_annotator(at)
        save_buttons = [b for b in at.button if b.label == "Save"]
        self.assertTrue(save_buttons)
        save_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(self.store_path.exists())
        rows = list(csv.DictReader(self.store_path.open(encoding="utf-8")))
        self.assertEqual(len(rows), 10)  # all 10 ontology classes, default "absent"

    def test_setting_a_class_present_and_saving_persists_it(self):
        at = self._app()
        at = self._set_annotator(at)
        person_radio = [r for r in at.radio if (r.key or "").endswith("_person_status")][0]
        person_radio.set_value("present")
        at.run(timeout=60)
        save_buttons = [b for b in at.button if b.label == "Save"]
        save_buttons[0].click()
        at.run(timeout=60)
        rows = list(csv.DictReader(self.store_path.open(encoding="utf-8")))
        person_row = [r for r in rows if r["class_name"] == "person"][0]
        self.assertEqual(person_row["status"], "present")
        self.assertEqual(person_row["instance_count"], "1")

    def test_bbox_entered_for_present_class_is_saved(self):
        at = self._app()
        at = self._set_annotator(at)
        person_radio = [r for r in at.radio if (r.key or "").endswith("_person_status")][0]
        person_radio.set_value("present")
        at.run(timeout=60)
        bbox_input = [t for t in at.text_input if (t.key or "").endswith("_person_bbox")][0]
        bbox_input.set_value("0.1,0.2,0.3,0.4")
        at.run(timeout=60)
        save_buttons = [b for b in at.button if b.label == "Save"]
        save_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        rows = list(csv.DictReader(self.store_path.open(encoding="utf-8")))
        person_row = [r for r in rows if r["class_name"] == "person"][0]
        self.assertEqual(person_row["bbox"], "0.100000,0.200000,0.300000,0.400000")

    def test_invalid_bbox_is_warned_and_not_saved(self):
        at = self._app()
        at = self._set_annotator(at)
        person_radio = [r for r in at.radio if (r.key or "").endswith("_person_status")][0]
        person_radio.set_value("present")
        at.run(timeout=60)
        bbox_input = [t for t in at.text_input if (t.key or "").endswith("_person_bbox")][0]
        bbox_input.set_value("not,a,valid,bbox")
        at.run(timeout=60)
        self.assertTrue(len(at.warning) > 0)
        save_buttons = [b for b in at.button if b.label == "Save"]
        save_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        rows = list(csv.DictReader(self.store_path.open(encoding="utf-8")))
        person_row = [r for r in rows if r["class_name"] == "person"][0]
        self.assertEqual(person_row["bbox"], "")

    def test_save_and_next_advances_image_index(self):
        at = self._app()
        at = self._set_annotator(at)
        save_next = [b for b in at.button if b.label == "Save & Next"][0]
        save_next.click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertIn("2 / 3", " ".join(m.value for m in at.markdown))

    def test_no_duplicate_rows_on_repeated_save(self):
        at = self._app()
        at = self._set_annotator(at)
        save_buttons = [b for b in at.button if b.label == "Save"]
        save_buttons[0].click()
        at.run(timeout=60)
        save_buttons = [b for b in at.button if b.label == "Save"]
        save_buttons[0].click()
        at.run(timeout=60)
        rows = list(csv.DictReader(self.store_path.open(encoding="utf-8")))
        self.assertEqual(len(rows), 10)


class ResumeTests(_IsolatedAppTestCase):
    def test_progress_reflects_prior_session_on_relaunch(self):
        sys.path.insert(0, str(ROOT / "src"))
        from doar.phase2c1 import store as store_mod
        from doar.phase2c1.schema import AnnotationRecord
        from doar.phase2b.ontology import CLASS_NAMES

        st = {}
        for cls in CLASS_NAMES:
            rec = AnnotationRecord(
                pilot_id="p2b_0000", image_id="x", source_image_group="g", class_name=cls,
                status="absent", annotator_id="tester1", annotation_timestamp="t",
            )
            store_mod.upsert(st, rec)
        store_mod.save_store(self.store_path, st)

        at = self._app()
        at = self._set_annotator(at)
        sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("1 / 3", sidebar_text)


class ReviewModeTests(_IsolatedAppTestCase):
    def _seed_primary_annotation(self):
        sys.path.insert(0, str(ROOT / "src"))
        from doar.phase2c1 import store as store_mod
        from doar.phase2c1.schema import AnnotationRecord
        from doar.phase2b.ontology import CLASS_NAMES

        st = {}
        for cls in CLASS_NAMES:
            status = "present" if cls == "person" else "absent"
            rec = AnnotationRecord(
                pilot_id="p2b_0000", image_id="x", source_image_group="g", class_name=cls,
                status=status, annotator_id="primary_annotator", annotation_timestamp="t",
                instance_count=1 if status == "present" else 0,
            )
            store_mod.upsert(st, rec)
        store_mod.save_store(self.store_path, st)

    def test_review_mode_hides_primary_label_until_own_judgment_saved(self):
        self._seed_primary_annotation()
        at = self._app()
        at = self._set_annotator(at, "reviewer1")
        at.sidebar.radio(key="mode").set_value("Review")
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        reveal_expanders = [e for e in at.expander if "Reveal" in (e.label or "")]
        self.assertEqual(len(reveal_expanders), 0)

    def test_reveal_unlocks_after_saving_own_judgment(self):
        self._seed_primary_annotation()
        at = self._app()
        at = self._set_annotator(at, "reviewer1")
        at.sidebar.radio(key="mode").set_value("Review")
        at.run(timeout=60)
        save_buttons = [b for b in at.button if "Save my independent judgment" in b.label]
        self.assertTrue(save_buttons)
        save_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        reveal_expanders = [e for e in at.expander if "Reveal" in (e.label or "")]
        self.assertGreater(len(reveal_expanders), 0)

    def test_agreement_outcome_updates_primary_row_not_a_new_row(self):
        self._seed_primary_annotation()
        at = self._app()
        at = self._set_annotator(at, "reviewer1")
        at.sidebar.radio(key="mode").set_value("Review")
        at.run(timeout=60)
        save_buttons = [b for b in at.button if "Save my independent judgment" in b.label]
        save_buttons[0].click()
        at.run(timeout=60)
        agree_buttons = [b for b in at.button if b.label == "Agreement"]
        self.assertTrue(agree_buttons)
        agree_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        rows = list(csv.DictReader(self.store_path.open(encoding="utf-8")))
        primary_rows = [r for r in rows if r["annotator_id"] == "primary_annotator"
                         and r["class_name"] == "person"]
        self.assertEqual(len(primary_rows), 1)
        self.assertEqual(primary_rows[0]["review_status"], "reviewed")
        self.assertEqual(primary_rows[0]["adjudication_status"], "agreement")
        reviewer_rows = [r for r in rows if r["annotator_id"] == "reviewer1"
                          and r["class_name"] == "person"]
        self.assertEqual(len(reviewer_rows), 1)


class ExportTests(_IsolatedAppTestCase):
    def test_export_writes_csv_and_json(self):
        at = self._app()
        at = self._set_annotator(at)
        save_buttons = [b for b in at.button if b.label == "Save"]
        save_buttons[0].click()
        at.run(timeout=60)
        export_buttons = [b for b in at.sidebar.button if "Export" in b.label]
        export_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue((self.export_dir / "phase2c1_annotations.csv").exists())
        self.assertTrue((self.export_dir / "phase2c1_annotations.json").exists())


if __name__ == "__main__":
    unittest.main()
