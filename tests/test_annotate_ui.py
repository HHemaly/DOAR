"""Tests for scripts/annotate_ui.py -- the Streamlit visual annotation UI
that replaced the console/timed workflow.

Two layers:
  - Pure-function tests (record_path/load_record/save_record/
    items_to_dataframe/dataframe_to_items/completion_marker) -- fast,
    no Streamlit runtime involved.
  - `streamlit.testing.v1.AppTest` integration tests -- actually execute
    the script's main() in a simulated session (real widget tree, real
    script reruns on interaction), the officially supported way to test
    a Streamlit app without a browser. Every AppTest test redirects
    ANNOTATIONS_DIR to a throwaway temp directory via the
    DOAR_ANNOTATIONS_DIR environment variable -- never touches the real
    annotations/ tree.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("annotate_ui", ROOT / "scripts" / "annotate_ui.py")
ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ui)

from streamlit.testing.v1 import AppTest  # noqa: E402


class ItemsDataframeRoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_fields(self):
        items = [
            {"label": "sun", "location": "top-right", "salient": True, "confidence": "clear", "note": ""},
            {"label": "small mark", "location": "bottom", "salient": False, "confidence": "ambiguous", "note": "faint"},
        ]
        df = ui.items_to_dataframe(items)
        back = ui.dataframe_to_items(df)
        self.assertEqual(back, items)

    def test_empty_items_produces_empty_dataframe_with_columns(self):
        df = ui.items_to_dataframe([])
        self.assertEqual(list(df.columns), ui.ITEM_COLUMNS)
        self.assertEqual(len(df), 0)

    def test_blank_label_rows_are_dropped(self):
        df = pd.DataFrame([
            {"label": "person", "location": "center", "salient": True, "confidence": "clear", "note": ""},
            {"label": "", "location": "nowhere", "salient": False, "confidence": "clear", "note": ""},
            {"label": "   ", "location": "", "salient": False, "confidence": "clear", "note": ""},
        ])
        items = ui.dataframe_to_items(df)
        self.assertEqual([i["label"] for i in items], ["person"])

    def test_invalid_confidence_normalized_to_clear(self):
        df = pd.DataFrame([{"label": "x", "location": "", "salient": False, "confidence": "very sure", "note": ""}])
        items = ui.dataframe_to_items(df)
        self.assertEqual(items[0]["confidence"], "clear")


class CompletionMarkerTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(ui.completion_marker({"complete": True, "items": [{"label": "x"}]}), "✅")

    def test_draft_not_complete(self):
        self.assertEqual(ui.completion_marker({"complete": False, "items": [{"label": "x"}]}), "\U0001F4DD")

    def test_untouched(self):
        self.assertEqual(ui.completion_marker({"complete": False, "items": []}), "⬜")


class RecordPersistenceTests(unittest.TestCase):
    def test_load_record_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = ui.ANNOTATIONS_DIR
            ui.ANNOTATIONS_DIR = Path(tmp)
            try:
                record = ui.load_record("A1", "fixture_img")
                self.assertEqual(record, {"annotator_id": "A1", "image_id": "fixture_img", "items": [], "complete": False})
            finally:
                ui.ANNOTATIONS_DIR = original

    def test_save_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = ui.ANNOTATIONS_DIR
            ui.ANNOTATIONS_DIR = Path(tmp)
            try:
                items = [{"label": "sun", "location": "top", "salient": True, "confidence": "clear", "note": ""}]
                ui.save_record("A1", "fixture_img", items, True)
                record = ui.load_record("A1", "fixture_img")
                self.assertEqual(record["items"], items)
                self.assertTrue(record["complete"])
            finally:
                ui.ANNOTATIONS_DIR = original

    def test_record_path_scheme_is_one_file_per_image_no_pass_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = ui.ANNOTATIONS_DIR
            ui.ANNOTATIONS_DIR = Path(tmp)
            try:
                path = ui.record_path("A2", "h38")
                self.assertEqual(path.name, "h38.json")
                self.assertEqual(path.parent.name, "A2")
            finally:
                ui.ANNOTATIONS_DIR = original


class _annotator_app_env:
    """Keeps DOAR_ANNOTATIONS_DIR/sys.argv pointed at a throwaway
    directory for the ENTIRE test body -- including any further
    `.run()` calls the test makes after the initial launch, since
    AppTest re-executes the full script (module-level ANNOTATIONS_DIR
    included) on every interaction, exactly like real Streamlit. Only
    restores the real environment on exit."""

    def __init__(self, annotator_id: str, annotations_dir: Path):
        self.annotator_id = annotator_id
        self.annotations_dir = annotations_dir

    def __enter__(self) -> AppTest:
        self._old_argv = sys.argv
        self._old_env = os.environ.get("DOAR_ANNOTATIONS_DIR")
        sys.argv = ["scripts/annotate_ui.py", "--annotator", self.annotator_id]
        os.environ["DOAR_ANNOTATIONS_DIR"] = str(self.annotations_dir)
        at = AppTest.from_file(str(ROOT / "scripts" / "annotate_ui.py"), default_timeout=30)
        at.run()
        return at

    def __exit__(self, *exc_info) -> None:
        sys.argv = self._old_argv
        if self._old_env is None:
            os.environ.pop("DOAR_ANNOTATIONS_DIR", None)
        else:
            os.environ["DOAR_ANNOTATIONS_DIR"] = self._old_env


class AppRunsWithoutErrorTests(unittest.TestCase):
    def test_app_runs_clean_for_annotator_a1(self):
        with tempfile.TemporaryDirectory() as tmp, _annotator_app_env("A1", Path(tmp)) as at:
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.title[0].value, "Annotator A1 -- Image 1 / 15 -- h38")

    def test_missing_annotator_shows_error_and_stops_before_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_argv = sys.argv
            old_env = os.environ.get("DOAR_ANNOTATIONS_DIR")
            sys.argv = ["scripts/annotate_ui.py"]  # no --annotator
            os.environ["DOAR_ANNOTATIONS_DIR"] = tmp
            try:
                at = AppTest.from_file(str(ROOT / "scripts" / "annotate_ui.py"), default_timeout=30)
                at.run()
                self.assertEqual(len(at.exception), 0)
                self.assertGreaterEqual(len(at.error), 1)
                self.assertEqual(len(at.title), 0)
            finally:
                sys.argv = old_argv
                if old_env is None:
                    os.environ.pop("DOAR_ANNOTATIONS_DIR", None)
                else:
                    os.environ["DOAR_ANNOTATIONS_DIR"] = old_env


class NavigationAndAutosaveTests(unittest.TestCase):
    def test_sidebar_jump_button_changes_current_image(self):
        with tempfile.TemporaryDirectory() as tmp, _annotator_app_env("A1", Path(tmp)) as at:
            # sidebar buttons come after the 3 nav buttons in DOM order.
            jump_button = [b for b in at.button if b.key == "jump_2"][0]
            jump_button.click().run()
            self.assertEqual(at.title[0].value, "Annotator A1 -- Image 3 / 15 -- p2b_0002")

    def test_previous_disabled_on_first_image(self):
        with tempfile.TemporaryDirectory() as tmp, _annotator_app_env("A1", Path(tmp)) as at:
            previous_buttons = [b for b in at.button if b.label == "Previous"]
            self.assertEqual(len(previous_buttons), 1)
            self.assertTrue(previous_buttons[0].disabled)

    def test_save_and_next_advances_and_autosave_persists_complete_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _annotator_app_env("A1", tmp_path) as at:
                at.checkbox[0].set_value(True).run()  # "Mark this drawing complete" for image 1 (h38)
                save_next = [b for b in at.button if b.label == "Save & Next"][0]
                save_next.click().run()

                self.assertEqual(at.title[0].value, "Annotator A1 -- Image 2 / 15 -- p2b_0001")
            h38_record = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
            self.assertTrue(h38_record["complete"])


class AnnotatorIsolationTests(unittest.TestCase):
    def test_a2_never_sees_a1_annotations_even_in_the_same_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "A1").mkdir(parents=True)
            (tmp_path / "A1" / "h38.json").write_text(json.dumps({
                "annotator_id": "A1", "image_id": "h38",
                "items": [{"label": "secret_a1_item", "location": "x", "salient": True, "confidence": "clear", "note": ""}],
                "complete": True,
            }), encoding="utf-8")

            with _annotator_app_env("A2", tmp_path) as at_a2:
                self.assertEqual(len(at_a2.exception), 0)

            # A2's editor for h38 must start empty -- never load A1's file.
            a2_h38_path = tmp_path / "A2" / "h38.json"
            self.assertTrue(a2_h38_path.exists())
            a2_record = json.loads(a2_h38_path.read_text(encoding="utf-8"))
            self.assertEqual(a2_record["items"], [])

            a1_record_untouched = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
            self.assertEqual(a1_record_untouched["items"][0]["label"], "secret_a1_item")


if __name__ == "__main__":
    unittest.main()
