"""Tests for scripts/annotate_ui.py -- the Streamlit visual annotation UI
that replaced the console/timed workflow.

Two layers:
  - Pure-function tests (record_path/load_record/save_record/build_item/
    completion_marker) -- fast, no Streamlit runtime involved.
  - `streamlit.testing.v1.AppTest` integration tests -- actually execute
    the script's main() in a simulated session (real widget tree, real
    script reruns on interaction), the officially supported way to test
    a Streamlit app without a browser. Every AppTest test redirects
    ANNOTATIONS_DIR to a throwaway temp directory via the
    DOAR_ANNOTATIONS_DIR environment variable -- never touches the real
    annotations/ tree.

The `st.data_editor`-based item entry was replaced by a plain `st.form`
after a real bug: checking the "Salient" checkbox inside the editor did
not survive the rerun the checkbox click itself triggers. The
`FormItemEntryTests` class below reproduces the exact scenario that bug
was found under (enter "car", check Salient, confidence=clear, Add Item)
and asserts salient=true actually lands in the saved JSON and survives
further reruns/navigation.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("annotate_ui", ROOT / "scripts" / "annotate_ui.py")
ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ui)

from streamlit.testing.v1 import AppTest  # noqa: E402


class BuildItemTests(unittest.TestCase):
    def test_builds_full_item(self):
        item = ui.build_item("car", "center", True, "clear", "a note")
        self.assertEqual(item, {"label": "car", "location": "center", "salient": True,
                                 "confidence": "clear", "note": "a note"})

    def test_blank_label_returns_none(self):
        self.assertIsNone(ui.build_item("   ", "center", True, "clear", ""))

    def test_invalid_confidence_normalized_to_clear(self):
        item = ui.build_item("x", "", False, "very sure", "")
        self.assertEqual(item["confidence"], "clear")

    def test_strips_whitespace(self):
        item = ui.build_item("  car  ", "  center  ", False, "clear", "  note  ")
        self.assertEqual(item["label"], "car")
        self.assertEqual(item["location"], "center")
        self.assertEqual(item["note"], "note")


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


def _widget(elements, key):
    matches = [e for e in elements if e.key == key]
    assert len(matches) == 1, f"expected exactly one widget with key={key!r}, found {len(matches)}"
    return matches[0]


def _current_generation_widget(elements, annotator_id, image_id, name):
    """Finds the add/edit form widget for `name` regardless of the
    internal form_gen counter's exact value -- only the CURRENT
    generation's widgets are ever present in the rendered tree at all
    (Streamlit only renders what the latest script execution declared),
    so a unique prefix match is generation-agnostic and doesn't couple
    the test to exactly how many times form_gen has been bumped so far."""
    prefix = f"{name}_{annotator_id}_{image_id}_"
    matches = [e for e in elements if e.key and e.key.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one current '{name}' widget, found {len(matches)}"
    return matches[0]


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


class NavigationTests(unittest.TestCase):
    def test_sidebar_jump_button_changes_current_image(self):
        with tempfile.TemporaryDirectory() as tmp, _annotator_app_env("A1", Path(tmp)) as at:
            jump_button = [b for b in at.button if b.key == "jump_2"][0]
            jump_button.click().run()
            self.assertEqual(at.title[0].value, "Annotator A1 -- Image 3 / 15 -- p2b_0002")

    def test_previous_disabled_on_first_image(self):
        with tempfile.TemporaryDirectory() as tmp, _annotator_app_env("A1", Path(tmp)) as at:
            previous_buttons = [b for b in at.button if b.label == "Previous"]
            self.assertEqual(len(previous_buttons), 1)
            self.assertTrue(previous_buttons[0].disabled)

    def test_save_and_next_advances_and_persists_complete_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _annotator_app_env("A1", tmp_path) as at:
                mark_complete = [c for c in at.checkbox if c.label == "Mark this drawing complete"][0]
                mark_complete.set_value(True).run()
                save_next = [b for b in at.button if b.label == "Save & Next"][0]
                save_next.click().run()

                self.assertEqual(at.title[0].value, "Annotator A1 -- Image 2 / 15 -- p2b_0001")
            h38_record = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
            self.assertTrue(h38_record["complete"])


class FormItemEntryTests(unittest.TestCase):
    """Reproduces the exact reported bug scenario end to end through the
    real Streamlit widget tree (not just the pure build_item() helper)."""

    def test_enter_car_check_salient_confidence_clear_add_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _annotator_app_env("A1", tmp_path) as at:
                # generation 0 is the widget generation for the very
                # first render of the (empty) add-item form.
                label_input = _widget(at.text_input, ui.widget_key("A1", "h38", 0, "label"))
                salient_checkbox = _widget(at.checkbox, ui.widget_key("A1", "h38", 0, "salient"))
                confidence_radio = _widget(at.radio, ui.widget_key("A1", "h38", 0, "confidence"))

                # 1. enter "car"
                label_input.set_value("car")
                # 2. check Salient
                salient_checkbox.set_value(True)
                # 3. set confidence=clear
                confidence_radio.set_value("clear")
                # 4. Add Item
                add_button = [b for b in at.button if b.label == "Add Item"][0]
                add_button.click().run()

                self.assertEqual(len(at.exception), 0)

                # 5. verify JSON contains salient=true
                saved = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
                self.assertEqual(len(saved["items"]), 1)
                self.assertEqual(saved["items"][0]["label"], "car")
                self.assertIs(saved["items"][0]["salient"], True)
                self.assertEqual(saved["items"][0]["confidence"], "clear")

                # 6. verify item stays visible after rerun (a fresh,
                # unrelated rerun -- not just the submission's own rerun).
                at.run()
                self.assertEqual(len(at.exception), 0)
                item_headers = [s.value for s in at.subheader if s.value.startswith("Items (")]
                self.assertEqual(item_headers, ["Items (1)"])
                rendered_text = " ".join(m.value for m in at.markdown)
                self.assertIn("car", rendered_text)

                # 7. verify Delete works
                delete_button = [b for b in at.button if b.key == "delete_A1_h38_0"][0]
                delete_button.click().run()
                self.assertEqual(len(at.exception), 0)
                after_delete = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
                self.assertEqual(after_delete["items"], [])
                item_headers_after_delete = [s.value for s in at.subheader if s.value.startswith("Items (")]
                self.assertEqual(item_headers_after_delete, ["Items (0)"])

    def test_navigation_does_not_lose_saved_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _annotator_app_env("A1", tmp_path) as at:
                label_input = _widget(at.text_input, ui.widget_key("A1", "h38", 0, "label"))
                salient_checkbox = _widget(at.checkbox, ui.widget_key("A1", "h38", 0, "salient"))
                label_input.set_value("car")
                salient_checkbox.set_value(True)
                add_button = [b for b in at.button if b.label == "Add Item"][0]
                add_button.click().run()

                # 8. navigate away and back -- item must still be there,
                # both on screen and on disk.
                save_next = [b for b in at.button if b.label == "Save & Next"][0]
                save_next.click().run()
                self.assertEqual(at.title[0].value, "Annotator A1 -- Image 2 / 15 -- p2b_0001")

                previous = [b for b in at.button if b.label == "Previous"][0]
                previous.click().run()
                self.assertEqual(at.title[0].value, "Annotator A1 -- Image 1 / 15 -- h38")

                item_headers = [s.value for s in at.subheader if s.value.startswith("Items (")]
                self.assertEqual(item_headers, ["Items (1)"])

            h38_record = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
            self.assertEqual(len(h38_record["items"]), 1)
            self.assertEqual(h38_record["items"][0]["label"], "car")
            self.assertIs(h38_record["items"][0]["salient"], True)

    def test_edit_item_prefills_form_and_updates_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with _annotator_app_env("A1", tmp_path) as at:
                label_input = _widget(at.text_input, ui.widget_key("A1", "h38", 0, "label"))
                label_input.set_value("car")
                add_button = [b for b in at.button if b.label == "Add Item"][0]
                add_button.click().run()

                edit_button = [b for b in at.button if b.key == "edit_A1_h38_0"][0]
                edit_button.click().run()
                self.assertEqual(len(at.exception), 0)

                # Edit bumps form_gen internally -- the prefilled label
                # widget now lives under a new generation's key.
                edit_label_input = _current_generation_widget(at.text_input, "A1", "h38", "label")
                self.assertEqual(edit_label_input.value, "car")

                edit_label_input.set_value("bus")
                update_button = [b for b in at.button if b.label == "Update Item"][0]
                update_button.click().run()

                saved = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
                self.assertEqual(len(saved["items"]), 1)
                self.assertEqual(saved["items"][0]["label"], "bus")


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
                # A2's page for h38 must render as empty -- never A1's item.
                item_headers = [s.value for s in at_a2.subheader if s.value.startswith("Items (")]
                self.assertEqual(item_headers, ["Items (0)"])
                rendered_text = " ".join(m.value for m in at_a2.markdown)
                self.assertNotIn("secret_a1_item", rendered_text)

                # Adding an item as A2 must never touch A1's file.
                label_input = _widget(at_a2.text_input, ui.widget_key("A2", "h38", 0, "label"))
                label_input.set_value("a2_item")
                add_button = [b for b in at_a2.button if b.label == "Add Item"][0]
                add_button.click().run()

            a2_record = json.loads((tmp_path / "A2" / "h38.json").read_text(encoding="utf-8"))
            self.assertEqual(a2_record["items"][0]["label"], "a2_item")

            a1_record_untouched = json.loads((tmp_path / "A1" / "h38.json").read_text(encoding="utf-8"))
            self.assertEqual(a1_record_untouched["items"][0]["label"], "secret_a1_item")


if __name__ == "__main__":
    unittest.main()
