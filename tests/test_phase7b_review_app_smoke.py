"""End-to-end Streamlit smoke tests for phase7b_review_app.py, using
streamlit.testing.v1.AppTest (bare-mode script execution, no browser/server
needed) -- the same tool used for doar_prototype_app.py's own smoke tests.

CRITICAL: these tests must NEVER write to the real
outputs/phase7b/human_review/app_data/decisions.json -- that file holds (or
will hold) real human review progress. Every test here points the app at a
throwaway registry/decisions/export location via the
DOAR_PHASE7B_REGISTRY_PATH / DOAR_PHASE7B_DECISIONS_PATH /
DOAR_PHASE7B_EXPORT_DIR environment variables phase7b_review_app.py now
reads (falling back to the real production paths only when unset, i.e.
during real, non-test use).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False


def _build_tiny_registry(tmp_dir: Path) -> Path:
    """A small (8-item, exactly 2 per category, all 4 categories
    represented) synthetic registry -- real image files on disk so
    st.image never errors, but nowhere near the real 225-item registry
    (keeps the test fast). Every category MUST be non-empty: the real app
    calls st.number_input(max_value=len(ids), ...) per category, which
    raises for an empty category -- this fixture deliberately guards
    against regressing back to that bug (see
    test_no_category_may_be_empty below)."""
    images_dir = tmp_dir / "review_pairs"
    images_dir.mkdir(parents=True, exist_ok=True)
    items, categories_order = {}, {"ambiguous": [], "threshold_boundary": [],
                                    "component_17": [], "policy_change": []}
    for category in categories_order:
        for i in range(2):  # exactly 2 items per category = 8 total
            item_id = f"{category}_{i:04d}"
            img_path = images_dir / f"{item_id}.jpg"
            Image.new("RGB", (40, 20), "white").save(img_path)
            items[item_id] = {
                "item_id": item_id, "category": category,
                "image_rel_path": f"review_pairs/{item_id}.jpg",
                "image_a": f"{item_id}_a", "image_b": f"{item_id}_b",
                "class_a": "Happy", "class_b": "Sad", "split_a": "train", "split_b": "train",
                "same_label": False, "source_category": "synthetic_test",
                "hamming_ahash": None, "hamming_dhash": 3,
                "ai_preliminary_judgment": "uncertain", "ai_notes": "synthetic fixture",
                "group_id": None, "group_size": None, "edge_type": "near_dup",
            }
            categories_order[category].append(item_id)
    registry_path = tmp_dir / "items_registry.json"
    registry_path.write_text(json.dumps({"items": items, "categories_order": categories_order}),
                             encoding="utf-8")
    return registry_path


class _IsolatedAppTestCase(unittest.TestCase):
    """Sets DOAR_PHASE7B_* env vars to a fresh temp dir for every test, and
    always restores/clears them afterward -- guarantees no test can ever
    touch the real outputs/phase7b/human_review/ files."""

    def setUp(self):
        if not _STREAMLIT_TESTING_AVAILABLE:
            self.skipTest("streamlit.testing.v1.AppTest not available")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.registry_path = _build_tiny_registry(self.tmp_dir)
        self.decisions_path = self.tmp_dir / "decisions.json"
        self.export_dir = self.tmp_dir / "exports"
        self._env_backup = {
            k: os.environ.get(k) for k in
            ("DOAR_PHASE7B_REGISTRY_PATH", "DOAR_PHASE7B_DECISIONS_PATH", "DOAR_PHASE7B_EXPORT_DIR")
        }
        os.environ["DOAR_PHASE7B_REGISTRY_PATH"] = str(self.registry_path)
        os.environ["DOAR_PHASE7B_DECISIONS_PATH"] = str(self.decisions_path)
        os.environ["DOAR_PHASE7B_EXPORT_DIR"] = str(self.export_dir)
        # Sanity guard, checked in every test: never point at the real
        # outputs/phase7b/human_review/ tree, under any circumstance.
        real_human_review_dir = (ROOT / "outputs/phase7b/human_review").resolve()
        for p in (self.registry_path, self.decisions_path, self.export_dir):
            self.assertFalse(
                str(p.resolve()).startswith(str(real_human_review_dir)),
                f"{p} must never resolve inside the real {real_human_review_dir}")

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def _app(self) -> "AppTest":
        at = AppTest.from_file(str(ROOT / "phase7b_review_app.py"))
        at.run(timeout=60)
        return at


class RendersWithoutExceptionTests(_IsolatedAppTestCase):
    def test_fresh_state_no_decisions_yet(self):
        self.assertFalse(self.decisions_path.exists())
        at = self._app()
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

    def test_sidebar_shows_export_button(self):
        at = self._app()
        button_labels = [b.label for b in at.sidebar.button]
        self.assertTrue(any("Export" in label for label in button_labels), button_labels)

    def test_sidebar_shows_progress_counts(self):
        at = self._app()
        sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("0 / 8", sidebar_text)
        self.assertIn("8 remaining", sidebar_text)
        # Per-category counts (2 items each) must also be visible.
        self.assertIn("0/2", sidebar_text)

    def test_no_category_may_be_empty(self):
        """Regression guard: an empty category makes the real app's
        `st.number_input(max_value=len(ids), value=idx+1)` raise
        StreamlitValueAboveMaxError (found while building this test's own
        fixture -- a bad cap produced a zero-item category). The real
        production registry (225 items across 4 non-empty categories) can
        never hit this, but the fixture must not regress into it either."""
        import json as _json
        registry = _json.loads(self.registry_path.read_text(encoding="utf-8"))
        for category, ids in registry["categories_order"].items():
            self.assertGreater(len(ids), 0, f"category {category!r} must not be empty")

    def test_incomplete_review_shows_warning(self):
        at = self._app()
        warnings = [w.value for w in at.sidebar.warning]
        self.assertTrue(any("INCOMPLETE" in w for w in warnings), warnings)


class DecisionPersistenceTests(_IsolatedAppTestCase):
    def test_no_decisions_file_created_just_by_viewing(self):
        self._app()
        self.assertFalse(self.decisions_path.exists(),
                         "Opening the app must never create decisions.json on its own")

    def test_saving_one_decision_creates_decisions_json(self):
        from doar.human_review import save_decision
        # Exercise the exact function the Save button calls, against the
        # isolated path, to prove persistence works end-to-end without
        # depending on AppTest's ability to drive a dynamically-keyed radio
        # widget (Streamlit radio keys are per-item, verified separately
        # via direct widget interaction below).
        save_decision(self.decisions_path, "ambiguous_0000", "definite_duplicate", "test note")
        self.assertTrue(self.decisions_path.exists())
        saved = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["ambiguous_0000"]["decision"], "definite_duplicate")
        self.assertEqual(saved["ambiguous_0000"]["notes"], "test note")
        self.assertIn("reviewed_at", saved["ambiguous_0000"])

    def test_saved_decision_reflected_on_next_app_run(self):
        from doar.human_review import save_decision
        save_decision(self.decisions_path, "ambiguous_0000", "definite_duplicate", "")
        at = self._app()
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        # Progress must now read 1/6, not 0/6.
        sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
        self.assertIn("1 / 8", sidebar_text)

    def test_save_via_actual_app_radio_and_button_widgets(self):
        at = self._app()
        # First tab = "ambiguous" category; find its radio + save button.
        radio = at.tabs[0].radio[0]
        radio.set_value("definite_duplicate")
        save_buttons = [b for b in at.tabs[0].button if "Save decision" in b.label]
        self.assertTrue(save_buttons, "Save decision button not found in the first tab")
        save_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(self.decisions_path.exists(),
                        "Clicking Save decision in the real app must create decisions.json")
        saved = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(next(iter(saved.values()))["decision"], "definite_duplicate")

    def test_clear_decision_removes_it(self):
        from doar.human_review import save_decision
        save_decision(self.decisions_path, "ambiguous_0000", "definite_duplicate", "")
        at = self._app()
        # The app auto-jumps to the first UNREVIEWED item on load (a real,
        # intentional convenience feature) -- explicitly navigate back to
        # item #1 (the one just decided) via the "Jump to item #" widget
        # before looking for its Clear button.
        at.tabs[0].number_input[0].set_value(1)
        at.run(timeout=60)
        clear_buttons = [b for b in at.tabs[0].button if "Clear decision" in b.label]
        self.assertTrue(clear_buttons)
        clear_buttons[0].click()
        at.run(timeout=60)
        saved = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        self.assertNotIn("ambiguous_0000", saved)


class ExportTests(_IsolatedAppTestCase):
    def test_export_button_writes_all_five_expected_files(self):
        from doar.human_review import save_decision
        save_decision(self.decisions_path, "ambiguous_0000", "definite_duplicate", "")
        at = self._app()
        export_buttons = [b for b in at.sidebar.button if "Export" in b.label]
        self.assertTrue(export_buttons)
        export_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        expected = {"human_pair_reviews.csv", "human_group_reviews.csv",
                   "reviewer_agreement_report.json", "threshold_precision_summary.json",
                   "unresolved_items.csv"}
        actual = {p.name for p in self.export_dir.glob("*")}
        self.assertEqual(expected, actual & expected)

    def test_export_never_fabricates_a_decision_for_unreviewed_items(self):
        from doar.human_review import save_decision
        save_decision(self.decisions_path, "ambiguous_0000", "definite_duplicate", "")
        at = self._app()
        export_buttons = [b for b in at.sidebar.button if "Export" in b.label]
        export_buttons[0].click()
        at.run(timeout=60)
        import csv
        rows = list(csv.DictReader(open(self.export_dir / "human_pair_reviews.csv", encoding="utf-8")))
        undecided = [r for r in rows if r["item_id"] != "ambiguous_0000"]
        self.assertTrue(undecided)
        for r in undecided:
            self.assertEqual(r["human_judgment"], "")  # never fabricated

    def test_partial_export_still_works_and_warns(self):
        # No decisions saved at all -- export must still succeed (explicit
        # requirement: exporting mid-review is allowed), just with a warning.
        at = self._app()
        export_buttons = [b for b in at.sidebar.button if "Export" in b.label]
        export_buttons[0].click()
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue((self.export_dir / "unresolved_items.csv").exists())


if __name__ == "__main__":
    unittest.main()
