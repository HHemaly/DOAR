"""Phase 2C.5 store tests -- synthetic data, real filesystem tmp dirs only."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import store as store_mod
from doar.phase2c5.schema import PartAnnotationRecord, PartInstance


def _rec(pilot_id="p2b_0000", target_name="eye", annotator_id="ann1", status="present"):
    instances = (PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1),
                               bbox_source="human_drawn"),) if status == "present" else ()
    return PartAnnotationRecord(pilot_id=pilot_id, target_name=target_name, status=status,
                                 annotator_id=annotator_id, annotation_timestamp="t", instances=instances)


class LoadSaveRoundTripTests(unittest.TestCase):
    def test_missing_file_returns_empty_store(self):
        with tempfile.TemporaryDirectory() as d:
            store = store_mod.load_store(Path(d) / "does_not_exist.csv")
            self.assertEqual(store, {})

    def test_save_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "store.csv"
            store = {}
            store_mod.upsert(store, _rec())
            store_mod.save_store(path, store)
            reloaded = store_mod.load_store(path)
            self.assertEqual(set(reloaded), set(store))

    def test_upsert_replaces_same_key_not_appends(self):
        store = {}
        store_mod.upsert(store, _rec(status="present"))
        store_mod.upsert(store, _rec(status="absent"))
        self.assertEqual(len(store), 1)
        self.assertEqual(next(iter(store.values())).status, "absent")

    def test_resume_after_simulated_crash(self):
        """upsert_and_save writes atomically -- reloading after a fresh
        process start (simulated by a brand-new load_store call) must see
        every previously saved row, mirroring the app's own resume path."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "store.csv"
            store = {}
            store_mod.upsert_and_save(path, store, _rec(pilot_id="p2b_0000"))
            store_mod.upsert_and_save(path, store, _rec(pilot_id="p2b_0001"))
            # simulate a fresh process picking the store back up
            resumed = store_mod.load_store(path)
            self.assertEqual(store_mod.annotated_pilot_ids(resumed), {"p2b_0000", "p2b_0001"})


class ProgressTests(unittest.TestCase):
    def test_progress_for_annotator_counts_only_complete_images(self):
        store = {}
        store_mod.upsert(store, _rec(pilot_id="p2b_0000", target_name="eye"))
        store_mod.upsert(store, _rec(pilot_id="p2b_0000", target_name="mouth"))
        store_mod.upsert(store, _rec(pilot_id="p2b_0001", target_name="eye"))
        progress = store_mod.progress_for_annotator(
            store, ["p2b_0000", "p2b_0001"], "ann1", ("eye", "mouth"))
        self.assertEqual(progress, {"done": 1, "total": 2})

    def test_progress_ignores_other_annotators(self):
        store = {}
        store_mod.upsert(store, _rec(pilot_id="p2b_0000", target_name="eye", annotator_id="other"))
        progress = store_mod.progress_for_annotator(store, ["p2b_0000"], "ann1", ("eye",))
        self.assertEqual(progress, {"done": 0, "total": 1})


if __name__ == "__main__":
    unittest.main()
