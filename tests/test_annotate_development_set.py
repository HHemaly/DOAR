"""Tests for scripts/annotate_development_set.py's Pass-1 timing-protocol
enforcement: the image must never be shown before the 60-second window
starts, and no annotation typed after the deadline may ever be recorded.
No real waiting, no real Tk window, and no real terminal are used --
`collect_items`'s clock/queue are injected, and `run_pass1`'s `open_fn`
is a recording stub.
"""
import importlib.util
import queue
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("annotate_development_set", SCRIPTS_DIR / "annotate_development_set.py")
ads = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ads)


class _FakeClock:
    """Returns a pre-scripted sequence of timestamps, one per call --
    lets a test simulate "time passing" deterministically without any
    real sleep."""

    def __init__(self, timestamps):
        self._timestamps = list(timestamps)

    def __call__(self):
        if len(self._timestamps) > 1:
            return self._timestamps.pop(0)
        return self._timestamps[0]  # hold the last value once exhausted


class CollectItemsDeadlineTests(unittest.TestCase):
    def test_stops_reading_once_deadline_passed_even_with_items_still_queued(self):
        q = queue.Queue()
        q.put("sun | top-left")
        q.put("tree | left")
        q.put("late-item-after-deadline | anywhere")  # must NEVER be consumed
        # clock() is called once per loop iteration, before each queue read:
        # iter1 -> 0.0 (read "sun"), iter2 -> 0.0 (read "tree"), iter3 -> 70.0
        # (past the 60s deadline -- returns WITHOUT reading the 3rd item).
        clock = _FakeClock([0.0, 0.0, 70.0])
        items, timed_out = ads.collect_items(q, 1, start_time=0.0, duration_seconds=60.0, clock=clock)
        labels = [i["label"] for i in items]
        self.assertEqual(labels, ["sun", "tree"])
        self.assertNotIn("late-item-after-deadline", labels)
        self.assertTrue(timed_out)

    def test_deadline_already_passed_before_first_read_returns_no_items(self):
        q = queue.Queue()
        q.put("sun | top-left")
        clock = _FakeClock([61.0])  # already past the 60s deadline from t=0
        items, timed_out = ads.collect_items(q, 1, start_time=0.0, duration_seconds=60.0, clock=clock)
        self.assertEqual(items, [])
        self.assertTrue(timed_out)

    def test_blank_line_ends_pass_early_without_timing_out(self):
        q = queue.Queue()
        q.put("sun | top-left")
        q.put("")
        clock = _FakeClock([0.0, 0.0, 5.0])
        items, timed_out = ads.collect_items(q, 1, start_time=0.0, duration_seconds=60.0, clock=clock)
        self.assertEqual([i["label"] for i in items], ["sun"])
        self.assertFalse(timed_out)

    def test_eof_ends_pass_without_timing_out(self):
        q = queue.Queue()
        q.put(None)
        clock = _FakeClock([0.0])
        items, timed_out = ads.collect_items(q, 1, start_time=0.0, duration_seconds=60.0, clock=clock)
        self.assertEqual(items, [])
        self.assertFalse(timed_out)

    def test_pass2_has_no_deadline_and_reads_until_blank_line(self):
        q = queue.Queue()
        q.put("a | loc")
        q.put("b | loc | ambiguous")
        q.put("")
        items, timed_out = ads.collect_items(q, 2)  # duration_seconds=None
        self.assertEqual(len(items), 2)
        self.assertEqual(items[1]["confidence"], "ambiguous")
        self.assertFalse(timed_out)


class ImageNotExposedBeforeTimingTests(unittest.TestCase):
    """`run_pass1` must call the "wait for start" gate strictly before it
    calls `open_fn` -- the actual code-level guarantee that the image is
    never shown before the 60-second window begins."""

    def test_open_fn_called_only_after_start_gate(self):
        call_order = []

        class RecordingQueue:
            """Stands in for the real stdin queue: records when `.get()`
            (the start-gate wait, and later the annotation reads) happens
            relative to when `open_fn` fires, then returns EOF so the
            pass ends immediately."""

            def get(self, timeout=None):
                call_order.append("queue_get")
                return None

        def fake_open_fn(image_path, duration_seconds):
            call_order.append("open_fn")

        image = {"image_id": "fixture_img", "relative_path": "does/not/matter.jpg"}
        tmp_root = Path(__file__).resolve().parent  # any existing dir; open_fn is faked, never touches disk

        # Redirect output_path to a throwaway location so this test never
        # writes into the real annotations/ directory.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original_annotations_dir = ads.ANNOTATIONS_DIR
            ads.ANNOTATIONS_DIR = Path(tmp)
            try:
                ads.run_pass1(
                    "TEST_ANNOTATOR", image, tmp_root, RecordingQueue(),
                    open_fn=fake_open_fn, clock=lambda: 0.0, duration_seconds=60.0,
                )
            finally:
                ads.ANNOTATIONS_DIR = original_annotations_dir

        self.assertIn("open_fn", call_order)
        first_queue_get_index = call_order.index("queue_get")
        open_fn_index = call_order.index("open_fn")
        self.assertLess(
            first_queue_get_index, open_fn_index,
            "the start-gate wait must happen before the image is opened",
        )

    def test_open_fn_never_called_if_pass_already_recorded(self):
        """Resumability: a pre-existing pass-1 file must short-circuit
        before ever touching open_fn -- re-running the tool must not
        re-expose an image for a pass that's already done."""
        call_order = []

        def fake_open_fn(image_path, duration_seconds):
            call_order.append("open_fn")

        image = {"image_id": "already_done", "relative_path": "does/not/matter.jpg"}
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original_annotations_dir = ads.ANNOTATIONS_DIR
            ads.ANNOTATIONS_DIR = Path(tmp)
            try:
                existing = Path(tmp) / "TEST_ANNOTATOR"
                existing.mkdir(parents=True)
                (existing / "already_done_pass1.json").write_text(
                    json.dumps({"annotator_id": "TEST_ANNOTATOR", "image_id": "already_done",
                                "pass": 1, "items": [], "elapsed_seconds": 1.0, "timed_out": False}),
                    encoding="utf-8")
                ads.run_pass1("TEST_ANNOTATOR", image, Path(tmp), queue.Queue(), open_fn=fake_open_fn)
            finally:
                ads.ANNOTATIONS_DIR = original_annotations_dir

        self.assertEqual(call_order, [])


if __name__ == "__main__":
    unittest.main()
