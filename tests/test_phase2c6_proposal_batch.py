"""Phase 2C.6 Stage 4 resumable/checkpointed batch pipeline tests --
synthetic fake predict functions only, no real model weights, no network."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c6 import proposal_batch as pb


class MakeBatchesTests(unittest.TestCase):
    def test_splits_into_correctly_sized_batches(self):
        batches = pb.make_batches([f"p{i}" for i in range(55)], batch_size=25)
        self.assertEqual([len(b) for b in batches], [25, 25, 5])

    def test_rejects_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            pb.make_batches(["a"], batch_size=0)


class PendingPilotIdsTests(unittest.TestCase):
    def test_completed_and_skipped_excluded_others_included(self):
        checkpoint = {
            "p1": {"status": "completed"}, "p2": {"status": "failed"},
            "p3": {"status": "skipped_not_assessable"}, "p4": {"status": "pending"},
        }
        todo = pb.pending_pilot_ids(["p1", "p2", "p3", "p4", "p5"], checkpoint)
        self.assertEqual(set(todo), {"p2", "p4", "p5"})


class CheckpointRoundTripTests(unittest.TestCase):
    def test_save_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.csv"
            checkpoint = {"p1": {"pilot_id": "p1", "status": "completed", "n_proposals": 3,
                                  "primary_model": "m", "fallback_used": "False",
                                  "error": "", "elapsed_s": "1.0", "timestamp": "t"}}
            pb.save_checkpoint(path, checkpoint)
            reloaded = pb.load_checkpoint(path)
            self.assertEqual(reloaded["p1"]["status"], "completed")

    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(pb.load_checkpoint("/no/such/file.csv"), {})


class AppendProposalRowsTests(unittest.TestCase):
    def test_writes_header_once_across_multiple_appends(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "proposals.csv"
            pb.append_proposal_rows(path, [{"model": "m", "pilot_id": "p1", "target": "eye",
                                             "instance_index": 0, "bbox_x": 0.1, "bbox_y": 0.1,
                                             "bbox_w": 0.1, "bbox_h": 0.1}])
            pb.append_proposal_rows(path, [{"model": "m", "pilot_id": "p2", "target": "eye",
                                             "instance_index": 0, "bbox_x": 0.2, "bbox_y": 0.2,
                                             "bbox_w": 0.1, "bbox_h": 0.1}])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "model,pilot_id,target,instance_index,bbox_x,bbox_y,bbox_w,bbox_h")
            self.assertEqual(len(lines), 3)

    def test_empty_rows_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "proposals.csv"
            pb.append_proposal_rows(path, [])
            self.assertFalse(path.exists())


def _fake_image_dir(tmp_dir: Path, pilot_ids: list[str], unreadable: set[str] = frozenset()) -> Path:
    images_dir = tmp_dir / "images"
    images_dir.mkdir()
    for pid in pilot_ids:
        content = b"" if pid in unreadable else b"fake image bytes"
        (images_dir / f"{pid}.jpg").write_bytes(content)
    return images_dir


def _fake_process_image_fn(pilot_id: str, image_path: Path):
    """A trivial stand-in for process_one_image: 'fails' on pilot_ids
    containing 'bad', 'skips' on zero-byte files, otherwise 'succeeds'
    with one fake proposal."""
    if image_path.stat().st_size == 0:
        return ({"pilot_id": pilot_id, "status": "skipped_not_assessable", "n_proposals": 0,
                 "primary_model": "", "fallback_used": "False", "error": "", "elapsed_s": "0",
                 "timestamp": "t"}, [])
    if "bad" in pilot_id:
        return ({"pilot_id": pilot_id, "status": "failed", "n_proposals": 0,
                 "primary_model": "fake", "fallback_used": "False", "error": "boom",
                 "elapsed_s": "0", "timestamp": "t"}, [])
    return ({"pilot_id": pilot_id, "status": "completed", "n_proposals": 1,
             "primary_model": "fake", "fallback_used": "False", "error": "", "elapsed_s": "0.1",
             "timestamp": "t"}, [{"model": "fake", "pilot_id": pilot_id, "target": "eye",
                                   "instance_index": 0, "bbox_x": 0.1, "bbox_y": 0.1,
                                   "bbox_w": 0.1, "bbox_h": 0.1}])


class RunBatchesTests(unittest.TestCase):
    def test_full_run_processes_every_image_and_checkpoints_each_batch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pilot_ids = [f"p{i:03d}" for i in range(12)]
            images_dir = _fake_image_dir(d, pilot_ids)
            checkpoint_path, proposals_path = d / "checkpoint.csv", d / "proposals.csv"
            summary = pb.run_batches(pilot_ids, images_dir, checkpoint_path=checkpoint_path,
                                      proposals_csv_path=proposals_path, batch_size=5,
                                      process_image_fn=_fake_process_image_fn, log=lambda *_: None)
            self.assertEqual(summary["n_completed"], 12)
            self.assertEqual(summary["n_total"], 12)
            checkpoint = pb.load_checkpoint(checkpoint_path)
            self.assertEqual(len(checkpoint), 12)

    def test_skipped_not_assessable_for_unreadable_image(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pilot_ids = ["p001", "p002_unreadable"]
            images_dir = _fake_image_dir(d, pilot_ids, unreadable={"p002_unreadable"})
            summary = pb.run_batches(pilot_ids, images_dir, checkpoint_path=d / "c.csv",
                                      proposals_csv_path=d / "p.csv", batch_size=5,
                                      process_image_fn=_fake_process_image_fn, log=lambda *_: None)
            self.assertEqual(summary["n_skipped_not_assessable"], 1)
            self.assertEqual(summary["n_completed"], 1)

    def test_failed_image_does_not_block_the_rest_of_the_batch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pilot_ids = ["p001", "p002_bad", "p003"]
            images_dir = _fake_image_dir(d, pilot_ids)
            summary = pb.run_batches(pilot_ids, images_dir, checkpoint_path=d / "c.csv",
                                      proposals_csv_path=d / "p.csv", batch_size=5,
                                      process_image_fn=_fake_process_image_fn, log=lambda *_: None)
            self.assertEqual(summary["n_failed"], 1)
            self.assertEqual(summary["n_completed"], 2)

    def test_interrupted_mid_run_resumes_only_unfinished_images(self):
        """Simulates a kill after the first batch: run_batches is called
        with only the first 5 pilot_ids (as if the process died before
        reaching the rest), then called again with the FULL list -- the
        second call must skip the first 5 (already checkpointed
        'completed') and only process the remaining 7."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pilot_ids = [f"p{i:03d}" for i in range(12)]
            images_dir = _fake_image_dir(d, pilot_ids)
            checkpoint_path, proposals_path = d / "checkpoint.csv", d / "proposals.csv"

            processed_order = []

            def tracking_fn(pilot_id, image_path):
                processed_order.append(pilot_id)
                return _fake_process_image_fn(pilot_id, image_path)

            # "first run" -- only sees the first 5 images (simulating a kill)
            pb.run_batches(pilot_ids[:5], images_dir, checkpoint_path=checkpoint_path,
                            proposals_csv_path=proposals_path, batch_size=5,
                            process_image_fn=tracking_fn, log=lambda *_: None)
            self.assertEqual(processed_order, pilot_ids[:5])

            # "restart" -- sees the FULL list, must only (re)process the remaining 7
            processed_order.clear()
            summary = pb.run_batches(pilot_ids, images_dir, checkpoint_path=checkpoint_path,
                                      proposals_csv_path=proposals_path, batch_size=5,
                                      process_image_fn=tracking_fn, log=lambda *_: None)
            self.assertEqual(set(processed_order), set(pilot_ids[5:]))
            self.assertEqual(summary["n_completed"], 12)
            # proposals.csv must contain exactly one header line despite two separate runs
            lines = proposals_path.read_text(encoding="utf-8").splitlines()
            header_lines = [ln for ln in lines if ln.startswith("model,pilot_id")]
            self.assertEqual(len(header_lines), 1)

    def test_failed_image_is_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pilot_ids = ["p001_bad"]
            images_dir = _fake_image_dir(d, pilot_ids)
            checkpoint_path, proposals_path = d / "checkpoint.csv", d / "proposals.csv"
            pb.run_batches(pilot_ids, images_dir, checkpoint_path=checkpoint_path,
                            proposals_csv_path=proposals_path, batch_size=5,
                            process_image_fn=_fake_process_image_fn, log=lambda *_: None)
            self.assertEqual(pb.load_checkpoint(checkpoint_path)["p001_bad"]["status"], "failed")

            attempts = []

            def eventually_succeeds(pilot_id, image_path):
                attempts.append(pilot_id)
                return ({"pilot_id": pilot_id, "status": "completed", "n_proposals": 0,
                         "primary_model": "fake", "fallback_used": "False", "error": "",
                         "elapsed_s": "0", "timestamp": "t"}, [])

            pb.run_batches(pilot_ids, images_dir, checkpoint_path=checkpoint_path,
                            proposals_csv_path=proposals_path, batch_size=5,
                            process_image_fn=eventually_succeeds, log=lambda *_: None)
            self.assertEqual(attempts, ["p001_bad"])  # retried, not skipped
            self.assertEqual(pb.load_checkpoint(checkpoint_path)["p001_bad"]["status"], "completed")

    def test_missing_image_file_recorded_as_failed_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            images_dir = d / "images"
            images_dir.mkdir()
            summary = pb.run_batches(["p_missing"], images_dir, checkpoint_path=d / "c.csv",
                                      proposals_csv_path=d / "p.csv", batch_size=5,
                                      process_image_fn=_fake_process_image_fn, log=lambda *_: None)
            self.assertEqual(summary["n_failed"], 1)


class ProcessOneImageTests(unittest.TestCase):
    def test_never_raises_on_predict_exception(self):
        def boom(_path):
            raise RuntimeError("simulated model crash")

        row, rows = pb.process_one_image(
            "p1", Path("fake.jpg"), predict_primary=boom,
            primary_meta={"model_name": "m", "checkpoint": "c", "prompt": "p", "threshold": 0.1},
            predict_fallback=None, fallback_meta=None, targets=("eye",),
            build_proposals=lambda *a, **k: {}, is_assessable=lambda p: True, timestamp_fn=lambda: "t")
        self.assertEqual(row["status"], "failed")
        self.assertIn("simulated model crash", row["error"])
        self.assertEqual(rows, [])

    def test_fallback_only_used_for_empty_targets(self):
        calls = []

        def primary(_path):
            calls.append("primary")
            return ["raw_primary"]

        def fallback(_path):
            calls.append("fallback")
            return ["raw_fallback"]

        # build_proposals needs real PartInstance-like objects with
        # .proposal_model/.bbox/.instance_index for the row-building step.
        class FakeInst:
            def __init__(self, model):
                self.proposal_model = model
                self.instance_index = 0
                self.bbox = (0.1, 0.1, 0.1, 0.1)

        def build_proposals(raw, *, model_name, **kwargs):
            if raw == ["raw_primary"]:
                return {"eye": [FakeInst(model_name)], "mouth": []}
            return {"eye": [], "mouth": [FakeInst(model_name)]}

        row, rows = pb.process_one_image(
            "p1", Path("fake.jpg"), predict_primary=primary,
            primary_meta={"model_name": "grounding_dino", "checkpoint": "c", "prompt": "p", "threshold": 0.25},
            predict_fallback=fallback,
            fallback_meta={"model_name": "owlv2", "checkpoint": "c2", "prompt": "p2", "threshold": 0.1},
            targets=("eye", "mouth"), build_proposals=build_proposals,
            is_assessable=lambda p: True, timestamp_fn=lambda: "t")
        self.assertEqual(calls, ["primary", "fallback"])
        self.assertEqual(row["fallback_used"], "True")
        targets_seen = {r["target"] for r in rows}
        self.assertEqual(targets_seen, {"eye", "mouth"})


if __name__ == "__main__":
    unittest.main()
