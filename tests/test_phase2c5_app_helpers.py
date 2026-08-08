"""Phase 2C.5 annotation-app-logic tests -- synthetic data only, no
Streamlit runtime needed."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import app_helpers as helpers
from doar.phase2c5.schema import PartInstance


def _proposal(index=0, bbox_source="model_proposed", model="grounding_dino:tiny"):
    return PartInstance(instance_index=index, bbox=(0.1, 0.1, 0.2, 0.2), bbox_source=bbox_source,
                         proposal_model=model, proposal_checkpoint="ckpt", proposal_prompt="eye.",
                         proposal_threshold=0.25, proposal_timestamp="t")


class AcceptEditRejectTests(unittest.TestCase):
    def test_accept_changes_source_keeps_provenance(self):
        accepted = helpers.accept_proposal_instance(_proposal())
        self.assertEqual(accepted.bbox_source, "human_accepted")
        self.assertEqual(accepted.proposal_model, "grounding_dino:tiny")

    def test_accept_rejects_non_proposal_input(self):
        with self.assertRaises(ValueError):
            helpers.accept_proposal_instance(_proposal(bbox_source="human_drawn", model=""))

    def test_edit_changes_bbox_and_source_keeps_provenance(self):
        edited = helpers.edit_proposal_instance(_proposal(), (0.5, 0.5, 0.1, 0.1))
        self.assertEqual(edited.bbox_source, "human_edited")
        self.assertEqual(edited.bbox, (0.5, 0.5, 0.1, 0.1))
        self.assertEqual(edited.proposal_model, "grounding_dino:tiny")

    def test_new_manual_instance_has_no_proposal_provenance(self):
        inst = helpers.new_manual_instance(0, (0.2, 0.2, 0.1, 0.1))
        self.assertEqual(inst.bbox_source, "human_drawn")
        self.assertEqual(inst.proposal_model, "")


class RenumberDeleteTests(unittest.TestCase):
    def test_renumber_reassigns_sequential_indices(self):
        instances = [_proposal(index=5), _proposal(index=9)]
        renumbered = helpers.renumber_instances(instances)
        self.assertEqual([i.instance_index for i in renumbered], [0, 1])

    def test_delete_instance_renumbers_remaining(self):
        instances = [helpers.new_manual_instance(0, (0.1, 0.1, 0.1, 0.1)),
                     helpers.new_manual_instance(1, (0.2, 0.2, 0.1, 0.1)),
                     helpers.new_manual_instance(2, (0.3, 0.3, 0.1, 0.1))]
        remaining = helpers.delete_instance(instances, 1)
        self.assertEqual(len(remaining), 2)
        self.assertEqual([i.instance_index for i in remaining], [0, 1])
        self.assertEqual(remaining[0].bbox, (0.1, 0.1, 0.1, 0.1))
        self.assertEqual(remaining[1].bbox, (0.3, 0.3, 0.1, 0.1))


class ProposalSummaryTests(unittest.TestCase):
    def test_reviewed_proposal_summary_counts_by_source(self):
        instances = [_proposal(0, "human_accepted"), _proposal(1, "human_edited"),
                     helpers.new_manual_instance(2, (0.1, 0.1, 0.1, 0.1))]
        summary = helpers.reviewed_proposal_summary(instances)
        self.assertEqual(summary["human_accepted"], 1)
        self.assertEqual(summary["human_edited"], 1)
        self.assertEqual(summary["human_drawn"], 1)
        self.assertEqual(summary["model_proposed"], 0)


class LoadProposalsCsvTests(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(helpers.load_proposals_csv("/no/such/file.csv"), {})

    def test_loads_and_groups_by_model_pilot_target(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "proposals.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["model", "pilot_id", "target", "instance_index",
                                                   "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
                w.writeheader()
                w.writerow({"model": "grounding_dino", "pilot_id": "p2b_0000", "target": "eye",
                            "instance_index": 0, "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.05, "bbox_h": 0.05})
            loaded = helpers.load_proposals_csv(path)
            self.assertIn(("grounding_dino", "p2b_0000", "eye"), loaded)
            self.assertEqual(loaded[("grounding_dino", "p2b_0000", "eye")][0]["bbox"], (0.1, 0.1, 0.05, 0.05))


class SeedInstancesTests(unittest.TestCase):
    def test_seed_instances_produces_model_proposed(self):
        boxes = [{"instance_index": 0, "bbox": (0.1, 0.1, 0.1, 0.1)}]
        seeded = helpers.seed_instances_from_proposals(
            boxes, model_name="owlv2:base", checkpoint="c", prompt="a eye", threshold=0.1, timestamp="t")
        self.assertEqual(len(seeded), 1)
        self.assertEqual(seeded[0].bbox_source, "model_proposed")


class ValidateTargetNameTests(unittest.TestCase):
    def test_valid_target_does_not_raise(self):
        helpers.validate_target_name("eye")

    def test_invalid_target_raises(self):
        with self.assertRaises(ValueError):
            helpers.validate_target_name("nose")


if __name__ == "__main__":
    unittest.main()
