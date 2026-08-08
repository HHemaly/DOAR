"""Phase 2C.6 Stage 1 canvas coordinate-conversion tests -- synthetic
data only, no Streamlit/browser runtime needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c6 import canvas_helpers as ch
from doar.phase2c5.schema import PartInstance


def _inst(index=0, bbox=(0.1, 0.2, 0.3, 0.4), source="model_proposed", model="m"):
    kwargs = dict(instance_index=index, bbox=bbox, bbox_source=source)
    if source == "model_proposed":
        kwargs["proposal_model"] = model
    return PartInstance(**kwargs)


class ComputeDisplaySizeTests(unittest.TestCase):
    def test_preserves_aspect_ratio_when_downscaling(self):
        w, h = ch.compute_display_size(2000, 1000, max_dim=800, min_dim=400)
        self.assertEqual(w, 800)
        self.assertAlmostEqual(h, 400, delta=1)

    def test_upscales_small_images_to_min_dim(self):
        w, h = ch.compute_display_size(100, 50, max_dim=800, min_dim=400)
        self.assertEqual(w, 400)
        self.assertAlmostEqual(h, 200, delta=1)

    def test_leaves_already_in_range_size_unchanged(self):
        w, h = ch.compute_display_size(600, 500, max_dim=800, min_dim=400)
        self.assertEqual((w, h), (600, 500))

    def test_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            ch.compute_display_size(0, 100)


class RoundTripCoordinateTests(unittest.TestCase):
    def test_normalized_to_canvas_to_normalized_round_trips(self):
        bbox = (0.1, 0.2, 0.3, 0.15)
        rect = ch.normalized_to_canvas_rect(bbox, 640, 480, bbox_source="human_drawn")
        recovered = ch.canvas_object_to_normalized_bbox(rect, 640, 480)
        for a, b in zip(bbox, recovered):
            self.assertAlmostEqual(a, b, places=6)

    def test_round_trip_survives_a_resize_between_conversions(self):
        """The whole point of always deriving from the CURRENT display
        size: converting at one display size, then converting the result
        back at a DIFFERENT display size, must not corrupt the box --
        this simulates a browser window resize between saves."""
        bbox = (0.25, 0.25, 0.2, 0.2)
        rect_a = ch.normalized_to_canvas_rect(bbox, 700, 700, bbox_source="human_drawn")
        recovered_a = ch.canvas_object_to_normalized_bbox(rect_a, 700, 700)
        rect_b = ch.normalized_to_canvas_rect(recovered_a, 400, 550, bbox_source="human_drawn")
        recovered_b = ch.canvas_object_to_normalized_bbox(rect_b, 400, 550)
        for a, b in zip(bbox, recovered_b):
            self.assertAlmostEqual(a, b, places=4)

    def test_scale_factor_is_respected_not_just_width_height(self):
        """Fabric.js reports a resize as a scaleX/scaleY change, not a new
        width/height -- this must be read correctly."""
        obj = {"left": 100, "top": 50, "width": 40, "height": 30, "scaleX": 2.0, "scaleY": 1.5}
        x, y, w, h = ch.canvas_object_to_normalized_bbox(obj, 400, 200)
        self.assertAlmostEqual(x, 0.25)
        self.assertAlmostEqual(y, 0.25)
        self.assertAlmostEqual(w, 80 / 400)
        self.assertAlmostEqual(h, 45 / 200)

    def test_clamps_out_of_bounds_box(self):
        obj = {"left": -50, "top": -50, "width": 900, "height": 900, "scaleX": 1.0, "scaleY": 1.0}
        x, y, w, h = ch.canvas_object_to_normalized_bbox(obj, 400, 400)
        self.assertGreaterEqual(x, 0.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(x + w, 1.0)
        self.assertLessEqual(y + h, 1.0)


class BuildInitialDrawingTests(unittest.TestCase):
    def test_one_rect_per_instance_in_order(self):
        instances = [_inst(0, (0.1, 0.1, 0.1, 0.1)), _inst(1, (0.5, 0.5, 0.1, 0.1))]
        drawing = ch.build_initial_drawing(instances, 400, 400)
        self.assertEqual(len(drawing["objects"]), 2)
        self.assertAlmostEqual(drawing["objects"][0]["left"], 40)
        self.assertAlmostEqual(drawing["objects"][1]["left"], 200)

    def test_proposal_boxes_are_dashed_human_boxes_are_not(self):
        instances = [_inst(0, source="model_proposed"), _inst(1, bbox=(0.5, 0.5, 0.1, 0.1), source="human_drawn")]
        drawing = ch.build_initial_drawing(instances, 400, 400)
        self.assertIsNotNone(drawing["objects"][0]["strokeDashArray"])
        self.assertIsNone(drawing["objects"][1]["strokeDashArray"])


class MatchCanvasBoxesToInstancesTests(unittest.TestCase):
    def test_untouched_proposal_matches_itself_not_moved(self):
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        result = ch.match_canvas_boxes_to_instances([(0.1, 0.1, 0.2, 0.2)], [seed])
        self.assertEqual(len(result.matched), 1)
        self.assertFalse(result.matched[0][2])
        self.assertEqual(result.deleted, ())
        self.assertEqual(result.new_boxes, ())

    def test_moved_proposal_flagged_as_moved(self):
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        result = ch.match_canvas_boxes_to_instances([(0.15, 0.15, 0.2, 0.2)], [seed])
        self.assertEqual(len(result.matched), 1)
        self.assertTrue(result.matched[0][2])

    def test_deleted_proposal_has_no_canvas_box(self):
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        result = ch.match_canvas_boxes_to_instances([], [seed])
        self.assertEqual(result.deleted, (seed,))
        self.assertEqual(result.matched, ())

    def test_brand_new_box_with_no_seed_is_a_new_box(self):
        result = ch.match_canvas_boxes_to_instances([(0.6, 0.6, 0.1, 0.1)], [])
        self.assertEqual(result.new_boxes, ((0.6, 0.6, 0.1, 0.1),))

    def test_multi_instance_matching_is_stable(self):
        seeds = [_inst(0, (0.1, 0.1, 0.1, 0.1), source="human_drawn"),
                 _inst(1, (0.6, 0.6, 0.1, 0.1), source="human_drawn")]
        # second seed moved slightly, first untouched, plus one brand-new box
        canvas_boxes = [(0.1, 0.1, 0.1, 0.1), (0.62, 0.62, 0.1, 0.1), (0.3, 0.3, 0.05, 0.05)]
        result = ch.match_canvas_boxes_to_instances(canvas_boxes, seeds)
        self.assertEqual(len(result.matched), 2)
        self.assertEqual(len(result.new_boxes), 1)
        moved_flags = {id(m[0]): m[2] for m in result.matched}
        self.assertFalse(moved_flags[id(seeds[0])])
        self.assertTrue(moved_flags[id(seeds[1])])


class ReconcileCanvasSessionTests(unittest.TestCase):
    def test_untouched_proposal_stays_model_proposed_in_working_list(self):
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        working = ch.reconcile_canvas_session([(0.1, 0.1, 0.2, 0.2)], [seed])
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].bbox_source, "model_proposed")

    def test_moved_proposal_becomes_edited_in_working_list(self):
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        # A realistic nudge/resize (still high IoU with the original) is
        # matched to the same seed and becomes an edit of it.
        working = ch.reconcile_canvas_session([(0.12, 0.12, 0.2, 0.2)], [seed])
        self.assertEqual(working[0].bbox_source, "human_edited")
        self.assertEqual(working[0].bbox, (0.12, 0.12, 0.2, 0.2))

    def test_proposal_dragged_far_away_is_treated_as_delete_plus_new_box(self):
        """A box moved far enough that it no longer overlaps its origin
        (IoU below match threshold) is indistinguishable from 'the
        original proposal was rejected and a fresh box was drawn' --
        documented, acceptable behavior, not a bug: both outcomes are
        explicitly human-reviewed states, never a silently-kept proposal."""
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        working = ch.reconcile_canvas_session([(0.7, 0.7, 0.2, 0.2)], [seed])
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].bbox_source, "human_drawn")

    def test_new_box_appears_as_human_drawn(self):
        working = ch.reconcile_canvas_session([(0.5, 0.5, 0.1, 0.1)], [])
        self.assertEqual(len(working), 1)
        self.assertEqual(working[0].bbox_source, "human_drawn")

    def test_deleted_proposal_absent_from_working_list(self):
        seed = _inst(0, (0.1, 0.1, 0.2, 0.2), source="model_proposed")
        working = ch.reconcile_canvas_session([], [seed])
        self.assertEqual(working, ())

    def test_working_list_is_renumbered_contiguously(self):
        seeds = [_inst(0, (0.1, 0.1, 0.1, 0.1), source="human_drawn"),
                 _inst(1, (0.6, 0.6, 0.1, 0.1), source="human_drawn")]
        working = ch.reconcile_canvas_session(
            [(0.1, 0.1, 0.1, 0.1), (0.6, 0.6, 0.1, 0.1), (0.3, 0.3, 0.05, 0.05)], seeds)
        self.assertEqual([i.instance_index for i in working], [0, 1, 2])


class SavableInstancesTests(unittest.TestCase):
    def test_drops_untouched_proposals(self):
        working = (_inst(0, source="model_proposed"),
                    _inst(1, bbox=(0.5, 0.5, 0.1, 0.1), source="human_accepted"))
        savable = ch.savable_instances(working)
        self.assertEqual(len(savable), 1)
        self.assertEqual(savable[0].bbox_source, "human_accepted")
        self.assertEqual(savable[0].instance_index, 0)  # renumbered after drop

    def test_keeps_edited_and_drawn(self):
        working = (_inst(0, source="human_edited", model=""),
                    _inst(1, bbox=(0.5, 0.5, 0.1, 0.1), source="human_drawn"))
        savable = ch.savable_instances(working)
        self.assertEqual(len(savable), 2)

    def test_all_proposals_gives_empty_savable_list(self):
        working = (_inst(0, source="model_proposed"),)
        self.assertEqual(ch.savable_instances(working), ())


class ResolveBboxSourceTests(unittest.TestCase):
    def test_untouched_proposal_stays_proposal(self):
        seed = _inst(0, source="model_proposed")
        self.assertEqual(ch.resolve_bbox_source(seed, moved=False), "model_proposed")

    def test_moved_proposal_becomes_edited(self):
        seed = _inst(0, source="model_proposed")
        self.assertEqual(ch.resolve_bbox_source(seed, moved=True), "human_edited")

    def test_moved_human_drawn_stays_edited_not_reverted(self):
        seed = _inst(0, bbox=(0.1, 0.1, 0.1, 0.1), source="human_drawn")
        self.assertEqual(ch.resolve_bbox_source(seed, moved=True), "human_edited")

    def test_untouched_human_accepted_stays_accepted(self):
        seed = _inst(0, source="human_accepted")
        self.assertEqual(ch.resolve_bbox_source(seed, moved=False), "human_accepted")


if __name__ == "__main__":
    unittest.main()
