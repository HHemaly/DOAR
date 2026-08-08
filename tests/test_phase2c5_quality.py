"""Phase 2C.5 Stage 8 quality/readiness metrics tests -- synthetic data."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import quality as quality_mod
from doar.phase2c5.schema import PartAnnotationRecord, PartInstance


def _rec(pilot_id, target_name, status, annotator_id="ann1", instances=None):
    if instances is None:
        instances = (PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1),
                                   bbox_source="human_drawn"),) if status == "present" else ()
    return PartAnnotationRecord(pilot_id=pilot_id, target_name=target_name, status=status,
                                 annotator_id=annotator_id, annotation_timestamp="t", instances=instances)


class SingleAnnotatorCaveatTests(unittest.TestCase):
    def test_caveat_text_present_and_mentions_not_inter_rater(self):
        self.assertIn("not inter-rater reliability", quality_mod.SINGLE_ANNOTATOR_CAVEAT.lower())

    def test_no_kappa_import_or_assignment_in_module(self):
        """The module's docstring/caveat legitimately mentions Cohen's
        kappa once, to explain why it is NOT computed -- check for an
        actual import or a `kappa = ...` assignment, not the word itself."""
        text = Path(ROOT / "src/doar/phase2c5/quality.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("import sklearn", text)
        self.assertNotIn("cohen_kappa", text)
        self.assertNotIn("kappa =", text)
        self.assertNotIn("kappa=", text)


class CompletionRateTests(unittest.TestCase):
    def test_completion_rate_counts_images_with_all_targets(self):
        store = {}
        for target in ("eye", "mouth"):
            store[f"p2b_0000__{target}__ann1"] = _rec("p2b_0000", target, "absent")
        store["p2b_0001__eye__ann1"] = _rec("p2b_0001", "eye", "absent")
        result = quality_mod.completion_rate(store, ["p2b_0000", "p2b_0001"], "ann1", ("eye", "mouth"))
        self.assertEqual(result["complete_images"], 1)
        self.assertEqual(result["incomplete_images"], 1)
        self.assertAlmostEqual(result["completion_rate"], 0.5)


class SupportCountsTests(unittest.TestCase):
    def test_support_counts_by_status(self):
        store = {
            "a": _rec("p1", "eye", "present"), "b": _rec("p2", "eye", "absent"),
            "c": _rec("p3", "eye", "uncertain", instances=()),
            "d": _rec("p4", "eye", "not_assessable", instances=()),
        }
        counts = quality_mod.support_counts(store, "eye")
        self.assertEqual(counts["present"], 1)
        self.assertEqual(counts["absent"], 1)
        self.assertEqual(counts["uncertain"], 1)
        self.assertEqual(counts["not_assessable"], 1)
        self.assertEqual(counts["n_reviewed"], 4)


class BboxCoverageTests(unittest.TestCase):
    def test_all_present_rows_have_bbox_by_construction(self):
        """Schema invariant guarantees this is always 1.0 -- still computed
        and reported explicitly per Stage 8's requirement, not assumed."""
        store = {"a": _rec("p1", "hand", "present")}
        result = quality_mod.bbox_coverage_among_present(store, "hand")
        self.assertEqual(result["bbox_coverage"], 1.0)

    def test_no_present_rows_gives_none_not_zero(self):
        store = {"a": _rec("p1", "hand", "absent")}
        result = quality_mod.bbox_coverage_among_present(store, "hand")
        self.assertIsNone(result["bbox_coverage"])


class InstanceCountStatsTests(unittest.TestCase):
    def test_multi_instance_counted_correctly(self):
        two_hands = (PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="human_drawn"),
                     PartInstance(instance_index=1, bbox=(0.5, 0.5, 0.1, 0.1), bbox_source="human_drawn"))
        store = {"a": _rec("p1", "hand", "present", instances=two_hands)}
        stats = quality_mod.instance_count_stats(store, "hand")
        self.assertEqual(stats["total_instances"], 2)
        self.assertEqual(stats["max_instances"], 2)


class ProposalReviewOutcomesTests(unittest.TestCase):
    def test_acceptance_and_rejection_rates(self):
        accepted = PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="human_accepted",
                                 proposal_model="grounding_dino:tiny", proposal_threshold=0.25)
        store = {"a": _rec("p1", "eye", "present", instances=(accepted,))}
        result = quality_mod.proposal_review_outcomes(store, "eye", n_proposal_boxes_offered=4)
        self.assertEqual(result["n_accepted_as_is"], 1)
        self.assertEqual(result["n_rejected_or_unreviewed"], 3)
        self.assertAlmostEqual(result["acceptance_rate"], 0.25)


class ReadinessSummaryTests(unittest.TestCase):
    def test_readiness_summary_includes_caveat_and_all_sections(self):
        store = {}
        summary = quality_mod.readiness_summary(store, ["p1"], "ann1", ("eye",))
        self.assertIn("single_annotator_caveat", summary)
        self.assertIn("completion", summary)
        self.assertIn("support_by_target", summary)
        self.assertIn("bbox_coverage_by_target", summary)


if __name__ == "__main__":
    unittest.main()
