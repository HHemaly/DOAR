"""Agreement/quality math tests -- synthetic data only. Verifies Cohen's
kappa and percent agreement against hand-computed values, and that no
statistic is fabricated when only one annotator exists."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import quality as quality_mod
from doar.phase2c1 import store as store_mod
from doar.phase2c1.schema import AnnotationRecord


def _rec(**overrides):
    defaults = dict(
        pilot_id="p2b_0000", image_id="abc", source_image_group="grp_1",
        class_name="person", status="present", annotator_id="ann1",
        annotation_timestamp="t", instance_count=1,
    )
    defaults.update(overrides)
    return AnnotationRecord(**defaults)


class SingleAnnotatorHonestyTests(unittest.TestCase):
    def test_zero_annotators_reports_insufficient(self):
        report = quality_mod.compute_agreement_report({})
        self.assertFalse(report["sufficient_annotators"])
        self.assertNotIn("cohens_kappa_overall", report)

    def test_one_annotator_reports_insufficient_not_a_fabricated_kappa(self):
        st = {}
        store_mod.upsert(st, _rec(annotator_id="only_one"))
        report = quality_mod.compute_agreement_report(st)
        self.assertFalse(report["sufficient_annotators"])
        self.assertEqual(report["distinct_annotators"], ["only_one"])
        self.assertNotIn("cohens_kappa_overall", report)


class ProvisionalVsHumanDistinctionTests(unittest.TestCase):
    """The hard requirement: a migrated Phase 2B legacy_provisional_human
    row must never be countable as a second genuine human annotator for
    human-human Cohen's kappa / percent agreement."""

    def test_one_human_plus_one_provisional_is_still_insufficient(self):
        st = {}
        store_mod.upsert(st, _rec(annotator_id="real_human", annotator_type="human"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0,
                                   annotator_id="phase2b_legacy", annotator_type="legacy_provisional_human"))
        report = quality_mod.compute_agreement_report(st)
        self.assertFalse(report["sufficient_annotators"])
        self.assertEqual(report["human_annotators"], ["real_human"])
        self.assertNotIn("cohens_kappa_overall", report)

    def test_two_humans_plus_provisional_excludes_provisional_from_pairing(self):
        st = {}
        for i in range(quality_mod.MIN_PAIRS_FOR_KAPPA):
            pid = f"p2b_{i:04d}"
            store_mod.upsert(st, _rec(pilot_id=pid, class_name="person", annotator_id="humanA",
                                       annotator_type="human", status="present", instance_count=1))
            store_mod.upsert(st, _rec(pilot_id=pid, class_name="person", annotator_id="humanB",
                                       annotator_type="human", status="present", instance_count=1))
            store_mod.upsert(st, _rec(pilot_id=pid, class_name="person", annotator_id="phase2b_legacy",
                                       annotator_type="legacy_provisional_human", status="absent",
                                       instance_count=0))
        report = quality_mod.compute_agreement_report(st)
        self.assertTrue(report["sufficient_annotators"])
        self.assertEqual(sorted(report["human_annotators"]), ["humanA", "humanB"])
        self.assertEqual(set(report["compared_pair"]), {"humanA", "humanB"})
        self.assertNotIn("phase2b_legacy", report["compared_pair"])
        # humanA/humanB agree on every judgment in this fixture -> kappa should be 1.0,
        # NOT diluted or affected by the disagreeing legacy row.
        self.assertEqual(report["percent_agreement_overall"], 1.0)

    def test_distinct_annotators_still_lists_everyone(self):
        st = {}
        store_mod.upsert(st, _rec(annotator_id="real_human", annotator_type="human"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0,
                                   annotator_id="phase2b_legacy", annotator_type="legacy_provisional_human"))
        report = quality_mod.compute_agreement_report(st)
        self.assertEqual(sorted(report["distinct_annotators"]), ["phase2b_legacy", "real_human"])


class ProvisionalReferenceComparisonTests(unittest.TestCase):
    def test_unavailable_with_no_provisional_annotator(self):
        st = {}
        store_mod.upsert(st, _rec(annotator_id="real_human", annotator_type="human"))
        result = quality_mod.compute_provisional_reference_comparison(st)
        self.assertFalse(result["available"])

    def test_unavailable_with_no_human_annotator(self):
        st = {}
        store_mod.upsert(st, _rec(annotator_id="phase2b_legacy", annotator_type="legacy_provisional_human"))
        result = quality_mod.compute_provisional_reference_comparison(st)
        self.assertFalse(result["available"])

    def test_available_and_correct_with_both_present(self):
        st = {}
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", class_name="person", annotator_id="real_human",
                                   annotator_type="human", status="present", instance_count=1))
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", class_name="person", annotator_id="phase2b_legacy",
                                   annotator_type="legacy_provisional_human", status="present", instance_count=1))
        store_mod.upsert(st, _rec(pilot_id="p2b_0001", class_name="person", annotator_id="real_human",
                                   annotator_type="human", status="absent", instance_count=0))
        store_mod.upsert(st, _rec(pilot_id="p2b_0001", class_name="person", annotator_id="phase2b_legacy",
                                   annotator_type="legacy_provisional_human", status="present", instance_count=1))
        result = quality_mod.compute_provisional_reference_comparison(st)
        self.assertTrue(result["available"])
        pair = result["per_pair"]["real_human_vs_phase2b_legacy"]
        self.assertEqual(pair["n_matched"], 2)
        self.assertEqual(pair["percent_agreement"], 0.5)

    def test_never_reports_a_kappa_value(self):
        """Kappa is deliberately never computed here -- the provisional
        side is not validated ground truth, so a formal inter-rater
        reliability statistic would be misleading."""
        st = {}
        store_mod.upsert(st, _rec(annotator_id="real_human", annotator_type="human"))
        store_mod.upsert(st, _rec(annotator_id="phase2b_legacy", annotator_type="legacy_provisional_human"))
        result = quality_mod.compute_provisional_reference_comparison(st)
        for pair in result.get("per_pair", {}).values():
            self.assertNotIn("cohens_kappa", pair)


class PercentAgreementTests(unittest.TestCase):
    def test_perfect_agreement(self):
        pairs = [("present", "present"), ("absent", "absent")]
        self.assertEqual(quality_mod.percent_agreement(pairs), 1.0)

    def test_zero_agreement(self):
        pairs = [("present", "absent"), ("absent", "present")]
        self.assertEqual(quality_mod.percent_agreement(pairs), 0.0)

    def test_empty_returns_none(self):
        self.assertIsNone(quality_mod.percent_agreement([]))


class CohensKappaTests(unittest.TestCase):
    def test_hand_computed_value(self):
        # po=0.8 (4/5), pe=0.48 -> kappa = 0.32/0.52 = 0.615384...
        pairs = [("present", "present"), ("absent", "absent"), ("present", "absent"),
                  ("absent", "absent"), ("present", "present")]
        kappa = quality_mod.cohens_kappa(pairs)
        self.assertAlmostEqual(kappa, 0.6153846153846155, places=9)

    def test_perfect_agreement_gives_kappa_one(self):
        pairs = [("present", "present"), ("absent", "absent"), ("present", "present")]
        self.assertAlmostEqual(quality_mod.cohens_kappa(pairs), 1.0)

    def test_empty_returns_none(self):
        self.assertIsNone(quality_mod.cohens_kappa([]))

    def test_single_category_returns_none_not_divide_by_zero(self):
        pairs = [("present", "present"), ("present", "present")]
        self.assertIsNone(quality_mod.cohens_kappa(pairs))


class KappaReportingThresholdTests(unittest.TestCase):
    def test_kappa_not_reported_below_min_pairs(self):
        st = {}
        for i in range(quality_mod.MIN_PAIRS_FOR_KAPPA - 1):
            for ann in ("annA", "annB"):
                store_mod.upsert(st, _rec(pilot_id=f"p2b_{i:04d}", class_name="person",
                                           annotator_id=ann, status="present", instance_count=1))
        report = quality_mod.compute_agreement_report(st)
        self.assertFalse(report["kappa_reported_overall"])
        self.assertIsNone(report["cohens_kappa_overall"])

    def test_kappa_reported_at_min_pairs(self):
        st = {}
        for i in range(quality_mod.MIN_PAIRS_FOR_KAPPA):
            for ann in ("annA", "annB"):
                store_mod.upsert(st, _rec(pilot_id=f"p2b_{i:04d}", class_name="person",
                                           annotator_id=ann, status="present", instance_count=1))
        report = quality_mod.compute_agreement_report(st)
        self.assertTrue(report["kappa_reported_overall"])


class CountDisagreementTests(unittest.TestCase):
    def test_no_pairs(self):
        result = quality_mod.count_disagreement([])
        self.assertEqual(result["n_compared"], 0)
        self.assertIsNone(result["disagreement_rate"])

    def test_all_agree(self):
        result = quality_mod.count_disagreement([(2, 2), (3, 3)])
        self.assertEqual(result["n_disagree"], 0)
        self.assertEqual(result["disagreement_rate"], 0.0)

    def test_some_disagree(self):
        result = quality_mod.count_disagreement([(2, 3), (1, 1)])
        self.assertEqual(result["n_disagree"], 1)
        self.assertEqual(result["disagreement_rate"], 0.5)
        self.assertEqual(result["mean_abs_diff"], 0.5)


class BboxIoUTests(unittest.TestCase):
    def test_identical_boxes_iou_one(self):
        box = (0.1, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(quality_mod.bbox_iou(box, box), 1.0)

    def test_no_overlap_iou_zero(self):
        a = (0.0, 0.0, 0.1, 0.1)
        b = (0.5, 0.5, 0.1, 0.1)
        self.assertEqual(quality_mod.bbox_iou(a, b), 0.0)

    def test_known_partial_overlap(self):
        # Two unit-ish boxes overlapping in a 0.5x0.5 region.
        a = (0.0, 0.0, 1.0, 1.0)
        b = (0.5, 0.5, 1.0, 1.0)
        # intersection = 0.5*0.5=0.25, union = 1+1-0.25=1.75
        self.assertAlmostEqual(quality_mod.bbox_iou(a, b), 0.25 / 1.75)


class UnresolvedDisagreementsTests(unittest.TestCase):
    def test_finds_unresolved_only(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person", adjudication_status="disagreement_unresolved",
                                   review_status="reviewed"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0,
                                   adjudication_status="agreement", review_status="reviewed"))
        unresolved = quality_mod.unresolved_disagreements(st)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].class_name, "person")


class ReviewCoverageTests(unittest.TestCase):
    def test_empty_store(self):
        result = quality_mod.review_coverage({})
        self.assertEqual(result["total_rows"], 0)
        self.assertIsNone(result["review_coverage_rate"])

    def test_mixed_review_statuses(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person", review_status="reviewed",
                                   adjudication_status="agreement"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0,
                                   review_status="unreviewed"))
        result = quality_mod.review_coverage(st)
        self.assertEqual(result["total_rows"], 2)
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["review_coverage_rate"], 0.5)


class ClassSupportTests(unittest.TestCase):
    def test_counts_by_status(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person", status="present", instance_count=1))
        store_mod.upsert(st, _rec(pilot_id="p2b_0001", class_name="person", status="absent",
                                   instance_count=0, annotator_id="ann2"))
        support = quality_mod.class_support(st)
        self.assertEqual(support["person"]["present"], 1)
        self.assertEqual(support["person"]["absent"], 1)


if __name__ == "__main__":
    unittest.main()
