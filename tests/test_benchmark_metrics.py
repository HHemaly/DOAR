"""Tests for src/doar/benchmark_metrics.py.

All annotation data here is SYNTHETIC FIXTURE DATA invented for this test
file only -- it validates the metric arithmetic and matching logic, and is
never used as a real annotation, never fed into the real 15-image
development benchmark, and never written anywhere under annotations/.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import benchmark_metrics as bm  # noqa: E402


def fixture_items(*labels: str, salient: bool = False) -> list[bm.HumanAnnotationItem]:
    return [bm.HumanAnnotationItem(label=label, location="fixture", salient=salient) for label in labels]


class VisualPrecisionRecallF1Tests(unittest.TestCase):
    def test_perfect_match(self):
        result = bm.visual_precision_recall_f1(["sun", "tree"], fixture_items("sun", "tree"))
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_partial_overlap(self):
        result = bm.visual_precision_recall_f1(["sun", "car"], fixture_items("sun", "tree"))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_empty_candidates_returns_none_precision(self):
        result = bm.visual_precision_recall_f1([], fixture_items("sun"))
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0.0)

    def test_empty_human_items_returns_none_recall(self):
        result = bm.visual_precision_recall_f1(["sun"], [])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["precision"], 0.0)

    def test_never_merges_two_annotators_implicitly(self):
        # Caller must pass ONE annotator's items -- verify the function
        # signature only accepts a single list, not two.
        import inspect
        sig = inspect.signature(bm.visual_precision_recall_f1)
        self.assertEqual(len(sig.parameters), 2)


class SalientRecallTests(unittest.TestCase):
    def test_all_salient_items_covered(self):
        result = bm.salient_recall(["sun", "person"], fixture_items("sun", "person"))
        self.assertEqual(result["salient_recall"], 1.0)

    def test_missed_salient_item(self):
        result = bm.salient_recall(["sun"], fixture_items("sun", "person"))
        self.assertEqual(result["salient_recall"], 0.5)

    def test_no_salient_items_is_none(self):
        result = bm.salient_recall(["sun"], [])
        self.assertIsNone(result["salient_recall"])


class HallucinationRateTests(unittest.TestCase):
    def test_no_hallucination(self):
        result = bm.hallucination_rate(["sun", "tree"], fixture_items("sun", "tree", "house"))
        self.assertEqual(result["hallucination_rate"], 0.0)

    def test_full_hallucination(self):
        result = bm.hallucination_rate(["dinosaur", "spaceship"], fixture_items("sun", "tree"))
        self.assertEqual(result["hallucination_rate"], 1.0)

    def test_empty_candidates_is_none(self):
        result = bm.hallucination_rate([], fixture_items("sun"))
        self.assertIsNone(result["hallucination_rate"])


class SalientItemsTests(unittest.TestCase):
    def test_filters_to_salient_true_only(self):
        items = fixture_items("sun", salient=True) + fixture_items("small mark", salient=False)
        salient = bm.salient_items(items)
        self.assertEqual([i.label for i in salient], ["sun"])

    def test_empty_when_none_salient(self):
        items = fixture_items("a", "b", salient=False)
        self.assertEqual(bm.salient_items(items), [])


class NormalizedSalienceTests(unittest.TestCase):
    """SALIENT = normalized union of salient items from either annotator;
    CORE_SALIENT = normalized intersection. Frozen definitions,
    BENCHMARK_SCHEMA.md Section 5. Callers pass each annotator's FULL
    exhaustive item list -- this function filters to salient=True
    itself."""

    def test_union_is_correct_with_partial_overlap(self):
        a = fixture_items("sun", "tree", "house", salient=True)
        b = fixture_items("sun", "tree", "car", salient=True)
        result = bm.normalized_salience(a, b)
        salient_labels = sorted(i.label for i in result["salient"])
        self.assertEqual(salient_labels, sorted(["sun", "tree", "house", "car"]))

    def test_intersection_is_correct_with_partial_overlap(self):
        a = fixture_items("sun", "tree", "house", salient=True)
        b = fixture_items("sun", "tree", "car", salient=True)
        result = bm.normalized_salience(a, b)
        core_labels = sorted(i.label for i in result["core_salient"])
        self.assertEqual(core_labels, sorted(["sun", "tree"]))

    def test_union_of_disjoint_lists_is_everything(self):
        a = fixture_items("sun", "tree", salient=True)
        b = fixture_items("dinosaur", "spaceship", salient=True)
        result = bm.normalized_salience(a, b)
        self.assertEqual(len(result["salient"]), 4)
        self.assertEqual(result["core_salient"], [])

    def test_intersection_of_identical_lists_is_everything(self):
        a = fixture_items("sun", "tree", "person", salient=True)
        b = fixture_items("sun", "tree", "person", salient=True)
        result = bm.normalized_salience(a, b)
        self.assertEqual(len(result["salient"]), 3)
        self.assertEqual(len(result["core_salient"]), 3)

    def test_duplicate_within_one_annotator_matched_at_most_once(self):
        a = fixture_items("sun", salient=True)
        b = fixture_items("sun", "sun", salient=True)
        result = bm.normalized_salience(a, b)
        # one core match consumes one "sun" from b; the leftover "sun" in
        # b is unmatched and still appears once in the union.
        self.assertEqual(len(result["core_salient"]), 1)
        self.assertEqual(len(result["salient"]), 2)

    def test_non_salient_items_are_excluded_from_both_sets(self):
        # An annotator's exhaustive list mixes salient and non-salient
        # items -- only salient=True items may ever appear in SALIENT/
        # CORE_SALIENT, regardless of label overlap.
        a = fixture_items("sun", salient=True) + fixture_items("small mark", salient=False)
        b = fixture_items("sun", salient=True) + fixture_items("small mark", salient=False)
        result = bm.normalized_salience(a, b)
        self.assertEqual(sorted(i.label for i in result["salient"]), ["sun"])
        self.assertEqual(sorted(i.label for i in result["core_salient"]), ["sun"])

    def test_original_annotator_lists_are_never_mutated(self):
        a = fixture_items("sun", "tree", salient=True)
        b = fixture_items("sun", "car", salient=True)
        a_before, b_before = list(a), list(b)
        bm.normalized_salience(a, b)
        self.assertEqual(a, a_before)
        self.assertEqual(b, b_before)

    def test_never_writes_back_to_disk(self):
        # These are frozen dataclasses built in memory from whatever the
        # caller loaded off disk -- this function has no filesystem
        # access at all, so "original A1/A2 records remain untouched" is
        # true by construction. Guard: the function accepts plain lists,
        # not file paths, so there is nothing it *could* write to.
        import inspect
        sig = inspect.signature(bm.normalized_salience)
        for param in sig.parameters.values():
            self.assertNotIn("path", param.name.lower())
            self.assertNotIn("file", param.name.lower())


class PrimaryAndSensitivitySalientRecallTests(unittest.TestCase):
    def test_primary_uses_union_sensitivity_uses_intersection(self):
        a = fixture_items("sun", "tree", "house", salient=True)
        b = fixture_items("sun", "tree", "car", salient=True)
        # Candidates cover sun/tree (the intersection) plus house (only A saw it).
        result = bm.primary_and_sensitivity_salient_recall(["sun", "tree", "house"], a, b)
        # primary: matched 3 of 4 union items (sun, tree, house) -> car missing
        self.assertAlmostEqual(result["primary_salient_recall"]["salient_recall"], 3 / 4)
        # sensitivity: matched both of the 2 intersection items (sun, tree) -> 1.0
        self.assertEqual(result["sensitivity_salient_recall"]["salient_recall"], 1.0)
        self.assertEqual(result["salient_count"], 4)
        self.assertEqual(result["core_salient_count"], 2)

    def test_takes_full_exhaustive_lists_and_filters_internally(self):
        # Non-salient items in the input must not affect the result.
        a = fixture_items("sun", salient=True) + fixture_items("small mark", salient=False)
        b = fixture_items("sun", salient=True) + fixture_items("other detail", salient=False)
        result = bm.primary_and_sensitivity_salient_recall(["sun"], a, b)
        self.assertEqual(result["salient_count"], 1)
        self.assertEqual(result["core_salient_count"], 1)

    def test_per_annotator_salient_recall_remains_independently_available(self):
        a = fixture_items("sun", "tree", salient=True)
        b = fixture_items("sun", "car", salient=True)
        # per-annotator numbers (existing function, unmerged) still work standalone.
        recall_a = bm.salient_recall(["sun"], bm.salient_items(a))
        recall_b = bm.salient_recall(["sun"], bm.salient_items(b))
        self.assertEqual(recall_a["salient_recall"], 0.5)
        self.assertEqual(recall_b["salient_recall"], 0.5)


class AbstentionRateTests(unittest.TestCase):
    def test_counts_unknown_entity_type(self):
        candidates = [{"entity_type": "sun"}, {"entity_type": "unknown"}, {"entity_type": "scribble"}]
        result = bm.abstention_rate(candidates)
        self.assertAlmostEqual(result["abstention_rate"], 2 / 3)

    def test_counts_uncertain_verification_status(self):
        candidates = [{"entity_type": "sun", "case_verification_status": "verified"},
                      {"entity_type": "person", "case_verification_status": "uncertain"}]
        result = bm.abstention_rate(candidates)
        self.assertEqual(result["abstention_rate"], 0.5)

    def test_empty_is_none(self):
        self.assertIsNone(bm.abstention_rate([])["abstention_rate"])


class VerifierCorrectionRateTests(unittest.TestCase):
    def test_confident_candidate_rejected_counts_as_corrected(self):
        candidates = [
            {"confidence": 0.9, "bbox": (1, 1, 2, 2), "case_verification_status": "rejected"},
            {"confidence": 0.9, "bbox": (1, 1, 2, 2), "case_verification_status": "verified"},
        ]
        result = bm.verifier_correction_rate(candidates)
        self.assertEqual(result["corrected"], 1)
        self.assertEqual(result["total"], 2)

    def test_no_bbox_candidates_excluded_from_denominator(self):
        candidates = [{"confidence": 0.9, "bbox": None, "case_verification_status": "rejected"}]
        result = bm.verifier_correction_rate(candidates)
        self.assertEqual(result["total"], 0)
        self.assertIsNone(result["verifier_correction_rate"])

    def test_low_confidence_rejection_not_counted_as_correction(self):
        candidates = [{"confidence": 0.1, "bbox": (1, 1, 2, 2), "case_verification_status": "rejected"}]
        result = bm.verifier_correction_rate(candidates)
        self.assertEqual(result["corrected"], 0)


class _FakeMatch:
    def __init__(self, matched_entity_ids):
        self.matched_entity_ids = matched_entity_ids


class EvidenceBackedClaimRateTests(unittest.TestCase):
    def test_all_backed(self):
        matches = [_FakeMatch(("e1",)), _FakeMatch(("e2", "e3"))]
        result = bm.evidence_backed_claim_rate(matches)
        self.assertEqual(result["evidence_backed_claim_rate"], 1.0)

    def test_unbacked_detected(self):
        matches = [_FakeMatch(("e1",)), _FakeMatch(())]
        result = bm.evidence_backed_claim_rate(matches)
        self.assertEqual(result["evidence_backed_claim_rate"], 0.5)

    def test_empty_is_none(self):
        self.assertIsNone(bm.evidence_backed_claim_rate([])["evidence_backed_claim_rate"])


class _FakeHypothesis:
    def __init__(self, supporting_rule_ids, support_level="WEAK_HYPOTHESIS"):
        self.supporting_rule_ids = supporting_rule_ids
        self.support_level = support_level


class UnsupportedClaimAndTraceabilityTests(unittest.TestCase):
    def test_all_supported(self):
        rule_matrix = {"RULE_A": {}, "RULE_B": {}}
        hyps = [_FakeHypothesis(("RULE_A", "RULE_B"))]
        self.assertEqual(bm.unsupported_claim_rate(hyps, rule_matrix)["unsupported_claim_rate"], 0.0)
        self.assertEqual(bm.rule_reference_traceability(hyps, rule_matrix)["traceability_rate"], 1.0)

    def test_detects_unknown_rule_id(self):
        rule_matrix = {"RULE_A": {}}
        hyps = [_FakeHypothesis(("RULE_A", "RULE_GHOST"))]
        self.assertAlmostEqual(bm.unsupported_claim_rate(hyps, rule_matrix)["unsupported_claim_rate"], 0.5)
        self.assertAlmostEqual(bm.rule_reference_traceability(hyps, rule_matrix)["traceability_rate"], 0.5)


class HypothesisDerivabilityTests(unittest.TestCase):
    def test_counts_images_and_levels(self):
        by_image = {
            "img1": [_FakeHypothesis(("R1",), "WEAK_HYPOTHESIS")],
            "img2": [],
            "img3": [_FakeHypothesis(("R2",), "WEAK_HYPOTHESIS"), _FakeHypothesis(("R3",), "MODERATE_HYPOTHESIS")],
        }
        result = bm.hypothesis_derivability(by_image)
        self.assertEqual(result["images_with_at_least_one_hypothesis"], 2)
        self.assertEqual(result["total_images"], 3)
        self.assertEqual(result["support_level_counts"], {"WEAK_HYPOTHESIS": 2, "MODERATE_HYPOTHESIS": 1})


class InterAnnotatorAgreementTests(unittest.TestCase):
    def test_identical_lists_perfect_agreement(self):
        a = fixture_items("sun", "tree", "person")
        b = fixture_items("sun", "tree", "person")
        result = bm.inter_annotator_agreement(a, b)
        self.assertEqual(result["matched_pairs"], 3)
        self.assertEqual(result["jaccard"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["adjudication_rate"], 0.0)

    def test_disjoint_lists_zero_agreement(self):
        a = fixture_items("sun", "tree")
        b = fixture_items("dinosaur", "spaceship")
        result = bm.inter_annotator_agreement(a, b)
        self.assertEqual(result["matched_pairs"], 0)
        self.assertEqual(result["jaccard"], 0.0)
        self.assertEqual(result["adjudication_rate"], 1.0)

    def test_partial_overlap(self):
        a = fixture_items("sun", "tree", "house")
        b = fixture_items("sun", "tree", "car")
        result = bm.inter_annotator_agreement(a, b)
        self.assertEqual(result["matched_pairs"], 2)
        # jaccard = 2 / (6 - 2) = 0.5 ; f1 = 2*2/6 = 0.6667
        self.assertAlmostEqual(result["jaccard"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["adjudication_rate"], 2 / 6)

    def test_greedy_matching_does_not_double_count_duplicate_matches(self):
        # Two "sun" items in B should not both match the single "sun" in A.
        a = fixture_items("sun")
        b = fixture_items("sun", "sun")
        result = bm.inter_annotator_agreement(a, b)
        self.assertEqual(result["matched_pairs"], 1)
        self.assertEqual(result["unmatched_items"], 1)

    def test_empty_both_lists(self):
        result = bm.inter_annotator_agreement([], [])
        self.assertEqual(result["matched_pairs"], 0)
        self.assertIsNone(result["jaccard"])
        self.assertIsNone(result["adjudication_rate"])

    def test_never_produces_a_merged_ground_truth_list(self):
        # The function's return dict must never contain a merged/combined
        # item list -- only counts and rates. This guards against a future
        # accidental change that starts treating agreement as truth.
        result = bm.inter_annotator_agreement(fixture_items("sun"), fixture_items("sun"))
        for key in result:
            self.assertNotIn("merged", key.lower())
            self.assertNotIn("combined", key.lower())
            self.assertNotIn("ground_truth", key.lower())


class UnmatchedCandidateRateTests(unittest.TestCase):
    def test_all_matched_is_zero(self):
        result = bm.unmatched_candidate_rate(["sun", "tree"], fixture_items("sun", "tree", "house"))
        self.assertEqual(result["unmatched_rate"], 0.0)

    def test_none_matched_is_one(self):
        result = bm.unmatched_candidate_rate(["dinosaur"], fixture_items("sun"))
        self.assertEqual(result["unmatched_rate"], 1.0)

    def test_empty_candidates_is_none(self):
        self.assertIsNone(bm.unmatched_candidate_rate([], fixture_items("sun"))["unmatched_rate"])


class RawVsVerifiedComparisonTests(unittest.TestCase):
    """Bug 5: a verifier 'correction' can be GOOD (drops a false
    detection) or BAD (drops a true one) -- these tests prove both
    directions are visible as separate, correctly-signed deltas rather
    than collapsing into one ambiguous number."""

    def test_verifier_removes_a_false_detection_precision_improves_recall_unchanged(self):
        # RAW has an extra hallucinated "dinosaur" the human never saw;
        # VERIFIED correctly drops it. Both find the real "sun".
        items = fixture_items("sun", salient=True)
        result = bm.raw_vs_verified_comparison(["sun", "dinosaur"], ["sun"], items)
        self.assertEqual(result["raw_observer"]["precision_recall_f1"]["precision"], 0.5)
        self.assertEqual(result["verified_only"]["precision_recall_f1"]["precision"], 1.0)
        self.assertAlmostEqual(result["delta_verified_minus_raw"]["precision"], 0.5)
        self.assertEqual(result["delta_verified_minus_raw"]["recall"], 0.0)

    def test_verifier_rejects_a_true_salient_detection_recall_and_salient_recall_worsen(self):
        # RAW correctly finds both "sun" and "person"; VERIFIED wrongly
        # drops the person (a real, salient, human-confirmed detection).
        items = fixture_items("sun", salient=True) + fixture_items("person", salient=True)
        result = bm.raw_vs_verified_comparison(["sun", "person"], ["sun"], items)
        self.assertEqual(result["raw_observer"]["precision_recall_f1"]["recall"], 1.0)
        self.assertEqual(result["verified_only"]["precision_recall_f1"]["recall"], 0.5)
        self.assertAlmostEqual(result["delta_verified_minus_raw"]["recall"], -0.5)
        self.assertAlmostEqual(result["delta_verified_minus_raw"]["salient_recall"], -0.5)

    def test_a1_and_a2_scored_independently_never_merged(self):
        items_a1 = fixture_items("sun")
        items_a2 = fixture_items("sun", "moon")
        result_a1 = bm.raw_vs_verified_comparison(["sun"], ["sun"], items_a1)
        result_a2 = bm.raw_vs_verified_comparison(["sun"], ["sun"], items_a2)
        # Different references -> different recall, proving no cross-annotator merge happened.
        self.assertEqual(result_a1["raw_observer"]["precision_recall_f1"]["recall"], 1.0)
        self.assertEqual(result_a2["raw_observer"]["precision_recall_f1"]["recall"], 0.5)

    def test_identical_raw_and_verified_gives_zero_deltas(self):
        items = fixture_items("sun", salient=True)
        result = bm.raw_vs_verified_comparison(["sun"], ["sun"], items)
        for delta in result["delta_verified_minus_raw"].values():
            self.assertEqual(delta, 0.0)

    def test_empty_verified_candidates_reports_none_precision_not_a_crash(self):
        items = fixture_items("sun")
        result = bm.raw_vs_verified_comparison(["sun"], [], items)
        self.assertIsNone(result["verified_only"]["precision_recall_f1"]["precision"])
        self.assertIsNone(result["delta_verified_minus_raw"]["precision"])


if __name__ == "__main__":
    unittest.main()
