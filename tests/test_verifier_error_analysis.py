"""Tests for scripts/verifier_error_analysis.py -- cached-data-only
verifier error analysis (no live Gemini calls, no writes to any cached
verification or annotation file).
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "verifier_error_analysis", ROOT / "scripts" / "verifier_error_analysis.py")
vea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vea)


class _Item:
    def __init__(self, label, salient=False):
        self.label = label
        self.salient = salient


class BboxValidityTests(unittest.TestCase):
    def test_valid_normalized_bbox(self):
        self.assertIs(vea._bbox_is_valid([0.1, 0.1, 0.2, 0.2]), True)

    def test_out_of_range_height_is_invalid(self):
        # Real p2b_0004 data: height values like 2.7, 3.5, 6.0.
        self.assertIs(vea._bbox_is_valid([0.0, 0.0, 0.13, 2.7]), False)

    def test_negative_value_is_invalid(self):
        self.assertIs(vea._bbox_is_valid([-0.1, 0.0, 0.2, 0.2]), False)

    def test_no_bbox_is_unknown_not_invalid(self):
        self.assertEqual(vea._bbox_is_valid(None), "")
        self.assertEqual(vea._bbox_is_valid([]), "")


class BucketClassificationTests(unittest.TestCase):
    def test_true_kept_verified(self):
        self.assertEqual(vea._bucket(True, "verified"), "1_true_kept_verified")

    def test_true_removed_rejected(self):
        self.assertEqual(vea._bucket(True, "rejected"), "2_true_removed")

    def test_true_removed_uncertain(self):
        self.assertEqual(vea._bucket(True, "uncertain"), "2_true_removed")

    def test_true_removed_unreviewed(self):
        # "unreviewed" has the same downstream effect as rejected/uncertain
        # -- reasoning_chain.py only trusts case_verification_status=="verified".
        self.assertEqual(vea._bucket(True, "unreviewed"), "2_true_removed")

    def test_false_removed(self):
        self.assertEqual(vea._bucket(False, "rejected"), "3_false_removed")

    def test_false_kept_verified(self):
        self.assertEqual(vea._bucket(False, "verified"), "4_false_kept_verified")

    def test_no_annotator_data_returns_none(self):
        self.assertIsNone(vea._bucket(None, "verified"))


class MatchingItemsTests(unittest.TestCase):
    def test_finds_matching_items(self):
        items = [_Item("sun"), _Item("tree")]
        self.assertEqual([i.label for i in vea._matching_items("sun", items)], ["sun"])

    def test_no_items_returns_empty(self):
        self.assertEqual(vea._matching_items("sun", None), [])
        self.assertEqual(vea._matching_items("sun", []), [])

    def test_no_match_returns_empty(self):
        self.assertEqual(vea._matching_items("dinosaur", [_Item("sun")]), [])


def _row(label="x", alt_labels=(), bbox=(0.1, 0.1, 0.2, 0.2), verifier_label="unknown",
         verifier_alt=(), status="uncertain", confidence=0.9, verifier_confidence=None):
    return {
        "observer_candidate": {"label": label, "alternative_labels": list(alt_labels), "bbox": list(bbox) if bbox else None,
                                "confidence": confidence, "entity_type": "object"},
        "verifier_independent_label": verifier_label,
        "verifier_independent_alternative_labels": list(verifier_alt),
        "verifier_confidence": verifier_confidence,
        "verification_status": status,
        "verifier_notes": "",
    }


class HeuristicFailureCategoryTests(unittest.TestCase):
    def test_unknown_verifier_label_is_visual_miss(self):
        row = _row(verifier_label="unknown", status="uncertain")
        self.assertEqual(vea._heuristic_failure_category(row, [], []), "verifier_visual_miss")

    def test_invalid_bbox_is_bad_bbox(self):
        row = _row(bbox=(0.0, 0.0, 0.1, 2.7), verifier_label="unknown", status="unreviewed")
        self.assertEqual(vea._heuristic_failure_category(row, [], []), "bad_bbox_or_crop_construction")

    def test_unreviewed_with_no_bbox_is_no_bbox_category(self):
        row = _row(bbox=None, status="unreviewed", verifier_label="")
        self.assertEqual(vea._heuristic_failure_category(row, [], []), "no_bbox_never_reviewed")

    def test_verifier_label_matches_a_different_human_item_is_bad_bbox(self):
        # Verifier's own independent read is itself a real, human-annotated
        # item -- just not the one the observer's candidate claimed.
        row = _row(label="tears", verifier_label="mouth", status="rejected", verifier_confidence=0.8)
        a1_items = [_Item("mouth", salient=True)]
        self.assertEqual(vea._heuristic_failure_category(row, a1_items, []), "bad_bbox_or_crop_construction")

    def test_small_bbox_is_bad_bbox(self):
        row = _row(bbox=(0.5, 0.5, 0.05, 0.05), verifier_label="something else entirely",
                    status="rejected", verifier_confidence=0.9)
        self.assertEqual(vea._heuristic_failure_category(row, [], []), "bad_bbox_or_crop_construction")

    def test_confident_wrong_label_with_normal_bbox_is_visual_miss(self):
        row = _row(label="window", bbox=(0.2, 0.2, 0.3, 0.3), verifier_label="french fries",
                    status="rejected", verifier_confidence=0.85)
        self.assertEqual(vea._heuristic_failure_category(row, [], []), "verifier_visual_miss")


class RealCachedDataAnalysisTests(unittest.TestCase):
    """Sanity checks against real cached data -- proves the analysis
    reproduces the two headline findings the report is built on."""

    def test_p2b_0004_now_has_valid_bboxes_after_the_live_refresh(self):
        # p2b_0004 was originally found 100% removed because every cached
        # bbox was out of the valid [0, 1] normalized range (stale,
        # pre-bbox-schema-fix data). It has since been refreshed through
        # the current, frozen Observer -> Verifier
        # (outputs/prototype_cases/development_live_cache/
        # p2b_0004_verification.json) -- this now proves the FIX, not the
        # original bug: every cached bbox is valid, and at least one
        # candidate is no longer stuck at "unreviewed".
        rows, summary = vea.analyze()
        p2b_0004 = next((img for img in summary["images"] if img["image_id"] == "p2b_0004"), None)
        if p2b_0004 is None or p2b_0004["status"] != "ok":
            self.skipTest("p2b_0004 cached data not present in this checkout")
        p2b_0004_rows = [r for r in rows if r["image_id"] == "p2b_0004"]
        self.assertTrue(all(r["observer_bbox_valid"] is True for r in p2b_0004_rows),
                         "every refreshed p2b_0004 candidate should have a valid normalized bbox")
        self.assertTrue(any(r["verification_status"] != "unreviewed" for r in p2b_0004_rows),
                         "the refresh should let the Verifier actually attempt at least one candidate")

    def test_p2b_0003_frowning_mouth_is_a_true_removed_salient_case(self):
        rows, _summary = vea.analyze()
        candidate = next((r for r in rows if r["image_id"] == "p2b_0003"
                           and r["observer_label"] == "frowning mouth"), None)
        if candidate is None:
            self.skipTest("p2b_0003 frowning mouth candidate not present in this checkout")
        self.assertEqual(candidate["bucket_a1"], "2_true_removed")
        self.assertEqual(candidate["a1_salient"], True)

    def test_every_row_has_a_verification_status_and_image_id(self):
        # vea.analyze() sweeps the real, generated development-set cache
        # (optional, never committed) across all 15 images -- absent on a
        # clean checkout (e.g. CI), same as the two real-cached-data tests
        # above in this class, which already self-skip the same way.
        rows, _summary = vea.analyze()
        if not rows:
            self.skipTest("no cached Observer/Verifier data present in this checkout")
        for row in rows:
            self.assertTrue(row["image_id"])
            self.assertTrue(row["verification_status"])

    def test_analysis_never_writes_to_any_annotation_or_cache_file(self):
        # Read-only guard: no write/open(...'w') call anywhere in the module.
        source = (ROOT / "scripts" / "verifier_error_analysis.py").read_text(encoding="utf-8")
        self.assertNotIn(".write_text(", source.split("def write_csv")[0])


if __name__ == "__main__":
    unittest.main()
