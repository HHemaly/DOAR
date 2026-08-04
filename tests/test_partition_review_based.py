"""Tests for `compute_duplicate_groups_from_review` (src/doar/partition.py),
added because the function was designed and manually smoke-tested against
the real completed Phase 7B human review this session, but had no
persisted unit test in the suite -- a real gap given Phase 1 Section 8
explicitly requires tests for "completed human-review export parsing,
dataset provenance from review decision to manifest, duplicate leakage
prevention."

These tests use small synthetic manifests/registries/decisions (not the
real 3,688-row dataset) so they run fast and pin down the override
semantics precisely, including the two concrete real-data cases this
session found where a naive blanket dHash threshold would have been
wrong (see HUMAN_REVIEW_EXPORT_VERIFICATION.md Section 6).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.partition import compute_duplicate_groups_from_review


def _row(image_id: str, dhash: str, sha256: str | None = None) -> dict:
    return {"image_id": image_id, "dhash": dhash, "sha256": sha256 or image_id}


def _registry(pairs: dict[str, tuple[str, str]]) -> dict:
    return {"items": {item_id: {"image_a": a, "image_b": b} for item_id, (a, b) in pairs.items()}}


def _decisions(choices: dict[str, str]) -> dict:
    return {item_id: {"decision": decision, "notes": "", "reviewed_at": "2026-08-04T00:00:00"}
            for item_id, decision in choices.items()}


class AutoMergeThresholdTests(unittest.TestCase):
    def test_low_distance_pair_auto_merges_with_no_review(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdefabcc")]  # distance 1
        result = compute_duplicate_groups_from_review(rows, _registry({}), _decisions({}))
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_near_edges"], 1)

    def test_high_distance_pair_does_not_auto_merge_with_no_review(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]  # distance 11
        result = compute_duplicate_groups_from_review(rows, _registry({}), _decisions({}))
        self.assertNotEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_near_edges"], 0)


class HumanOverrideTests(unittest.TestCase):
    """Mirrors the two real cases found this session at component_17,
    distance=3, reviewed different_drawings -- proving the override isn't
    merely theoretical."""

    def test_reviewed_different_drawings_blocks_an_otherwise_auto_merged_low_distance_pair(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdefabcc")]  # distance 1, would auto-merge
        registry = _registry({"item_001": ("a", "b")})
        decisions = _decisions({"item_001": "different_drawings"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertNotEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_near_edges"], 0)
        self.assertEqual(result["n_human_forced_split_edges"], 1)

    def test_reviewed_definite_duplicate_forces_merge_of_an_otherwise_unmerged_high_distance_pair(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]  # distance 11, would not auto-merge
        registry = _registry({"item_002": ("a", "b")})
        decisions = _decisions({"item_002": "definite_duplicate"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_human_forced_merge_edges"], 1)

    def test_same_drawing_transformed_is_also_a_forced_merge(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]
        registry = _registry({"item_003": ("a", "b")})
        decisions = _decisions({"item_003": "same_drawing_transformed"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])

    def test_uncertain_forces_neither_merge_nor_split(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]  # distance 11
        registry = _registry({"item_004": ("a", "b")})
        decisions = _decisions({"item_004": "uncertain"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        # Not auto-merged (distance too high) and not human-forced either.
        self.assertNotEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_human_forced_merge_edges"], 0)
        self.assertEqual(result["n_human_forced_split_edges"], 0)


class ExactShaAlwaysWinsTests(unittest.TestCase):
    def test_exact_sha256_merges_even_if_reviewed_different_drawings(self):
        # sha256 equality is defined as always winning, regardless of any
        # human review verdict recorded for the same pair.
        rows = [_row("a", "abcdefabcdefabcd", sha256="same"), _row("b", "abcdefabcdef1234", sha256="same")]
        registry = _registry({"item_005": ("a", "b")})
        decisions = _decisions({"item_005": "different_drawings"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_exact_edges"], 1)


class ContradictionHandlingTests(unittest.TestCase):
    def test_same_pair_reviewed_twice_with_conflicting_verdicts_is_flagged_and_resolved_toward_merge(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]  # distance 11
        registry = _registry({"item_006": ("a", "b"), "item_007": ("a", "b")})
        decisions = _decisions({"item_006": "definite_duplicate", "item_007": "different_drawings"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_flagged_contradictions"], 1)
        contradiction = result["flagged_contradictions"][0]
        self.assertEqual(set(contradiction["item_ids"]), {"item_006", "item_007"})

    def test_same_pair_reviewed_twice_with_both_duplicate_flavors_is_not_a_contradiction(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]
        registry = _registry({"item_008": ("a", "b"), "item_009": ("a", "b")})
        decisions = _decisions({"item_008": "definite_duplicate", "item_009": "same_drawing_transformed"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_flagged_contradictions"], 0)


class UnaffectedRowsAndProvenanceTests(unittest.TestCase):
    def test_image_with_no_review_and_no_close_hash_stays_in_its_own_group(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("c", "ffffffff00000000")]
        result = compute_duplicate_groups_from_review(rows, _registry({}), _decisions({}))
        self.assertNotEqual(result["group_of"]["a"], result["group_of"]["c"])
        self.assertEqual(result["n_groups"], 2)

    def test_every_forced_merge_edge_traces_back_to_a_source(self):
        rows = [_row("a", "abcdefabcdefabcd"), _row("b", "abcdefabcdef1234")]
        registry = _registry({"item_010": ("a", "b")})
        decisions = _decisions({"item_010": "definite_duplicate"})
        result = compute_duplicate_groups_from_review(rows, registry, decisions)
        self.assertEqual(len(result["human_forced_merge_edges"]), 1)
        edge = result["human_forced_merge_edges"][0]
        self.assertEqual(edge["source"], "human_review")
        self.assertEqual({edge["a"], edge["b"]}, {"a", "b"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
