"""Tests for src/doar/relationship_features.py -- deterministic,
descriptive spatial-relationship features computed from verified,
bbox-bearing VisualEntity records only.

Proves: only verified+bbox entities are considered; entity_count/figures/
pairwise_distances/spatially_separated_entity_ids/central_entity_ids/repeated_labels
are all populated correctly; a single or zero-entity drawing degrades
gracefully (no crash, no fabricated relationship claim).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar import relationship_features as rf  # noqa: E402
from doar.visual_entity import VisualEntity  # noqa: E402


def _entity(entity_id, canonical_label, bbox, *, status="verified"):
    return VisualEntity(
        entity_id=entity_id, entity_type="object", canonical_label=canonical_label,
        candidate_labels=((canonical_label, 0.9),), aliases_en=(), aliases_ar=(),
        broader_categories=(), possible_subtypes=(), visual_similarities=(),
        bbox=bbox, crop_ref=None, dominant_colors=None, relative_size=None,
        page_position=None, shape_features=None, line_features=None,
        detector="visual_observer:test", checkpoint="", prompt="",
        confidence=0.9, model_validation_status="UNKNOWN", case_verification_status=status,
        evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
        related_rule_ids=(), source="visual_observer", query=None, timestamp="2026-08-14T00:00:00+00:00",
    )


class EmptyAndSingleEntityTests(unittest.TestCase):
    def test_no_entities_returns_all_empty(self):
        result = rf.compute_relationship_features([])
        self.assertEqual(result["entity_count"], 0)
        self.assertEqual(result["figures"], [])
        self.assertEqual(result["spatially_separated_entity_ids"], [])
        self.assertEqual(result["central_entity_ids"], [])
        self.assertEqual(result["repeated_labels"], {})

    def test_single_entity_has_no_pairwise_distances_or_isolation(self):
        entities = [_entity("e1", "sun", (0.1, 0.1, 0.2, 0.2))]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(result["entity_count"], 1)
        self.assertEqual(result["pairwise_distances"], [])
        self.assertEqual(result["spatially_separated_entity_ids"], [])
        self.assertEqual(result["central_entity_ids"], ["e1"])
        self.assertEqual(result["figures"][0]["relative_size"], 1.0)


class VerificationGateTests(unittest.TestCase):
    def test_unverified_entities_are_excluded(self):
        entities = [
            _entity("e1", "sun", (0.1, 0.1, 0.2, 0.2), status="verified"),
            _entity("e2", "moon", (0.5, 0.5, 0.2, 0.2), status="uncertain"),
            _entity("e3", "star", (0.7, 0.7, 0.1, 0.1), status="rejected"),
        ]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(result["entity_count"], 1)
        self.assertEqual({f["entity_id"] for f in result["figures"]}, {"e1"})

    def test_entities_with_no_bbox_are_excluded(self):
        entities = [_entity("e1", "sun", None, status="verified")]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(result["entity_count"], 0)


class RelationshipComputationTests(unittest.TestCase):
    def test_relative_size_is_largest_equals_one(self):
        entities = [
            _entity("big", "house", (0.0, 0.0, 0.5, 0.5)),
            _entity("small", "flower", (0.6, 0.6, 0.1, 0.1)),
        ]
        result = rf.compute_relationship_features(entities)
        sizes = {f["entity_id"]: f["relative_size"] for f in result["figures"]}
        self.assertEqual(sizes["big"], 1.0)
        self.assertLess(sizes["small"], 1.0)

    def test_isolated_entity_is_the_one_farthest_from_all_others(self):
        # A tight 4-entity cluster (6 small pairwise distances) plus one
        # far outlier (4 large distances) -- with 6 small distances the
        # median (index len//2==5 of 10) is guaranteed to still fall
        # inside the small group, so the outlier's nearest distance is
        # unambiguously greater than the median, not merely tied with it.
        entities = [
            _entity("a", "person", (0.10, 0.10, 0.02, 0.02)),
            _entity("b", "person", (0.12, 0.10, 0.02, 0.02)),
            _entity("c", "person", (0.10, 0.12, 0.02, 0.02)),
            _entity("d", "person", (0.12, 0.12, 0.02, 0.02)),
            _entity("e", "sun", (0.90, 0.90, 0.02, 0.02)),
        ]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(result["spatially_separated_entity_ids"], ["e"])

    def test_central_entity_is_closest_to_image_center(self):
        entities = [
            _entity("center", "circle", (0.45, 0.45, 0.1, 0.1)),
            _entity("corner", "star", (0.0, 0.0, 0.05, 0.05)),
        ]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(result["central_entity_ids"], ["center"])

    def test_repeated_labels_only_counts_labels_appearing_more_than_once(self):
        entities = [
            _entity("t1", "tree", (0.0, 0.0, 0.1, 0.1)),
            _entity("t2", "tree", (0.2, 0.2, 0.1, 0.1)),
            _entity("s1", "sun", (0.5, 0.5, 0.1, 0.1)),
        ]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(result["repeated_labels"], {"tree": 2})

    def test_pairwise_distances_cover_every_pair_exactly_once(self):
        entities = [
            _entity("a", "x", (0.0, 0.0, 0.1, 0.1)),
            _entity("b", "y", (0.3, 0.3, 0.1, 0.1)),
            _entity("c", "z", (0.6, 0.6, 0.1, 0.1)),
        ]
        result = rf.compute_relationship_features(entities)
        self.assertEqual(len(result["pairwise_distances"]), 3)
        pairs = {frozenset((d["a"], d["b"])) for d in result["pairwise_distances"]}
        self.assertEqual(pairs, {frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))})


if __name__ == "__main__":
    unittest.main()
