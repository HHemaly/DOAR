"""Tests for the canonical, append-only, multi-source FeatureStore.

Covers, per the task's required test list: multiple observations for one
feature, conflicting observations preserved, human correction does not
erase the original source, and not-assessable state represented
correctly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.ontology.feature_schema import FeatureType
from doar.ontology.observation import FeatureObservation, not_assessable
from doar.ontology.store import FeatureStore, ObservationStatus


def _obs(feature_id="f", value=True, source="existing_cv", value_type=FeatureType.BOOLEAN, **kw):
    return FeatureObservation(feature_id=feature_id, value=value, value_type=value_type, assessable=True, source=source, **kw)


class SingleAndMultiSourceTests(unittest.TestCase):
    def test_no_observation_is_not_assessable(self):
        store = FeatureStore()
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.NOT_ASSESSABLE)
        self.assertIsNone(eff.value)
        self.assertFalse(eff.assessable)

    def test_single_ai_source_is_ai_only(self):
        store = FeatureStore()
        store.add_observation(_obs(source="existing_cv"))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.AI_ONLY)
        self.assertTrue(eff.value)

    def test_single_human_source_without_confirm_correct_is_single_source(self):
        store = FeatureStore()
        store.add_observation(_obs(source="human_observer_note"))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.SINGLE_SOURCE)

    def test_two_agreeing_sources_are_corroborated(self):
        store = FeatureStore()
        store.add_observation(_obs(source="vlm_a", value=True))
        store.add_observation(_obs(source="vlm_b", value=True))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.CORROBORATED)
        self.assertTrue(eff.value)

    def test_two_disagreeing_sources_are_conflict_and_not_assessable_as_a_value(self):
        store = FeatureStore()
        store.add_observation(_obs(source="detector", value=False))
        store.add_observation(_obs(source="vlm_a", value=True))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.CONFLICT)
        self.assertFalse(eff.assessable)
        self.assertIsNone(eff.value)

    def test_multiple_observations_are_all_preserved_in_history(self):
        store = FeatureStore()
        store.add_observation(_obs(source="detector", value=False))
        store.add_observation(_obs(source="vlm_a", value=True))
        store.add_observation(_obs(source="vlm_b", value=True))
        history = store.history("f")
        self.assertEqual(len(history), 3)
        self.assertEqual([o.source for o in history], ["detector", "vlm_a", "vlm_b"])

    def test_repeated_observations_from_the_same_source_stay_single_source_status(self):
        store = FeatureStore()
        store.add_observation(_obs(source="existing_cv", value=True))
        store.add_observation(_obs(source="existing_cv", value=True))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.AI_ONLY)
        self.assertEqual(len(store.history("f")), 2)


class NotAssessableTests(unittest.TestCase):
    def test_not_assessable_observation_is_represented_correctly(self):
        store = FeatureStore()
        store.add_observation(not_assessable("f", value_type=FeatureType.BOOLEAN, source="detector", reason="no detector implemented"))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.NOT_ASSESSABLE)
        self.assertIsNone(eff.value)
        self.assertIn("no detector", eff.reason)

    def test_not_assessable_observation_does_not_count_as_a_source_for_corroboration(self):
        store = FeatureStore()
        store.add_observation(not_assessable("f", value_type=FeatureType.BOOLEAN, source="detector_a", reason="failed"))
        store.add_observation(_obs(source="detector_b", value=True))
        eff = store.effective("f")
        # Only one assessable source contributed a real value.
        self.assertEqual(eff.status, ObservationStatus.AI_ONLY)
        self.assertTrue(eff.value)


class HumanCorrectionTests(unittest.TestCase):
    def test_human_confirm_does_not_erase_ai_observation(self):
        store = FeatureStore()
        store.add_observation(_obs(source="existing_cv", value=True))
        store.human_confirm(_obs(source="human_reviewer:alex", value=True))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.HUMAN_CONFIRMED)
        self.assertEqual(len(store.history("f")), 2)
        self.assertEqual(store.history("f")[0].source, "existing_cv")

    def test_human_correct_overrides_effective_value_but_keeps_original(self):
        store = FeatureStore()
        store.add_observation(_obs(source="existing_cv", value=False))
        store.human_correct(_obs(source="human_reviewer:alex", value=True))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.HUMAN_CORRECTED)
        self.assertTrue(eff.value)
        history = store.history("f")
        self.assertEqual(len(history), 2)
        self.assertFalse(history[0].value)  # original AI claim intact
        self.assertTrue(history[1].value)

    def test_human_correction_wins_over_prior_conflict(self):
        store = FeatureStore()
        store.add_observation(_obs(source="detector", value=False))
        store.add_observation(_obs(source="vlm_a", value=True))
        self.assertEqual(store.effective("f").status, ObservationStatus.CONFLICT)
        store.human_correct(_obs(source="human_reviewer:alex", value=True))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.HUMAN_CORRECTED)
        self.assertTrue(eff.value)
        self.assertEqual(len(store.history("f")), 3)

    def test_second_human_correction_supersedes_first_but_both_kept(self):
        store = FeatureStore()
        store.human_correct(_obs(source="human_reviewer:alex", value=True))
        store.human_correct(_obs(source="human_reviewer:sam", value=False))
        eff = store.effective("f")
        self.assertEqual(eff.status, ObservationStatus.HUMAN_CORRECTED)
        self.assertFalse(eff.value)
        self.assertEqual(len(store.history("f")), 2)

    def test_internal_append_guard_rejects_feature_id_mismatch(self):
        # Defensive invariant: _append's feature_id key must always match
        # the observation's own feature_id. Not reachable through the
        # public add_observation/human_confirm/human_correct API (all
        # three derive the key from the observation itself), but this
        # guard exists so a future internal call site cannot silently
        # file an observation under the wrong feature.
        store = FeatureStore()
        obs = _obs(feature_id="other", source="existing_cv")
        with self.assertRaises(ValueError):
            store._append("f", "observe", obs)


class FeatureIdsAndIterationTests(unittest.TestCase):
    def test_feature_ids_lists_every_feature_touched(self):
        store = FeatureStore()
        store.add_observation(_obs(feature_id="a", source="existing_cv"))
        store.add_observation(_obs(feature_id="b", source="existing_cv"))
        self.assertEqual(set(store.feature_ids()), {"a", "b"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
