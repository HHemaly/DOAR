"""Unit tests for the DOAR V2 ontology foundation's basic building
blocks: three-valued logic, the feature ontology schema/loader, and the
FeatureObservation invariant."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.ontology.feature_schema import (
    FeatureDefinition,
    FeatureOntology,
    FeatureType,
    load_feature_ontology,
)
from doar.ontology.observation import FeatureObservation, not_assessable
from doar.ontology.tristate import TriState, from_bool, tri_and, tri_not, tri_or


class TriStateLogicTests(unittest.TestCase):
    def test_and_true_true_is_true(self):
        self.assertIs(tri_and(TriState.TRUE, TriState.TRUE), TriState.TRUE)

    def test_and_true_unknown_is_unknown(self):
        self.assertIs(tri_and(TriState.TRUE, TriState.UNKNOWN), TriState.UNKNOWN)

    def test_and_false_unknown_is_false(self):
        self.assertIs(tri_and(TriState.FALSE, TriState.UNKNOWN), TriState.FALSE)

    def test_and_false_true_is_false(self):
        self.assertIs(tri_and(TriState.FALSE, TriState.TRUE), TriState.FALSE)

    def test_or_true_unknown_is_true(self):
        self.assertIs(tri_or(TriState.TRUE, TriState.UNKNOWN), TriState.TRUE)

    def test_or_false_unknown_is_unknown(self):
        self.assertIs(tri_or(TriState.FALSE, TriState.UNKNOWN), TriState.UNKNOWN)

    def test_or_false_false_is_false(self):
        self.assertIs(tri_or(TriState.FALSE, TriState.FALSE), TriState.FALSE)

    def test_not_unknown_is_unknown(self):
        self.assertIs(tri_not(TriState.UNKNOWN), TriState.UNKNOWN)

    def test_not_true_is_false(self):
        self.assertIs(tri_not(TriState.TRUE), TriState.FALSE)

    def test_from_bool_none_is_unknown(self):
        self.assertIs(from_bool(None), TriState.UNKNOWN)

    def test_tristate_has_no_python_truthiness(self):
        with self.assertRaises(TypeError):
            bool(TriState.TRUE)

    def test_and_variadic_three_values(self):
        self.assertIs(tri_and(TriState.TRUE, TriState.TRUE, TriState.UNKNOWN), TriState.UNKNOWN)
        self.assertIs(tri_and(TriState.TRUE, TriState.FALSE, TriState.UNKNOWN), TriState.FALSE)

    def test_or_variadic_three_values(self):
        self.assertIs(tri_or(TriState.FALSE, TriState.FALSE, TriState.UNKNOWN), TriState.UNKNOWN)
        self.assertIs(tri_or(TriState.FALSE, TriState.TRUE, TriState.UNKNOWN), TriState.TRUE)


class FeatureOntologyLoaderTests(unittest.TestCase):
    def test_loads_the_real_config_file_without_error(self):
        ontology = load_feature_ontology()
        self.assertGreater(len(ontology), 0)
        self.assertIn("composition.bounding_box_coverage", ontology)

    def test_prerequisite_chain_is_declared_for_hands_present(self):
        ontology = load_feature_ontology()
        self.assertEqual(
            ontology.requirements_of("hands_present"),
            ("human_figure_present", "hands_region_visible"),
        )

    def test_undefined_requirement_raises(self):
        defs = [FeatureDefinition(
            feature_id="a", name="a", category="c", value_type=FeatureType.BOOLEAN,
            definition="x", requirements=("undefined_feature",),
        )]
        with self.assertRaises(ValueError):
            FeatureOntology(defs)

    def test_duplicate_feature_id_raises(self):
        d = FeatureDefinition(feature_id="a", name="a", category="c", value_type=FeatureType.BOOLEAN, definition="x")
        with self.assertRaises(ValueError):
            FeatureOntology([d, d])

    def test_categorical_without_allowed_values_raises(self):
        with self.assertRaises(ValueError):
            FeatureDefinition(feature_id="a", name="a", category="c", value_type=FeatureType.CATEGORICAL, definition="x")

    def test_invalid_value_type_raises(self):
        with self.assertRaises(ValueError):
            FeatureDefinition(feature_id="a", name="a", category="c", value_type="not_a_real_type", definition="x")


class FeatureObservationInvariantTests(unittest.TestCase):
    def test_not_assessable_with_value_raises(self):
        with self.assertRaises(ValueError):
            FeatureObservation(
                feature_id="f", value=True, value_type=FeatureType.BOOLEAN,
                assessable=False, source="test", not_assessable_reason="x",
            )

    def test_not_assessable_without_reason_raises(self):
        with self.assertRaises(ValueError):
            FeatureObservation(
                feature_id="f", value=None, value_type=FeatureType.BOOLEAN,
                assessable=False, source="test",
            )

    def test_not_assessable_helper_is_valid(self):
        obs = not_assessable("f", value_type=FeatureType.BOOLEAN, source="test", reason="no detector")
        self.assertFalse(obs.assessable)
        self.assertIsNone(obs.value)
        self.assertEqual(obs.not_assessable_reason, "no detector")

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            FeatureObservation(
                feature_id="f", value=True, value_type=FeatureType.BOOLEAN,
                assessable=True, source="test", raw_confidence=1.5,
            )

    def test_empty_source_raises(self):
        with self.assertRaises(ValueError):
            FeatureObservation(
                feature_id="f", value=True, value_type=FeatureType.BOOLEAN,
                assessable=True, source="",
            )

    def test_assessable_observation_round_trips_to_dict(self):
        obs = FeatureObservation(
            feature_id="f", value=0.5, value_type=FeatureType.CONTINUOUS,
            assessable=True, source="existing_cv", raw_confidence=0.9,
        )
        data = obs.to_dict()
        self.assertEqual(data["value"], 0.5)
        self.assertEqual(data["source"], "existing_cv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
