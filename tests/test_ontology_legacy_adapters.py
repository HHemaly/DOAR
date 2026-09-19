"""Tests for the seam between the existing, already-tested evidence
pipeline and the new ontology layer, and the required minimal
end-to-end demonstration on a real (synthetically drawn, non-dataset)
image: no mocking anywhere in this file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.evidence_schema import EvidenceItem, unavailable_item
from doar.ontology.engine import RuleOutcome
from doar.ontology.legacy_adapters import (
    evidence_item_to_observation,
    evidence_set_to_store,
    legacy_rule_to_ontology_rule,
    load_legacy_rules_as_ontology_rules,
    run_ontology_demo_on_image,
)
from doar.ontology.rule import RuleFramework
from doar.evidence_schema import EvidenceSet
from doar.rule_schema import load_rules_v2


class EvidenceItemConversionTests(unittest.TestCase):
    def test_available_item_converts_to_assessable_observation(self):
        item = EvidenceItem(
            evidence_id="e1", feature_id="composition.bounding_box_coverage", value=0.5, unit="ratio",
            status="available", confidence=0.9, extractor="analysis.py", extractor_version="v1",
            validation_status="ok", location=None, visualization_reference=None, reason="measured",
        )
        obs = evidence_item_to_observation(item)
        self.assertTrue(obs.assessable)
        self.assertEqual(obs.value, 0.5)
        self.assertEqual(obs.raw_confidence, 0.9)

    def test_unavailable_item_converts_to_not_assessable_observation(self):
        item = unavailable_item("e1", "semantic.face.eyes", category="semantic_objects", reason="no detector")
        obs = evidence_item_to_observation(item)
        self.assertFalse(obs.assessable)
        self.assertIsNone(obs.value)
        self.assertEqual(obs.not_assessable_reason, "no detector")

    def test_store_conversion_derives_centroid_scalars(self):
        item = EvidenceItem(
            evidence_id="e1", feature_id="composition.centroid_normalized", value=[0.3, 0.7], unit="frac",
            status="available", confidence=0.8, extractor="analysis.py", extractor_version="v1",
            validation_status="ok", location=None, visualization_reference=None, reason="measured",
        )
        store = evidence_set_to_store(EvidenceSet(items=[item]))
        self.assertAlmostEqual(store.effective("composition.centroid_x").value, 0.3)
        self.assertAlmostEqual(store.effective("composition.centroid_y").value, 0.7)


class LegacyRuleConversionTests(unittest.TestCase):
    def test_every_registry_rule_converts_without_error(self):
        rules = load_legacy_rules_as_ontology_rules()
        self.assertEqual(len(rules), 19)

    def test_every_converted_rule_is_classical_projective(self):
        # None of the 19 rules in resources/psychology_sources/rules_registry.json
        # is empirically supported (its own `scientific_support` fields say so) --
        # every one must be tagged CLASSICAL_PROJECTIVE, never
        # EMPIRICAL_EXPRESSIVE, by this adapter.
        for rule in load_legacy_rules_as_ontology_rules():
            self.assertEqual(rule.framework, RuleFramework.CLASSICAL_PROJECTIVE, rule.rule_id)

    def test_coverage_full_rule_premise_matches_high_coverage(self):
        rules = {r.rule_id: r for r in load_legacy_rules_as_ontology_rules()}
        rule = rules["PSY_AR_SIZE_FULL_015"]
        self.assertEqual(rule.required_feature_ids(), ("composition.bounding_box_coverage",))

    def test_unmapped_scientific_support_raises_rather_than_defaulting(self):
        raw = load_rules_v2()[0]
        import dataclasses
        poisoned = dataclasses.replace(raw, scientific_support="a_brand_new_unmapped_value")
        with self.assertRaises(ValueError):
            legacy_rule_to_ontology_rule(poisoned)


class EndToEndDemoTests(unittest.TestCase):
    """Real analyze_image -> evidence_adapter -> ontology store -> ontology
    rule engine chain, on real (non-dataset) synthetic images."""

    def _run(self, image: Image.Image) -> dict:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        return run_ontology_demo_on_image(path, Path(temp.name) / "out")

    def test_full_page_drawing_produces_a_provisional_match_end_to_end(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        result = self._run(image)
        self.assertEqual(result["rule_count"], 19)
        self.assertGreaterEqual(result["outcome_counts"].get("MATCHED_PROVISIONAL", 0)
                                 + result["outcome_counts"].get("MATCHED", 0), 1)
        self.assertEqual(result["outcome_counts"].get("NOT_ASSESSABLE"), 13)

    def test_output_files_are_real_and_contain_traces(self):
        image = Image.new("RGB", (150, 150), "white")
        ImageDraw.Draw(image).ellipse((10, 10, 140, 140), fill="black")
        result = self._run(image)
        trace_path = Path(result["output_files"]["rule_trace"])
        observations_path = Path(result["output_files"]["observations"])
        self.assertTrue(trace_path.exists())
        self.assertTrue(observations_path.exists())
        traces = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(len(traces), 19)
        for trace in traces:
            self.assertIn(trace["outcome"], RuleOutcome.ALL)
            self.assertIn("premise_trace", trace)

    def test_tier2_rules_never_matched_or_not_matched(self):
        image = Image.new("RGB", (150, 150), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 130, 130), fill="black")
        result = self._run(image)
        traces = json.loads(Path(result["output_files"]["rule_trace"]).read_text(encoding="utf-8"))
        tier2_ids = {
            "PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003",
            "PSY_AR_ANIMAL_TIGER_WOLF_004", "PSY_AR_ANIMAL_FOX_005", "PSY_AR_ANIMAL_SQUIRREL_006",
            "PSY_AR_ANIMAL_LION_007", "PSY_AR_GEOMETRY_008", "PSY_AR_STARS_009",
            "PSY_AR_FLOWERS_CLOUDS_SUN_010", "PSY_AR_CIRCLES_011", "PSY_AR_TRANSPORT_012", "PSY_AR_HEARTS_013",
        }
        for trace in traces:
            if trace["rule_id"] in tier2_ids:
                self.assertEqual(trace["outcome"], "NOT_ASSESSABLE", trace["rule_id"])
                self.assertNotIn(trace["outcome"], ("MATCHED", "MATCHED_PROVISIONAL", "NOT_MATCHED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
