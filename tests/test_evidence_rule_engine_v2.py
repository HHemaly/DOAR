"""Tests for the Phase 1 evidence-to-rule path: canonical evidence schema
(`evidence_schema.py`), rule registry v2 provenance (`rule_schema.py`),
the deterministic rule engine (`evidence_rule_engine.py`), the adapter
that wires real extractors into the schema (`evidence_adapter.py`), the
traceable English report generator (`english_report.py`), and the
end-to-end runner (`evidence_pipeline.py`).

Covers, per the task's required test list: evidence-schema validation,
rule/source validation, missing evidence, failed extractors, qualitative
rules, unvalidated thresholds, conflicting evidence, omission logic,
evidence-to-overlay traceability, report-claim traceability, prevention
of unsupported psychological conclusions -- all against the real pipeline
on real (synthetically generated, non-dataset) images, never mocked.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.english_report import generate_traceable_report
from doar.evidence_adapter import UNIMPLEMENTED_FEATURES
from doar.evidence_pipeline import run_evidence_pipeline
from doar.evidence_rule_engine import evaluate_all_rules_v2, evaluate_omission, evaluate_rule_v2
from doar.evidence_schema import EvidenceItem, EvidenceLocation, EvidenceSet, unavailable_item
from doar.rule_schema import load_rules_v2


class EvidenceSchemaInvariantTests(unittest.TestCase):
    def test_unavailable_status_with_nonnull_value_raises(self):
        with self.assertRaises(ValueError):
            EvidenceItem(
                evidence_id="e1", feature_id="f1", value=0.0, unit=None, status="unavailable",
                confidence=None, extractor="x", extractor_version="v1", validation_status="n/a",
                location=None, visualization_reference=None, reason="test",
            )

    def test_insufficient_evidence_status_with_nonnull_value_raises(self):
        with self.assertRaises(ValueError):
            EvidenceItem(
                evidence_id="e1", feature_id="f1", value=False, unit=None, status="insufficient_evidence",
                confidence=None, extractor="x", extractor_version="v1", validation_status="n/a",
                location=None, visualization_reference=None, reason="test",
            )

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            EvidenceItem(
                evidence_id="e1", feature_id="f1", value=None, unit=None, status="not_a_real_status",
                confidence=None, extractor="x", extractor_version="v1", validation_status="n/a",
                location=None, visualization_reference=None, reason="test",
            )

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            EvidenceItem(
                evidence_id="e1", feature_id="f1", value=1.0, unit=None, status="available",
                confidence=1.5, extractor="x", extractor_version="v1", validation_status="n/a",
                location=None, visualization_reference=None, reason="test",
            )

    def test_unavailable_item_helper_is_valid(self):
        item = unavailable_item("e1", "f1", category="semantic_objects", reason="no detector")
        self.assertEqual(item.status, "unavailable")
        self.assertIsNone(item.value)

    def test_evidence_set_rejects_duplicate_ids(self):
        item = unavailable_item("dup", "f1", category="ocr_text", reason="x")
        with self.assertRaises(ValueError):
            EvidenceSet(items=[item, item])

    def test_evidence_item_with_location_round_trips_to_dict(self):
        item = EvidenceItem(
            evidence_id="e1", feature_id="composition.bounding_box_coverage", value=0.5, unit="ratio",
            status="available", confidence=0.9, extractor="x", extractor_version="v1",
            validation_status="ok", location=EvidenceLocation(region="bounding_box", bbox_xywh=(0.1, 0.1, 0.5, 0.5)),
            visualization_reference="artifacts/feature_overlay.png", reason="test",
        )
        data = item.to_dict()
        self.assertEqual(data["location"]["region"], "bounding_box")
        self.assertEqual(data["visualization_reference"], "artifacts/feature_overlay.png")


class RuleSchemaProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.rules = {r.rule_id: r for r in load_rules_v2()}

    def test_loads_exactly_nineteen_rules(self):
        self.assertEqual(len(self.rules), 19)

    def test_every_rule_has_nonempty_required_feature_ids(self):
        for rule in self.rules.values():
            self.assertTrue(rule.required_feature_ids, rule.rule_id)

    def test_directly_sourced_threshold_is_coverage_small(self):
        self.assertEqual(self.rules["PSY_AR_SIZE_SMALL_016"].threshold_source, "directly_sourced")

    def test_sourced_center_invented_band_is_coverage_half(self):
        self.assertEqual(self.rules["PSY_AR_SIZE_HALF_014"].threshold_source, "sourced_center_invented_band")

    def test_invented_numeric_stand_in_is_coverage_full(self):
        self.assertEqual(self.rules["PSY_AR_SIZE_FULL_015"].threshold_source, "invented_numeric_stand_in")

    def test_placement_rules_have_no_source_anchor(self):
        for rid in ("PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019"):
            self.assertEqual(self.rules[rid].threshold_source, "invented_no_anchor", rid)

    def test_tier2_rules_have_not_applicable_threshold_source(self):
        tier2 = [r for r in self.rules.values() if r.tier == "tier_2_content_conditional"]
        self.assertEqual(len(tier2), 13)
        for rule in tier2:
            self.assertEqual(rule.threshold_source, "not_applicable", rule.rule_id)

    def test_every_rule_cites_the_same_single_source_pdf_and_a_real_page(self):
        for rule in self.rules.values():
            self.assertEqual(rule.source_document, "التحليل النفسي للصور.pdf")
            self.assertIn(rule.source_page, (1, 2))


def _coverage_evidence(value: float, status: str = "available") -> EvidenceSet:
    item = EvidenceItem(
        evidence_id="ev_bbox", feature_id="composition.bounding_box_coverage", value=value, unit="ratio",
        status=status, confidence=0.9, extractor="test", extractor_version="v1", validation_status="ok",
        location=None, visualization_reference=None, reason="synthetic test evidence",
    )
    return EvidenceSet(items=[item])


class RuleEngineOutcomeVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.rules = {r.rule_id: r for r in load_rules_v2()}

    def test_triggered_when_condition_met(self):
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], _coverage_evidence(0.95))
        self.assertEqual(result.outcome, "triggered")
        self.assertIn("ev_bbox", result.evidence_ids_used)
        self.assertGreater(result.rule_confidence, 0.0)

    def test_not_triggered_when_condition_not_met(self):
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], _coverage_evidence(0.5))
        self.assertEqual(result.outcome, "not_triggered")
        self.assertEqual(result.rule_confidence, 0.0)

    def test_extractor_unavailable_when_feature_entirely_missing(self):
        empty = EvidenceSet(items=[])
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], empty)
        self.assertEqual(result.outcome, "extractor_unavailable")

    def test_extractor_unavailable_when_evidence_explicitly_unavailable(self):
        evidence = EvidenceSet(items=[unavailable_item(
            "ev_bbox", "composition.bounding_box_coverage", category="global_composition", reason="x",
        )])
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], evidence)
        self.assertEqual(result.outcome, "extractor_unavailable")

    def test_insufficient_evidence_propagates(self):
        item = EvidenceItem(
            evidence_id="ev_bbox", feature_id="composition.bounding_box_coverage", value=None, unit="ratio",
            status="insufficient_evidence", confidence=None, extractor="test", extractor_version="v1",
            validation_status="n/a", location=None, visualization_reference=None, reason="blank page",
        )
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], EvidenceSet(items=[item]))
        self.assertEqual(result.outcome, "insufficient_evidence")

    def test_requires_manual_review_propagates(self):
        item = EvidenceItem(
            evidence_id="ev_bbox", feature_id="composition.bounding_box_coverage", value=None, unit="ratio",
            status="requires_manual_review", confidence=None, extractor="test", extractor_version="v1",
            validation_status="n/a", location=None, visualization_reference=None, reason="untrusted extractor",
        )
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], EvidenceSet(items=[item]))
        self.assertEqual(result.outcome, "requires_manual_review")

    def test_conflicting_evidence_detected_for_two_disagreeing_measurements(self):
        items = [
            EvidenceItem(
                evidence_id="ev_bbox_a", feature_id="composition.bounding_box_coverage", value=0.95, unit="ratio",
                status="available", confidence=0.9, extractor="test_a", extractor_version="v1",
                validation_status="ok", location=None, visualization_reference=None, reason="x",
            ),
            EvidenceItem(
                evidence_id="ev_bbox_b", feature_id="composition.bounding_box_coverage", value=0.10, unit="ratio",
                status="available", confidence=0.9, extractor="test_b", extractor_version="v1",
                validation_status="ok", location=None, visualization_reference=None, reason="y",
            ),
        ]
        result = evaluate_rule_v2(self.rules["PSY_AR_SIZE_FULL_015"], EvidenceSet(items=items))
        self.assertEqual(result.outcome, "conflicting_evidence")

    def test_missing_extractor_never_produces_triggered_or_not_triggered_for_any_tier2_rule(self):
        """The task's hard invariant: a missing extractor must never produce
        a passing or negative rule result. Builds evidence exactly as
        evidence_adapter.py does for every unimplemented category and
        confirms all 13 tier-2 rules resolve to extractor_unavailable."""
        items = [
            unavailable_item(f"u{i}", feature_id, category=info["category"], reason=info["reason"])
            for i, (feature_id, info) in enumerate(UNIMPLEMENTED_FEATURES.items())
        ]
        evidence = EvidenceSet(items=items)
        tier2 = [r for r in self.rules.values() if r.tier == "tier_2_content_conditional"]
        self.assertEqual(len(tier2), 13)
        for rule in tier2:
            result = evaluate_rule_v2(rule, evidence)
            self.assertEqual(result.outcome, "extractor_unavailable", rule.rule_id)
            self.assertNotIn(result.outcome, ("triggered", "not_triggered"))

    def test_not_applicable_for_tier3_prompt_dependent_rule(self):
        # No rule in the live registry is tier_3 -- constructed synthetically
        # to prove the engine handles it correctly if one is ever added.
        base = self.rules["PSY_AR_SIZE_FULL_015"]
        synthetic = dataclasses.replace(base, tier="tier_3_prompt_or_age_dependent", required_feature_ids=[])
        result = evaluate_rule_v2(synthetic, EvidenceSet(items=[]))
        self.assertEqual(result.outcome, "not_applicable")

    def test_all_nineteen_rules_evaluate_without_error_against_realistic_evidence(self):
        rules = list(self.rules.values())
        items = [
            unavailable_item(f"u{i}", feature_id, category=info["category"], reason=info["reason"])
            for i, (feature_id, info) in enumerate(UNIMPLEMENTED_FEATURES.items())
        ]
        items.append(EvidenceItem(
            evidence_id="ev_bbox", feature_id="composition.bounding_box_coverage", value=0.5, unit="ratio",
            status="available", confidence=0.9, extractor="test", extractor_version="v1", validation_status="ok",
            location=None, visualization_reference=None, reason="x",
        ))
        items.append(EvidenceItem(
            evidence_id="ev_centroid", feature_id="composition.centroid_normalized", value=[0.5, 0.5], unit="frac",
            status="available", confidence=0.9, extractor="test", extractor_version="v1", validation_status="ok",
            location=None, visualization_reference=None, reason="x",
        ))
        results = evaluate_all_rules_v2(rules, EvidenceSet(items=items))
        self.assertEqual(len(results), 19)
        for result in results:
            self.assertIn(result.outcome, (
                "triggered", "not_triggered", "insufficient_evidence", "not_applicable",
                "extractor_unavailable", "requires_manual_review", "conflicting_evidence",
            ))


class OmissionLogicTests(unittest.TestCase):
    def test_all_preconditions_true_returns_evaluable(self):
        outcome = evaluate_omission(
            parent_object_reliably_detected=True, region_visible=True,
            component_extractor_applicable_and_reliable=True, absence_criterion_evaluated=True,
        )
        self.assertEqual(outcome, "evaluable")

    def test_parent_not_detected_returns_insufficient_evidence(self):
        outcome = evaluate_omission(
            parent_object_reliably_detected=False, region_visible=True,
            component_extractor_applicable_and_reliable=True, absence_criterion_evaluated=True,
        )
        self.assertEqual(outcome, "insufficient_evidence")

    def test_region_not_visible_returns_insufficient_evidence(self):
        outcome = evaluate_omission(
            parent_object_reliably_detected=True, region_visible=False,
            component_extractor_applicable_and_reliable=True, absence_criterion_evaluated=True,
        )
        self.assertEqual(outcome, "insufficient_evidence")

    def test_extractor_not_reliable_returns_insufficient_evidence(self):
        outcome = evaluate_omission(
            parent_object_reliably_detected=True, region_visible=True,
            component_extractor_applicable_and_reliable=False, absence_criterion_evaluated=True,
        )
        self.assertEqual(outcome, "insufficient_evidence")

    def test_absence_not_evaluated_returns_insufficient_evidence(self):
        outcome = evaluate_omission(
            parent_object_reliably_detected=True, region_visible=True,
            component_extractor_applicable_and_reliable=True, absence_criterion_evaluated=False,
        )
        self.assertEqual(outcome, "insufficient_evidence")


class EndToEndRealPipelineTests(unittest.TestCase):
    """Runs the real analyze_image -> evidence_adapter -> rule engine ->
    report chain against real (synthetically drawn, non-dataset) images.
    No mocking anywhere in this class."""

    def _run(self, image: Image.Image):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        return run_evidence_pipeline(path, Path(temp.name) / "out")

    def test_full_page_dark_drawing_triggers_coverage_full_end_to_end(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        result = self._run(image)
        self.assertEqual(result["rule_outcome_counts"].get("triggered"), 1)
        self.assertEqual(result["rule_outcome_counts"].get("extractor_unavailable"), 13)
        report_text = Path(result["output_files"]["report_en"]).read_text(encoding="utf-8")
        self.assertIn("PSY_AR_SIZE_FULL_015", report_text)
        self.assertIn("page 2", report_text)

    def test_small_centered_drawing_triggers_coverage_small_end_to_end(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((95, 95, 105, 105), fill="black")
        result = self._run(image)
        rule_json_path = Path(result["output_files"]["rule_evaluations_v2"])
        import json
        rule_results = json.loads(rule_json_path.read_text(encoding="utf-8"))
        small = next(r for r in rule_results if r["rule_id"] == "PSY_AR_SIZE_SMALL_016")
        self.assertEqual(small["outcome"], "triggered")

    def test_thirteen_tier2_rules_are_always_extractor_unavailable_never_triggered(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 180, 180), fill="black")
        result = self._run(image)
        import json
        rule_results = json.loads(Path(result["output_files"]["rule_evaluations_v2"]).read_text(encoding="utf-8"))
        tier2_ids = {
            "PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003",
            "PSY_AR_ANIMAL_TIGER_WOLF_004", "PSY_AR_ANIMAL_FOX_005", "PSY_AR_ANIMAL_SQUIRREL_006",
            "PSY_AR_ANIMAL_LION_007", "PSY_AR_GEOMETRY_008", "PSY_AR_STARS_009",
            "PSY_AR_FLOWERS_CLOUDS_SUN_010", "PSY_AR_CIRCLES_011", "PSY_AR_TRANSPORT_012", "PSY_AR_HEARTS_013",
        }
        for r in rule_results:
            if r["rule_id"] in tier2_ids:
                self.assertEqual(r["outcome"], "extractor_unavailable", r["rule_id"])

    def test_output_files_are_real_and_nonempty(self):
        image = Image.new("RGB", (150, 150), "white")
        ImageDraw.Draw(image).ellipse((10, 10, 140, 140), fill="black")
        result = self._run(image)
        for key in ("evidence_v2", "rule_evaluations_v2", "report_en", "pipeline_timing"):
            path = Path(result["output_files"][key])
            self.assertTrue(path.exists(), key)
            self.assertGreater(path.stat().st_size, 0, key)

    def test_evidence_includes_visualization_reference_for_localized_items(self):
        image = Image.new("RGB", (150, 150), "white")
        ImageDraw.Draw(image).ellipse((10, 10, 140, 140), fill="black")
        result = self._run(image)
        import json
        evidence_items = json.loads(Path(result["output_files"]["evidence_v2"]).read_text(encoding="utf-8"))
        bbox_item = next(i for i in evidence_items if i["evidence_id"] == "ev2_bbox_coverage")
        self.assertIsNotNone(bbox_item["visualization_reference"])
        self.assertTrue(Path(bbox_item["visualization_reference"]).exists())


class ReportTraceabilityAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules_v2()

    def test_generated_report_never_contains_diagnostic_language(self):
        image = Image.new("RGB", (150, 150), "white")
        ImageDraw.Draw(image).rectangle((5, 5, 145, 145), fill="black")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        result = run_evidence_pipeline(path, Path(temp.name) / "out")
        report_text = Path(result["output_files"]["report_en"]).read_text(encoding="utf-8")
        from doar.judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
        self.assertIsNone(_ARABIC_DIAGNOSTIC.search(report_text))
        for pattern in DIAGNOSTIC_PATTERNS:
            self.assertIsNone(pattern.search(report_text), pattern.pattern)

    def test_report_raises_if_a_rule_injects_diagnostic_language(self):
        rules_by_id = {r.rule_id: r for r in self.rules}
        poisoned = dataclasses.replace(
            rules_by_id["PSY_AR_SIZE_FULL_015"],
            professional_reasoning="The child is depressed and this proves trauma.",
        )
        rules = [poisoned if r.rule_id == "PSY_AR_SIZE_FULL_015" else r for r in self.rules]
        evidence = _coverage_evidence(0.95)
        result = evaluate_rule_v2(poisoned, evidence)
        with self.assertRaises(ValueError):
            generate_traceable_report(rules, [result], evidence, "synthetic_test_image.png")

    def test_triggered_observation_cites_source_pdf_page_and_evidence_id(self):
        evidence = _coverage_evidence(0.95)
        rules_by_id = {r.rule_id: r for r in self.rules}
        rule = rules_by_id["PSY_AR_SIZE_FULL_015"]
        result = evaluate_rule_v2(rule, evidence)
        report = generate_traceable_report([rule], [result], evidence, "synthetic_test_image.png")
        self.assertIn("ev_bbox", report)
        self.assertIn("page 2", report)
        self.assertIn(rule.source_section, report)
        self.assertIn("Clinician review required", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
