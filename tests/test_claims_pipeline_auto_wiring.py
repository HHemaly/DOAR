"""Tests for DOAR-TRACE Phase 1.5 Section 8: automatic judges_v2.json,
generated_claims.json, verification_report.json persistence for every
case, and that failed claims are removed/downgraded rather than shown."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.claims_pipeline import build_claims_and_verification
from doar.registry_v2_build import build_registry_v2


class AutomaticPersistenceTests(unittest.TestCase):
    def _analyze(self, image: Image.Image):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        out = Path(temp.name) / "out"
        analyze_image(path, out)
        return out

    def test_judges_v2_json_written_automatically(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        out = self._analyze(image)
        path = out / "judges_v2.json"
        self.assertTrue(path.exists())
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(doc.keys()), {
            "quality_judge", "feature_judge", "detection_judge", "relation_judge",
            "model_judge", "rule_judge", "aggregation_judge", "language_judge",
        })

    def test_generated_claims_json_written_automatically(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        out = self._analyze(image)
        path = out / "generated_claims.json"
        self.assertTrue(path.exists())
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(doc["claim_count"], 1)
        for claim in doc["claims"]:
            self.assertIn("claim_id", claim)
            self.assertIn("claim_type", claim)
            self.assertIn("evidence_ids", claim)
            self.assertIn("rule_ids", claim)
            self.assertIn("source_ids", claim)
            self.assertIn("construct_id", claim)

    def test_verification_report_json_written_automatically(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        out = self._analyze(image)
        path = out / "verification_report.json"
        self.assertTrue(path.exists())
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("all_passed", doc)
        self.assertEqual(doc["claim_count"], len(doc["claims"]))

    def test_real_case_claims_all_pass_because_they_are_built_from_real_evidence(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        out = self._analyze(image)
        claims_doc = json.loads((out / "generated_claims.json").read_text(encoding="utf-8"))
        verification_doc = json.loads((out / "verification_report.json").read_text(encoding="utf-8"))
        # Every claim here is mechanically built FROM the case's own real
        # evidence, so it must always verify -- this is the honest baseline
        # proving the wiring works, not a case designed to fail.
        self.assertEqual(claims_doc["rejected_count"], 0)
        self.assertTrue(verification_doc["all_passed"])


class FailedClaimDowngradeTests(unittest.TestCase):
    """Seeds a deliberately invalid claim directly through the pipeline
    (bypassing the normally-safe structured_report.py construction) to
    prove the pipeline downgrades/removes it rather than displaying it."""

    def test_seeded_fabricated_evidence_id_claim_is_marked_rejected(self):
        registry_v2 = build_registry_v2()
        structured_analysis = {
            "individual_rule_suggestions": [{
                "rule_id": "PSY_AR_SIZE_FULL_015", "parent_safe_wording": "poisoned claim",
                "evidence_ids": ["ev_totally_fabricated"], "reference_ids": [], "target_construct": None,
            }],
            "combined_drawing_level_hypotheses": [],
        }
        analysis = {"evidence": [], "rule_evaluations": [
            {"rule_id": "PSY_AR_SIZE_FULL_015", "status": "weak_support"},
        ], "objective_features": {}}
        generated_claims, verification_report = build_claims_and_verification(structured_analysis, analysis, registry_v2)
        self.assertEqual(generated_claims["rejected_count"], 1)
        self.assertEqual(generated_claims["claims"][0]["status"], "rejected_fallback_to_observation_only")
        self.assertFalse(verification_report["all_passed"])

    def test_seeded_diagnostic_language_claim_is_marked_rejected(self):
        registry_v2 = build_registry_v2()
        structured_analysis = {
            "individual_rule_suggestions": [],
            "combined_drawing_level_hypotheses": [{
                "target_construct": "low_mood_or_emotional_distress_pattern",
                "allowed_wording": ["The child is depressed."],
                "supporting_texts": [],
                "contributing_evidence_ids": [], "contributing_rule_ids": [],
            }],
        }
        analysis = {"evidence": [], "rule_evaluations": [], "objective_features": {}}
        generated_claims, verification_report = build_claims_and_verification(structured_analysis, analysis, registry_v2)
        self.assertEqual(generated_claims["rejected_count"], 1)
        self.assertFalse(verification_report["all_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
