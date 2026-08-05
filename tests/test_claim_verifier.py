"""Tests for claim_verifier.py (DOAR-TRACE 4F). Seeds deliberately invalid
claims and proves the deterministic verifier catches each failure mode.
No LLM or free VLM report is needed to make a claim fail here -- every
seeded claim below is hand-constructed."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.claim_verifier import (
    Claim, verify_claim, verify_claims, verify_evidence_ids, verify_no_diagnostic_claims,
    verify_no_omitted_contradictions, verify_numeric_agreement, verify_rule_ids,
    verify_source_ids, verify_unavailable_wording,
)
from doar.registry_v2_build import build_registry_v2

REGISTRY_V2 = build_registry_v2()


def _real_case():
    temp = tempfile.TemporaryDirectory()
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
    path = Path(temp.name) / "image.png"
    image.save(path)
    out = Path(temp.name) / "out"
    result = analyze_image(path, out)
    return temp, result.to_dict()


class NumericAgreementTests(unittest.TestCase):
    def setUp(self):
        self.temp, self.analysis = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_correct_claimed_value_passes(self):
        real_value = next(e["value"] for e in self.analysis["evidence"] if e["evidence_id"] == "ev_bbox_coverage")
        claim = Claim(text="x", claimed_value=real_value, claimed_value_evidence_id="ev_bbox_coverage")
        result = verify_numeric_agreement(claim, self.analysis)
        self.assertTrue(result.passed)

    def test_seeded_wrong_numeric_value_is_caught(self):
        claim = Claim(text="The bounding box covers 12% of the page.",
                       claimed_value=0.12, claimed_value_evidence_id="ev_bbox_coverage")
        result = verify_numeric_agreement(claim, self.analysis)
        self.assertFalse(result.passed)
        self.assertIn("0.12", result.reason)

    def test_seeded_unknown_evidence_id_for_numeric_claim_is_caught(self):
        claim = Claim(text="x", claimed_value=0.5, claimed_value_evidence_id="ev_does_not_exist")
        result = verify_numeric_agreement(claim, self.analysis)
        self.assertFalse(result.passed)

    def test_no_claimed_value_trivially_passes(self):
        claim = Claim(text="A qualitative observation with no number.")
        self.assertTrue(verify_numeric_agreement(claim, self.analysis).passed)


class EvidenceAndRuleIdTests(unittest.TestCase):
    def setUp(self):
        self.temp, self.analysis = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_real_evidence_id_passes(self):
        claim = Claim(text="x", evidence_ids=["ev_bbox_coverage"])
        self.assertTrue(verify_evidence_ids(claim, self.analysis).passed)

    def test_seeded_fabricated_evidence_id_is_caught(self):
        claim = Claim(text="x", evidence_ids=["ev_totally_made_up_12345"])
        result = verify_evidence_ids(claim, self.analysis)
        self.assertFalse(result.passed)
        self.assertIn("ev_totally_made_up_12345", result.reason)

    def test_real_rule_id_passes(self):
        claim = Claim(text="x", rule_ids=["PSY_AR_SIZE_FULL_015"])
        self.assertTrue(verify_rule_ids(claim, self.analysis).passed)

    def test_seeded_fabricated_rule_id_is_caught(self):
        claim = Claim(text="x", rule_ids=["PSY_AR_NOT_A_REAL_RULE_999"])
        result = verify_rule_ids(claim, self.analysis)
        self.assertFalse(result.passed)

    def test_objective_feature_evidence_id_is_recognized(self):
        # Any real feature evidence_id from the 59 persisted features (4B) must count.
        feature_evidence_id = next(iter(self.analysis["objective_features"].values()))["evidence_id"]
        claim = Claim(text="x", evidence_ids=[feature_evidence_id])
        self.assertTrue(verify_evidence_ids(claim, self.analysis).passed)


class SourceIdTests(unittest.TestCase):
    def test_real_reference_id_passes(self):
        claim = Claim(text="x", source_ids=["REF_SIZE_2004"])
        self.assertTrue(verify_source_ids(claim, REGISTRY_V2).passed)

    def test_seeded_fabricated_source_id_is_caught(self):
        claim = Claim(text="x", source_ids=["REF_FAKE_STUDY_THAT_DOES_NOT_EXIST_2099"])
        result = verify_source_ids(claim, REGISTRY_V2)
        self.assertFalse(result.passed)
        self.assertIn("REF_FAKE_STUDY_THAT_DOES_NOT_EXIST_2099", result.reason)


class UnavailableWordingTests(unittest.TestCase):
    def setUp(self):
        self.temp, self.analysis = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_hedged_claim_about_missing_detector_rule_passes(self):
        claim = Claim(text="Eye style was not evaluated in this release; no detector exists.",
                       rule_ids=["PSY_AR_EYES_WIDE_001"])
        self.assertTrue(verify_unavailable_wording(claim, self.analysis).passed)

    def test_seeded_unhedged_claim_about_missing_detector_rule_is_caught(self):
        claim = Claim(text="The child's eyes are wide, indicating an outgoing personality.",
                       rule_ids=["PSY_AR_EYES_WIDE_001"])
        result = verify_unavailable_wording(claim, self.analysis)
        self.assertFalse(result.passed)
        self.assertIn("PSY_AR_EYES_WIDE_001", result.reason)

    def test_claim_about_an_executed_rule_needs_no_hedge(self):
        claim = Claim(text="The drawing covers nearly the full page.", rule_ids=["PSY_AR_SIZE_FULL_015"])
        self.assertTrue(verify_unavailable_wording(claim, self.analysis).passed)


class DiagnosticLanguageTests(unittest.TestCase):
    def test_safe_cautious_claim_passes(self):
        claim = Claim(text="Several features form a possible low-mood pattern. This is not a diagnosis.")
        self.assertTrue(verify_no_diagnostic_claims(claim).passed)

    def test_seeded_forbidden_english_claim_is_caught(self):
        for forbidden in (
            "The child is depressed.",
            "The child has anxiety.",
            "This drawing proves abuse.",
        ):
            with self.subTest(forbidden=forbidden):
                claim = Claim(text=forbidden)
                result = verify_no_diagnostic_claims(claim)
                self.assertFalse(result.passed, forbidden)

    def test_seeded_forbidden_arabic_claim_is_caught(self):
        claim = Claim(text="الطفل يعاني من اكتئاب.")
        result = verify_no_diagnostic_claims(claim)
        self.assertFalse(result.passed)


class OmittedContradictionTests(unittest.TestCase):
    def test_no_contradictions_trivially_passes(self):
        result = verify_no_omitted_contradictions([Claim(text="x")], {"cross_theme_contradictions": []})
        self.assertTrue(result.passed)

    def test_seeded_omitted_contradiction_is_caught(self):
        structured = {"cross_theme_contradictions": [{
            "construct_a": "introversion", "construct_b": "extraversion",
            "rule_ids_a": ["PSY_AR_PLACE_LEFT_018"], "rule_ids_b": ["PSY_AR_PLACE_RIGHT_019"],
        }]}
        claims = [Claim(text="The drawing is placed on the left.", rule_ids=["PSY_AR_PLACE_LEFT_018"])]
        result = verify_no_omitted_contradictions(claims, structured)
        self.assertFalse(result.passed)
        self.assertIn("introversion", result.reason)

    def test_contradiction_mentioned_in_claims_passes(self):
        structured = {"cross_theme_contradictions": [{
            "construct_a": "introversion", "construct_b": "extraversion",
            "rule_ids_a": ["PSY_AR_PLACE_LEFT_018"], "rule_ids_b": ["PSY_AR_PLACE_RIGHT_019"],
        }]}
        claims = [Claim(text="Both left and right placement patterns were observed, which conflict.",
                         rule_ids=["PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019"])]
        result = verify_no_omitted_contradictions(claims, structured)
        self.assertTrue(result.passed)


class VerifyClaimAndBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp, self.analysis = _real_case()
        self.addCleanup(self.temp.cleanup)

    def test_verify_claim_runs_all_checks_and_a_clean_claim_passes_everything(self):
        real_value = next(e["value"] for e in self.analysis["evidence"] if e["evidence_id"] == "ev_bbox_coverage")
        claim = Claim(
            text="The drawing's bounding box covers a large fraction of the page. This is not a diagnosis.",
            evidence_ids=["ev_bbox_coverage"], rule_ids=["PSY_AR_SIZE_FULL_015"],
            source_ids=["REF_GENERAL_LIMITS_1998"], claimed_value=real_value,
            claimed_value_evidence_id="ev_bbox_coverage",
        )
        report = verify_claim(claim, self.analysis, REGISTRY_V2)
        self.assertTrue(report.passed, report.failed_checks)
        self.assertEqual(len(report.checks), 6)

    def test_verify_claim_a_poisoned_claim_fails_at_least_one_check(self):
        claim = Claim(text="The child is depressed.", evidence_ids=["ev_fabricated"])
        report = verify_claim(claim, self.analysis, REGISTRY_V2)
        self.assertFalse(report.passed)
        failed_names = {c.check_name for c in report.failed_checks}
        self.assertIn("no_diagnostic_claims", failed_names)
        self.assertIn("valid_evidence_ids", failed_names)

    def test_verify_claims_batch_aggregates_correctly(self):
        good = Claim(text="Observation only, no diagnosis.", rule_ids=["PSY_AR_SIZE_FULL_015"])
        bad = Claim(text="The child has anxiety.")
        result = verify_claims([good, bad], self.analysis, REGISTRY_V2)
        self.assertFalse(result["all_passed"])
        self.assertTrue(result["per_claim"][0]["passed"])
        self.assertFalse(result["per_claim"][1]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
