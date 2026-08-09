"""DOAR V1 rule integration: the VisualFinding -> canonical Evidence
adapter (visual_evidence.py). Proves provenance survives conversion and
that `rule_eligible` (the ONE trust channel the rule engine checks) is
set correctly per validation_status -- never upgraded because a detection
happened to be confident."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.schemas import Evidence  # noqa: E402
from doar.visual_evidence import (  # noqa: E402
    VisualFinding, rule_eligible_visual_evidence, visual_finding_to_evidence, visual_findings_to_evidence,
)


def _finding(label, *, validation_status, evidence_status, bbox=(0.1, 0.2, 0.3, 0.4), confidence=0.87,
             detector="grounding_dino_object_classes", checkpoint="IDEA-Research/grounding-dino-tiny",
             prompt="person. face. hand.", source="initial_scan", query=None,
             related_rule_ids=("EN_COMPILED_HOUSE_023",), rule_mapping_status="MAPPED"):
    return VisualFinding(
        label=label, finding_id=f"vf_test_{label}_{validation_status}", free_form_label=None, bbox=bbox,
        confidence=confidence, detector=detector, checkpoint=checkpoint, prompt=prompt,
        validation_status=validation_status, evidence_status=evidence_status,
        rule_mapping_status=rule_mapping_status, related_rule_ids=related_rule_ids, source=source,
        query=query, timestamp="2026-08-09T00:00:00+00:00")


class VisualFindingToEvidenceTests(unittest.TestCase):
    def test_returns_canonical_evidence_dataclass(self):
        f = _finding("house", validation_status="VALIDATED", evidence_status="validated_evidence")
        ev = visual_finding_to_evidence(f)
        self.assertIsInstance(ev, Evidence)

    def test_provenance_fully_preserved(self):
        f = _finding("house", validation_status="VALIDATED", evidence_status="validated_evidence")
        ev = visual_finding_to_evidence(f)
        self.assertEqual(ev.value["label"], "house")
        self.assertEqual(ev.value["bbox"], (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(ev.confidence, 0.87)
        self.assertIn("grounding_dino_object_classes", ev.method)
        self.assertIn("IDEA-Research/grounding-dino-tiny", ev.method)
        self.assertEqual(ev.value["prompt"], "person. face. hand.")
        self.assertEqual(ev.value["source"], "initial_scan")
        self.assertEqual(ev.value["source_finding_id"], f.finding_id)
        self.assertEqual(ev.value["validation_status"], "VALIDATED")
        self.assertEqual(ev.value["related_rule_ids"], ["EN_COMPILED_HOUSE_023"])
        self.assertEqual(ev.evidence_id, f"ev_visual_{f.finding_id}")

    def test_on_demand_query_text_preserved(self):
        f = _finding("kite", validation_status="UNKNOWN",
                      evidence_status="experimental_evidence_technical_view_only",
                      source="on_demand_search", query="kite", related_rule_ids=())
        ev = visual_finding_to_evidence(f)
        self.assertEqual(ev.value["query"], "kite")
        self.assertEqual(ev.value["source"], "on_demand_search")

    def test_validated_finding_is_rule_eligible(self):
        f = _finding("house", validation_status="VALIDATED", evidence_status="validated_evidence")
        ev = visual_finding_to_evidence(f)
        self.assertTrue(ev.value["rule_eligible"])
        self.assertEqual(ev.limitations, [])

    def test_experimental_finding_is_never_rule_eligible(self):
        f = _finding("heart", validation_status="EXPERIMENTAL",
                      evidence_status="experimental_evidence_technical_view_only")
        ev = visual_finding_to_evidence(f)
        self.assertFalse(ev.value["rule_eligible"])
        self.assertTrue(any("must not activate a psychological rule" in lim for lim in ev.limitations))

    def test_unknown_finding_is_never_rule_eligible(self):
        f = _finding("sun", validation_status="UNKNOWN",
                      evidence_status="experimental_evidence_technical_view_only")
        ev = visual_finding_to_evidence(f)
        self.assertFalse(ev.value["rule_eligible"])

    def test_disabled_finding_is_never_rule_eligible(self):
        f = _finding("circle", validation_status="DISABLED_FOR_RULES", evidence_status="not_used")
        ev = visual_finding_to_evidence(f)
        self.assertFalse(ev.value["rule_eligible"])
        self.assertTrue(any("must not enter rule reasoning" in lim for lim in ev.limitations))

    def test_confidence_alone_never_upgrades_eligibility(self):
        # A very high-confidence EXPERIMENTAL detection must still be
        # ineligible -- confidence is not a proxy for validation status.
        f = _finding("heart", validation_status="EXPERIMENTAL",
                      evidence_status="experimental_evidence_technical_view_only", confidence=0.999)
        ev = visual_finding_to_evidence(f)
        self.assertFalse(ev.value["rule_eligible"])


class VisualFindingsToEvidenceTests(unittest.TestCase):
    def test_converts_every_finding_regardless_of_status(self):
        findings = [
            _finding("house", validation_status="VALIDATED", evidence_status="validated_evidence"),
            _finding("heart", validation_status="EXPERIMENTAL",
                     evidence_status="experimental_evidence_technical_view_only"),
            _finding("sun", validation_status="UNKNOWN",
                     evidence_status="experimental_evidence_technical_view_only"),
            _finding("circle", validation_status="DISABLED_FOR_RULES", evidence_status="not_used"),
        ]
        evidence = visual_findings_to_evidence(findings)
        self.assertEqual(len(evidence), 4)
        self.assertEqual({e.value["label"] for e in evidence}, {"house", "heart", "sun", "circle"})


class RuleEligibleVisualEvidenceTests(unittest.TestCase):
    def test_filters_to_validated_only(self):
        findings = [
            _finding("house", validation_status="VALIDATED", evidence_status="validated_evidence"),
            _finding("heart", validation_status="EXPERIMENTAL",
                     evidence_status="experimental_evidence_technical_view_only"),
            _finding("sun", validation_status="UNKNOWN",
                     evidence_status="experimental_evidence_technical_view_only"),
            _finding("circle", validation_status="DISABLED_FOR_RULES", evidence_status="not_used"),
        ]
        evidence = visual_findings_to_evidence(findings)
        eligible = rule_eligible_visual_evidence(evidence)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].value["label"], "house")

    def test_ignores_non_visual_evidence(self):
        other = Evidence(evidence_id="ev_bbox_coverage", kind="measurement", value=0.4,
                          method="bounding_box", confidence=1.0)
        self.assertEqual(rule_eligible_visual_evidence([other]), [])


if __name__ == "__main__":
    unittest.main()
