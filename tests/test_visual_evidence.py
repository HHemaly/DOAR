"""DOAR MVP: visual_evidence.py -- the broad visual-object evidence store.

All tests use fake, injected predict_fns (never real model weights),
matching the project-wide injectable-backend pattern. Exercises: status
mapping (VALIDATED/EXPERIMENTAL/UNKNOWN/DISABLED_FOR_RULES never blurred),
rule-mapping (curated policy links + text-heuristic fallback, and that the
heuristic never upgrades trust), absence-is-not-evidence, persistence
round-trip, on-demand search, and the rule-evidence trace.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7.detector_policy import EXPERIMENTAL_AUTOMATIC, build_eye_policy_entry  # noqa: E402
from doar.visual_evidence import (  # noqa: E402
    VisualFinding, build_finding, build_rule_evidence_trace, compute_rule_mapping, find_matching,
    load_detections, run_and_persist_initial_scan, run_initial_visual_scan, save_detections, search_visual,
)

GD_OBJECT_RESULTS = {
    "person": (True, 0.9, None), "face": (True, 0.8, None), "hand": (False, 0.0, None),
    "tree": (True, 0.7, None), "house": (False, 0.0, None), "animal": (True, 0.6, None),
    "star": (False, 0.0, None),
}
OWL_OBJECT_RESULTS = {"heart": (True, 0.99, None)}
GD_PARTS_RESULTS = {"mouth": (True, 0.5, None)}
EYE_COMBO_RESULTS = {"eye": (True, 0.4, (0.1, 0.1, 0.2, 0.2))}

FAKE_REGISTRY_V2 = {"rules": [
    {"rule_id": "TEST_RULE_FLOWER_001", "observable": "a bright flower in the garden",
     "faithful_source_quote": "flower symbol"},
]}


def _eye_entry(status=EXPERIMENTAL_AUTOMATIC):
    return build_eye_policy_entry(
        status=status, best_model="grounding_dino_parts+owlv2_parts_fallback",
        best_model_checkpoint="ckpt", prompt="eye. mouth.", threshold=0.25,
        precision=0.88, recall=0.73, balanced_accuracy=0.57, n_ground_truth_present=128,
        localization_validated=False, rationale="test")


def _open_vocab_query(hit_target: str, *, detected=True, confidence=0.55):
    def query(image_path: str, target: str) -> list[tuple]:
        if target == hit_target:
            return [(detected, confidence, None, "fake_open_vocab", "ckpt", target)]
        return []
    return query


def _predict_fns(*, with_open_vocab_hit="flower"):
    return {
        "grounding_dino_object_classes": lambda p: GD_OBJECT_RESULTS,
        "owlv2_object_classes": lambda p: OWL_OBJECT_RESULTS,
        "grounding_dino_parts": lambda p: GD_PARTS_RESULTS,
        "grounding_dino_parts+owlv2_parts_fallback": lambda p: EYE_COMBO_RESULTS,
        "open_vocab_query": _open_vocab_query(with_open_vocab_hit),
    }


class ComputeRuleMappingTests(unittest.TestCase):
    def test_policy_target_uses_curated_related_rule_ids(self):
        from doar.phase2c7.detector_policy import OBJECT_CLASS_POLICY
        status, ids = compute_rule_mapping("face", FAKE_REGISTRY_V2, OBJECT_CLASS_POLICY)
        self.assertEqual(status, "MAPPED")
        self.assertEqual(ids, ("EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038",))

    def test_policy_target_with_no_related_rules_is_unmapped(self):
        from doar.phase2c7.detector_policy import OBJECT_CLASS_POLICY
        status, ids = compute_rule_mapping("person", FAKE_REGISTRY_V2, OBJECT_CLASS_POLICY)
        self.assertEqual(status, "UNMAPPED")
        self.assertEqual(ids, ())

    def test_non_policy_label_falls_back_to_text_heuristic_match(self):
        status, ids = compute_rule_mapping("flower", FAKE_REGISTRY_V2, {})
        self.assertEqual(status, "MAPPED")
        self.assertEqual(ids, ("TEST_RULE_FLOWER_001",))

    def test_non_policy_label_with_no_heuristic_match_is_unmapped(self):
        status, ids = compute_rule_mapping("bicycle", FAKE_REGISTRY_V2, {})
        self.assertEqual(status, "UNMAPPED")
        self.assertEqual(ids, ())

    def test_short_words_are_ignored_by_the_heuristic(self):
        # "sun" is 3 letters -- below the 4-char heuristic floor -- so it
        # can never spuriously match via short, generic word overlap.
        status, ids = compute_rule_mapping("sun", FAKE_REGISTRY_V2, {})
        self.assertEqual(status, "UNMAPPED")
        self.assertEqual(ids, ())


class BuildFindingTests(unittest.TestCase):
    def test_returns_none_for_non_detection(self):
        from doar.phase2c7.detector_policy import OBJECT_CLASS_POLICY
        f = build_finding("hand", detected=False, confidence=0.0, bbox=None, model="m",
                           checkpoint="c", prompt="p", policy=OBJECT_CLASS_POLICY, registry_v2={})
        self.assertIsNone(f)

    def test_validated_automatic_maps_to_validated_evidence(self):
        from doar.phase2c7.detector_policy import OBJECT_CLASS_POLICY
        f = build_finding("person", detected=True, confidence=0.9, bbox=None, model="m",
                           checkpoint="c", prompt="p", policy=OBJECT_CLASS_POLICY, registry_v2={})
        self.assertEqual(f.validation_status, "VALIDATED")
        self.assertEqual(f.evidence_status, "validated_evidence")
        self.assertIsNone(f.free_form_label)

    def test_experimental_automatic_never_becomes_validated_evidence(self):
        from doar.phase2c7.detector_policy import OBJECT_CLASS_POLICY
        f = build_finding("heart", detected=True, confidence=0.99, bbox=None, model="m",
                           checkpoint="c", prompt="p", policy=OBJECT_CLASS_POLICY, registry_v2={})
        self.assertEqual(f.validation_status, "EXPERIMENTAL")
        self.assertEqual(f.evidence_status, "experimental_evidence_technical_view_only")

    def test_label_outside_policy_is_unknown_not_validated(self):
        f = build_finding("flower", detected=True, confidence=0.5, bbox=None, model="m",
                           checkpoint="c", prompt="p", policy={}, registry_v2=FAKE_REGISTRY_V2)
        self.assertEqual(f.validation_status, "UNKNOWN")
        self.assertEqual(f.evidence_status, "experimental_evidence_technical_view_only")
        self.assertEqual(f.free_form_label, "flower")


class RunInitialVisualScanTests(unittest.TestCase):
    def test_produces_findings_only_for_real_detections(self):
        findings = run_initial_visual_scan(
            "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
            model_predict_fns=_predict_fns())
        labels = {f.label for f in findings}
        # Detected: person, face, tree, animal (GD object classes), heart
        # (OWLv2), mouth (GD parts), eye (combo), flower (open-vocab extra).
        self.assertEqual(labels, {"person", "face", "tree", "animal", "heart", "mouth", "eye", "flower"})
        # Never a finding for something that wasn't detected.
        self.assertNotIn("hand", labels)
        self.assertNotIn("house", labels)
        self.assertNotIn("star", labels)
        # DISABLED targets (circle, vehicle) are never even dispatched.
        self.assertNotIn("circle", labels)
        self.assertNotIn("vehicle", labels)

    def test_extra_target_gets_unknown_status(self):
        findings = run_initial_visual_scan(
            "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
            model_predict_fns=_predict_fns())
        flower = next(f for f in findings if f.label == "flower")
        self.assertEqual(flower.validation_status, "UNKNOWN")
        self.assertEqual(flower.source, "initial_scan")

    def test_works_with_no_open_vocab_query_fn_injected(self):
        fns = _predict_fns()
        del fns["open_vocab_query"]
        findings = run_initial_visual_scan(
            "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2, model_predict_fns=fns)
        self.assertNotIn("flower", {f.label for f in findings})
        self.assertIn("eye", {f.label for f in findings})


class PersistenceRoundTripTests(unittest.TestCase):
    def test_save_and_load_detections_round_trips_exactly(self):
        findings = run_initial_visual_scan(
            "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
            model_predict_fns=_predict_fns())
        with tempfile.TemporaryDirectory() as tmp:
            save_detections(tmp, findings)
            loaded = load_detections(tmp)
            self.assertEqual(loaded, findings)

    def test_load_detections_on_missing_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_detections(tmp), [])

    def test_run_and_persist_writes_detections_and_refreshes_judges(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(
                json.dumps({"module_availability": {"detection": "unavailable"}}), encoding="utf-8")
            findings = run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns())
            self.assertTrue(findings)
            doc = json.loads((case_dir / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["status"], "available")
            self.assertEqual(doc["n_findings"], len(findings))
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["detection"], "available")

    def test_run_and_persist_is_a_no_op_on_judges_when_judges_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns())
            self.assertFalse((case_dir / "judges.json").exists())
            self.assertTrue((case_dir / "detections.json").exists())


class FindMatchingTests(unittest.TestCase):
    def _finding(self, label, free_form_label=None):
        return VisualFinding(
            label=label, free_form_label=free_form_label, bbox=None, confidence=0.5, detector="m",
            checkpoint="c", prompt="p", validation_status="UNKNOWN",
            evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
            related_rule_ids=(), source="initial_scan", query=None, timestamp="t")

    def test_exact_and_case_insensitive_match(self):
        findings = [self._finding("Dog")]
        self.assertEqual(len(find_matching(findings, "dog")), 1)

    def test_substring_match_on_free_form_label(self):
        findings = [self._finding("weapon_or_weapon_like_object", free_form_label="pointy knife")]
        self.assertEqual(len(find_matching(findings, "knife")), 1)

    def test_no_match_returns_empty(self):
        findings = [self._finding("cat")]
        self.assertEqual(find_matching(findings, "dog"), [])


class SearchVisualTests(unittest.TestCase):
    def test_found_hit_is_persisted_and_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "image.png").write_bytes(b"\x89PNG\r\n")
            finding = search_visual(case_dir, "bicycle", registry_v2=FAKE_REGISTRY_V2,
                                     open_vocab_predict_fn=_open_vocab_query("bicycle"))
            self.assertIsNotNone(finding)
            self.assertEqual(finding.source, "on_demand_search")
            self.assertEqual(finding.query, "bicycle")
            self.assertEqual(finding.validation_status, "UNKNOWN")
            loaded = load_detections(case_dir)
            self.assertEqual(loaded, [finding])

    def test_no_hit_returns_none_and_does_not_write_a_negative_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "image.png").write_bytes(b"\x89PNG\r\n")
            finding = search_visual(case_dir, "unicorn", registry_v2=FAKE_REGISTRY_V2,
                                     open_vocab_predict_fn=_open_vocab_query("bicycle"))
            self.assertIsNone(finding)
            self.assertEqual(load_detections(case_dir), [])

    def test_no_image_in_case_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            finding = search_visual(tmp, "bicycle", registry_v2=FAKE_REGISTRY_V2,
                                     open_vocab_predict_fn=_open_vocab_query("bicycle"))
            self.assertIsNone(finding)


class BuildRuleEvidenceTraceTests(unittest.TestCase):
    def test_one_row_per_rule_id_finding_pair(self):
        validated = VisualFinding(
            label="face", free_form_label=None, bbox=None, confidence=0.8, detector="m", checkpoint="c",
            prompt="p", validation_status="VALIDATED", evidence_status="validated_evidence",
            rule_mapping_status="MAPPED", related_rule_ids=("RULE_A",), source="initial_scan",
            query=None, timestamp="t")
        experimental = VisualFinding(
            label="heart", free_form_label=None, bbox=None, confidence=0.9, detector="m", checkpoint="c",
            prompt="p", validation_status="EXPERIMENTAL",
            evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="MAPPED",
            related_rule_ids=("RULE_B",), source="initial_scan", query=None, timestamp="t")
        trace = build_rule_evidence_trace([validated, experimental])
        by_rule = {row["rule_id"]: row for row in trace}
        self.assertTrue(by_rule["RULE_A"]["can_activate"])
        self.assertFalse(by_rule["RULE_B"]["can_activate"])

    def test_finding_with_no_related_rules_contributes_no_row(self):
        f = VisualFinding(
            label="person", free_form_label=None, bbox=None, confidence=0.9, detector="m", checkpoint="c",
            prompt="p", validation_status="VALIDATED", evidence_status="validated_evidence",
            rule_mapping_status="UNMAPPED", related_rule_ids=(), source="initial_scan", query=None,
            timestamp="t")
        self.assertEqual(build_rule_evidence_trace([f]), [])


if __name__ == "__main__":
    unittest.main()
