"""DOAR V1 rule integration: the full missing connection, end to end --
visual_evidence.py::integrate_visual_findings_into_case, wired into
run_and_persist_initial_scan, reaching case_output.py's synthesis tail.

Uses a REAL case built via the real analyze_image pipeline (no mocking of
the objective/measurement side) with FAKE (no real model weights)
VisualFinding objects standing in for a real visual scan -- matching this
project's existing convention that model-loading code (`load_real_*`) is
never exercised by the test suite, while everything downstream of a
detection is fully real and fully tested.

Two registry scenarios are tested side by side, deliberately:
  - a SYNTHETIC registry with one enabled static_detector rule, to prove
    the wiring genuinely activates a rule, reaches aggregation/synthesis,
    and stays invisible to Parent View's raw internals -- when the gate
    IS open;
  - the REAL, unmodified rules_registry_v2.json, to prove today's actual
    production behavior is a correct, honest abstention -- the gate is
    closed for every static_detector rule, so nothing is fabricated.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image  # noqa: E402
from doar.expert_review import load_review, submit_review  # noqa: E402
from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.visual_evidence import (  # noqa: E402
    VisualFinding, integrate_visual_findings_into_case, load_detections, save_detections,
)
from doar.visual_qa import answer_with_visual_grounding  # noqa: E402


def _real_case(tmp_dir: str) -> Path:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image(path, case_dir)
    return case_dir


def _house_finding(*, validated=True) -> VisualFinding:
    return VisualFinding(
        label="house", finding_id="vf_test_house_001", free_form_label=None, bbox=(0.1, 0.2, 0.3, 0.2),
        confidence=0.9, detector="grounding_dino_object_classes",
        checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand.",
        validation_status=("VALIDATED" if validated else "EXPERIMENTAL"),
        evidence_status=("validated_evidence" if validated
                          else "experimental_evidence_technical_view_only"),
        rule_mapping_status="MAPPED", related_rule_ids=("EN_COMPILED_HOUSE_023",),
        source="initial_scan", query=None, timestamp="2026-08-09T00:00:00+00:00")


def _registry_with_house_enabled() -> dict:
    """A deep copy of the REAL, current rules_registry_v2.json with ONLY
    EN_COMPILED_HOUSE_023's `allowed_output_level` flipped to
    `individual_heuristic_only` (and a real-shaped confidence_ceiling
    filled in, since disabled rules carry None) -- every other field is
    the real, complete, curator-authored content. Simulates "what if a
    registry curator enabled this one rule" using real data everywhere
    except the one field this session is not authorized to flip in the
    actual committed registry file."""
    registry_v2 = copy.deepcopy(build_registry_v2())
    for rule in registry_v2["rules"]:
        if rule["rule_id"] == "EN_COMPILED_HOUSE_023":
            rule["allowed_output_level"] = "individual_heuristic_only"
            rule["confidence_ceiling"] = 0.2
    return registry_v2


class SyntheticGateOpenTests(unittest.TestCase):
    """Proves the mechanism is real -- a controlled scenario, clearly not
    a claim about current production behavior."""

    def test_validated_finding_produces_weak_support_rule_reaching_analysis_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            new_rows = integrate_visual_findings_into_case(
                case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            self.assertEqual(len(new_rows), 1)
            self.assertEqual(new_rows[0]["status"], "weak_support")
            self.assertEqual(new_rows[0]["rule_id"], "EN_COMPILED_HOUSE_023")

            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            triggered = [r for r in analysis["rule_evaluations"] if r.get("visual_evidence_sourced")]
            self.assertEqual(len(triggered), 1)
            self.assertEqual(triggered[0]["status"], "weak_support")
            self.assertEqual(triggered[0]["matched_evidence_ids"], ["ev_visual_vf_test_house_001"])

    def test_canonical_evidence_id_is_findable_in_evidence_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            evidence = json.loads((case_dir / "evidence.json").read_text(encoding="utf-8"))
            ids = {e["evidence_id"] for e in evidence}
            self.assertIn("ev_visual_vf_test_house_001", ids)

    def test_triggered_rule_reaches_structured_analysis_synthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            structured = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
            all_rule_ids = {s["rule_id"] for s in structured.get("individual_rule_suggestions", [])}
            for hyp in structured.get("combined_drawing_level_hypotheses", []):
                all_rule_ids.update(hyp.get("contributing_rule_ids", []))
            self.assertIn("EN_COMPILED_HOUSE_023", all_rule_ids)

    def test_parent_safe_content_present_with_no_technical_internals_leaked(self):
        # structured_analysis.json is a shared TECHNICAL artifact -- it
        # legitimately carries evidence_ids/bboxes elsewhere for Technical
        # View (Stage 6 explicitly wants that). The actual Stage 5 safety
        # boundary is narrower: the specific PARENT-FACING fields
        # doar_prototype_app.py's Parent View literally reads and displays
        # (parent_safe_wording / possible_interpretation / supporting_texts)
        # must never themselves contain detector/model/evidence-id/bbox
        # internals -- checked here directly against those fields, not the
        # whole document.
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            structured = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
            suggestion = next(s for s in structured["individual_rule_suggestions"]
                               if s["rule_id"] == "EN_COMPILED_HOUSE_023")
            parent_facing_text = " ".join([
                suggestion["parent_safe_wording"], suggestion["possible_interpretation"],
                suggestion["contextual_question"],
            ])
            self.assertIn("family life", parent_facing_text)
            for leaked in ("grounding_dino", "IDEA-Research", "vf_test_house_001",
                           "ev_visual_vf_test_house_001", "bbox", "0.9"):
                self.assertNotIn(leaked, parent_facing_text,
                                  f"{leaked!r} leaked into a Parent-View-facing field")

    def test_detections_json_and_clinician_review_untouched_by_resynthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            save_detections(case_dir, [_house_finding()])
            before_detections = (case_dir / "detections.json").read_text(encoding="utf-8")
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            after_detections = (case_dir / "detections.json").read_text(encoding="utf-8")
            self.assertEqual(before_detections, after_detections)

    def test_existing_submitted_review_status_survives_resynthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            submit_review(case_dir, reviewer_name="Dr. Test", action="confirm", target_label="house")
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            review = load_review(case_dir)
            self.assertEqual(review["status"], "submitted")
            self.assertEqual(len(review["history"]), 1)  # not reset, not duplicated
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["clinician_review"], "submitted")

    def test_case_reload_preserves_the_complete_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            save_detections(case_dir, [_house_finding()])
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            # Simulate a fresh reload: reread every artifact from disk.
            detections_reloaded = load_detections(case_dir)
            analysis_reloaded = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            structured_reloaded = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(len(detections_reloaded), 1)
            self.assertTrue(any(r.get("visual_evidence_sourced") for r in analysis_reloaded["rule_evaluations"]))
            all_ids = {s["rule_id"] for s in structured_reloaded.get("individual_rule_suggestions", [])}
            self.assertIn("EN_COMPILED_HOUSE_023", all_ids)

    def test_qa_can_explain_which_rule_used_which_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            resp = answer_with_visual_grounding(case_dir, "which rules were evaluated?", analysis)
            self.assertEqual(resp["source_module"], "rule_evaluations")
            self.assertIn("EN_COMPILED_HOUSE_023", resp["answer"])
            self.assertIn("ev_visual_vf_test_house_001", resp["evidence_ids"])

    def test_qa_evidence_question_surfaces_the_visual_evidence_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=_registry_with_house_enabled())
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            resp = answer_with_visual_grounding(case_dir, "what evidence exists?", analysis)
            self.assertEqual(resp["source_module"], "evidence")
            self.assertIn("ev_visual_vf_test_house_001", resp["evidence_ids"])


class SyntheticGateOpenSafetyTests(unittest.TestCase):
    def test_experimental_finding_never_triggers_even_an_enabled_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            new_rows = integrate_visual_findings_into_case(
                case_dir, [_house_finding(validated=False)], registry_v2=_registry_with_house_enabled())
            self.assertEqual(new_rows, [])

    def test_on_demand_search_path_never_calls_rule_integration(self):
        # search_visual is exercised in test_visual_evidence.py; here we
        # confirm the CONTRACT: only run_and_persist_initial_scan chains
        # into integrate_visual_findings_into_case. A case with only a
        # search_visual-style (on_demand_search) finding must never gain a
        # visual_evidence_sourced rule row simply from being persisted.
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            on_demand = VisualFinding(
                label="house", finding_id="vf_test_on_demand", free_form_label=None, bbox=None,
                confidence=0.9, detector="grounding_dino_open_query:test", checkpoint="test",
                prompt="house.", validation_status="VALIDATED", evidence_status="validated_evidence",
                rule_mapping_status="MAPPED", related_rule_ids=("EN_COMPILED_HOUSE_023",),
                source="on_demand_search", query="house", timestamp="2026-08-09T00:00:00+00:00")
            save_detections(case_dir, [on_demand])
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            # The old-registry engine (rules.py) already populates baseline
            # missing_detector rows for every case regardless -- what must
            # NEVER happen is a visual_evidence_sourced row appearing just
            # because save_detections() was called directly (bypassing
            # run_and_persist_initial_scan, which is the only real caller
            # of integrate_visual_findings_into_case).
            self.assertFalse(any(r.get("visual_evidence_sourced") for r in analysis.get("rule_evaluations", [])))
            self.assertFalse(any(e.get("kind") == "visual_detection" for e in analysis.get("evidence", [])))


class RealRegistryTodayIntegrationTests(unittest.TestCase):
    """The honest production answer, exercised through the FULL
    integration path (not just the rule-engine unit, see
    test_rule_engine_v2_visual.py for that) with the REAL, unmodified
    registry."""

    def test_validated_house_finding_correctly_abstains_with_real_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            registry_v2 = build_registry_v2()
            new_rows = integrate_visual_findings_into_case(case_dir, [_house_finding()], registry_v2=registry_v2)
            # A row IS produced (EN_COMPILED_HOUSE_023's observable really
            # does match this validated finding) -- but its status must
            # never be a fabricated "weak_support"; the registry's own
            # allowed_output_level="disabled" gate is respected, so the
            # honest outcome is "missing_detector", never a trigger.
            self.assertTrue(all(r["status"] != "weak_support" for r in new_rows))
            self.assertTrue(any(r["rule_id"] == "EN_COMPILED_HOUSE_023" and r["status"] == "missing_detector"
                                 for r in new_rows))
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertFalse(any(r.get("visual_evidence_sourced") and r["status"] == "weak_support"
                                  for r in analysis["rule_evaluations"]))
            # The finding still becomes canonical evidence -- full
            # technical traceability even though no rule triggered.
            self.assertTrue(any(e.get("kind") == "visual_detection" for e in analysis["evidence"]))

    def test_missing_analysis_json_is_a_safe_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "no_such_case"
            case_dir.mkdir()
            result = integrate_visual_findings_into_case(case_dir, [_house_finding()],
                                                           registry_v2=build_registry_v2())
            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
