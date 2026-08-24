"""Milestone 2 focused tests -- three-tab restructure, simplified Parent
View, Psychologist View, Visual Consistency Judge (AUDIT ONLY), the
strengthened response-Judge checklist, and psychologist feedback
persistence.

No live API/network access is required anywhere in this file -- every
Gemini-shaped provider used here is a stub; `GeminiVisualConsistencyJudge`
and `GeminiJudge` are only constructed to inspect their prompt/gate logic,
never called over the network.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

from doar import human_interaction as hi
from doar.live_case_bundle import build_live_case_bundle
from doar.psychologist_feedback import export_all_feedback, load_feedback, submit_feedback
from doar.timed_analysis import analyze_image_with_timing


def _entity_dict(entity_id: str, canonical_label: str, status: str) -> dict:
    return {
        "entity_id": entity_id, "entity_type": "object", "canonical_label": canonical_label,
        "candidate_labels": [[canonical_label, 0.9]], "aliases_en": [], "aliases_ar": [],
        "broader_categories": [], "possible_subtypes": [], "visual_similarities": [],
        "bbox": None, "crop_ref": None, "dominant_colors": None, "relative_size": None,
        "page_position": None, "shape_features": None, "line_features": None,
        "detector": "test-detector", "checkpoint": "n/a", "prompt": canonical_label,
        "confidence": 0.9, "model_validation_status": "EXPERIMENTAL",
        "case_verification_status": status, "evidence_status": "present",
        "rule_mapping_status": "unmapped", "related_rule_ids": [], "source": "initial_scan",
        "query": None, "timestamp": "2026-08-19T00:00:00",
    }


def _build_case(tmp_dir: str) -> Path:
    case_dir = Path(tmp_dir) / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "drawing.png"
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    image.save(path)
    analyze_image_with_timing(str(path), str(case_dir), None)
    return case_dir


class _StubVisualConsistencyJudge:
    def audit(self, image_path, structured_observations):
        return {
            "supported_observations": ["a filled rectangle"],
            "likely_missed_observations": [],
            "disputed_observations": [],
            "unavailable_checks": [],
            "overall_agreement": "high",
        }


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ThreeTabRenderTests(unittest.TestCase):
    def test_english_renders_three_tabs_without_exception(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.session_state["language"] = "en"
            at.run(timeout=90)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            self.assertEqual(len(at.tabs), 3)
            headers = " ".join(h.value for h in at.header) + " ".join(s.value for s in at.subheader)
            self.assertIn("Psychologist feedback", headers)

    def test_arabic_renders_without_exception_and_is_rtl(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.session_state["language"] = "ar"
            at.run(timeout=90)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            markdown_html = " ".join(m.value for m in at.markdown)
            self.assertIn('dir="rtl"', markdown_html)

    def test_parent_view_has_no_rule_id_jargon(self):
        """Milestone 2 Section 3: Parent View must never show raw rule IDs
        (e.g. EN_COMPILED_*/PSY_AR_*) -- that detail belongs only in the
        Psychologist/Technical views."""
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.session_state["language"] = "en"
            at.run(timeout=90)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            parent_tab = at.tabs[0]
            parent_text = " ".join(m.value for m in parent_tab.markdown)
            self.assertNotIn("EN_COMPILED_", parent_text)
            self.assertNotIn("PSY_AR_", parent_text)


class VisualConsistencyJudgeSeparationTests(unittest.TestCase):
    """Requirement 7: the Visual Consistency Judge is AUDIT ONLY and can
    never automatically become case evidence."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="doar_m2_")
        case_dir = Path(self._tmp) / "case"
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / "drawing.png"
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
        image.save(path)
        analyze_image_with_timing(str(path), str(case_dir), None)
        (case_dir / "detections.json").write_text(
            json.dumps({"entities": [_entity_dict("e-sun", "sun", "verified")]}), encoding="utf-8")
        self.bundle = build_live_case_bundle(case_dir)

    def test_audit_runs_offline_with_a_stub_judge(self):
        # The stub judge never actually opens the image, so any truthy
        # relative_path is enough -- this test's case_dir lives in a
        # tempdir outside ROOT, where build_live_case_bundle correctly
        # sets relative_path=None (see test_live_case_bundle.py); that's
        # the right adapter behavior, not something to fake around except
        # for this narrow, stub-only check.
        bundle = {**self.bundle, "image": {**self.bundle["image"], "relative_path": "drawing.png"}}
        result = hi.run_visual_consistency_audit(bundle, judge=_StubVisualConsistencyJudge())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["audit"]["overall_agreement"], "high")

    def test_audit_unavailable_with_no_judge_configured(self):
        result = hi.run_visual_consistency_audit(self.bundle, judge=None)
        self.assertEqual(result["status"], "unavailable")

    def test_visual_audit_is_never_wired_into_answer_question(self):
        """The audit function is never called from inside answer_question's
        own category answerers -- structurally, an audit finding cannot
        reach a StructuredAnswer unless a future change explicitly wires it
        in (and deterministic_verify would then require it to be labeled;
        see the next test)."""
        sa = hi.answer_question("What did you find in this drawing?", self.bundle)
        self.assertNotIn("AUDIT", sa.answer.upper().replace("AUDIT ONLY", ""))
        for claim in sa.claims:
            self.assertNotIn("visual_audit", str(claim))

    def test_deterministic_verify_blocks_unlabeled_visual_audit_usage(self):
        """Forward-looking guard (checklist item 10): if some future
        answerer sets used_visual_audit=True without including the
        AUDIT_ONLY label in the answer text, deterministic_verify must
        fail it -- mirrors the existing external-research separation
        check."""
        result = {"answer": "The drawing shows a sun.", "evidence_ids": [], "rule_ids": [], "source_ids": [],
                  "used_visual_audit": True}
        verification = hi.deterministic_verify(result, self.bundle)
        self.assertFalse(verification["all_passed"])
        self.assertIn("visual consistency audit used but not labeled as a separate, audit-only finding",
                      verification["extra_checks_failed"])

    def test_labeled_visual_audit_usage_passes(self):
        result = {"answer": f"The drawing shows a sun. {hi._VISUAL_AUDIT_LABEL}: extra context.",
                  "evidence_ids": [], "rule_ids": [], "source_ids": [], "used_visual_audit": True}
        verification = hi.deterministic_verify(result, self.bundle)
        self.assertNotIn("visual consistency audit used but not labeled as a separate, audit-only finding",
                          verification["extra_checks_failed"])


class JudgeChecklistTests(unittest.TestCase):
    """Requirement 8: the fixed DOAR checklist is present in the semantic
    Judge's system prompt, and GeminiJudge never overrides a deterministic
    failure (construction only -- no network call is made)."""

    def test_system_prompt_covers_the_fixed_checklist(self):
        prompt = hi._JUDGE_SYSTEM_PROMPT
        for phrase in (
            "Evidence grounding", "Verified/uncertain/rejected fidelity", "Missing detector != absence",
            "Page-assessability gates", "Rule eligibility/prerequisites", "Source validity",
            "No diagnosis/causal invention", "No single weak observation escalated",
            "External research separated", "Visual-audit findings separated",
            "Alternatives/limitations preserved", "directly answer the question",
        ):
            self.assertIn(phrase, prompt)

    def test_gemini_judge_never_calls_out_on_a_hard_deterministic_failure(self):
        judge = hi.GeminiJudge(api_key="dummy-test-key-not-a-real-secret")
        result = {"answer": "This confirms the child has anxiety, definitely.", "evidence_ids": [],
                  "rule_ids": [], "source_ids": [], "used_external_research": False}
        verification = {"all_passed": False, "per_claim": [], "extra_checks_failed": ["unsupported diagnostic language in answer text"]}
        out = judge.judge(question="What did you find?", result=result, verification=verification, package={})
        self.assertEqual(out["judge_mode"], "gemini+deterministic_gate")
        self.assertNotEqual(out["verdict"], "PASS")


class PsychologistFeedbackPersistenceTests(unittest.TestCase):
    def test_submit_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            submit_feedback(case_dir, reviewer_name="Dr. Test", verdict="partially_agree", comment="Reasonable overall.")
            loaded = load_feedback(case_dir)
            self.assertEqual(len(loaded["entries"]), 1)
            self.assertEqual(loaded["entries"][0]["verdict"], "partially_agree")
            self.assertEqual(loaded["entries"][0]["reviewer_name"], "Dr. Test")

    def test_invalid_verdict_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            with self.assertRaises(ValueError):
                submit_feedback(case_dir, reviewer_name="Dr. Test", verdict="strongly_agree")

    def test_export_all_feedback_aggregates_across_cases(self):
        with tempfile.TemporaryDirectory() as d:
            cases_dir = Path(d)
            for name, verdict in (("case_a", "agree"), ("case_b", "disagree")):
                case_dir = cases_dir / name
                case_dir.mkdir()
                submit_feedback(case_dir, reviewer_name="Dr. Test", verdict=verdict)
            exported = export_all_feedback(cases_dir)
            self.assertEqual(len(exported), 2)
            self.assertEqual({e["case_id"] for e in exported}, {"case_a", "case_b"})


class CategorizedPsychologistFeedbackTests(unittest.TestCase):
    """Requirement (research-runtime task): the psychologist can rate
    overall/emotion/concern-domain interpretation and explanation
    usefulness separately, each Agree/Partially agree/Disagree/Cannot
    assess -- the smallest additive extension to the append-only schema."""

    def test_submit_categorized_and_load_round_trips(self):
        from doar.psychologist_feedback import FEEDBACK_CATEGORIES, submit_categorized_feedback
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            ratings = {cat: "agree" for cat in FEEDBACK_CATEGORIES}
            ratings["concern_domain_interpretation"] = "partially_agree"
            submit_categorized_feedback(case_dir, reviewer_name="Dr. Test", ratings=ratings, comment="Good detail.")
            loaded = load_feedback(case_dir)
            self.assertEqual(len(loaded["entries"]), 1)
            self.assertEqual(loaded["entries"][0]["ratings"]["concern_domain_interpretation"], "partially_agree")
            self.assertEqual(loaded["entries"][0]["ratings"]["overall_interpretation"], "agree")

    def test_unknown_category_is_rejected(self):
        from doar.psychologist_feedback import submit_categorized_feedback
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            with self.assertRaises(ValueError):
                submit_categorized_feedback(case_dir, reviewer_name="Dr. Test",
                                            ratings={"not_a_real_category": "agree"})

    def test_unknown_verdict_within_a_category_is_rejected(self):
        from doar.psychologist_feedback import submit_categorized_feedback
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            with self.assertRaises(ValueError):
                submit_categorized_feedback(case_dir, reviewer_name="Dr. Test",
                                            ratings={"overall_interpretation": "strongly_agree"})

    def test_legacy_and_categorized_entries_coexist_append_only(self):
        from doar.psychologist_feedback import submit_categorized_feedback
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            case_dir.mkdir()
            submit_feedback(case_dir, reviewer_name="Dr. A", verdict="agree")
            submit_categorized_feedback(case_dir, reviewer_name="Dr. B",
                                        ratings={"overall_interpretation": "disagree"})
            loaded = load_feedback(case_dir)
            self.assertEqual(len(loaded["entries"]), 2)
            self.assertIn("verdict", loaded["entries"][0])
            self.assertIn("ratings", loaded["entries"][1])


if __name__ == "__main__":
    unittest.main()
