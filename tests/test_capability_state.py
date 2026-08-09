"""DOAR V1.1 Stage 5: the canonical, per-case capability-state source
(judges.json["module_availability"]) and its consumers. Extends the
EXISTING mechanism (case_output.refresh_module_availability was already
built for detection/clinician_review in a prior phase) rather than
inventing a second, redundant schema.
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

from doar.analysis import analyze_image  # noqa: E402
from doar.parent_view import build_overall_result_summary  # noqa: E402


def _real_case(tmp_dir: str) -> Path:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image(path, case_dir)
    return case_dir


class ModuleAvailabilityShapeTests(unittest.TestCase):
    def test_new_case_has_the_full_canonical_key_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            availability = judges["module_availability"]
            for key in ("detection", "visual_detection", "open_world_search", "objective_features",
                        "expressive_model", "rules", "expert_review", "clinician_review"):
                self.assertIn(key, availability, key)

    def test_objective_features_not_unavailable_for_a_real_successful_case(self):
        # "partial" is the honest, expected outcome for this fixture (some
        # features legitimately require a loaded emotion checkpoint, which
        # this test doesn't supply) -- the meaningful assertion is that the
        # mechanism ran at all (never "unavailable" when objective_features
        # genuinely computed something).
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertIn(judges["module_availability"]["objective_features"], ("available", "partial"))

    def test_objective_features_unavailable_when_analysis_has_none(self):
        import doar.judges as judges_mod
        analysis = {
            "composition": {"foreground_coverage": 0.1, "empty_space_ratio": 0.9, "centroid_normalized": None},
            "segmentation": {"candidate_disagreement": 0.0},
            "evidence": [], "rule_evaluations": [], "safety_disclaimer": "not a diagnosis. غير تشخيصي",
            "quality": {"supported": True}, "emotion": {"status": "unavailable"}, "concerns": [],
            "objective_features": {},
        }
        result = judges_mod.run_judges(analysis)
        self.assertEqual(result["module_availability"]["objective_features"], "unavailable")

    def test_visual_detection_and_open_world_search_start_unavailable_before_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["visual_detection"], "unavailable")
            self.assertEqual(judges["module_availability"]["open_world_search"], "unavailable")

    def test_expressive_model_mirrors_emotion_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            availability = judges["module_availability"]
            self.assertEqual(availability["expressive_model"], availability["emotion_model"])

    def test_expert_review_mirrors_clinician_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            availability = judges["module_availability"]
            self.assertEqual(availability["expert_review"], availability["clinician_review"])


class RefreshAfterVisualScanTests(unittest.TestCase):
    def test_refresh_module_availability_sets_visual_detection_and_open_world_search(self):
        from doar.case_output import refresh_module_availability
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(json.dumps({
                "module_availability": {"detection": "unavailable", "visual_detection": "unavailable",
                                         "open_world_search": "unavailable"}}), encoding="utf-8")
            refresh_module_availability(case_dir, detection="available", visual_detection="available",
                                         open_world_search="available")
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["visual_detection"], "available")
            self.assertEqual(judges["module_availability"]["open_world_search"], "available")

    def test_expert_review_submission_sets_expert_review_key(self):
        from doar.case_output import refresh_module_availability
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(json.dumps({
                "module_availability": {"clinician_review": "not_submitted", "expert_review": "not_submitted"}}),
                encoding="utf-8")
            refresh_module_availability(case_dir, clinician_review="submitted", expert_review="submitted")
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["expert_review"], "submitted")


class OverallResultSummaryCapabilityAwareTests(unittest.TestCase):
    def test_no_capabilities_keeps_the_pre_visual_detection_wording(self):
        sentences = build_overall_result_summary({}, {}, "en")
        self.assertTrue(any("were not analyzed" in s for s in sentences))

    def test_visual_detection_available_uses_the_updated_honest_wording(self):
        sentences = build_overall_result_summary({}, {}, "en", capabilities={"visual_detection": "available"})
        self.assertTrue(any("automatically scanned for known objects" in s for s in sentences))
        self.assertFalse(any("were not analyzed" in s for s in sentences))

    def test_visual_detection_unavailable_keeps_the_honest_absence_wording(self):
        sentences = build_overall_result_summary({}, {}, "en", capabilities={"visual_detection": "unavailable"})
        self.assertTrue(any("were not analyzed" in s for s in sentences))

    def test_no_technical_internals_leak_into_either_wording(self):
        for capabilities in (None, {"visual_detection": "available"}, {"visual_detection": "unavailable"}):
            sentences = build_overall_result_summary({}, {}, "en", capabilities=capabilities)
            blob = " ".join(sentences)
            for leaked in ("grounding_dino", "checkpoint", "bbox", "confidence"):
                self.assertNotIn(leaked, blob.lower())


if __name__ == "__main__":
    unittest.main()
