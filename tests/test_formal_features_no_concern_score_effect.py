"""Phase G0 -- Step 9's CRITICAL REGRESSION TEST: computing/persisting the
new OBJECTIVE_ONLY formal-feature measurements must not change ANY
psychological output -- concern-domain support, evidence-family support,
consistency status, or the final synthesis text -- unless a later,
explicitly validated phase deliberately wires them in.

Two independent proofs:
1. STRUCTURAL guard -- case_interpretation.py (concern-domain aggregation)
   and reasoning_chain.py/drawing_synthesis.py (the rule engine) never
   import formal_features at all, mirroring
   tests/test_phase2c4_detectors.py's own "no forbidden import" pattern.
2. BEHAVIORAL guard -- build the SAME case's full CaseInterpretation both
   BEFORE formal_features.json exists and AFTER it has been computed and
   written to disk, and assert the two are byte-for-byte identical
   (via to_dict()), proving the aggregator's output cannot depend on
   whether the new artifact is present.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar import case_interpretation as ci
from doar import drawing_synthesis, reasoning_chain
from doar.case_artifacts import resolve_analysis_artifacts
from doar.case_output import write_versioned
from doar.formal_features import compute_formal_features, serialize_formal_features
from doar.live_case_bundle import build_live_case_bundle
from doar.timed_analysis import analyze_image_with_timing


class NoForbiddenImportTests(unittest.TestCase):
    """Mirrors test_phase2c4_detectors.py's
    test_no_real_model_weights_touched_by_this_test structural-guard
    pattern: read the SOURCE of the aggregation/rule-engine modules and
    assert they never mention formal_features."""

    def test_case_interpretation_never_imports_formal_features(self):
        source = inspect.getsource(ci)
        self.assertNotIn("formal_features", source)

    def test_reasoning_chain_never_imports_formal_features(self):
        source = inspect.getsource(reasoning_chain)
        self.assertNotIn("formal_features", source)

    def test_drawing_synthesis_never_imports_formal_features(self):
        source = inspect.getsource(drawing_synthesis)
        self.assertNotIn("formal_features", source)


class ConcernScoresUnchangedByFormalFeaturesTests(unittest.TestCase):
    def _build_case(self, tmp_dir: str) -> Path:
        case_dir = Path(tmp_dir) / "case"
        case_dir.mkdir(parents=True, exist_ok=True)
        path = case_dir / "drawing.png"
        image = Image.new("RGB", (300, 300), "white")
        draw = ImageDraw.Draw(image)
        # A drawing with real content across several existing rule
        # observables (light lines, small size, off-center placement) so
        # build_concern_domains has real, non-trivial support to compare.
        draw.line((120, 120, 180, 180), fill=(210, 210, 210), width=1)
        draw.ellipse((100, 100, 160, 160), outline="black", width=1)
        image.save(path)
        analyze_image_with_timing(str(path), str(case_dir), None)
        (case_dir / "detections.json").write_text(json.dumps({"entities": []}), encoding="utf-8")
        return case_dir

    def test_case_interpretation_identical_before_and_after_formal_features_computed(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d)

            bundle_before = build_live_case_bundle(case_dir)
            interp_before = ci.build_case_interpretation(bundle_before)
            before_dict = interp_before.to_dict()

            # Compute AND PERSIST formal_features.json -- the exact
            # artifact a "Compute formal measurements" UI action writes.
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            resolved = resolve_analysis_artifacts(analysis, case_dir)
            image_path = case_dir / Path(analysis["image_path"]).name
            formal_row = compute_formal_features(str(image_path), resolved)
            write_versioned(case_dir / "formal_features.json",
                             {"status": "available", "features": serialize_formal_features(formal_row)})
            self.assertTrue((case_dir / "formal_features.json").exists())

            bundle_after = build_live_case_bundle(case_dir)
            interp_after = ci.build_case_interpretation(bundle_after)
            after_dict = interp_after.to_dict()

            self.assertEqual(before_dict, after_dict,
                              "CaseInterpretation changed after formal_features.json was computed/persisted -- "
                              "the OBJECTIVE_ONLY formal-feature artifact must never affect concern-domain "
                              "aggregation, consistency, or synthesis.")

    def test_build_concern_domains_identical_before_and_after(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._build_case(d)
            bundle = build_live_case_bundle(case_dir)
            before = tuple(c.to_dict() for c in ci.build_concern_domains(bundle))

            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            resolved = resolve_analysis_artifacts(analysis, case_dir)
            image_path = case_dir / Path(analysis["image_path"]).name
            formal_row = compute_formal_features(str(image_path), resolved)
            write_versioned(case_dir / "formal_features.json",
                             {"status": "available", "features": serialize_formal_features(formal_row)})

            bundle_again = build_live_case_bundle(case_dir)
            after = tuple(c.to_dict() for c in ci.build_concern_domains(bundle_again))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
