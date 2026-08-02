"""Regression tests for case-output versioning.

Found during the 2026-08-02 audit (CURRENT_STATE_AUDIT.md Section 6):
re-analyzing a case silently overwrote analysis.json/evidence.json/rules.json/
concerns.json/judges.json/emotion.json/detections.json with no history
retained. write_versioned() (case_output.py) fixes this by archiving the
prior content under <case>/versions/ before overwriting, without changing the
path any existing reader uses for the current version.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class WriteVersionedTests(unittest.TestCase):
    def test_first_write_creates_no_version_history(self):
        from doar.case_output import write_versioned
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.json"
            write_versioned(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})
            self.assertFalse((Path(tmp) / "versions").exists())

    def test_changed_rewrite_archives_previous_version(self):
        from doar.case_output import write_versioned
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.json"
            write_versioned(path, {"a": 1})
            write_versioned(path, {"a": 2})
            # Current file reflects the latest write.
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 2})
            # The previous version is preserved, not lost.
            versions_dir = Path(tmp) / "versions"
            archived = list(versions_dir.glob("analysis.v*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(json.loads(archived[0].read_text(encoding="utf-8")), {"a": 1})
            history = (versions_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(history), 1)

    def test_identical_rewrite_does_not_create_a_spurious_version(self):
        from doar.case_output import write_versioned
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.json"
            write_versioned(path, {"a": 1})
            write_versioned(path, {"a": 1})
            self.assertFalse((Path(tmp) / "versions").exists())

    def test_three_writes_produce_two_archived_versions_in_order(self):
        from doar.case_output import write_versioned
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.json"
            write_versioned(path, {"a": 1})
            write_versioned(path, {"a": 2})
            write_versioned(path, {"a": 3})
            archived = sorted((Path(tmp) / "versions").glob("analysis.v*.json"))
            self.assertEqual(len(archived), 2)
            self.assertEqual(json.loads(archived[0].read_text(encoding="utf-8")), {"a": 1})
            self.assertEqual(json.loads(archived[1].read_text(encoding="utf-8")), {"a": 2})


class AnalyzeImageReanalysisTests(unittest.TestCase):
    def test_reanalyzing_a_case_preserves_prior_analysis_json(self):
        import numpy as np
        from PIL import Image
        from doar.analysis import analyze_image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "drawing.png"
            Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(image_path)
            case_dir = Path(tmp) / "case"
            analyze_image(str(image_path), str(case_dir))
            first_run = (case_dir / "analysis.json").read_text(encoding="utf-8")
            analyze_image(str(image_path), str(case_dir))
            # A second run on an unchanged image typically reproduces
            # identical deterministic output, so no new version is forced --
            # but if segmentation/quality output ever differs run-to-run, the
            # prior version must be archived, never silently lost.
            second_run = (case_dir / "analysis.json").read_text(encoding="utf-8")
            if first_run != second_run:
                self.assertTrue((case_dir / "versions").exists())


if __name__ == "__main__":
    unittest.main()
