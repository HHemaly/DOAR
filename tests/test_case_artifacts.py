"""DOAR V1.1 Problem A: case-relative artifact path resolution.

Regression coverage for a real bug: analysis.json stores artifact paths
case-relative (e.g. "artifacts/foreground_mask.png"); a consumer running
from a different cwd than the case directory (the Streamlit app's
process cwd is the repo root, not the case dir) must resolve them
against the case directory, never rely on Image.open()'s implicit
cwd-relative resolution.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.case_artifacts import resolve_analysis_artifacts, resolve_artifact_path  # noqa: E402


class ResolveArtifactPathTests(unittest.TestCase):
    def test_relative_path_resolved_against_case_dir(self):
        result = resolve_artifact_path("/case/dir", "artifacts/foreground_mask.png")
        self.assertEqual(Path(result).as_posix(), "/case/dir/artifacts/foreground_mask.png")

    def test_absolute_path_returned_unchanged(self):
        absolute = str(Path("/somewhere/else/mask.png"))
        result = resolve_artifact_path("/case/dir", absolute)
        self.assertEqual(str(result), absolute)

    def test_works_regardless_of_process_cwd(self):
        # The whole point of this function: it must not depend on
        # os.getcwd() at all -- only on the case_dir argument.
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            result = resolve_artifact_path("/case/dir", "artifacts/foreground_mask.png")
            self.assertEqual(Path(result).as_posix(), "/case/dir/artifacts/foreground_mask.png")
        finally:
            os.chdir(original_cwd)


class ResolveAnalysisArtifactsTests(unittest.TestCase):
    def test_resolves_flat_and_nested_artifact_paths(self):
        analysis = {
            "artifacts": {
                "foreground_mask": "artifacts/foreground_mask.png",
                "normalized_image": "artifacts/normalized.png",
                "candidate_masks": {"colour_distance": "artifacts/candidate_colour_distance.png"},
            },
            "other_field": "unchanged",
        }
        resolved = resolve_analysis_artifacts(analysis, "/case/dir")
        self.assertEqual(Path(resolved["artifacts"]["foreground_mask"]).as_posix(),
                          "/case/dir/artifacts/foreground_mask.png")
        self.assertEqual(Path(resolved["artifacts"]["candidate_masks"]["colour_distance"]).as_posix(),
                          "/case/dir/artifacts/candidate_colour_distance.png")
        self.assertEqual(resolved["other_field"], "unchanged")
        # Original dict must not be mutated.
        self.assertEqual(analysis["artifacts"]["foreground_mask"], "artifacts/foreground_mask.png")

    def test_missing_artifacts_key_is_a_no_op(self):
        analysis = {"other_field": "x"}
        self.assertEqual(resolve_analysis_artifacts(analysis, "/case/dir"), analysis)

    def test_empty_artifacts_dict_is_a_no_op(self):
        analysis = {"artifacts": {}}
        self.assertEqual(resolve_analysis_artifacts(analysis, "/case/dir"), analysis)


class RealFileResolutionTests(unittest.TestCase):
    """Proves the fix against a real file on disk, reproducing the exact
    failure mode: opening an artifact from a process whose cwd is NOT the
    case directory."""

    def test_foreground_mask_opens_correctly_from_a_different_cwd(self):
        import os

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            (case_dir / "artifacts").mkdir(parents=True)
            mask_path = case_dir / "artifacts" / "foreground_mask.png"
            Image.new("L", (10, 10), 0).save(mask_path)

            analysis = {"artifacts": {"foreground_mask": "artifacts/foreground_mask.png"}}
            resolved = resolve_analysis_artifacts(analysis, case_dir)

            original_cwd = os.getcwd()
            try:
                os.chdir(tempfile.gettempdir())  # simulate the app's cwd != case_dir
                # Unresolved path would fail here (reproduces the real bug):
                with self.assertRaises(FileNotFoundError):
                    Image.open(analysis["artifacts"]["foreground_mask"])
                # Resolved path succeeds regardless of cwd:
                img = Image.open(resolved["artifacts"]["foreground_mask"])
                img.load()
            finally:
                os.chdir(original_cwd)

    def test_reopening_a_case_from_a_fresh_process_still_resolves(self):
        # Simulates "reopen a previous case": only the saved analysis.json
        # and case_dir path are available, no in-memory state survives.
        import json

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            (case_dir / "artifacts").mkdir(parents=True)
            Image.new("L", (10, 10), 0).save(case_dir / "artifacts" / "foreground_mask.png")
            analysis_path = case_dir / "analysis.json"
            analysis_path.write_text(json.dumps(
                {"artifacts": {"foreground_mask": "artifacts/foreground_mask.png"}}), encoding="utf-8")

            # Fresh "reload": read analysis.json back from disk, resolve, open.
            reloaded = json.loads(analysis_path.read_text(encoding="utf-8"))
            resolved = resolve_analysis_artifacts(reloaded, case_dir)
            img = Image.open(resolved["artifacts"]["foreground_mask"])
            img.load()


if __name__ == "__main__":
    unittest.main()
