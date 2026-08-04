"""Regression guards required by this task:
  - the Phase 7B human-review app stays separate from the new prototype
  - the new prototype never touches test-split guard bypass mechanics
  - the new prototype fails safely (clear error, no fabricated output)
    when required files/models are unavailable
"""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


class ReviewAppSeparationTests(unittest.TestCase):
    def test_phase7b_review_app_does_not_import_prototype_modules(self):
        imports = _imported_names(ROOT / "phase7b_review_app.py")
        forbidden = {"doar.chat", "doar.parent_view", "doar.profile", "doar.timed_analysis"}
        self.assertEqual(imports & forbidden, set())

    def test_prototype_app_does_not_import_phase7b_review_modules(self):
        imports = _imported_names(ROOT / "doar_prototype_app.py")
        self.assertNotIn("doar.human_review", imports)

    def test_prototype_app_does_not_import_partition_module(self):
        # analyze-image and the prototype have no business touching
        # duplicate-detection/partition logic at all.
        imports = _imported_names(ROOT / "doar_prototype_app.py")
        self.assertNotIn("doar.partition", imports)
        chat_imports = _imported_names(ROOT / "src" / "doar" / "chat.py")
        self.assertNotIn("doar.partition", chat_imports)


class NoTestSplitAccessTests(unittest.TestCase):
    """The prototype performs single-image inference only -- it has no
    concept of a dataset split at all, and must never gain one silently."""

    def test_prototype_source_never_references_test_unlock_mechanics(self):
        source = (ROOT / "doar_prototype_app.py").read_text(encoding="utf-8")
        for forbidden in ("unlock_test", "unlock-test", "confirm_final_evaluation",
                          "confirm-final-evaluation", "require_test_access"):
            self.assertNotIn(forbidden, source)

    def test_chat_and_related_modules_never_reference_test_unlock_mechanics(self):
        for rel in ("chat.py", "profile.py", "parent_view.py", "timed_analysis.py"):
            source = (ROOT / "src" / "doar" / rel).read_text(encoding="utf-8")
            for forbidden in ("unlock_test", "confirm_final_evaluation", "require_test_access"):
                self.assertNotIn(forbidden, source, f"{rel} unexpectedly references {forbidden!r}")

    def test_prototype_modules_never_read_a_manifest_or_split_column(self):
        for rel in ("chat.py", "profile.py", "parent_view.py", "timed_analysis.py"):
            source = (ROOT / "src" / "doar" / rel).read_text(encoding="utf-8")
            self.assertNotIn("manifest.csv", source)


class SafeFailureTests(unittest.TestCase):
    def test_chat_on_missing_case_raises_instead_of_fabricating(self):
        from doar.chat import respond_to_chat
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                respond_to_chat(Path(d) / "no_such_case", "hello", "en")

    def test_analyze_image_with_timing_propagates_real_errors(self):
        from doar.timed_analysis import analyze_image_with_timing
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                analyze_image_with_timing(Path(d) / "does_not_exist.png", Path(d) / "out")

    def test_profile_load_on_missing_file_returns_none_not_a_fake_profile(self):
        from doar.profile import load_profile
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_profile(d))

    def test_analyze_image_with_unavailable_checkpoint_reports_failed_not_silent(self):
        from doar.timed_analysis import analyze_image_with_timing
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image, ImageDraw
            path = Path(d) / "drawing.png"
            # Must pass the quality gate (a blank white image does not) so the
            # checkpoint-loading code path is actually reached.
            image = Image.new("RGB", (200, 200), "white")
            ImageDraw.Draw(image).rectangle((5, 5, 195, 195), fill="black")
            image.save(path)
            result, _ = analyze_image_with_timing(path, Path(d) / "out",
                                                    emotion_checkpoint=Path(d) / "no_such_checkpoint.pt")
            self.assertEqual(result.emotion["status"], "failed")


if __name__ == "__main__":
    unittest.main()
