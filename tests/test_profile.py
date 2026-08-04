from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.profile import ChildProfile, ALLOWED_AGE_RANGES, load_profile, save_profile


class ChildProfileValidationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        ChildProfile()  # must not raise

    def test_rejects_unknown_age_range(self):
        with self.assertRaises(ValueError):
            ChildProfile(age_range="thirty")

    def test_rejects_unknown_language(self):
        with self.assertRaises(ValueError):
            ChildProfile(language="fr")

    def test_all_allowed_age_ranges_accepted(self):
        for age in ALLOWED_AGE_RANGES:
            ChildProfile(age_range=age)  # must not raise

    def test_to_dict_tags_source_as_user_provided(self):
        profile = ChildProfile(age_range="4-5", parent_concern="withdrawn lately")
        data = profile.to_dict()
        self.assertEqual(data["source"], "user_provided")
        self.assertEqual(data["parent_concern"], "withdrawn lately")


class ChildProfilePersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            profile = ChildProfile(age_range="6-7", gender="female",
                                    drawing_instruction="draw your family",
                                    date="2026-08-03", parent_concern="none")
            save_profile(d, profile)
            reloaded = load_profile(d)
            self.assertEqual(reloaded, profile)

    def test_missing_profile_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_profile(d))

    def test_save_writes_versioned_json_not_raw_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            save_profile(d, ChildProfile(age_range="2-3"))
            save_profile(d, ChildProfile(age_range="8-9"))  # changed value
            reloaded = load_profile(d)
            self.assertEqual(reloaded.age_range, "8-9")
            # write_versioned archives the prior version on change
            self.assertTrue((Path(d) / "versions" / "history.jsonl").exists())


class ChildProfileIsolationFromInferenceTests(unittest.TestCase):
    """Regression guard: profile.py must never be importable from the
    modules that decide the emotion prediction or rule/concern outcome --
    child context must never influence those, only be displayed alongside
    them (see profile.py's own module docstring)."""

    def _imports(self, module_path: Path) -> set[str]:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
        return names

    def test_rules_module_does_not_import_profile(self):
        imports = self._imports(ROOT / "src" / "doar" / "rules.py")
        self.assertFalse(any("profile" in name for name in imports))

    def test_emotion_module_does_not_import_profile(self):
        imports = self._imports(ROOT / "src" / "doar" / "emotion.py")
        self.assertFalse(any("profile" in name for name in imports))

    def test_concerns_module_does_not_import_profile(self):
        imports = self._imports(ROOT / "src" / "doar" / "concerns.py")
        self.assertFalse(any("profile" in name for name in imports))


if __name__ == "__main__":
    unittest.main()
