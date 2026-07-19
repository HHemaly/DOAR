"""Item 2 — the `ml` extra must NOT pull PyTorch; `deep` provides it."""

from __future__ import annotations
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ExtrasSeparationTests(unittest.TestCase):
    def setUp(self):
        self.extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"]["optional-dependencies"]

    def test_ml_extra_has_no_torch(self):
        ml = " ".join(self.extras["ml"]).lower()
        self.assertNotIn("torch", ml)
        self.assertIn("scikit-learn", ml)

    def test_deep_extra_has_torch(self):
        deep = " ".join(self.extras["deep"]).lower()
        self.assertIn("torch", deep)
        self.assertIn("torchvision", deep)

    def test_setup_script_installs_ml_cv_dev_without_torch(self):
        import re
        script = (ROOT / "scripts" / "setup_windows.ps1").read_text(encoding="utf-8")
        self.assertIn('".[ml,cv,dev]"', script)
        # No EXECUTED line (starting with '&') may install torch — torch install
        # appears only as user GUIDANCE inside the help message.
        executed_torch = [ln for ln in script.splitlines()
                          if re.match(r"^\s*&.*pip install\s+torch", ln)]
        self.assertEqual(executed_torch, [], f"torch is executed, not guided: {executed_torch}")
        # It must, however, still instruct the user via the official selector.
        self.assertIn("pytorch.org/get-started/locally", script)
        self.assertIn("nvidia-smi", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
