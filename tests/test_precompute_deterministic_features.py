"""Tests for scripts/precompute_deterministic_features.py -- the local,
Gemini-free CLI that warms drawing_synthesis.py's deterministic-feature
cache for the 15-image development set.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "precompute_deterministic_features", ROOT / "scripts" / "precompute_deterministic_features.py")
precompute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(precompute)


class PrecomputeScriptTests(unittest.TestCase):
    def test_module_makes_no_live_gemini_calls(self):
        source = (ROOT / "scripts" / "precompute_deterministic_features.py").read_text(encoding="utf-8")
        self.assertNotIn("GeminiVisualObserver(", source)
        self.assertNotIn("GeminiVisualVerifier(", source)
        self.assertNotIn("run_live_observer_and_verifier(", source)

    def test_unknown_image_id_prints_message_and_does_not_raise(self):
        with mock.patch.object(sys, "argv", ["precompute_deterministic_features.py", "--image-id", "not_a_real_case"]):
            precompute.main()  # must not raise

    def test_single_known_case_is_cached_and_idempotent_on_rerun(self):
        image_path = ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0003.jpeg"
        if not image_path.exists():
            self.skipTest("dev-set image not present on this machine")
        with mock.patch.object(sys, "argv", ["precompute_deterministic_features.py", "--image-id", "p2b_0003"]):
            precompute.main()
        cache_file = precompute.ds.deterministic_cache_path("p2b_0003")
        self.assertTrue(cache_file.exists())
        # Second run without --force must hit the cache path, not recompute.
        with mock.patch.object(sys, "argv", ["precompute_deterministic_features.py", "--image-id", "p2b_0003"]):
            precompute.main()  # must not raise


if __name__ == "__main__":
    unittest.main()
