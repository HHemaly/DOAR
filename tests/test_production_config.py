"""DOAR V1.1 Stage 3: centralized production analysis configuration.

Proves: the config is resolved automatically (no model choice needed by
a caller), never invents a fallback checkpoint when the recommended one
is missing, and the resulting emotion.py call (checkpoint=None when
unavailable) correctly reaches the existing "unavailable" branch rather
than a raw exception or a fabricated result.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar import emotion  # noqa: E402
from doar.production_config import (  # noqa: E402
    EXPRESSIVE_MODEL_CHECKPOINT_PATH, EXPRESSIVE_MODEL_IDENTIFIER, ProductionAnalysisConfig,
    resolve_production_config,
)


class ResolveProductionConfigTests(unittest.TestCase):
    def test_returns_frozen_expressive_model_identifier(self):
        config = resolve_production_config()
        self.assertEqual(config.expressive_model_identifier, EXPRESSIVE_MODEL_IDENTIFIER)

    def test_available_flag_matches_real_filesystem_state(self):
        config = resolve_production_config()
        self.assertEqual(config.expressive_model_available, EXPRESSIVE_MODEL_CHECKPOINT_PATH.exists())

    def test_unavailable_checkpoint_is_none_not_a_dead_path(self):
        # Never pass a path to a file we already know doesn't exist --
        # checkpoint must be exactly None in that case (matches
        # emotion.py's own "no checkpoint supplied" contract).
        config = resolve_production_config()
        if not config.expressive_model_available:
            self.assertIsNone(config.expressive_model_checkpoint)
            self.assertIsNotNone(config.expressive_model_unavailable_reason)
            self.assertIn(str(EXPRESSIVE_MODEL_CHECKPOINT_PATH), config.expressive_model_unavailable_reason)

    def test_visual_detector_policy_path_is_the_frozen_phase2c7_artifact(self):
        config = resolve_production_config()
        self.assertIn("phase2c7", config.visual_detector_policy_path)
        self.assertIn("visual_detector_policy.json", config.visual_detector_policy_path)

    def test_to_dict_is_json_serializable(self):
        import json
        config = resolve_production_config()
        json.dumps(config.to_dict())  # must not raise

    def test_config_is_immutable(self):
        config = resolve_production_config()
        with self.assertRaises(Exception):
            config.expressive_model_identifier = "something_else"


class EmotionUnavailableBranchTests(unittest.TestCase):
    """Confirms the production config's None-when-missing contract feeds
    correctly into emotion.py's ALREADY-CORRECT unavailable() branch --
    never a raw exception, never fabricated probabilities."""

    def test_none_checkpoint_from_production_config_yields_unavailable_not_failed(self):
        config = ProductionAnalysisConfig(
            expressive_model_identifier="test_id", expressive_model_checkpoint=None,
            expressive_model_available=False, expressive_model_unavailable_reason="test reason",
            visual_detector_policy_path="test_path", visual_detector_policy_version="test_version")
        result = emotion.predict("fake_image.png", config.expressive_model_checkpoint)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["top_class"])
        self.assertEqual(result["probabilities"], {name: None for name in emotion.CLASSES})

    def test_never_fabricates_probabilities_when_unavailable(self):
        result = emotion.predict("fake_image.png", None)
        self.assertTrue(all(v is None for v in result["probabilities"].values()))
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["top_class"])


if __name__ == "__main__":
    unittest.main()
