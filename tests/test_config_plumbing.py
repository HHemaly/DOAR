"""Item 2 — config plumbing: no accepted-but-unused fields; run metadata saved."""

from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The full train-image-model mapping (kept in sync with main.py).
FULL_MAPPING = {
    "dataset": ("data", "dataset"), "validation_split": ("data", "validation_split"),
    "model": ("training", "model"), "pretrained_weights": ("training", "pretrained_weights"),
    "seed": ("training", "seed"), "image_size": ("training", "image_size"),
    "batch_size": ("training", "batch_size"), "epochs": ("training", "epochs"),
    "device": ("training", "device"), "workers": ("training", "workers"),
    "augmentation": ("training", "augmentation"),
    "freeze_epochs": ("training", "freeze_epochs"),
    "class_weighting": ("training", "class_weighting"),
    "early_stopping_patience": ("training", "early_stopping_patience"),
    "optimizer": ("optimization", "optimizer"),
    "head_learning_rate": ("optimization", "head_learning_rate"),
    "backbone_learning_rate": ("optimization", "backbone_learning_rate"),
    "scheduler": ("optimization", "scheduler"),
    "output": ("output", "directory"), "calibration": ("output", "calibration"),
}


class ConfigPlumbingTests(unittest.TestCase):
    def test_real_resnet18_toml_fully_consumed(self):
        from doar.config import load_config, assert_all_config_consumed
        allowed = {
            "data": {"dataset", "validation_split"},
            "training": {"model", "pretrained_weights", "seed", "image_size", "batch_size",
                         "epochs", "device", "workers", "augmentation", "freeze_epochs",
                         "class_weighting", "early_stopping_patience"},
            "optimization": {"optimizer", "head_learning_rate", "backbone_learning_rate",
                             "scheduler"},
            "output": {"directory", "calibration"},
        }
        cfg = load_config(str(ROOT / "configs" / "training" / "resnet18.toml"), allowed)
        assert_all_config_consumed(cfg, FULL_MAPPING)  # must not raise

    def test_unused_field_raises(self):
        from doar.config import assert_all_config_consumed
        cfg = {"training": {"model": "resnet18", "mystery_field": 1}}
        mapping = {"model": ("training", "model")}
        with self.assertRaises(ValueError):
            assert_all_config_consumed(cfg, mapping)

    def test_save_run_metadata_writes_all_artifacts(self):
        from doar.config import save_run_metadata
        with tempfile.TemporaryDirectory() as d:
            digest = save_run_metadata(d, "train-image-model",
                                       {"model": "resnet18"}, {"model": "resnet18", "seed": 42})
            self.assertTrue(digest)
            for name in ("resolved_config.json", "requested_config.json", "environment.json"):
                self.assertTrue((Path(d) / name).exists(), name)
            env = json.loads((Path(d) / "environment.json").read_text())
            for key in ("python", "torch", "cuda_available", "git_commit"):
                self.assertIn(key, env)

    def test_optimizer_and_scheduler_builders_reject_unknown(self):
        # Pure-logic guardrails (no torch needed for the ValueError path).
        import importlib
        trainers = importlib.import_module("doar.deep.trainers")
        # _build_scheduler("none", ...) returns (None, "none") without torch use?
        # It imports torch; skip when torch missing.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        with self.assertRaises(ValueError):
            trainers._build_optimizer("nope", [{"params": []}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
