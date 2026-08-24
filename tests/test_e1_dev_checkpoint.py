"""Focused tests for the E1-development emotion-checkpoint adapter
(src/doar/deep/e1_dev_checkpoint.py) -- demo-readiness task.

Skips gracefully (rather than failing) if the E1 checkpoint file isn't
present on the machine running the tests -- this adapter's whole purpose
is to use a local, git-ignored artifact under DOAR-work, which is not
expected to exist in every environment (e.g. CI)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.deep.e1_dev_checkpoint import (  # noqa: E402
    E1_DEV_CHECKPOINT_PATH, MODEL_VERSION, is_e1_dev_checkpoint, predict_e1_dev_checkpoint,
)

_CHECKPOINT_AVAILABLE = E1_DEV_CHECKPOINT_PATH.exists()


class IsE1DevCheckpointTests(unittest.TestCase):
    def test_e1_shaped_payload_is_recognized(self):
        payload = {"model_state": {}, "epoch": 3, "model_key": "mobilenet_v3_small", "val_macro_f1": 0.7}
        self.assertTrue(is_e1_dev_checkpoint(payload))

    def test_production_shaped_payload_is_not_misidentified(self):
        payload = {"model_state": {}, "classes": ("Angry", "Fear", "Happy", "Sad"), "model_name": "resnet18"}
        self.assertFalse(is_e1_dev_checkpoint(payload))

    def test_non_dict_payload_is_not_misidentified(self):
        self.assertFalse(is_e1_dev_checkpoint(["not", "a", "dict"]))


@unittest.skipUnless(_CHECKPOINT_AVAILABLE, "E1 dev checkpoint not present on this machine")
class PredictE1DevCheckpointTests(unittest.TestCase):
    def _sample_image(self) -> str:
        from PIL import Image
        import tempfile
        path = Path(tempfile.mkdtemp()) / "sample.png"
        Image.new("RGB", (300, 300), "white").save(path)
        return str(path)

    def test_checkpoint_file_is_never_modified(self):
        import hashlib
        before = hashlib.sha256(E1_DEV_CHECKPOINT_PATH.read_bytes()).hexdigest()
        before_mtime = E1_DEV_CHECKPOINT_PATH.stat().st_mtime
        import torch
        payload = torch.load(E1_DEV_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        predict_e1_dev_checkpoint(self._sample_image(), E1_DEV_CHECKPOINT_PATH, payload)
        after = hashlib.sha256(E1_DEV_CHECKPOINT_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(before_mtime, E1_DEV_CHECKPOINT_PATH.stat().st_mtime)

    def test_produces_real_probabilities_over_the_four_classes(self):
        import torch
        payload = torch.load(E1_DEV_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        result = predict_e1_dev_checkpoint(self._sample_image(), E1_DEV_CHECKPOINT_PATH, payload)
        self.assertEqual(result["status"], "available")
        self.assertEqual(set(result["probabilities"]), {"Angry", "Fear", "Happy", "Sad"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)
        self.assertEqual(result["model_version"], MODEL_VERSION)
        self.assertIn("E1_DEVELOPMENT", result["model_version"])
        self.assertIn("not the final thesis-selected model", result["development_model_disclaimer"].lower())

    def test_rejects_a_checkpoint_with_a_different_model_key(self):
        payload = {"model_state": {}, "epoch": 1, "model_key": "resnet18"}
        with self.assertRaises(ValueError):
            predict_e1_dev_checkpoint(self._sample_image(), E1_DEV_CHECKPOINT_PATH, payload)


class EmotionDispatchTests(unittest.TestCase):
    @unittest.skipUnless(_CHECKPOINT_AVAILABLE, "E1 dev checkpoint not present on this machine")
    def test_emotion_predict_dispatches_to_the_e1_adapter(self):
        from doar import emotion
        from PIL import Image
        import tempfile
        path = Path(tempfile.mkdtemp()) / "sample.png"
        Image.new("RGB", (300, 300), "white").save(path)
        result = emotion.predict(str(path), str(E1_DEV_CHECKPOINT_PATH))
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["model_version"], MODEL_VERSION)

    def test_a_normal_production_checkpoint_shape_is_unaffected(self):
        """Sanity check on the detection logic only (no real production
        checkpoint is expected to exist on this machine) -- confirms
        is_e1_dev_checkpoint() would correctly defer to the existing
        deep.inference.predict_image() path for a real one."""
        production_shaped = {"model_state": {}, "classes": ("Angry", "Fear", "Happy", "Sad"),
                             "model_name": "efficientnet_b0", "image_size": 224,
                             "preprocessing_version": "torchvision_weights_derived"}
        self.assertFalse(is_e1_dev_checkpoint(production_shaped))


if __name__ == "__main__":
    unittest.main()
