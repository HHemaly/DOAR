"""Tests for this session's new densenet121 registry/embedding support."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: F401
    _TORCH = True
except ImportError:
    _TORCH = False


@unittest.skipUnless(_TORCH, "torch/torchvision not installed")
class DenseNet121RegistryTests(unittest.TestCase):
    def test_registered_in_model_names(self):
        from doar.deep import MODEL_NAMES
        self.assertIn("densenet121", MODEL_NAMES)

    def test_build_model_forward_pass(self):
        from doar.deep.registry import build_model
        import torch as _torch
        model = build_model("densenet121", classes=4, pretrained=False)
        out = model(_torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 4))

    def test_classifier_is_plain_linear_not_sequential(self):
        from doar.deep.registry import build_model
        import torch.nn as nn
        model = build_model("densenet121", classes=4, pretrained=False)
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.out_features, 4)

    def test_freeze_then_unfreeze(self):
        from doar.deep.registry import build_model, freeze_backbone, unfreeze_all
        model = build_model("densenet121", classes=4, pretrained=False)
        freeze_backbone(model)
        trainable = sum(p.requires_grad for p in model.parameters())
        total = sum(1 for _ in model.parameters())
        self.assertLess(trainable, total)
        self.assertGreater(trainable, 0)  # classifier stays trainable
        unfreeze_all(model)
        self.assertEqual(sum(p.requires_grad for p in model.parameters()), total)

    def test_resolve_weights_default_and_scratch(self):
        from doar.deep.registry import resolve_weights
        weights, resolved_id = resolve_weights("densenet121", "DEFAULT")
        self.assertIsNotNone(weights)
        self.assertIn("DenseNet121_Weights", resolved_id)
        weights_none, resolved_none = resolve_weights("densenet121", "none")
        self.assertIsNone(weights_none)
        self.assertEqual(resolved_none, "scratch")

    def test_finetuned_embedding_extraction_penultimate_layer(self):
        import torch as _torch
        from doar.deep.registry import build_model
        from doar.deep.embeddings import _finetuned_extractor
        with tempfile.TemporaryDirectory() as d:
            model = build_model("densenet121", classes=4, pretrained=False)
            ckpt_path = Path(d) / "fake.pt"
            _torch.save({
                "model_name": "densenet121", "classes": ("Angry", "Fear", "Happy", "Sad"),
                "model_state": model.state_dict(), "image_size": 224,
                "preprocessing_version": "torchvision_weights_derived",
            }, ckpt_path)
            extractor, image_size, meta = _finetuned_extractor(str(ckpt_path), "cpu")
            with _torch.no_grad():
                out = extractor(_torch.randn(2, 3, 224, 224))
            self.assertEqual(tuple(out.shape), (2, 1024))  # DenseNet121's real penultimate width
            self.assertEqual(meta["source_model_name"], "densenet121")

    def test_unknown_model_still_rejected(self):
        from doar.deep.registry import build_model
        with self.assertRaises(ValueError):
            build_model("not_a_real_model", classes=4)


if __name__ == "__main__":
    unittest.main()
