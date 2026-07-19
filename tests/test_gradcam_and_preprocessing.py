"""A5/A6 — Grad-CAM CPU smoke (real tiny checkpoint) + preprocessing consistency."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa
    import torchvision  # noqa
    _TORCH = True
except Exception:
    _TORCH = False


def _tiny_checkpoint(path, model_name="small_cnn", image_size=32):
    from doar.deep.registry import build_model
    from doar.dataset import CLASSES
    from doar.deep.preprocessing import resolve_preprocessing, preprocessing_hash
    model = build_model(model_name, len(CLASSES), pretrained=False)
    spec = resolve_preprocessing(model_name, image_size)
    torch.save({"model_state": model.state_dict(), "model_name": model_name,
                "classes": CLASSES, "image_size": image_size,
                "preprocessing_spec": spec, "preprocessing_hash": preprocessing_hash(spec)},
               path)


def _img(path, size=32):
    from PIL import Image
    import numpy as np
    Image.fromarray((np.random.RandomState(0).rand(size, size, 3) * 255).astype("uint8")).save(path)


class GradCamLayerTests(unittest.TestCase):
    def test_vit_is_explicitly_unsupported(self):
        from doar.explain.gradcam import target_layer_name
        with self.assertRaises(ValueError):
            target_layer_name("vit_b_16")


@unittest.skipUnless(_TORCH, "torch not installed")
class GradCamSmokeTests(unittest.TestCase):
    def test_small_cnn_gradcam_produces_heatmap_and_overlay(self):
        from doar.explain.gradcam import generate_gradcam
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "m.pt"
            _tiny_checkpoint(ckpt)
            img = Path(d) / "x.png"
            _img(img)
            res = generate_gradcam(str(img), str(ckpt), Path(d) / "cam", device="cpu")
            self.assertEqual(res["attribution_type"], "visual_classifier_attention")
            self.assertTrue(Path(res["raw_heatmap"]).exists())
            self.assertIn("not causal", res["disclaimer"].lower())

    def test_resnet18_gradcam_layer_resolves(self):
        from doar.explain.gradcam import _resolve_layer
        from doar.deep.registry import build_model
        model = build_model("resnet18", 4, pretrained=False)
        layer = _resolve_layer(model, "resnet18")
        self.assertIsNotNone(layer)


@unittest.skipUnless(_TORCH, "torch not installed")
class PreprocessingConsistencyTests(unittest.TestCase):
    def test_finetuned_small_cnn_penultimate_extraction(self):
        # A5: small_cnn (nn.Sequential) must not use model.classifier[-1].
        from doar.deep.embeddings import _finetuned_extractor
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "m.pt"
            _tiny_checkpoint(ckpt, "small_cnn")
            model, image_size, meta = _finetuned_extractor(str(ckpt), "cpu")
            img = torch.randn(1, 3, 32, 32)
            out = model(img)
            self.assertEqual(out.shape[0], 1)
            self.assertGreater(out.shape[1], 4)   # penultimate width > n_classes
            self.assertIn("checkpoint_hash", meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
