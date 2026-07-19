"""A4 — pretrained-weight resolution (CPU torchvision, no downloads)."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torchvision  # noqa
    _TV = True
except Exception:
    _TV = False


@unittest.skipUnless(_TV, "torchvision not installed")
class ResolveWeightsTests(unittest.TestCase):
    def test_default_resolves_to_enum_member(self):
        from doar.deep.registry import resolve_weights
        w, wid = resolve_weights("resnet18", "DEFAULT")
        self.assertIsNotNone(w)
        self.assertIn("ResNet18_Weights", wid)

    def test_scratch_and_none(self):
        from doar.deep.registry import resolve_weights
        for spec in ("none", "scratch", False, None, ""):
            w, wid = resolve_weights("resnet18", spec)
            self.assertIsNone(w)
            self.assertEqual(wid, "scratch")

    def test_explicit_member(self):
        from doar.deep.registry import resolve_weights
        w, wid = resolve_weights("resnet18", "IMAGENET1K_V1")
        self.assertIsNotNone(w)
        self.assertTrue(wid.endswith("IMAGENET1K_V1"))

    def test_invalid_member_raises_with_model_name(self):
        from doar.deep.registry import resolve_weights
        with self.assertRaises(ValueError) as ctx:
            resolve_weights("resnet18", "NOPE_V9")
        self.assertIn("resnet18", str(ctx.exception))

    def test_small_cnn_is_scratch(self):
        from doar.deep.registry import resolve_weights
        w, wid = resolve_weights("small_cnn", "DEFAULT")
        self.assertIsNone(w)
        self.assertEqual(wid, "scratch")

    def test_all_supported_models_resolve_default(self):
        from doar.deep.registry import resolve_weights
        for name in ("mobilenet_v3_small", "mobilenet_v3_large", "resnet18",
                     "resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"):
            w, wid = resolve_weights(name, "DEFAULT")
            self.assertIsNotNone(w, name)

    def test_preprocessing_derived_from_weights_object(self):
        from doar.deep.registry import resolve_weights
        from doar.deep.preprocessing import resolve_preprocessing, preprocessing_hash
        w, _ = resolve_weights("resnet18", "DEFAULT")
        spec = resolve_preprocessing("resnet18", 224, weights_object=w)
        self.assertEqual(spec["transform_source"], "torchvision_weights.transforms()")
        self.assertIsNotNone(spec["crop"])
        self.assertTrue(preprocessing_hash(spec))


if __name__ == "__main__":
    unittest.main(verbosity=2)
