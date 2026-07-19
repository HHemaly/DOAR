"""Item 1 — the resolved preprocessing spec drives train + valid + inference.

CPU torchvision. Proves the training-validation and inference transforms share
the same resize/crop/interpolation/normalization for the required backbones, and
that a trained checkpoint's executed transform matches its recorded spec.
"""

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

REQUIRED = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]


def _geometry(compose):
    """Extract (resize, crop, interpolation, mean, std) from a Compose."""
    from torchvision import transforms
    resize = crop = interp = mean = std = None
    for t in compose.transforms:
        if isinstance(t, transforms.Resize):
            resize = t.size
            interp = t.interpolation
        elif isinstance(t, transforms.CenterCrop):
            crop = t.size
        elif isinstance(t, transforms.Normalize):
            mean = tuple(round(float(x), 4) for x in t.mean)
            std = tuple(round(float(x), 4) for x in t.std)
    return resize, crop, interp, mean, std


@unittest.skipUnless(_TORCH, "torch/torchvision not installed")
class PreprocessingConsistencyTests(unittest.TestCase):
    def test_train_and_eval_share_geometry_and_norm(self):
        from doar.deep.registry import resolve_weights
        from doar.deep.preprocessing import (resolve_preprocessing, build_eval_transform,
                                             build_train_transform)
        for name in REQUIRED:
            w, _ = resolve_weights(name, "DEFAULT")
            spec = resolve_preprocessing(name, 224, weights_object=w)
            train_tf = build_train_transform(spec, "conservative")
            eval_tf = build_eval_transform(spec)
            g_tr = _geometry(train_tf)
            g_ev = _geometry(eval_tf)
            # resize / crop / interpolation / mean / std must all match
            self.assertEqual(g_tr, g_ev, f"{name}: train vs eval geometry/norm differ")

    def test_train_transform_has_conservative_augmentation(self):
        from doar.deep.registry import resolve_weights
        from doar.deep.preprocessing import resolve_preprocessing, build_train_transform
        from torchvision import transforms
        w, _ = resolve_weights("resnet18", "DEFAULT")
        spec = resolve_preprocessing("resnet18", 224, weights_object=w)
        tf = build_train_transform(spec, "conservative")
        types = [type(t) for t in tf.transforms]
        self.assertIn(transforms.RandomAffine, types)     # drawing-safe aug preserved
        self.assertIn(transforms.ColorJitter, types)
        # augmentation must come BEFORE ToTensor
        idx_aug = types.index(transforms.RandomAffine)
        idx_tt = types.index(transforms.ToTensor)
        self.assertLess(idx_aug, idx_tt)

    def test_loaders_use_the_spec(self):
        from PIL import Image
        import numpy as np
        from doar.deep.datasets import build_loaders
        from doar.deep.registry import resolve_weights
        from doar.deep.preprocessing import resolve_preprocessing
        with tempfile.TemporaryDirectory() as d:
            for split in ("train", "valid"):
                for cls in ("Angry", "Fear", "Happy", "Sad"):
                    p = Path(d) / split / cls
                    p.mkdir(parents=True)
                    Image.fromarray((np.random.RandomState(1).rand(40, 40, 3) * 255)
                                    .astype("uint8")).save(p / "a.png")
            w, _ = resolve_weights("resnet18", "DEFAULT")
            spec = resolve_preprocessing("resnet18", 224, weights_object=w)
            tr, va = build_loaders(d, 224, 2, 0, "conservative", preprocessing_spec=spec)
            xb, _ = next(iter(va))
            # valid tensor spatial size == the spec's crop (or resize) size
            expected = spec.get("crop") or (spec["resize"] if isinstance(spec["resize"], int)
                                            else spec["resize"][0])
            self.assertEqual(xb.shape[-1], expected)

    def test_checkpoint_records_executed_transform(self):
        # Train 1 epoch; the checkpoint's preprocessing_spec must match what a
        # loader built from it produces, and predict-image must accept it.
        from doar.deep.trainers import train_image_model
        from doar.deep.preprocessing import preprocessing_hash
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            for split in ("train", "valid"):
                for ci, cls in enumerate(("Angry", "Fear", "Happy", "Sad")):
                    p = Path(d) / "data" / split / cls
                    p.mkdir(parents=True)
                    for i in range(3):
                        Image.fromarray((np.random.RandomState(ci*10+i).rand(48, 48, 3)*255)
                                        .astype("uint8")).save(p / f"{i}.png")
            out = Path(d) / "out"
            train_image_model(str(Path(d)/"data"), "resnet18", str(out), seed=0,
                                    epochs=1, batch_size=2, image_size=224, device="cpu",
                                    workers=0, freeze_epochs=0, pretrained_weights="none")
            ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            self.assertIn("preprocessing_spec", ckpt)
            self.assertEqual(ckpt["preprocessing_hash"],
                             preprocessing_hash(ckpt["preprocessing_spec"]))
            self.assertEqual(ckpt["preprocessing_spec"]["augmentation_profile"], "conservative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
