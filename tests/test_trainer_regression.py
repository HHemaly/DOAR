"""A2/A3 — gradient accumulation + optimizer/scheduler preservation (CPU torch).

Skips cleanly when torch is unavailable. Uses a tiny synthetic ImageFolder so it
runs on CPU quickly. Synthetic data — NOT real dataset results.
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


def _tiny_dataset(root, per_class=4, size=32):
    from PIL import Image
    import numpy as np
    for split in ("train", "valid"):
        for ci, cls in enumerate(("Angry", "Fear", "Happy", "Sad")):
            d = Path(root) / split / cls
            d.mkdir(parents=True)
            for i in range(per_class):
                arr = (np.random.RandomState(ci * 100 + i).rand(size, size, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{i}.png")


@unittest.skipUnless(_TORCH, "torch/torchvision not installed")
class GradAccumTests(unittest.TestCase):
    def _run(self, grad_accum_steps, batch_size=4):
        from doar.deep.trainers import train_image_model
        with tempfile.TemporaryDirectory() as d:
            _tiny_dataset(Path(d) / "data")
            out = Path(d) / "out"
            res = train_image_model(
                str(Path(d) / "data"), "small_cnn", str(out), seed=0, epochs=1,
                batch_size=batch_size, image_size=32, device="cpu", workers=0,
                freeze_epochs=0, grad_accum_steps=grad_accum_steps,
                pretrained_weights="none")
            return res["executed_config"]

    def test_accumulation_reduces_optimizer_steps_same_samples(self):
        # 16 train samples, batch 4 -> 4 batches.
        no_accum = self._run(grad_accum_steps=1)     # steps == 4
        accum2 = self._run(grad_accum_steps=2)        # steps == 2
        self.assertEqual(no_accum["optimizer_step_count"], 4)
        self.assertEqual(accum2["optimizer_step_count"], 2)   # fewer steps
        # same number of samples processed (effective batch differs)
        self.assertEqual(no_accum["physical_batch_size"], 4)
        self.assertEqual(accum2["effective_batch_size"], 8)

    def test_grad_accum_validation(self):
        with self.assertRaises(ValueError):
            self._run(grad_accum_steps=0)


@unittest.skipUnless(_TORCH, "torch/torchvision not installed")
class UnfreezeOptimizerTests(unittest.TestCase):
    def _run(self, optimizer, scheduler, head_lr=3e-4, backbone_lr=1e-4):
        from doar.deep.trainers import train_image_model
        with tempfile.TemporaryDirectory() as d:
            _tiny_dataset(Path(d) / "data")
            out = Path(d) / "out"
            # freeze_epochs=1 with epochs=2 forces the unfreeze transition.
            train_image_model(
                str(Path(d) / "data"), "resnet18", str(out), seed=0, epochs=2,
                batch_size=4, image_size=32, device="cpu", workers=0,
                freeze_epochs=1, optimizer_name=optimizer, scheduler_name=scheduler,
                head_learning_rate=head_lr, backbone_learning_rate=backbone_lr,
                pretrained_weights="none")
            import json
            return json.loads((out / "executed_config.json").read_text())

    def test_optimizers_and_schedulers_survive_unfreeze(self):
        for opt in ("adamw", "adam", "sgd"):
            for sched in ("reduce_on_plateau", "cosine", "none"):
                cfg = self._run(opt, sched)
                self.assertEqual(cfg["optimizer"], opt)
                self.assertEqual(cfg["scheduler"], sched)
                # differential LRs preserved in executed config
                self.assertNotEqual(cfg["head_learning_rate"], cfg["backbone_learning_rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
