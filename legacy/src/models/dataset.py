"""
dataset.py — PyTorch Dataset + transforms driven by the leak-safe split.csv.

Augmentation is applied to the TRAIN split only. Val/test use deterministic
resize + normalise. Class order is fixed (sorted) so label indices are stable
across runs.
"""

from __future__ import annotations
import csv
from collections import defaultdict


def load_split(split_csv: str):
    """Return (rows, classes, class_to_idx). rows = list of {path,class,split}."""
    with open(split_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({r["class"] for r in rows})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return rows, classes, class_to_idx


def class_weights(rows, class_to_idx, split="train"):
    """Inverse-frequency weights for weighted loss (imbalance handling)."""
    import torch
    counts = defaultdict(int)
    for r in rows:
        if r["split"] == split:
            counts[r["class"]] += 1
    total = sum(counts.values()) or 1
    n_cls = len(class_to_idx)
    w = [0.0] * n_cls
    for c, i in class_to_idx.items():
        w[i] = total / (n_cls * counts[c]) if counts[c] else 0.0
    return torch.tensor(w, dtype=torch.float32)


def build_transforms(img_size: int = 224, train: bool = False):
    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])


class DrawingDataset:
    """Torch Dataset over one split. Import torch lazily so the module loads
    even when torch is absent (dataset inspection doesn't need it)."""

    def __init__(self, rows, split, class_to_idx, img_size=224, train=False):
        from torch.utils.data import Dataset  # noqa
        self.samples = [(r["path"], class_to_idx[r["class"]])
                        for r in rows if r["split"] == split]
        self.tf = build_transforms(img_size, train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label, path


def make_torch_dataset(*args, **kwargs):
    """Factory that returns a real torch Dataset subclass instance."""
    from torch.utils.data import Dataset

    class _TD(Dataset, DrawingDataset):
        pass

    return _TD(*args, **kwargs)
