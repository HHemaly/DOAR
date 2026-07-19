from __future__ import annotations

from pathlib import Path

from ..dataset import CLASSES
from .augmentations import build_transforms


def build_loaders(
    dataset: str | Path, image_size: int, batch_size: int, workers: int,
    augmentation: str, weighted_sampler: bool = False, preprocessing_spec: dict | None = None,
):
    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    root = Path(dataset)
    # Item 1: when a resolved preprocessing spec is supplied, BOTH loaders use it
    # (validation uses build_eval_transform(spec) exactly; training uses the same
    # geometry + normalization with augmentation inserted). Fall back to the
    # generic transform only when no spec is given (backward compatible).
    if preprocessing_spec is not None:
        from .preprocessing import build_eval_transform, build_train_transform
        train_tf = build_train_transform(preprocessing_spec, augmentation)
        valid_tf = build_eval_transform(preprocessing_spec)
    else:
        train_tf = build_transforms(image_size, True, augmentation)
        valid_tf = build_transforms(image_size, False)
    train = ImageFolder(root / "train", train_tf)
    valid = ImageFolder(root / "valid", valid_tf)
    expected = {name: index for index, name in enumerate(CLASSES)}
    if train.class_to_idx != expected or valid.class_to_idx != expected:
        raise ValueError(f"Class mapping must be {expected}; got {train.class_to_idx}")
    sampler = None
    shuffle = True
    if weighted_sampler:
        counts = torch.bincount(torch.tensor(train.targets), minlength=len(CLASSES)).float()
        weights = (1 / counts.clamp_min(1))[torch.tensor(train.targets)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=workers, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
    )
    return train_loader, valid_loader
