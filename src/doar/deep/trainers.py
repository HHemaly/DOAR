from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from .datasets import build_loaders
from .registry import build_model, freeze_backbone, unfreeze_all


def _seed(value: int):
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _macro_f1(truth, predictions) -> float:
    scores = []
    for index in range(len(CLASSES)):
        tp = sum(t == index and p == index for t, p in zip(truth, predictions))
        fp = sum(t != index and p == index for t, p in zip(truth, predictions))
        fn = sum(t == index and p != index for t, p in zip(truth, predictions))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(scores))


def train_image_model(
    dataset: str, model_name: str, output: str, seed: int = 42, epochs: int = 30,
    batch_size: int = 16, image_size: int = 224, device: str = "auto",
    augmentation: str = "conservative", patience: int = 7, workers: int = 0,
    pretrained: bool = True, freeze_epochs: int = 3, resume: str | None = None,
    configuration_hash: str | None = None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    _seed(seed)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    train_loader, valid_loader = build_loaders(
        dataset, image_size, batch_size, workers, augmentation
    )
    model = build_model(model_name, len(CLASSES), pretrained).to(selected_device)
    if freeze_epochs and model_name != "small_cnn":
        freeze_backbone(model)
    counts = torch.bincount(torch.tensor(train_loader.dataset.targets), minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts.clamp_min(1))
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(selected_device), label_smoothing=.05)
    resume_payload = None
    start_epoch = 0
    if resume:
        resume_payload = torch.load(resume, map_location=selected_device, weights_only=False)
        if resume_payload.get("model_name") != model_name:
            raise ValueError("Resume checkpoint model name does not match --model")
        if tuple(resume_payload.get("classes", ())) != CLASSES:
            raise ValueError("Resume checkpoint class mapping mismatch")
        model.load_state_dict(resume_payload["model_state"])
        start_epoch = int(resume_payload["epoch"]) + 1
        if start_epoch >= freeze_epochs:
            unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4 if start_epoch >= freeze_epochs else 3e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=.3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=selected_device.startswith("cuda"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], -1.0, 0
    if resume_payload:
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state"):
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if resume_payload.get("scaler_state"):
            scaler.load_state_dict(resume_payload["scaler_state"])
        history = list(resume_payload.get("history", []))
        best = float(resume_payload.get("best_valid_macro_f1", -1.0))
        stale = int(resume_payload.get("early_stopping_stale_epochs", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if epoch == freeze_epochs and model_name != "small_cnn" and start_epoch <= freeze_epochs:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=2, factor=.3
            )
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(selected_device), labels.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=selected_device.split(":")[0],
                                enabled=selected_device.startswith("cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(labels)
        model.eval()
        truth, predictions = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                logits = model(images.to(selected_device))
                predictions.extend(logits.argmax(1).cpu().tolist())
                truth.extend(labels.tolist())
        macro_f1 = _macro_f1(truth, predictions)
        scheduler.step(macro_f1)
        record = {
            "epoch": epoch, "train_loss": train_loss / len(train_loader.dataset),
            "valid_macro_f1": macro_f1, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        checkpoint = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "epoch": epoch, "model_name": model_name, "classes": CLASSES,
            "seed": seed, "image_size": image_size, "preprocessing_version": "imagenet_v1",
            "model_version": "doar_deep_v1", "calibration_status": "uncalibrated",
            "validation": record,
            "history": history,
            "best_valid_macro_f1": max(best, macro_f1),
            "early_stopping_stale_epochs": stale,
            "model_family": "deep_image",
            "configuration_sha256": configuration_hash,
        }
        torch.save(checkpoint, output / "last.pt")
        if macro_f1 > best:
            best, stale = macro_f1, 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    result = {
        "model": model_name, "seed": seed, "device": selected_device,
        "selection_split": "valid", "test_used": False, "best_valid_macro_f1": best,
        "epochs_completed": len(history), "training_seconds": time.perf_counter() - started,
        "checkpoint": str(output / "best.pt"), "history": history,
    }
    (output / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
