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


_HEAD_NAMES = ("fc", "classifier", "heads")


def _build_param_groups(model, head_lr: float, backbone_lr: float):
    """Head vs backbone parameter groups over CURRENTLY-trainable params.
    Reused at init and after unfreeze so differential LRs survive (A3)."""
    head_params, backbone_params = [], []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if pname.split(".")[0] in _HEAD_NAMES else backbone_params).append(param)
    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups


def _build_optimizer(name: str, param_groups):
    import torch
    name = (name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups)
    if name == "adam":
        return torch.optim.Adam(param_groups)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(name: str, optimizer):
    import torch
    name = (name or "reduce_on_plateau").lower()
    if name in ("reduce_on_plateau", "plateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.3), "max_metric"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10), "step"
    if name in ("none", "constant"):
        return None, "none"
    raise ValueError(f"Unsupported scheduler: {name}")


def train_image_model(
    dataset: str, model_name: str, output: str, seed: int = 42, epochs: int = 30,
    batch_size: int = 16, image_size: int = 224, device: str = "auto",
    augmentation: str = "conservative", patience: int = 7, workers: int = 0,
    pretrained: bool = True, freeze_epochs: int = 3, resume: str | None = None,
    configuration_hash: str | None = None,
    *,
    class_weighting: bool = True, optimizer_name: str = "adamw",
    head_learning_rate: float = 3e-4, backbone_learning_rate: float = 1e-4,
    scheduler_name: str = "reduce_on_plateau", calibration: str | None = None,
    grad_accum_steps: int = 1, pretrained_weights: str = "DEFAULT",
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    _seed(seed)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    # A4: resolve the exact torchvision weights (or scratch). `pretrained=False`
    # forces scratch; otherwise the pretrained_weights spec is honoured.
    from .registry import resolve_weights
    weights_spec = pretrained_weights if pretrained else "none"
    _weights_obj, _resolved_weights_id = resolve_weights(model_name, weights_spec)
    use_pretrained = _weights_obj is not None
    # Item 1: resolve weights + preprocessing spec BEFORE building the loaders so
    # the SAME resolved spec drives training and validation transforms (derived
    # from the weights object where available so recorded == executed).
    from .preprocessing import resolve_preprocessing, preprocessing_hash
    _pp_spec = resolve_preprocessing(model_name, image_size, weights_object=_weights_obj)
    _pp_spec["resolved_weights_id"] = _resolved_weights_id
    _pp_spec["augmentation_profile"] = augmentation
    _pp_hash = preprocessing_hash(_pp_spec)
    train_loader, valid_loader = build_loaders(
        dataset, image_size, batch_size, workers, augmentation,
        preprocessing_spec=_pp_spec,
    )
    model = build_model(model_name, len(CLASSES), weights_spec).to(selected_device)
    if freeze_epochs and model_name != "small_cnn":
        freeze_backbone(model)
    if class_weighting:
        counts = torch.bincount(torch.tensor(train_loader.dataset.targets), minlength=len(CLASSES)).float()
        weights = (counts.sum() / (len(CLASSES) * counts.clamp_min(1))).to(selected_device)
    else:
        weights = None
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
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
    # Differential learning rates: classification head vs backbone. Params are
    # matched by the same attribute names used by freeze_backbone (fc/classifier/
    # heads); everything else is the backbone.
    param_groups = _build_param_groups(model, head_learning_rate, backbone_learning_rate)
    optimizer = _build_optimizer(optimizer_name, param_groups)
    scheduler, scheduler_mode = _build_scheduler(scheduler_name, optimizer)
    # Gradient accumulation (A2): validate and record the effective batch size.
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    effective_batch_size = batch_size * grad_accum_steps
    optimizer_step_count = 0
    _use_amp = selected_device.startswith("cuda")
    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)
    except (AttributeError, TypeError):  # torch 2.2/2.3
        scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], -1.0, 0
    if resume_payload:
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state") and scheduler is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if resume_payload.get("scaler_state"):
            scaler.load_state_dict(resume_payload["scaler_state"])
        history = list(resume_payload.get("history", []))
        best = float(resume_payload.get("best_valid_macro_f1", -1.0))
        stale = int(resume_payload.get("early_stopping_stale_epochs", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if epoch == freeze_epochs and model_name != "small_cnn" and start_epoch <= freeze_epochs:
            # A3: after unfreezing, rebuild param groups and PRESERVE the
            # requested optimizer, differential LRs and scheduler (previously the
            # transition hardcoded AdamW + ReduceLROnPlateau).
            unfreeze_all(model)
            param_groups = _build_param_groups(model, head_learning_rate, backbone_learning_rate)
            optimizer = _build_optimizer(optimizer_name, param_groups)
            scheduler, scheduler_mode = _build_scheduler(scheduler_name, optimizer)
        model.train()
        train_loss = 0.0
        n_batches = len(train_loader)
        # A2: one zero_grad before each accumulation cycle; step only at the
        # configured interval; handle the final incomplete interval; unscale +
        # clip only before a step; report the true (unscaled) epoch loss.
        optimizer.zero_grad(set_to_none=True)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(selected_device), labels.to(selected_device)
            with torch.autocast(device_type=selected_device.split(":")[0],
                                enabled=selected_device.startswith("cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            # Divide by accumulation count so summed grads match a large batch.
            scaler.scale(loss / grad_accum_steps).backward()
            train_loss += float(loss.detach()) * len(labels)   # unscaled per-sample loss
            is_step = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == n_batches)
            if is_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
        model.eval()
        truth, predictions = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                logits = model(images.to(selected_device))
                predictions.extend(logits.argmax(1).cpu().tolist())
                truth.extend(labels.tolist())
        macro_f1 = _macro_f1(truth, predictions)
        if scheduler is not None:
            if scheduler_mode == "max_metric":
                scheduler.step(macro_f1)
            elif scheduler_mode == "step":
                scheduler.step()
        record = {
            "epoch": epoch, "train_loss": train_loss / len(train_loader.dataset),
            "valid_macro_f1": macro_f1, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        checkpoint = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "epoch": epoch, "model_name": model_name, "classes": CLASSES,
            "seed": seed, "image_size": image_size,
            "preprocessing_version": _pp_spec["preprocessing_version"],
            "preprocessing_hash": _pp_hash, "preprocessing_spec": _pp_spec,
            "resolved_weights_id": _resolved_weights_id,
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
    # What was ACTUALLY executed (Item 2) — recorded regardless of the config.
    executed_config = {
        "model_name": model_name, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "image_size": image_size,
        "device": selected_device, "workers": workers,
        "augmentation": augmentation, "freeze_epochs": freeze_epochs,
        "class_weighting": class_weighting, "optimizer": optimizer_name,
        "head_learning_rate": head_learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "scheduler": scheduler_name, "calibration": calibration,
        "grad_accum_steps": grad_accum_steps, "pretrained": use_pretrained,
        "pretrained_weights": pretrained_weights,
        "resolved_weights_id": _resolved_weights_id,
        "early_stopping_patience": patience,
        "physical_batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "optimizer_step_count": optimizer_step_count,
    }
    (output / "executed_config.json").write_text(
        json.dumps(executed_config, indent=2), encoding="utf-8")

    # Optional validation-only calibration of the best checkpoint (Item 2/7).
    calibration_result = None
    if calibration and str(calibration).lower() in ("temperature_scaling", "temperature"):
        try:
            from .calibration import calibrate_checkpoint
            calibration_result = calibrate_checkpoint(
                str(output / "best.pt"), dataset, output / "calibration", device)
        except Exception as exc:  # pragma: no cover - needs data+torch
            calibration_result = {"status": "failed", "error": str(exc)}

    result = {
        "model": model_name, "seed": seed, "device": selected_device,
        "selection_split": "valid", "test_used": False, "best_valid_macro_f1": best,
        "epochs_completed": len(history), "training_seconds": time.perf_counter() - started,
        "checkpoint": str(output / "best.pt"), "history": history,
        "executed_config": executed_config,
        "calibration": calibration_result,
    }
    (output / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
