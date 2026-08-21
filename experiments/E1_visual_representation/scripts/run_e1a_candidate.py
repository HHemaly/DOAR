#!/usr/bin/env python
"""Run E1-A for ONE candidate: smoke test -> (if OK) full frozen-backbone
head-training run on ALL 2599 train / 284 valid images, seed=42, max 30
epochs, early stopping patience=5 on validation Macro-F1. Writes
checkpoints/<key>/ and raw/<key>_result.json.

Usage: python run_e1a_candidate.py --model resnet18 [--smoke-only] [--max-epochs 30]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "scripts"))

import e1a_common as ec  # noqa: E402

IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS_DEFAULT = 30
PATIENCE = 5
HEAD_LR = 3e-4
WEIGHT_DECAY = 1e-4
AUGMENTATION = "conservative"
WORKERS = 0  # CPU-only, small dataset -- avoids Windows multiprocessing overhead

CNN_VIT_MODELS = {
    "mobilenet_v3_small": "mobilenet_v3_small",
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
    "convnext_tiny": "convnext_tiny",
    "vit_b_16": "vit_b_16",
}
FOUNDATION_MODELS = ("clip_vit_b32", "dinov2_vits14", "siglip2_vit_b16")
ALL_MODELS = tuple(CNN_VIT_MODELS) + FOUNDATION_MODELS


def build_cnn_vit_pipeline(tv_name: str, train_rows, valid_rows, device: str):
    from doar.deep.preprocessing import resolve_preprocessing
    from doar.deep.registry import build_model, freeze_backbone, resolve_weights

    weights_obj, weights_id = resolve_weights(tv_name, "DEFAULT")
    pp_spec = resolve_preprocessing(tv_name, IMAGE_SIZE, weights_object=weights_obj)
    model = build_model(tv_name, len(ec.CLASSES), pretrained=True).to(device)
    freeze_backbone(model)
    train_loader, valid_loader, train_ds, valid_ds = ec.build_image_loaders(
        train_rows, valid_rows, pp_spec, AUGMENTATION, BATCH_SIZE, WORKERS)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, train_loader, valid_loader, train_ds, {
        "resolved_weights_id": weights_id, "preprocessing_spec": pp_spec,
        "n_params": n_params, "n_trainable": n_trainable, "image_size": IMAGE_SIZE,
    }


def build_foundation_pipeline(key: str, train_rows, valid_rows, device: str, cache_dir: Path):
    import torch

    t0 = time.perf_counter()
    train_cache = ec.extract_embeddings_cached(key, train_rows, cache_dir / f"{key}_train.npz", device)
    valid_cache = ec.extract_embeddings_cached(key, valid_rows, cache_dir / f"{key}_valid.npz", device)
    extraction_seconds = time.perf_counter() - t0
    embed_dim = train_cache["embeddings"].shape[1]
    class_to_idx = {c: i for i, c in enumerate(ec.CLASSES)}
    x_train = torch.tensor(train_cache["embeddings"], dtype=torch.float32)
    y_train = torch.tensor([class_to_idx[str(c)] for c in train_cache["labels"]], dtype=torch.long)
    x_valid = torch.tensor(valid_cache["embeddings"], dtype=torch.float32)
    y_valid = torch.tensor([class_to_idx[str(c)] for c in valid_cache["labels"]], dtype=torch.long)

    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_ds.targets = y_train.tolist()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(ec.SEED))
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)

    model = torch.nn.Linear(embed_dim, len(ec.CLASSES)).to(device)
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters())
    embed_cache_mb = (train_cache["embeddings"].nbytes + valid_cache["embeddings"].nbytes) / 1e6
    return model, train_loader, valid_loader, train_ds, {
        "resolved_weights_id": ec.FOUNDATION_SPECS[key], "embedding_dimension": int(embed_dim),
        "n_params": n_params, "n_trainable": n_params, "image_size": None,
        "embedding_extraction_seconds": extraction_seconds, "embedding_cache_mb": embed_cache_mb,
    }


def run_smoke(model_key: str, device: str, cache_dir: Path) -> dict:
    """Tiny 2-batch forward+backward check -- catches technical failures
    (missing weights, shape mismatch, OOM) before committing to a full run."""
    manifest_dir = EXP_DIR / "raw"
    rows = ec.load_manifest_rows(manifest_dir / "e1a_train_valid_manifest.csv")
    train_rows = [r for r in rows if r["split"] == "train"][:32]
    valid_rows = [r for r in rows if r["split"] == "valid"][:16]
    t0 = time.perf_counter()
    try:
        if model_key in CNN_VIT_MODELS:
            model, train_loader, valid_loader, _, meta = build_cnn_vit_pipeline(
                CNN_VIT_MODELS[model_key], train_rows, valid_rows, device)
        else:
            model, train_loader, valid_loader, _, meta = build_foundation_pipeline(
                model_key, train_rows, valid_rows, device, cache_dir / "smoke")
        history, best_epoch, best_f1 = ec.run_epoch_loop(
            model, train_loader, valid_loader, device=device, max_epochs=1, patience=1,
            head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY, class_weights=None,
            output_dir=EXP_DIR / "checkpoints" / f"{model_key}_smoke", model_key=model_key)
        elapsed = time.perf_counter() - t0
        per_image = elapsed / (len(train_rows) + len(valid_rows))
        return {"status": "ok", "elapsed_seconds": elapsed, "seconds_per_image": per_image, "meta": meta}
    except Exception as exc:
        import traceback
        return {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}


def run_full(model_key: str, device: str, cache_dir: Path, max_epochs: int) -> dict:
    manifest_dir = EXP_DIR / "raw"
    rows = ec.load_manifest_rows(manifest_dir / "e1a_train_valid_manifest.csv")
    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]
    assert len(train_rows) == 2599 and len(valid_rows) == 284, "manifest split sizes drifted!"

    ec.set_seed(ec.SEED)
    if model_key in CNN_VIT_MODELS:
        model, train_loader, valid_loader, train_ds, meta = build_cnn_vit_pipeline(
            CNN_VIT_MODELS[model_key], train_rows, valid_rows, device)
    else:
        model, train_loader, valid_loader, train_ds, meta = build_foundation_pipeline(
            model_key, train_rows, valid_rows, device, cache_dir)
    class_weights = ec.compute_class_weights(train_ds.targets)

    started = time.perf_counter()
    history, best_epoch, best_f1 = ec.run_epoch_loop(
        model, train_loader, valid_loader, device=device, max_epochs=max_epochs, patience=PATIENCE,
        head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY, class_weights=class_weights,
        output_dir=EXP_DIR / "checkpoints" / model_key, model_key=model_key)
    total_seconds = time.perf_counter() - started

    # Latency: single-image inference timing over 30 validation images, model in eval mode.
    import torch
    model.eval()
    latencies = []
    with torch.no_grad():
        for x, _ in valid_loader:
            for i in range(min(30, x.shape[0])):
                t0 = time.perf_counter()
                model(x[i:i + 1].to(device))
                latencies.append(time.perf_counter() - t0)
            break
    latency_ms = float(sum(latencies) / len(latencies) * 1000) if latencies else None

    result = {
        "model_key": model_key, "status": "ok", "seed": ec.SEED,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_valid_macro_f1": best_f1, "training_seconds": total_seconds,
        "seconds_per_epoch": total_seconds / max(1, len(history)),
        "latency_ms_per_image": latency_ms, "meta": meta,
        "max_epochs": max_epochs, "patience": PATIENCE, "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE if model_key in CNN_VIT_MODELS else 64,
        "augmentation": AUGMENTATION if model_key in CNN_VIT_MODELS else "none_cached_embedding",
        "device": device, "hardware": platform.processor() or platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (EXP_DIR / "raw" / f"{model_key}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (EXP_DIR / "raw" / f"{model_key}_epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DEFAULT)
    args = parser.parse_args()

    device = "cpu"  # confirmed no CUDA available this session
    cache_dir = EXP_DIR / "embeddings"

    print(f"=== SMOKE TEST: {args.model} ===")
    smoke = run_smoke(args.model, device, cache_dir)
    print(json.dumps(smoke, indent=2)[:2000])
    (EXP_DIR / "raw" / f"{args.model}_smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    if smoke["status"] != "ok":
        print(f"SMOKE FAILED for {args.model} -- not running full training.")
        return
    if args.smoke_only:
        return

    print(f"=== FULL RUN: {args.model} ===")
    result = run_full(args.model, device, cache_dir, args.max_epochs)
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
