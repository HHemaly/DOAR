"""E1-A2 — Standardized Frozen Linear-Probe Benchmark: shared, portable
library. No hardcoded paths, no hardcoded device. Every function takes
`dataset_root`/`output_root`/`device` explicitly.

Embedding definition (Section 7): the representation immediately BEFORE
the original classifier/head -- never the 4-class logits. For torchvision
CNN/ViT backbones this means replacing the final classifier layer with
nn.Identity (same technique already used, read-only, by
src/doar/deep/embeddings.py::_finetuned_extractor, reused here for a
FRESH ImageNet-pretrained backbone rather than a fine-tuned DOAR
checkpoint). For CLIP/DINOv2/SigLIP2, `encode_image`/the model's own
forward pass already returns the pre-classification representation by
construction (these architectures have no classification head at all in
their base pretrained form).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

T0_MANIFEST = ROOT / "outputs" / "t0_automated" / "final_partition" / "partition_manifest.csv"
EXPECTED_T0_SHA256 = "4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d"
CLASSES = ("Angry", "Fear", "Happy", "Sad")
SEED = 42

# ---------------------------------------------------------------------------
# Candidate registry -- one entry per E1-A2 candidate. `family` selects the
# extraction strategy; `builder` args are resolved lazily (torch/torchvision/
# open_clip/torch.hub are only imported inside the functions that need them,
# so this module imports cleanly without any deep-learning package present).
# ---------------------------------------------------------------------------

TORCHVISION_MODELS = {
    "mobilenet_v3_small": {"head_attr": ("classifier", -1)},
    "resnet18": {"head_attr": ("fc", None)},
    "efficientnet_b0": {"head_attr": ("classifier", -1)},
    "densenet121": {"head_attr": ("classifier", None)},
    "convnext_tiny": {"head_attr": ("classifier", -1)},
    "vit_b_16": {"head_attr": ("heads.head", None)},
}
FOUNDATION_MODELS = {
    "clip_vit_b32": {"family": "openclip", "model_name": "ViT-B-32-quickgelu", "pretrained": "openai"},
    "dinov2_vits14": {"family": "dinov2", "hub_name": "dinov2_vits14"},
    "siglip2_vit_b16": {"family": "openclip", "model_name": "ViT-B-16-SigLIP2", "pretrained": "webli"},
}
ALL_CANDIDATES = tuple(TORCHVISION_MODELS) + tuple(FOUNDATION_MODELS)


def set_seed(seed: int = SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    import torch
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False. "
                            "Use --device auto or --device cpu, or run on a CUDA-enabled machine.")
    return device_arg


def hardware_report(device: str) -> dict:
    import torch
    report = {"device": device, "pytorch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if device.startswith("cuda") and torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["cuda_version"] = torch.version.cuda
        report["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return report


# ---------------------------------------------------------------------------
# Portable T0 manifest resolution (never reads the T0 manifest's own
# machine-specific `path` column; always dataset_root / relative_path).
# ---------------------------------------------------------------------------


def verify_t0_manifest_sha256() -> str:
    actual = hashlib.sha256(T0_MANIFEST.read_bytes()).hexdigest()
    if actual != EXPECTED_T0_SHA256:
        raise RuntimeError(f"T0 manifest SHA-256 mismatch: expected {EXPECTED_T0_SHA256}, got {actual}")
    return actual


def load_split_rows(dataset_root: str | Path) -> dict[str, list[dict]]:
    manifest_sha = verify_t0_manifest_sha256()
    with open(T0_MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_split: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for r in rows:
        split = r["new_split"]
        if split not in by_split:
            continue
        by_split[split].append({
            "image_id": r["image_id"], "relative_path": r["relative_path"], "class": r["class"],
            "split": split, "path": str(Path(dataset_root) / r["relative_path"]),
        })
    assert len(by_split["train"]) == 2599 and len(by_split["valid"]) == 284 and len(by_split["test"]) == 512, (
        "T0 split sizes drifted from the frozen spec -- refusing to proceed.")
    by_split["_manifest_sha256"] = manifest_sha
    return by_split


# ---------------------------------------------------------------------------
# Embedding extraction (Section 7): ONE common abstraction,
# extract_embedding(model_key, image_batch) -> penultimate representation.
# ---------------------------------------------------------------------------


def _torchvision_penultimate_model(model_key: str, device: str):
    """Builds a FRESH ImageNet-pretrained torchvision model with its final
    classifier replaced by nn.Identity, so a forward pass returns the
    penultimate (pre-logit) representation -- reuses
    src/doar/deep/registry.py::build_model/resolve_weights unmodified."""
    import torch.nn as nn

    from doar.deep.registry import build_model, resolve_weights
    weights_obj, weights_id = resolve_weights(model_key, "DEFAULT")
    model = build_model(model_key, len(CLASSES), pretrained=True)
    attr, index = TORCHVISION_MODELS[model_key]["head_attr"]
    target = model
    parts = attr.split(".")
    for p in parts[:-1]:
        target = getattr(target, p)
    last_attr = parts[-1]
    if index is None:
        setattr(target, last_attr, nn.Identity())
    else:
        getattr(target, last_attr)[index] = nn.Identity()
    model = model.to(device).eval()
    return model, weights_obj, weights_id


def build_extractor(model_key: str, device: str):
    """Returns (extractor_callable(tensor_batch) -> embedding_tensor,
    preprocessing_spec, weights_id, embedding_dim_hint_or_None)."""
    import torch

    from doar.deep.preprocessing import resolve_preprocessing
    if model_key in TORCHVISION_MODELS:
        model, weights_obj, weights_id = _torchvision_penultimate_model(model_key, device)
        pp_spec = resolve_preprocessing(model_key, 224, weights_object=weights_obj)

        def extractor(batch):
            with torch.no_grad():
                out = model(batch)
                return out.flatten(1)
        return extractor, pp_spec, weights_id, None

    spec = FOUNDATION_MODELS[model_key]
    if spec["family"] == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(spec["model_name"], pretrained=spec["pretrained"])
        model = model.to(device).eval()
        weights_id = f"open_clip:{spec['model_name']}:{spec['pretrained']}"
        pp_spec = {"family": "openclip", "model_name": spec["model_name"], "weights_id": spec["pretrained"],
                   "preprocessing_version": "open_clip_native"}

        def extractor(batch):
            with torch.no_grad():
                return model.encode_image(batch).flatten(1)
        return extractor, pp_spec, weights_id, preprocess

    if spec["family"] == "dinov2":
        from torchvision import transforms
        model = torch.hub.load("facebookresearch/dinov2", spec["hub_name"]).to(device).eval()
        weights_id = f"torch_hub:facebookresearch/dinov2:{spec['hub_name']}"
        pp_spec = {"family": "dinov2", "model_name": spec["hub_name"], "weights_id": "facebookresearch/dinov2",
                   "preprocessing_version": "dinov2_resize256_centercrop224_imagenet_norm"}
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        def extractor(batch):
            with torch.no_grad():
                out = model(batch)
                return (out["x_norm_clstoken"] if isinstance(out, dict) else out).flatten(1)
        return extractor, pp_spec, weights_id, preprocess

    raise ValueError(model_key)


def preprocessing_hash(pp_spec: dict) -> str:
    return hashlib.sha256(json.dumps(pp_spec, sort_keys=True).encode()).hexdigest()


def _cache_key(model_key: str, manifest_sha: str, split: str, weights_id: str, pp_hash: str) -> str:
    payload = {"model_key": model_key, "manifest_sha256": manifest_sha, "split": split,
               "weights_id": weights_id, "preprocessing_hash": pp_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_and_cache_embeddings(
    model_key: str, rows: list[dict], split: str, manifest_sha: str, output_root: Path, device: str,
    batch_size: int = 16, force: bool = False,
) -> dict:
    """Cache invalidates on ANY of: model_key, manifest_sha256, split,
    weights_id, preprocessing_hash (Section 7). Never silently reused if
    any differs -- the cache key IS the hash of all five."""
    from PIL import Image
    import torch

    extractor, pp_spec, weights_id, custom_preprocess = build_extractor(model_key, device)
    pp_hash = preprocessing_hash(pp_spec)
    key = _cache_key(model_key, manifest_sha, split, weights_id, pp_hash)

    cache_dir = Path(output_root) / "embeddings" / model_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{split}.npz"
    meta_path = cache_dir / f"{split}_meta.json"

    if not force and npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") == key and set(meta.get("image_ids", [])) == {r["image_id"] for r in rows}:
            payload = np.load(npz_path, allow_pickle=False)
            return {"embeddings": payload["embeddings"], "image_ids": payload["image_ids"],
                    "labels": payload["labels"], "cache_hit": True, "weights_id": weights_id,
                    "preprocessing_spec": pp_spec, "embedding_dimension": int(payload["embeddings"].shape[1])}

    if model_key in TORCHVISION_MODELS:
        from doar.deep.preprocessing import build_eval_transform
        transform = build_eval_transform(pp_spec)
    else:
        transform = custom_preprocess

    embeddings, ids, labels, latencies = [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        tensors = [transform(Image.open(r["path"]).convert("RGB")) for r in batch_rows]
        tensor = torch.stack(tensors).to(device)
        t0 = time.perf_counter()
        values = extractor(tensor).cpu().numpy().astype(np.float32)
        latencies.append((time.perf_counter() - t0) / len(batch_rows))
        embeddings.append(values)
        ids.extend(r["image_id"] for r in batch_rows)
        labels.extend(r["class"] for r in batch_rows)
    total_seconds = time.perf_counter() - started
    matrix = np.concatenate(embeddings, axis=0)

    np.savez_compressed(npz_path, embeddings=matrix, image_ids=np.asarray(ids), labels=np.asarray(labels))
    meta = {
        "cache_key": key, "model_key": model_key, "manifest_sha256": manifest_sha, "split": split,
        "weights_id": weights_id, "preprocessing_hash": pp_hash, "preprocessing_spec": pp_spec,
        "image_ids": ids, "embedding_dimension": int(matrix.shape[1]),
        "extraction_seconds_total": total_seconds, "encoder_latency_seconds_per_image_mean": float(np.mean(latencies)),
        "n_images": len(ids), "cache_size_bytes": npz_path.stat().st_size,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"embeddings": matrix, "image_ids": np.asarray(ids), "labels": np.asarray(labels), "cache_hit": False,
           "weights_id": weights_id, "preprocessing_spec": pp_spec, "embedding_dimension": int(matrix.shape[1]),
           "extraction_seconds_total": total_seconds,
           "encoder_latency_seconds_per_image_mean": float(np.mean(latencies)) if latencies else None}


def encoder_parameter_count(model_key: str, device: str = "cpu") -> int:
    """FULL frozen encoder parameter count -- never the trainable-head
    count alone (Section 8's explicit correction)."""
    if model_key in TORCHVISION_MODELS:
        model, _, _ = _torchvision_penultimate_model(model_key, device)
        return sum(p.numel() for p in model.parameters())
    # For foundation models, re-derive the underlying nn.Module to count
    # parameters -- build_extractor's closures don't expose it directly,
    # so re-resolve via the same code path used to build the extractor.
    spec = FOUNDATION_MODELS[model_key]
    import torch
    if spec["family"] == "openclip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(spec["model_name"], pretrained=spec["pretrained"])
        return sum(p.numel() for p in model.parameters())
    if spec["family"] == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", spec["hub_name"])
        return sum(p.numel() for p in model.parameters())
    raise ValueError(model_key)


# ---------------------------------------------------------------------------
# Standardized linear-probe epoch loop -- IDENTICAL for every candidate
# (Section 2/3). No augmentation (deterministic cached embeddings).
# ---------------------------------------------------------------------------


def compute_class_weights(train_targets: list[int]) -> np.ndarray:
    counts = np.bincount(train_targets, minlength=len(CLASSES)).astype(np.float64)
    return counts.sum() / (len(CLASSES) * np.clip(counts, 1, None))


def run_linear_probe(
    x_train, y_train, x_valid, y_valid, valid_ids, *, device: str, max_epochs: int, patience: int,
    head_lr: float, weight_decay: float, class_weights, output_dir: Path, model_key: str,
    resume: bool = False,
):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    embed_dim = x_train.shape[1]
    model = torch.nn.Linear(embed_dim, len(CLASSES)).to(device)

    history, best_f1, best_epoch, stale, start_epoch = [], -1.0, -1, 0, 0
    last_ckpt = output_dir / "last.pt"
    if resume and last_ckpt.exists():
        payload = torch.load(last_ckpt, map_location=device, weights_only=False)
        if payload.get("embedding_dimension") == embed_dim and payload.get("model_key") == model_key:
            model.load_state_dict(payload["model_state"])
            history = payload.get("history", [])
            start_epoch = payload["epoch"] + 1
            best_f1 = payload.get("best_valid_macro_f1", -1.0)
            best_epoch = payload.get("best_epoch", -1)
            stale = payload.get("stale_epochs", 0)

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    valid_ds = TensorDataset(torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)

    for epoch in range(start_epoch, max_epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(y)
            train_correct += int((logits.argmax(1) == y).sum())
            train_n += len(y)

        model.eval()
        val_loss, val_n, val_truth, val_pred, val_probs = 0.0, 0, [], [], []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += float(loss.detach()) * len(y)
                val_n += len(y)
                val_truth.extend(y.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
                val_probs.extend(torch.softmax(logits, dim=1).cpu().tolist())

        runtime = time.perf_counter() - t0
        val_macro_f1 = f1_score(val_truth, val_pred, average="macro", zero_division=0)
        is_best = val_macro_f1 > best_f1
        record = {
            "model": model_key, "epoch": epoch, "train_loss": train_loss / max(1, train_n),
            "val_loss": val_loss / max(1, val_n), "train_accuracy": train_correct / max(1, train_n),
            "val_accuracy": accuracy_score(val_truth, val_pred),
            "val_balanced_accuracy": balanced_accuracy_score(val_truth, val_pred),
            "val_macro_f1": val_macro_f1, "learning_rate": optimizer.param_groups[0]["lr"],
            "runtime_seconds": runtime, "is_best_epoch": is_best,
        }
        history.append(record)
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "model_key": model_key,
                   "embedding_dimension": embed_dim, "history": history, "best_valid_macro_f1": max(best_f1, val_macro_f1),
                   "best_epoch": best_epoch if not is_best else epoch, "stale_epochs": 0 if is_best else stale + 1},
                  last_ckpt)
        if is_best:
            best_f1, best_epoch, stale = val_macro_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "model_key": model_key,
                       "embedding_dimension": embed_dim, "val_macro_f1": val_macro_f1}, output_dir / "best.pt")
            with open(output_dir / "best_valid_predictions.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "true_class", "predicted_class",
                           "prob_Angry", "prob_Fear", "prob_Happy", "prob_Sad"])
                for iid, t, p, probs in zip(valid_ids, val_truth, val_pred, val_probs):
                    w.writerow([iid, CLASSES[t], CLASSES[p], *[f"{v:.6f}" for v in probs]])
        else:
            stale += 1
        if stale >= patience:
            break

    (output_dir / "epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history, best_epoch, best_f1
