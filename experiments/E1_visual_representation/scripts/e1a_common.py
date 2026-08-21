"""E1-A shared library: manifest-based Dataset, unified frozen-backbone
head-training epoch loop (reused identically for image-input CNN/ViT
models and cached-embedding foundation models), and embedding extraction
for CLIP/DINOv2/SigLIP2. Reuses src/doar/deep/{registry,preprocessing,
augmentations}.py unmodified for CNN/ViT model construction and
transforms; does NOT reuse deep/datasets.py (ImageFolder-based, incompatible
with the T0 partition's manifest-driven split membership) or deep/
trainers.py (train_image_model's history schema lacks several fields E1-A's
required tables need, and freeze_epochs=0 semantics don't cleanly express
"never unfreeze") -- both would need modification to fit E1-A's exact
requirements, so a purpose-built (but small, and reusing all the pure
model/transform building blocks) loop is used instead."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ("Angry", "Fear", "Happy", "Sad")
SEED = 42


def set_seed(seed: int = SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ManifestImageDataset:
    """torch.utils.data.Dataset over the E1-A manifest CSV (image_id, path,
    class, split) -- NOT torchvision.ImageFolder, because the T0 partition's
    split membership is defined by the manifest, not by directory layout."""

    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.targets = [self.class_to_idx[r["class"]] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["class"]]


def build_image_loaders(train_rows, valid_rows, preprocessing_spec, augmentation, batch_size, workers=0):
    import torch
    from torch.utils.data import DataLoader

    from doar.deep.preprocessing import build_eval_transform, build_train_transform
    train_tf = build_train_transform(preprocessing_spec, augmentation)
    valid_tf = build_eval_transform(preprocessing_spec)
    train_ds = ManifestImageDataset(train_rows, train_tf)
    valid_ds = ManifestImageDataset(valid_rows, valid_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                               generator=torch.Generator().manual_seed(SEED))
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, valid_loader, train_ds, valid_ds


# ---------------------------------------------------------------------------
# Foundation-model embedding extraction (CLIP / DINOv2 / SigLIP2). Mirrors
# the pattern in src/doar/deep/embeddings.py::_extractor, but parameterized
# with the CORRECT per-model pretrained tag (embeddings.py hardcodes
# pretrained="laion2b_s34b_b79k" for every openclip: backbone, which is
# wrong for SigLIP2 and not the canonical choice for a "CLIP" baseline) --
# written here rather than editing that shared production module.
# ---------------------------------------------------------------------------

FOUNDATION_SPECS = {
    # ViT-B-32-quickgelu (not plain ViT-B-32): the openai CLIP weights were
    # trained with QuickGELU activation; open_clip's plain ViT-B-32 config
    # does not enable it and warns loudly that results will be numerically
    # wrong without this suffix. Fixed here rather than ignoring the warning.
    "clip_vit_b32": {"family": "openclip", "model_name": "ViT-B-32-quickgelu", "pretrained": "openai"},
    "dinov2_vits14": {"family": "dinov2", "hub_name": "dinov2_vits14"},
    "siglip2_vit_b16": {"family": "openclip", "model_name": "ViT-B-16-SigLIP2", "pretrained": "webli"},
}


def build_foundation_extractor(key: str, device: str):
    spec = FOUNDATION_SPECS[key]
    if spec["family"] == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["model_name"], pretrained=spec["pretrained"])
        model = model.to(device).eval()
        return ("openclip", model), preprocess
    if spec["family"] == "dinov2":
        import torch
        from torchvision import transforms
        model = torch.hub.load("facebookresearch/dinov2", spec["hub_name"]).to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return ("dinov2", model), preprocess
    raise ValueError(key)


def extract_embeddings_cached(key: str, rows: list[dict], cache_path: Path, device: str, batch_size: int = 8):
    """Single forward pass per image (no gradient, no augmentation -- the
    encoder is frozen and never re-run per epoch); cached to .npz keyed by
    image_id so a re-run of head training never re-extracts."""
    import torch
    from PIL import Image

    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_ids = set(payload["image_ids"].tolist())
        if all(r["image_id"] in cached_ids for r in rows):
            return payload

    (extractor_kind, model), preprocess = build_foundation_extractor(key, device)
    embeddings, ids, labels = [], [], []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        tensors = [preprocess(Image.open(r["path"]).convert("RGB")) for r in batch]
        with torch.no_grad():
            tensor = torch.stack(tensors).to(device)
            if extractor_kind == "openclip":
                values = model.encode_image(tensor)
            else:
                out = model(tensor)
                values = out["x_norm_clstoken"] if isinstance(out, dict) else out
            values = values.flatten(1).cpu().numpy().astype(np.float32)
        embeddings.append(values)
        ids.extend(r["image_id"] for r in batch)
        labels.extend(r["class"] for r in batch)
    elapsed = time.perf_counter() - started
    matrix = np.concatenate(embeddings, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=matrix, image_ids=np.asarray(ids),
                        labels=np.asarray(labels), extraction_seconds=elapsed)
    return {"embeddings": matrix, "image_ids": np.asarray(ids), "labels": np.asarray(labels)}


# ---------------------------------------------------------------------------
# Unified epoch loop -- identical code path for image-input models (model =
# full frozen-backbone torchvision net) and embedding-input models (model =
# a single nn.Linear over cached embeddings). Only the loaders' item shape
# differs (image tensor vs. embedding vector); the loop itself never knows
# which kind it is training.
# ---------------------------------------------------------------------------


def compute_class_weights(train_targets: list[int]) -> "np.ndarray":
    counts = np.bincount(train_targets, minlength=len(CLASSES)).astype(np.float64)
    return counts.sum() / (len(CLASSES) * np.clip(counts, 1, None))


def run_epoch_loop(
    model, train_loader, valid_loader, *, device: str, max_epochs: int, patience: int,
    head_lr: float, weight_decay: float, class_weights, output_dir: Path, model_key: str,
):
    import torch
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=head_lr, weight_decay=weight_decay)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)

    history = []
    best_f1, best_epoch, stale = -1.0, -1, 0
    for epoch in range(max_epochs):
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
        val_loss, val_n = 0.0, 0
        val_truth, val_pred = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += float(loss.detach()) * len(y)
                val_n += len(y)
                val_truth.extend(y.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())

        runtime = time.perf_counter() - t0
        train_acc = train_correct / max(1, train_n)
        val_acc = accuracy_score(val_truth, val_pred)
        val_bal_acc = balanced_accuracy_score(val_truth, val_pred)
        val_macro_f1 = f1_score(val_truth, val_pred, average="macro", zero_division=0)
        lr_now = optimizer.param_groups[0]["lr"]

        is_best = val_macro_f1 > best_f1
        record = {
            "model": model_key, "epoch": epoch, "train_loss": train_loss / max(1, train_n),
            "val_loss": val_loss / max(1, val_n), "train_accuracy": train_acc,
            "val_accuracy": val_acc, "val_balanced_accuracy": val_bal_acc,
            "val_macro_f1": val_macro_f1, "learning_rate": lr_now,
            "runtime_seconds": runtime, "is_best_epoch": is_best,
        }
        history.append(record)
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "model_key": model_key},
                   output_dir / "last.pt")
        if is_best:
            best_f1, best_epoch, stale = val_macro_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "model_key": model_key,
                       "val_macro_f1": val_macro_f1},
                       output_dir / "best.pt")
            with open(output_dir / "best_valid_predictions.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["true_class_idx", "pred_class_idx"])
                w.writerows(zip(val_truth, val_pred))
        else:
            stale += 1
        if stale >= patience:
            break

    with open(output_dir / "epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history, best_epoch, best_f1
