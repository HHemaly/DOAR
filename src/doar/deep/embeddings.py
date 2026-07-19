from __future__ import annotations

import csv
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from .augmentations import build_transforms


def _torchvision_extractor(backbone: str, device: str):
    import torch
    from torchvision import models
    from torchvision.models.feature_extraction import create_feature_extractor
    specifications = {
        "resnet18": (models.resnet18, "flatten"),
        "efficientnet_b0": (models.efficientnet_b0, "flatten"),
        "convnext_tiny": (models.convnext_tiny, "flatten"),
        "vit_b_16": (models.vit_b_16, "getitem_5"),
    }
    if backbone not in specifications:
        raise ValueError(f"Unsupported torchvision embedding backbone: {backbone}")
    constructor, node = specifications[backbone]
    model = constructor(weights="DEFAULT").to(device).eval()
    return create_feature_extractor(model, return_nodes={node: "embedding"}), 224


def _extractor(backbone: str, device: str):
    if backbone in ("resnet18", "efficientnet_b0", "convnext_tiny", "vit_b_16"):
        return _torchvision_extractor(backbone, device)
    if backbone.startswith("dinov2"):
        import torch
        model = torch.hub.load("facebookresearch/dinov2", backbone).to(device).eval()
        return model, 224
    if backbone.startswith("openclip:"):
        import open_clip
        name = backbone.split(":", 1)[1]
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained="laion2b_s34b_b79k")
        return (model.to(device).eval(), preprocess), 224
    raise ValueError(f"Unsupported embedding backbone: {backbone}")


def extract_embeddings(
    manifest: str | Path, output: str | Path, backbone: str,
    device: str = "auto", batch_size: int = 16, force: bool = False,
) -> dict:
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Install deep dependencies with: pip install -e ".[deep]"') from exc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest)
    fingerprint = hashlib.sha256(
        manifest.read_bytes() + json.dumps({"backbone": backbone, "v": "embedding_v1"},
                                           sort_keys=True).encode()
    ).hexdigest()
    metadata_path = output / "embedding_metadata.json"
    array_path = output / "embeddings.npz"
    if not force and metadata_path.exists() and array_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("cache_fingerprint") == fingerprint:
            return {**cached, "cache_hit": True}
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    extractor, image_size = _extractor(backbone, selected)
    transform = extractor[1] if backbone.startswith("openclip:") else build_transforms(image_size, False)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("readable", "").lower() in ("true", "1")]
    embeddings, ids, splits, labels, failures = [], [], [], [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        tensors, accepted = [], []
        for row in batch_rows:
            try:
                tensors.append(transform(Image.open(row["path"]).convert("RGB")))
                accepted.append(row)
            except Exception as exc:
                failures.append({"image_id": row["image_id"], "path": row["path"],
                                 "error": str(exc), "traceback": traceback.format_exc()})
        if not tensors:
            continue
        with torch.no_grad():
            tensor = torch.stack(tensors).to(selected)
            if backbone.startswith("openclip:"):
                values = extractor[0].encode_image(tensor)
            else:
                values = extractor(tensor)
                if isinstance(values, dict):
                    values = values["embedding"]
            values = values.flatten(1).cpu().numpy().astype(np.float32)
        embeddings.append(values)
        ids.extend(row["image_id"] for row in accepted)
        splits.extend(row["split"] for row in accepted)
        labels.extend(row["class"] for row in accepted)
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), np.float32)
    np.savez_compressed(array_path, embeddings=matrix, image_ids=np.asarray(ids),
                        splits=np.asarray(splits), labels=np.asarray(labels))
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "path", "error", "traceback"])
        writer.writeheader()
        writer.writerows(failures)
    # Record the TRUE preprocessing per backbone family (D5). CLIP and DINOv2 use
    # their own transforms, not the ImageNet pipeline — labelling them
    # "imagenet_v1" was a reproducibility-metadata bug.
    if backbone.startswith("openclip:"):
        preprocessing_version = "openclip_native_preprocess"
    elif backbone.startswith("dinov2"):
        preprocessing_version = "dinov2_native_preprocess"
    else:
        preprocessing_version = "imagenet_v1"
    metadata = {
        "status": "available", "backbone": backbone, "model_version": "official_pretrained",
        "preprocessing_version": preprocessing_version, "preprocessing_hash": hashlib.sha256(
            f"{backbone}:{image_size}:{preprocessing_version}".encode()).hexdigest(),
        "embedding_dimension": int(matrix.shape[1]) if matrix.size else 0,
        "images": len(ids), "failures": len(failures), "device": selected,
        "cache_fingerprint": fingerprint, "cache_hit": False,
        "splits": sorted(set(splits)), "classes": list(CLASSES),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def embed_image(image: str | Path, backbone: str, device: str = "auto") -> np.ndarray:
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Install deep dependencies with: pip install -e ".[deep]"') from exc
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    extractor, image_size = _extractor(backbone, selected)
    transform = extractor[1] if backbone.startswith("openclip:") else build_transforms(image_size, False)
    tensor = transform(Image.open(image).convert("RGB"))[None].to(selected)
    with torch.no_grad():
        if backbone.startswith("openclip:"):
            value = extractor[0].encode_image(tensor)
        else:
            value = extractor(tensor)
            if isinstance(value, dict):
                value = value["embedding"]
    return value.flatten(1)[0].cpu().numpy().astype(np.float32)
