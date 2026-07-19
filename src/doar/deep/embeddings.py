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


def _finetuned_extractor(checkpoint: str, device: str):
    """Penultimate-layer embeddings from a trained emotion checkpoint (Item 5).

    Rebuilds the checkpoint's model and replaces its final classification head
    with nn.Identity, so a forward pass returns the penultimate representation.
    Returns (model, image_size, extraction_meta)."""
    import torch
    import torch.nn as nn
    from .registry import build_model
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = payload["model_name"]
    image_size = int(payload.get("image_size", 224))
    model = build_model(model_name, len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model_state"])
    # Replace the final head with Identity to expose penultimate features.
    if model_name == "small_cnn":
        # small_cnn is a plain nn.Sequential; the last module is the classifier
        # Linear. Replace it directly (A5 — it has no .classifier attribute).
        model[-1] = nn.Identity()
    elif model_name in ("resnet18", "resnet50"):
        model.fc = nn.Identity()
    elif model_name in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0",
                        "convnext_tiny"):
        model.classifier[-1] = nn.Identity()
    elif model_name == "vit_b_16":
        model.heads.head = nn.Identity()
    else:
        raise ValueError(f"No penultimate mapping for model: {model_name}")
    model = model.to(device).eval()
    meta = {
        "extraction_layer": "penultimate_head_identity",
        "source_checkpoint": checkpoint,
        "source_model_name": model_name,
        "checkpoint_hash": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "preprocessing_version": payload.get("preprocessing_version", "imagenet_v1"),
    }
    return model, image_size, meta


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
    if backbone.startswith("finetuned:"):
        model, image_size, _meta = _finetuned_extractor(backbone.split(":", 1)[1], device)
        return model, image_size
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
    # Preprocessing spec is the single source of truth (Item 6): it drives BOTH
    # the transform below and the recorded metadata, so they cannot drift. DINOv2
    # now gets its real resize-256 -> center-crop-224 transform.
    from .preprocessing import resolve_preprocessing, build_eval_transform, preprocessing_hash
    _ckpt_meta = None
    if backbone.startswith("finetuned:"):
        _, _, _ckpt_meta = _finetuned_extractor(backbone.split(":", 1)[1], selected)
    pp_spec = resolve_preprocessing(backbone, image_size, _ckpt_meta)
    if backbone.startswith("openclip:"):
        transform = extractor[1]                       # open_clip's own preprocess
    else:
        transform = build_eval_transform(pp_spec)
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
    finetuned_meta = _ckpt_meta or {}
    model_version = ("finetuned_emotion_checkpoint" if backbone.startswith("finetuned:")
                     else "official_pretrained")
    metadata = {
        "status": "available", "backbone": backbone, "model_version": model_version,
        # Item 6: version, hash and full transform params come from ONE spec.
        "preprocessing_version": pp_spec["preprocessing_version"],
        "preprocessing_hash": preprocessing_hash(pp_spec),
        "preprocessing_spec": pp_spec,
        "embedding_dimension": int(matrix.shape[1]) if matrix.size else 0,
        "images": len(ids), "failures": len(failures), "device": selected,
        "cache_fingerprint": fingerprint, "cache_hit": False,
        "splits": sorted(set(splits)), "classes": list(CLASSES),
        **finetuned_meta,
    }
    # B5: artifact provenance so features/embeddings can be verified as coming
    # from the same manifest with matching sample-ID sets.
    from ..provenance import build_embedding_provenance
    metadata["provenance"] = build_embedding_provenance(
        manifest, ids, class_order=list(CLASSES),
        extraction_config_hash=fingerprint, backbone=backbone,
        revision=pp_spec.get("revision", "unknown"),
        checkpoint_hash=finetuned_meta.get("checkpoint_hash"),
        preprocessing_hash=preprocessing_hash(pp_spec),
        embedding_dimension=int(matrix.shape[1]) if matrix.size else 0)
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
    # A5: use the SAME canonical preprocessing resolver as batch extraction.
    from .preprocessing import resolve_preprocessing, build_eval_transform
    _ckpt_meta = None
    if backbone.startswith("finetuned:"):
        _, _, _ckpt_meta = _finetuned_extractor(backbone.split(":", 1)[1], selected)
    pp_spec = resolve_preprocessing(backbone, image_size, _ckpt_meta)
    transform = extractor[1] if backbone.startswith("openclip:") else build_eval_transform(pp_spec)
    tensor = transform(Image.open(image).convert("RGB"))[None].to(selected)
    with torch.no_grad():
        if backbone.startswith("openclip:"):
            value = extractor[0].encode_image(tensor)
        else:
            value = extractor(tensor)
            if isinstance(value, dict):
                value = value["embedding"]
    return value.flatten(1)[0].cpu().numpy().astype(np.float32)
