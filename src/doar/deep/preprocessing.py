"""
preprocessing.py — one source of truth for model preprocessing (Item 6).

A single spec per backbone family drives BOTH the actual transform and the
recorded metadata, so they can never drift (the DINOv2 bug was: transform =
resize-224 while metadata claimed "dinov2_native_preprocess"). Each spec records
model name, weight identifier, revision, transform parameters (resize/crop/
mean/std/interpolation) and a preprocessing hash computed over those parameters.

Inference fails when the checkpoint's preprocessing hash is incompatible with the
transform that would be applied.
"""

from __future__ import annotations
import hashlib
import json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PreprocessingMismatch(RuntimeError):
    """Raised when a checkpoint's preprocessing is incompatible with inference."""


def resolve_preprocessing(backbone: str, image_size: int = 224,
                          checkpoint_meta: dict | None = None) -> dict:
    """Return the canonical preprocessing spec for a backbone/checkpoint."""
    if backbone.startswith("openclip:"):
        name = backbone.split(":", 1)[1]
        return {
            "family": "openclip",
            "model_name": name,
            "weights_id": "laion2b_s34b_b79k",
            "revision": "open_clip_default",
            "preprocessing_version": "openclip_native_preprocess",
            # OpenCLIP supplies its own transform object; parameters are fixed by
            # the pretrained tag, which we hash for reproducibility.
            "resize": None, "crop": None, "mean": None, "std": None,
            "interpolation": "open_clip_default",
            "transform_source": "open_clip.create_model_and_transforms",
        }
    if backbone.startswith("dinov2"):
        # DINOv2's actual recommended transform: resize 256 -> center-crop 224,
        # ImageNet normalization. (Previously ran resize-224 while claiming native.)
        return {
            "family": "dinov2",
            "model_name": backbone,
            "weights_id": "facebookresearch/dinov2",
            "revision": "hub_main",
            "preprocessing_version": "dinov2_resize256_centercrop224_imagenet_norm",
            "resize": 256, "crop": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "interpolation": "bicubic",
            "transform_source": "doar_dinov2_transform",
        }
    if backbone.startswith("finetuned:"):
        img = int((checkpoint_meta or {}).get("image_size", image_size))
        return {
            "family": "finetuned",
            "model_name": (checkpoint_meta or {}).get("source_model_name", "unknown"),
            "weights_id": (checkpoint_meta or {}).get("checkpoint_hash", "unknown"),
            "revision": "doar_deep_v1",
            "preprocessing_version": "resize_square_imagenet_norm",
            "resize": [img, img], "crop": None, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "interpolation": "bilinear",
            "transform_source": "doar_build_transforms_eval",
        }
    # torchvision pretrained backbones + trained emotion models.
    return {
        "family": "torchvision",
        "model_name": backbone,
        "weights_id": "DEFAULT",
        "revision": "torchvision_default",
        "preprocessing_version": "resize_square_imagenet_norm",
        "resize": [image_size, image_size], "crop": None,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "interpolation": "bilinear",
        "transform_source": "doar_build_transforms_eval",
    }


def preprocessing_hash(spec: dict) -> str:
    """Hash the transform-relevant parameters (not just a label)."""
    relevant = {k: spec.get(k) for k in
                ("family", "model_name", "weights_id", "resize", "crop",
                 "mean", "std", "interpolation", "preprocessing_version")}
    canonical = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_eval_transform(spec: dict):
    """Build the actual eval transform from the spec (torch-gated). Returns None
    for openclip (the caller uses open_clip's own preprocess)."""
    if spec["family"] == "openclip":
        return None
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - needs [deep]
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    interp = {"bilinear": transforms.InterpolationMode.BILINEAR,
              "bicubic": transforms.InterpolationMode.BICUBIC}.get(
                  spec.get("interpolation"), transforms.InterpolationMode.BILINEAR)
    steps = []
    resize = spec["resize"]
    steps.append(transforms.Resize(resize, interpolation=interp))
    if spec.get("crop"):
        steps.append(transforms.CenterCrop(spec["crop"]))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(spec["mean"], spec["std"]))
    return transforms.Compose(steps)


def assert_preprocessing_compatible(checkpoint_hash: str | None,
                                    applied_spec: dict) -> None:
    """Fail inference when the checkpoint's stored preprocessing hash does not
    match the transform that will actually be applied (Item 6). A missing stored
    hash (older checkpoints) is allowed but should be treated as unverified."""
    if not checkpoint_hash:
        return
    applied = preprocessing_hash(applied_spec)
    if checkpoint_hash != applied:
        raise PreprocessingMismatch(
            f"Checkpoint preprocessing hash {checkpoint_hash[:12]} does not match "
            f"the applied transform hash {applied[:12]} "
            f"({applied_spec.get('preprocessing_version')}). Refusing to run "
            f"inference with incompatible preprocessing.")
