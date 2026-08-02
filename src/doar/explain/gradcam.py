"""
gradcam.py — Grad-CAM visual explainability for CNN emotion models (Item 15).

Produces a raw heatmap (.npy) and an overlay (.png) for the predicted (or a
selected) target class. This is VISUAL classifier attention only. It is kept
strictly separate from objective-feature importance and must never be presented
as psychological or causal evidence.

Requires torch (device="auto", clean CPU fallback). The target-layer selection
per architecture is pure and CPU-tested.
"""

from __future__ import annotations
from pathlib import Path

DISCLAIMER = (
    "Grad-CAM shows which image regions most influenced the CLASSIFIER's "
    "prediction. It does not prove those regions carry psychological meaning and "
    "is not causal or clinical evidence. It is a visual attribution only and is "
    "unrelated to objective-feature importance."
)


# Grad-CAM needs convolutional feature maps; ViT has none -> unsupported.
_CAM_UNSUPPORTED = {"vit_b_16"}


def target_layer_name(model_name: str) -> str:
    """Name of the conv container to hook per architecture (pure, CPU-testable)."""
    if model_name in _CAM_UNSUPPORTED:
        raise ValueError(
            f"Grad-CAM is not supported for {model_name!r}: it has no convolutional "
            f"feature maps. Use an attention-based explanation for ViT instead.")
    return {
        "resnet18": "layer4", "resnet50": "layer4",
        "mobilenet_v3_small": "features", "mobilenet_v3_large": "features",
        "efficientnet_b0": "features", "convnext_tiny": "features",
        "small_cnn": "<sequential>",
    }.get(model_name, "features")


def _resolve_layer(model, model_name):
    import torch.nn as nn
    if model_name in _CAM_UNSUPPORTED:
        raise ValueError(
            f"Grad-CAM is not supported for {model_name!r} (no conv feature maps).")
    if model_name == "small_cnn":
        # Plain nn.Sequential with no .features: hook the LAST Conv2d module.
        convs = [m for m in model if isinstance(m, nn.Conv2d)]
        if not convs:
            raise ValueError("small_cnn has no Conv2d layer to hook for Grad-CAM.")
        return convs[-1]
    name = target_layer_name(model_name)
    module = getattr(model, name, None)
    if module is None:
        raise ValueError(f"Model {model_name!r} has no attribute {name!r} for Grad-CAM.")
    # For sequential feature stacks, hook the last conv-bearing block.
    if hasattr(module, "__getitem__"):
        try:
            return module[-1]
        except Exception:
            return module
    return module


def generate_gradcam(image_path, checkpoint, output_dir, device="auto",
                     target_class: int | None = None) -> dict:
    try:
        import torch
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - needs [deep]
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    from ..dataset import CLASSES
    from ..deep.registry import build_model
    from ..deep.augmentations import build_transforms
    from ..deep.preprocessing import resolve_preprocessing, build_eval_transform

    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    payload = torch.load(checkpoint, map_location=selected, weights_only=False)
    model_name = payload["model_name"]
    model = build_model(model_name, len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(selected).eval()

    layer = _resolve_layer(model, model_name)
    activations, gradients = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: activations.__setitem__("v", o.detach()))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("v", go[0].detach()))

    # A5: same canonical preprocessing as training/inference.
    spec = payload.get("preprocessing_spec") or resolve_preprocessing(
        model_name, payload["image_size"])
    tf = build_eval_transform(spec) or build_transforms(payload["image_size"], False)
    pil = Image.open(image_path).convert("RGB")
    x = tf(pil).unsqueeze(0).to(selected)

    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(1).item())
    model.zero_grad()
    logits[0, target_class].backward()

    acts = activations["v"][0]
    grads = gradients["v"][0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    h1.remove()
    h2.remove()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "gradcam_raw.npy"
    np.save(raw_path, cam)

    overlay_path = out / "gradcam_overlay.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cam_img = np.array(Image.fromarray((cam * 255).astype("uint8")).resize(pil.size))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(pil)
        ax.imshow(cam_img, cmap="jet", alpha=0.45)
        ax.axis("off")
        ax.set_title(f"Grad-CAM -> {CLASSES[target_class]}", fontsize=9)
        fig.text(0.5, 0.01, "Classifier attention, not psychological meaning.",
                 ha="center", fontsize=7, color="#555")
        fig.savefig(overlay_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        overlay_path = None

    return {
        "method": "grad_cam",
        "attribution_type": "visual_classifier_attention",
        "target_class": CLASSES[target_class],
        "raw_heatmap": str(raw_path),
        "overlay": str(overlay_path) if overlay_path else None,
        "disclaimer": DISCLAIMER,
    }
