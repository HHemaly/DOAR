"""
gradcam.py — Grad-CAM heatmaps for the trained classifier.

IMPORTANT INTERPRETATION NOTE (embedded into every saved figure):
    A Grad-CAM heatmap shows which image regions most influenced the CLASSIFIER's
    prediction. It does NOT prove those regions carry psychological meaning and is
    not clinical evidence.

Requires torch + a trained checkpoint. Degrades gracefully (returns None) if
torch is unavailable.
"""

from __future__ import annotations
import os

DISCLAIMER = ("Grad-CAM shows classifier attention, not psychological meaning. "
              "Not clinical evidence.")


def generate_gradcam(image_path: str, checkpoint: str, out_path: str,
                     target_class: int | None = None) -> str | None:
    try:
        import torch
        import numpy as np
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models.classifier import build_model, gradcam_target_layer
    from src.models.dataset import build_transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt["classes"]
    model_name = ckpt.get("model_name", "transfer")
    img_size = ckpt.get("img_size", 224)

    model = build_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    layer = gradcam_target_layer(model, model_name)
    if layer is None:
        return None

    activations, gradients = {}, {}

    def fwd_hook(_m, _i, o):
        activations["v"] = o.detach()

    def bwd_hook(_m, gi, go):
        gradients["v"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    tf = build_transforms(img_size, train=False)
    pil = Image.open(image_path).convert("RGB")
    x = tf(pil).unsqueeze(0).to(device)

    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(1).item())
    model.zero_grad()
    logits[0, target_class].backward()

    acts = activations["v"][0]           # C,H,W
    grads = gradients["v"][0]            # C,H,W
    weights = grads.mean(dim=(1, 2))     # C
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    h1.remove(); h2.remove()

    # Upsample CAM to image size and overlay
    import numpy as np
    cam_img = np.array(Image.fromarray((cam * 255).astype("uint8")).resize(pil.size))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(pil); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(pil)
    axes[1].imshow(cam_img, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM -> predicted: {classes[target_class]}")
    axes[1].axis("off")
    fig.suptitle(DISCLAIMER, fontsize=8, y=0.02, color="#555")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
