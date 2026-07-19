from __future__ import annotations

from . import MODEL_NAMES

# Builder name in torchvision for each supported model (small_cnn has none).
_TV_BUILDER = {
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "resnet18": "resnet18", "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0", "convnext_tiny": "convnext_tiny",
    "vit_b_16": "vit_b_16",
}


def resolve_weights(name: str, spec):
    """Resolve a pretrained-weights spec to a torchvision weights object + id (A4).

    spec may be:
      * True / "DEFAULT"           -> the model's DEFAULT weights
      * "none" / "scratch" / False / "" / None -> None (random init)
      * an explicit enum member    -> e.g. "IMAGENET1K_V1" / "IMAGENET1K_V2"
    Returns (weights_object_or_None, resolved_id_str). Raises ValueError on an
    invalid member with the list of valid names for that model.
    """
    if name == "small_cnn":
        return None, "scratch"
    from torchvision import models
    enum = models.get_model_weights(_TV_BUILDER[name])   # WeightsEnum class
    if spec in (False, None, "", "none", "scratch") or (isinstance(spec, str) and spec.lower() in ("none", "scratch")):
        return None, "scratch"
    if spec in (True, "DEFAULT") or (isinstance(spec, str) and spec.upper() == "DEFAULT"):
        w = enum.DEFAULT
        return w, f"{enum.__name__}.{w.name}"
    member = str(spec).split(".")[-1]                    # accept "ResNet18_Weights.IMAGENET1K_V1"
    try:
        w = enum[member]
    except KeyError:
        valid = [m.name for m in enum]
        raise ValueError(
            f"Invalid pretrained weights {spec!r} for model {name!r}. "
            f"Valid: DEFAULT, none/scratch, or one of {valid}.")
    return w, f"{enum.__name__}.{w.name}"


def build_model(name: str, classes: int = 4, pretrained=True):
    """Build a model. `pretrained` may be a bool or a weights spec string (A4)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown image model {name!r}; choose from {MODEL_NAMES}")
    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    if name == "small_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(.25), nn.Linear(128, classes),
        )
    weights, _ = resolve_weights(name, pretrained)
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    else:
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, classes)
    return model


def freeze_backbone(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Re-enable the final classifier using common torchvision naming.
    for name in ("fc", "classifier", "heads"):
        layer = getattr(model, name, None)
        if layer is not None:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def unfreeze_all(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
