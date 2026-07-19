from __future__ import annotations

from . import MODEL_NAMES


def build_model(name: str, classes: int = 4, pretrained: bool = True):
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
    weights = "DEFAULT" if pretrained else None
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
