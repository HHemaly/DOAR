"""
classifier.py — model factory.

Two options, per the audit's "simplest defensible" recommendation:
  - "baseline"  : a small from-scratch CNN (sanity floor, fast on CPU)
  - "transfer"  : a pretrained backbone (ResNet18 by default; MobileNetV3 or
                  EfficientNet-B0 selectable) with a fresh classification head.

The final model is chosen on validation performance, not hardcoded.
"""

from __future__ import annotations


def build_model(name: str, num_classes: int, pretrained: bool = True):
    import torch.nn as nn

    name = name.lower()

    if name == "baseline":
        return _SmallCNN(num_classes)

    from torchvision import models

    if name in ("transfer", "resnet18"):
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "mobilenet":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "efficientnet":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise ValueError(f"Unknown model name: {name}")


def gradcam_target_layer(model, name: str):
    """Return the conv layer to hook for Grad-CAM, per architecture."""
    name = name.lower()
    if name in ("transfer", "resnet18"):
        return model.layer4[-1]
    if name == "mobilenet":
        return model.features[-1]
    if name == "efficientnet":
        return model.features[-1]
    if name == "baseline":
        return model.features[-1]
    return None


class _SmallCNN:
    """Tiny CNN baseline. Defined via a factory to avoid importing torch at
    module import time."""

    def __new__(cls, num_classes: int):
        import torch.nn as nn

        class SmallCNN(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return SmallCNN(num_classes)
