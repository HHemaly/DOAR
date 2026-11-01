"""DINOv2 ViT-S/14 support for the frozen E1-B adaptation code (`_frozen_e1b`).

Why this exists
---------------
`_frozen_e1b.py` is the archived E1-B notebook code, extracted verbatim. On T0
its adaptation shortlist was SigLIP2, CLIP and EfficientNet-B0, so its
`build_model_and_transforms()` only knows those three. On T1 the pre-registered
shortlist rule (top-3 10-seed probe means) selected `dinov2_vits14`, which the
archived builder rejects with `ValueError: dinov2_vits14`.

This module adds DINOv2 WITHOUT editing the archived file: `install(e1b)` wraps
`e1b.build_model_and_transforms` so that `dinov2_vits14` is built here and every
other model key goes to the original, unchanged code. Everything else in the
adaptation protocol (optimiser, learning rates, epochs, patience, batch size,
label smoothing, class weights, AMP, early stopping, checkpoints, outputs) is the
archived code path.

Choices (each mirrors an existing, already-used component)
----------------------------------------------------------
* Backbone: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`, the
  same call and weights as the B1 linear-probe screening (e1a2_common.py).
* Features: the normalised CLS token (`x_norm_clstoken`), as in the probe.
* Eval preprocessing: Resize(256, bicubic) -> CenterCrop(224) -> ToTensor ->
  ImageNet Normalize, identical to the probe ("dinov2_resize256_centercrop224_imagenet_norm").
* Train preprocessing: the eval pipeline with the archived conservative
  augmentation inserted before ToTensor, by the archived
  `build_openclip_train_transform()` (the rule used for CLIP and SigLIP2).
* Head: one Linear(384 -> 4) named `head` (the archived optimiser puts `head.*`
  at HEAD_LR and everything else at ENCODER_LR).
* frozen: backbone frozen, head trained. full: everything trained.
  partial: the last transformer blocks are unfrozen, from the top, until they
  hold >= PARTIAL_TARGET (25%) of the backbone's parameters, plus the final
  norm, exactly the rule the archived code applies to OpenCLIP towers
  (`configure_openclip_trainability`: last blocks + final norm).
"""

from __future__ import annotations

HUB_REPO = "facebookresearch/dinov2"
HUB_NAME = "dinov2_vits14"
MODEL_KEY = "dinov2_vits14"
PREPROCESSING_VERSION = "dinov2_resize256_centercrop224_imagenet_norm"


def eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _classifier_class():
    import torch.nn as nn

    class DinoV2Classifier(nn.Module):
        def __init__(self, backbone, embedding_dimension, n_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(embedding_dimension, n_classes)

        @property
        def visual(self):          # the archived parameter_report() reads model.visual / model.head
            return self.backbone

        def forward(self, images):
            out = self.backbone(images)
            features = out["x_norm_clstoken"] if isinstance(out, dict) else out
            return self.head(features.flatten(1))

    return DinoV2Classifier


def transformer_blocks(backbone) -> list:
    """Blocks in forward order; handles chunked block lists (BlockChunk)."""
    import torch.nn as nn
    out = []
    for b in backbone.blocks:
        if isinstance(b, nn.ModuleList):          # BlockChunk: Identity padding + real blocks
            out += [x for x in b if not isinstance(x, nn.Identity)]
        else:
            out.append(b)
    return out


def configure_trainability(model, mode, *, partial_target, freeze_module, unfreeze_module, parameter_count):
    freeze_module(model.backbone)
    unfreeze_module(model.head)
    if mode == "frozen":
        return
    if mode == "full":
        unfreeze_module(model.backbone)
        return
    if mode != "partial":
        raise ValueError(mode)
    blocks = transformer_blocks(model.backbone)
    if not blocks:
        raise RuntimeError("Cannot identify DINOv2 blocks.")
    target = parameter_count(model.backbone) * partial_target
    accumulated, selected = 0, []
    for block in reversed(blocks):
        selected.append(block)
        accumulated += parameter_count(block)
        if accumulated >= target:
            break
    for block in selected:
        unfreeze_module(block)
    if getattr(model.backbone, "norm", None) is not None:
        unfreeze_module(model.backbone.norm)


def build(e1b, mode, backbone=None):
    """Return (model, train_transform, eval_transform, embedding_dimension) like the archived builder."""
    import torch
    if backbone is None:
        backbone = torch.hub.load(HUB_REPO, HUB_NAME)
    ev = eval_transform()
    tr = e1b.build_openclip_train_transform(ev)
    dim = int(backbone.embed_dim)
    model = _classifier_class()(backbone, dim, len(e1b.CLASSES))
    configure_trainability(model, mode, partial_target=e1b.PARTIAL_TARGET, freeze_module=e1b.freeze_module,
                           unfreeze_module=e1b.unfreeze_module, parameter_count=e1b.parameter_count)
    return model, tr, ev, dim


def install(e1b) -> None:
    """Route `dinov2_vits14` to this builder; all other models keep the archived code. Idempotent."""
    if getattr(e1b, "_DINOV2_EXTENSION", False):
        return
    original = e1b.build_model_and_transforms

    def build_model_and_transforms(model_key, mode):
        if model_key == MODEL_KEY:
            return build(e1b, mode)
        return original(model_key, mode)

    e1b.build_model_and_transforms = build_model_and_transforms
    e1b._DINOV2_EXTENSION = True
