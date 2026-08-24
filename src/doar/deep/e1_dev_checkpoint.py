"""Additive, read-only inference adapter for an E1-experiment checkpoint,
used ONLY as an explicitly-labeled DEVELOPMENT expressive-model stand-in
when the frozen production checkpoint (production_config.py's
EXPRESSIVE_MODEL_CHECKPOINT_PATH) is not present on this machine.

The E1 checkpoint (torch.save'd by DOAR-work/experiments/E1_visual_
representation/scripts/e1a_common.py::run_epoch_loop) stores only
{"model_state", "epoch", "model_key", "val_macro_f1"} -- it never embeds
"classes"/"model_name"/"image_size"/"preprocessing_spec" the way
production checkpoints do, so `deep.inference.predict_image` cannot load
it directly (it would KeyError on payload["classes"]).

Every piece of metadata this adapter supplies below was read verbatim
from that SAME experiment's own recorded artifacts, never guessed:
  - class order:        DOAR-work/.../scripts/e1a_common.py CLASSES =
                         ("Angry", "Fear", "Happy", "Sad") -- IDENTICAL,
                         char-for-char, to this repo's doar/dataset.py
                         CLASSES.
  - architecture:        DOAR-work/.../scripts/run_e1a_candidate.py
                         build_cnn_vit_pipeline() calls THIS repo's own
                         doar.deep.registry.build_model(tv_name, ...) --
                         same function, same weights class, so
                         model_state is a directly loadable state_dict
                         for build_model("mobilenet_v3_small", 4).
  - full vs. head-only:  run_epoch_loop() saves model.state_dict() (the
                         WHOLE model -- freeze_backbone() only stops
                         gradient updates via requires_grad=False, it
                         does not remove the backbone's parameters from
                         the state_dict).
  - preprocessing spec:  DOAR-work/.../raw/mobilenet_v3_small_result.json
                         "meta.preprocessing_spec" -- computed by THIS
                         repo's doar.deep.preprocessing.resolve_
                         preprocessing() during E1 training itself, saved
                         verbatim below.
  - candidate selection: mobilenet_v3_small has the HIGHEST best_valid_
                         macro_f1 (0.7480) among every E1 candidate with
                         a completed (non-smoke-only) full run -- see
                         DOAR-work/.../raw/*_result.json. siglip2_vit_b16
                         only has a smoke-test checkpoint and was
                         excluded. The two foundation-model candidates
                         that beat some CNN/ViT scores (clip_vit_b32
                         0.7244, dinov2_vits14 0.6400) use a DIFFERENT
                         checkpoint shape (a linear probe over cached
                         embeddings, not an image-in/logits-out model) --
                         out of scope for this minimal adapter.

This checkpoint file itself is NEVER modified, re-saved, or copied --
loaded read-only, in place, from DOAR-work.

NOT a claimed final thesis model. `MODEL_VERSION` below is deliberately
prefixed "E1_DEVELOPMENT_" so every consumer (analysis.json, the
Psychologist/Technical views, Ask DOAR) can tell this apart from the
eventual, formally-selected E1 winner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..dataset import CLASSES

E1_WORK_ROOT = Path(r"C:\Users\ZZ01G7865\Downloads\DOAR\DOAR-work\experiments\E1_visual_representation")
E1_DEV_CHECKPOINT_PATH = E1_WORK_ROOT / "checkpoints" / "mobilenet_v3_small" / "best.pt"
E1_DEV_MODEL_KEY = "mobilenet_v3_small"
E1_DEV_MODEL_NAME = "mobilenet_v3_small"
E1_DEV_VAL_MACRO_F1 = 0.7480453959583941  # DOAR-work/.../raw/mobilenet_v3_small_result.json
MODEL_VERSION = "E1_DEVELOPMENT_mobilenet_v3_small_seed42_not_final_thesis_model"

# Verbatim from DOAR-work/.../raw/mobilenet_v3_small_result.json "meta.preprocessing_spec"
# (itself produced by THIS repo's doar.deep.preprocessing.resolve_preprocessing() during
# E1 training) -- never recomputed/guessed here.
E1_DEV_PREPROCESSING_SPEC = {
    "family": "torchvision",
    "model_name": "mobilenet_v3_small",
    "weights_id": "IMAGENET1K_V1",
    "revision": "torchvision_weights_transform",
    "preprocessing_version": "torchvision_weights_derived",
    "resize": 256,
    "crop": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "bilinear",
    "transform_source": "torchvision_weights.transforms()",
}
E1_DEV_IMAGE_SIZE = 224


def is_e1_dev_checkpoint(payload: dict) -> bool:
    """True for the shape run_epoch_loop() actually saves -- no "classes"
    key, unlike every production checkpoint (deep/inference.py's own
    contract requires one). Used by emotion.py to decide which loader
    applies, without guessing from the file path."""
    return isinstance(payload, dict) and "classes" not in payload and "model_state" in payload


def predict_e1_dev_checkpoint(image_path: str, checkpoint_path: str | Path, payload: dict) -> dict:
    """Same return-dict shape as deep.inference.predict_image() (every
    downstream consumer -- analysis.py, case_interpretation.py,
    live_case_bundle.py, the Streamlit views -- only depends on that
    shape, never on which function produced it), with a clearly-labeled
    DEVELOPMENT model_version/model_name and E1's own recorded
    val_macro_f1 preserved for provenance."""
    import torch
    from PIL import Image

    from .preprocessing import build_eval_transform
    from .registry import build_model

    checkpoint_path = Path(checkpoint_path)
    if payload.get("model_key") != E1_DEV_MODEL_KEY:
        raise ValueError(f"Unexpected E1 dev checkpoint model_key: {payload.get('model_key')!r}")

    model = build_model(E1_DEV_MODEL_NAME, len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    transform = build_eval_transform(E1_DEV_PREPROCESSING_SPEC)
    tensor = transform(Image.open(image_path).convert("RGB"))
    with torch.no_grad():
        logits = model(tensor[None])[0].cpu().numpy()
    exp = np.exp(logits - logits.max())
    probabilities = exp / exp.sum()
    order = np.argsort(probabilities)[::-1]
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())

    return {
        "status": "available",
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
        "top_class": CLASSES[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "top_two_margin": float(probabilities[order[0]] - probabilities[order[1]]),
        "entropy": entropy,
        "uncertainty": "high" if probabilities[order[0]] < .5 else "moderate",
        "raw_probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
        "calibration_status": "uncalibrated",
        "model_name": E1_DEV_MODEL_NAME,
        "model_family": "deep_image",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "model_version": MODEL_VERSION,
        "preprocessing_version": E1_DEV_PREPROCESSING_SPEC["preprocessing_version"],
        "evidence_id": "ev_emotion_prediction",
        "e1_val_macro_f1": E1_DEV_VAL_MACRO_F1,
        "e1_checkpoint_epoch": payload.get("epoch"),
        "e1_source_checkpoint_path": str(checkpoint_path),
        "development_model_disclaimer": (
            "DEVELOPMENT model from E1 candidate screening (mobilenet_v3_small, "
            "highest validation macro-F1 among fully-trained E1 candidates as of this session) "
            "-- NOT the final thesis-selected model."),
    }
