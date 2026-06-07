"""
visual_claim_validator.py — CLIP-based visual claim verification.

Uses the same CLIP model already loaded by Part B, so no extra dependencies.
Validates each visual_object / visual_symbol claim by running CLIP on the
relevant image crop (or full image as fallback).

Thresholds are conservative:
  - Standard objects:        ≥ 0.70
  - Sensitive/eye features:  ≥ 0.75
  - Animal type specifics:   ≥ 0.72
  - Concerning symbols:      ≥ 0.85

Returns validator_status: "verified" / "uncertain" / "rejected"
"""

from __future__ import annotations
import json, os

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")
def _cfg():
    try:
        with open(_CFG_PATH) as f: return json.load(f)
    except Exception: return {}

CFG    = _cfg()
VC_CFG = CFG.get("object_detection", {}).get("visual_claim_validator", {})

# Threshold lookup
_THRESH = {
    "default":           float(VC_CFG.get("default_verify_threshold",   0.60)),
    "visual_object":     float(VC_CFG.get("object_symbol_threshold",    0.70)),
    "visual_symbol":     float(VC_CFG.get("object_symbol_threshold",    0.70)),
    "eye_feature":       float(VC_CFG.get("eye_feature_threshold",      0.75)),
    "animal_type":       float(VC_CFG.get("animal_type_threshold",      0.72)),
    "concerning_symbol": float(VC_CFG.get("concerning_symbol_threshold",0.85)),
    "safety_warning":    float(VC_CFG.get("concerning_symbol_threshold",0.85)),
}


def _get_threshold(claim: dict) -> float:
    ct = claim.get("claim_type", "")
    ev = claim.get("evidence", {})
    cat = ev.get("category", "")
    label = ev.get("label", "").lower()

    if ct == "safety_warning" or cat == "symbols_concerning":
        return _THRESH["concerning_symbol"]
    if "eye" in label or "eyes" in label:
        return _THRESH["eye_feature"]
    if ct == "visual_symbol":
        return _THRESH["visual_symbol"]
    if ct == "visual_object":
        return _THRESH["visual_object"]
    return _THRESH["default"]


def _clip_score_for_label(clip_model, clip_preprocess, clip_tokenize,
                           image_pil: Image.Image, label: str) -> float:
    """Run CLIP and return cosine similarity for a single text prompt vs image."""
    import torch
    device = "cuda" if hasattr(clip_model, "visual") else "cpu"
    try:
        img_tensor = clip_preprocess(image_pil).unsqueeze(0)
        text_tokens= clip_tokenize([f"a drawing of {label}"]).to(device)
        with torch.no_grad():
            img_feat  = clip_model.encode_image(img_tensor.to(device))
            txt_feat  = clip_model.encode_text(text_tokens)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score     = (img_feat * txt_feat).sum().item()
        return float(max(0.0, score))
    except Exception:
        return 0.0


def validate_visual_claim(claim: dict, image_path: str,
                           clip_model=None, clip_preprocess=None,
                           clip_tokenize=None) -> dict:
    """
    Validate a single visual_object / visual_symbol / safety_warning claim.

    If clip_model is None, marks as "uncertain" (cannot validate without model).
    """
    ct = claim.get("claim_type", "")
    if ct not in ("visual_object", "visual_symbol", "safety_warning"):
        return claim  # not a visual claim; skip

    if clip_model is None or clip_preprocess is None or clip_tokenize is None:
        claim["validator_status"] = "uncertain"
        claim["validator_note"]   = "CLIP model not available; visual claim cannot be verified."
        claim["show_to_user"]     = False
        return claim

    threshold = _get_threshold(claim)
    ev        = claim.get("evidence", {})
    label     = ev.get("label", "")
    bbox      = None

    # Try to find bbox from the detected object list (passed via evidence)
    # If no bbox, use full image
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if bbox:
            x1, y1, x2, y2 = bbox
            crop = img.crop((x1, y1, x2, y2))
        else:
            crop = img
    except Exception:
        claim["validator_status"] = "uncertain"
        claim["validator_note"]   = "Could not open image for visual validation."
        claim["show_to_user"]     = False
        return claim

    score = _clip_score_for_label(clip_model, clip_preprocess, clip_tokenize, crop, label)
    quality = float(claim.get("confidence", 0.5))  # already quality-adjusted from builder

    claim["clip_validation_score"] = round(score, 3)
    claim["validation_threshold"]  = threshold

    if score >= threshold:
        claim["validator_status"] = "verified"
        claim["validator_note"]   = (
            f"CLIP confirms label '{label}' with score {score:.3f} "
            f"(threshold {threshold:.2f})."
        )
        claim["confidence"]   = round(min(score, quality), 3)
        claim["show_to_user"] = True
    elif score >= threshold * 0.75:
        claim["validator_status"] = "uncertain"
        claim["validator_note"]   = (
            f"CLIP score {score:.3f} is below threshold {threshold:.2f} "
            f"for '{label}'. Visual presence is unclear."
        )
        claim["show_to_user"] = False
    else:
        claim["validator_status"] = "rejected"
        claim["validator_note"]   = (
            f"CLIP score {score:.3f} is well below threshold {threshold:.2f} "
            f"for '{label}'. Claim rejected as unsupported."
        )
        claim["confidence"]   = 0.0
        claim["show_to_user"] = False
    return claim


def validate_all_visual_claims(claims: list[dict], image_path: str,
                                clip_model=None, clip_preprocess=None,
                                clip_tokenize=None) -> list[dict]:
    """Run visual validation on all relevant claims."""
    return [
        validate_visual_claim(c, image_path, clip_model, clip_preprocess, clip_tokenize)
        for c in claims
    ]
