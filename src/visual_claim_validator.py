"""
visual_claim_validator.py — CLIP-based visual claim verification (bbox-aware).

FIXED (previously-confirmed defect): the validator now uses the detected
bounding box to crop the region and validates THAT crop. The full image is used
only as an explicitly-logged fallback when no bbox is available. Detector score,
CLIP cosine similarity, threshold, crop path, and validation status are all
stored SEPARATELY. Raw CLIP cosine similarity is never called a "probability".

Thresholds (conservative, from config/thresholds.json):
  - Standard objects:        >= 0.70
  - Eye features:            >= 0.75
  - Animal type specifics:   >= 0.72
  - Concerning symbols:      >= 0.85
"""

from __future__ import annotations
import json
import os

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")


def _cfg():
    try:
        with open(_CFG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


CFG = _cfg()
VC_CFG = CFG.get("object_detection", {}).get("visual_claim_validator", {})

_THRESH = {
    "default":           float(VC_CFG.get("default_verify_threshold",   0.60)),
    "visual_object":     float(VC_CFG.get("object_symbol_threshold",    0.70)),
    "visual_symbol":     float(VC_CFG.get("object_symbol_threshold",    0.70)),
    "eye_feature":       float(VC_CFG.get("eye_feature_threshold",      0.75)),
    "animal_type":       float(VC_CFG.get("animal_type_threshold",      0.72)),
    "concerning_symbol": float(VC_CFG.get("concerning_symbol_threshold", 0.85)),
    "safety_warning":    float(VC_CFG.get("concerning_symbol_threshold", 0.85)),
}


def _get_threshold(claim: dict) -> float:
    ct = claim.get("claim_type", "")
    ev = claim.get("evidence", {})
    cat = ev.get("category", "")
    label = ev.get("label", "").lower()
    if ct == "safety_warning" or cat == "symbols_concerning":
        return _THRESH["concerning_symbol"]
    if "eye" in label:
        return _THRESH["eye_feature"]
    if ct == "visual_symbol":
        return _THRESH["visual_symbol"]
    if ct == "visual_object":
        return _THRESH["visual_object"]
    return _THRESH["default"]


def _clip_similarity(clip_model, clip_preprocess, clip_tokenize,
                     image_pil, label: str) -> float:
    """Return raw normalised CLIP cosine similarity (NOT a probability)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        img_tensor = clip_preprocess(image_pil).unsqueeze(0).to(device)
        text_tokens = clip_tokenize([f"a drawing of {label}"]).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(img_tensor)
            txt_feat = clip_model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat * txt_feat).sum().item()
        return float(max(0.0, sim))
    except Exception:
        return 0.0


def _extract_bbox(claim: dict):
    """Find a bbox for this claim from its own evidence."""
    ev = claim.get("evidence", {})
    for key in ("bbox", "bounding_box", "box"):
        if ev.get(key):
            return ev[key]
    return None


def validate_visual_claim(claim: dict, image_path: str,
                          clip_model=None, clip_preprocess=None,
                          clip_tokenize=None, crops_dir: str = None) -> dict:
    """
    Validate one visual_object / visual_symbol / safety_warning claim.

    Uses the claim's bbox to crop and validate the region. Full image is a
    logged fallback only. Stores detector_score, clip_similarity, threshold,
    crop_path, and validation_method separately on the claim.
    """
    ct = claim.get("claim_type", "")
    if ct not in ("visual_object", "visual_symbol", "safety_warning"):
        return claim

    # Preserve the raw detector confidence separately BEFORE CLIP touches anything
    claim.setdefault("detector_score", claim.get("evidence", {}).get("raw_confidence"))

    if clip_model is None or clip_preprocess is None or clip_tokenize is None:
        claim["validator_status"] = "uncertain"
        claim["validation_method"] = "none"
        claim["validator_note"] = "CLIP model not available; visual claim cannot be verified."
        claim["show_to_user"] = False
        return claim

    threshold = _get_threshold(claim)
    label = claim.get("evidence", {}).get("label", "")
    bbox = _extract_bbox(claim)

    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
    except Exception:
        claim["validator_status"] = "uncertain"
        claim["validation_method"] = "none"
        claim["validator_note"] = "Could not open image for visual validation."
        claim["show_to_user"] = False
        return claim

    crop_path = None
    if bbox:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = img.crop((x1, y1, x2, y2))
            validation_method = "clip_crop"
            if crops_dir:
                os.makedirs(crops_dir, exist_ok=True)
                crop_path = os.path.join(
                    crops_dir, f"crop_{label}_{x1}_{y1}.png".replace(" ", "_"))
                crop.save(crop_path)
        except Exception:
            crop = img
            validation_method = "clip_fullimage_fallback"
            claim["validator_note_fallback"] = "bbox invalid; fell back to full image."
    else:
        crop = img
        validation_method = "clip_fullimage_fallback"

    similarity = _clip_similarity(clip_model, clip_preprocess, clip_tokenize, crop, label)

    # Store everything SEPARATELY — no conflation of similarity with probability
    claim["clip_similarity"] = round(similarity, 4)
    claim["validation_threshold"] = threshold
    claim["validation_method"] = validation_method
    claim["crop_path"] = crop_path

    if validation_method == "clip_fullimage_fallback":
        claim["validator_note"] = (
            f"No bounding box available; validated on FULL IMAGE (fallback). "
            f"CLIP similarity {similarity:.3f} vs threshold {threshold:.2f}."
        )

    if similarity >= threshold:
        claim["validator_status"] = "verified"
        claim["show_to_user"] = True
        claim.setdefault("validator_note",
                         f"CLIP crop similarity {similarity:.3f} >= threshold {threshold:.2f}.")
    elif similarity >= threshold * 0.75:
        claim["validator_status"] = "uncertain"
        claim["show_to_user"] = False
        claim.setdefault("validator_note",
                         f"CLIP similarity {similarity:.3f} below threshold {threshold:.2f}.")
    else:
        claim["validator_status"] = "rejected"
        claim["show_to_user"] = False
        claim.setdefault("validator_note",
                         f"CLIP similarity {similarity:.3f} well below threshold {threshold:.2f}.")
    return claim


def validate_all_visual_claims(claims, image_path, clip_model=None,
                               clip_preprocess=None, clip_tokenize=None,
                               crops_dir=None):
    return [validate_visual_claim(c, image_path, clip_model, clip_preprocess,
                                  clip_tokenize, crops_dir) for c in claims]
