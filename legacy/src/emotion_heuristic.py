"""
emotion_heuristic.py — Feature-based emotional tendency estimation.

IMPORTANT: No trained emotion classifier is currently available in this system.
This module provides a HEURISTIC, feature-based soft estimate only.

Output is clearly labelled:
  method:     "heuristic"
  confidence: "low"
  note:       "No trained emotion model is currently available."

This estimate should NEVER be presented as a trained model prediction.
It is a temporary fallback to be replaced when a real model checkpoint
is integrated.
"""

from __future__ import annotations
import json, os

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")
def _cfg():
    try:
        with open(_CFG_PATH) as f: return json.load(f)
    except Exception: return {}

CFG = _cfg()
PT  = CFG.get("psychological_rules", {})

NO_MODEL_NOTE = "No trained emotion model is currently available. This is a heuristic estimate only and must not be used as a model prediction."


def _cf(doc): return doc.get("color_features",       {})
def _cp(doc): return doc.get("composition_features",  {})
def _sf(doc): return doc.get("stroke_features",       {})
def _iq(doc): return doc.get("image_quality",          {})


def _score_sad(doc: dict) -> float:
    """Dark palette + large empty space + small drawing + lower-left bias → sad tendency."""
    score = 0.0
    cf, cp, sf = _cf(doc), _cp(doc), _sf(doc)
    dark   = cf.get("dark_dominance",    0.0)
    empty  = cp.get("empty_space_ratio", 0.0)
    content= cp.get("drawn_content_ratio", 1.0)
    quad   = cp.get("quadrant_bias", "")
    warm   = cf.get("warm_dominance",    0.0)
    ndiv   = cf.get("color_diversity_count", 5)

    if dark   > float(PT.get("dark_dominance_moderate", 0.20)):  score += 0.25
    if dark   > float(PT.get("dark_dominance_high",     0.35)):  score += 0.10
    if empty  > float(PT.get("empty_space_high",        0.60)):  score += 0.20
    if content < float(PT.get("empty_space_low",        0.20)):  score += 0.20
    if quad in ("lower_left",):                                   score += 0.10
    if warm   < 0.10:                                             score += 0.10
    if ndiv   < int(PT.get("color_diversity_low",       3)):     score += 0.10
    return min(score, 1.0)


def _score_happy(doc: dict) -> float:
    """Warm bright palette + high content + many colors → happy tendency."""
    score = 0.0
    cf, cp = _cf(doc), _cp(doc)
    warm  = cf.get("warm_dominance",       0.0)
    bright= cf.get("mean_brightness",      0.0)
    ndiv  = cf.get("color_diversity_count",0)
    sat   = cf.get("mean_saturation",      0.0)
    content= cp.get("drawn_content_ratio", 0.0)

    if warm   > 0.25:                                              score += 0.25
    if bright > 0.55:                                              score += 0.15
    if ndiv   >= 5:                                                score += 0.20
    if sat    > 0.40:                                              score += 0.15
    if content> 0.40:                                              score += 0.15
    return min(score, 1.0)


def _score_angry(doc: dict) -> float:
    """Dark + fragmented + red-dominant → angry tendency."""
    score = 0.0
    cf, sf = _cf(doc), _sf(doc)
    dark  = cf.get("dark_dominance",   0.0)
    red   = cf.get("color_ratios",{}).get("red", 0.0)
    frag  = sf.get("fragmentation_ratio", 0.0)
    stroke= sf.get("stroke_darkness_estimate", 0.0)

    if dark   > float(PT.get("dark_dominance_moderate", 0.20)):   score += 0.20
    if red    > float(PT.get("red_ratio_high",         0.30)):    score += 0.25
    if frag   > float(PT.get("fragmentation_moderate", 0.35)):    score += 0.20
    if stroke > float(PT.get("stroke_darkness_high",   0.65)):    score += 0.15
    return min(score, 1.0)


def _score_fear(doc: dict) -> float:
    """Very small + very empty + concerning symbols → fear tendency."""
    score = 0.0
    cp, cf = _cp(doc), _cf(doc)
    content  = cp.get("drawn_content_ratio", 1.0)
    empty    = cp.get("empty_space_ratio",   0.0)
    concerns = doc.get("concerning_symbol_flags", [])
    dark     = cf.get("dark_dominance", 0.0)

    if content < float(PT.get("empty_space_low",     0.20)):  score += 0.25
    if empty   > float(PT.get("empty_space_high",    0.60)):  score += 0.20
    if len(concerns) > 0:                                     score += 0.30
    if dark    > float(PT.get("dark_dominance_high", 0.35)):  score += 0.15
    return min(score, 1.0)


def estimate_emotional_tendency(doc: dict) -> dict:
    """
    Compute heuristic emotional tendency scores.

    Returns a structured dict clearly labelled as heuristic/low-confidence.
    The 'feature_based_emotional_tendency' key prevents confusion with
    a real model prediction field.
    """
    iq = _iq(doc)
    quality = float(iq.get("quality_score", 1.0))

    raw = {
        "sad":   _score_sad(doc),
        "happy": _score_happy(doc),
        "angry": _score_angry(doc),
        "fear":  _score_fear(doc),
    }

    # Apply quality dampening — poor image → lower effective scores
    dampened = {k: round(v * quality, 3) for k, v in raw.items()}

    best_emotion = max(dampened, key=dampened.get)
    best_score   = dampened[best_emotion]

    # If best score is below 0.20 there is no clear tendency
    if best_score < 0.20:
        best_emotion = "neutral_or_unclear"
        best_score   = 0.0

    return {
        "feature_based_emotional_tendency": {
            "estimated_emotion":  best_emotion,
            "confidence_numeric": best_score,
            "confidence_label":   "low",
            "method":             "heuristic",
            "note":               NO_MODEL_NOTE,
            "raw_scores":         dampened,
            "quality_applied":    round(quality, 3),
            "safe_to_display":    True,
            "display_warning": (
                "This is NOT a trained model prediction. "
                "It is a temporary feature-based soft estimate and should not be "
                "presented to parents as an emotion classification result."
            ),
        }
    }
