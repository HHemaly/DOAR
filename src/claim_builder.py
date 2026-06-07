"""
claim_builder.py — Converts extracted doc fields, rule activations, OCR, and
emotion estimates into structured Claim objects.

Each Claim carries:
  - a unique ID
  - the raw assertion text
  - its type (visual_numeric / visual_object / visual_symbol / ocr_text /
              emotion_prediction / psychological_interpretation /
              safety_warning / parent_guidance)
  - the source component that generated it
  - supporting evidence (numeric or object refs)
  - a confidence value
  - validator_status (initially "pending")
  - show_to_user flag (initially False, set by validators)
"""

from __future__ import annotations
import json, os, re
from typing import Any

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")
def _cfg():
    try:
        with open(_CFG_PATH) as f: return json.load(f)
    except Exception: return {}

CFG = _cfg()
SH  = CFG.get("show_to_user_thresholds", {})
PT  = CFG.get("psychological_rules", {})

_claim_counter = [0]

def _next_id() -> str:
    _claim_counter[0] += 1
    return f"C{_claim_counter[0]:04d}"

def _make(claim_type, claim_text, source, evidence, confidence,
          validator_status="pending", show_to_user=False,
          raw_value=None, sensitive=False) -> dict:
    return {
        "claim_id":        _next_id(),
        "claim":           claim_text,
        "claim_type":      claim_type,
        "source":          source,
        "evidence":        evidence,
        "confidence":      round(confidence, 3),
        "raw_value":       raw_value,
        "sensitive":       sensitive,
        "validator_status": validator_status,
        "show_to_user":    show_to_user,
    }


# ---------------------------------------------------------------------------
# Visual numeric claims (from Part A features)
# ---------------------------------------------------------------------------

def build_numeric_claims(doc: dict) -> list[dict]:
    cf  = doc.get("color_features",       {})
    cp  = doc.get("composition_features", {})
    sf  = doc.get("stroke_features",      {})
    iq  = doc.get("image_quality",        {})
    q   = float(iq.get("quality_score", 1.0))
    claims = []

    # Empty space
    empty = cp.get("empty_space_ratio", None)
    if empty is not None:
        pct = round(empty * 100, 1)
        claims.append(_make(
            "visual_numeric",
            f"The drawing has approximately {pct}% empty space.",
            "composition_features",
            {"empty_space_ratio": empty},
            confidence=min(0.92, q),
            raw_value=empty,
        ))

    # Content area
    content = cp.get("drawn_content_ratio", None)
    if content is not None:
        pct = round(content * 100, 1)
        claims.append(_make(
            "visual_numeric",
            f"The drawn content covers approximately {pct}% of the page.",
            "composition_features",
            {"drawn_content_ratio": content},
            confidence=min(0.92, q),
            raw_value=content,
        ))

    # Dark dominance
    dark = cf.get("dark_dominance", None)
    if dark is not None:
        pct = round(dark * 100, 1)
        claims.append(_make(
            "visual_numeric",
            f"Approximately {pct}% of the colored area uses dark tones.",
            "color_features",
            {"dark_dominance": dark},
            confidence=min(0.90, q),
            raw_value=dark,
        ))

    # Color diversity
    ndiv = cf.get("color_diversity_count", None)
    if ndiv is not None:
        claims.append(_make(
            "visual_numeric",
            f"The drawing uses {ndiv} distinct color(s).",
            "color_features",
            {"color_diversity_count": ndiv},
            confidence=min(0.90, q),
            raw_value=ndiv,
        ))

    # Warm dominance
    warm = cf.get("warm_dominance", None)
    if warm is not None:
        pct = round(warm * 100, 1)
        claims.append(_make(
            "visual_numeric",
            f"Approximately {pct}% of the colored area uses warm tones.",
            "color_features",
            {"warm_dominance": warm},
            confidence=min(0.90, q),
            raw_value=warm,
        ))

    # Fragmentation
    frag = sf.get("fragmentation_ratio", None)
    if frag is not None:
        pct = round(frag * 100, 1)
        claims.append(_make(
            "visual_numeric",
            f"Approximately {pct}% of detected contours are small/fragmented marks.",
            "stroke_features",
            {"fragmentation_ratio": frag},
            confidence=min(0.80, q),
            raw_value=frag,
        ))

    return claims


# ---------------------------------------------------------------------------
# Visual object / symbol claims (from Part B)
# ---------------------------------------------------------------------------

def build_object_claims(doc: dict) -> list[dict]:
    iq = doc.get("image_quality", {})
    q  = float(iq.get("quality_score", 1.0))
    objects   = doc.get("detected_objects",        [])
    concerns  = doc.get("concerning_symbol_flags", [])
    claims    = []

    for obj in objects:
        top   = obj.get("top_match", {})
        label = top.get("label", "unknown")
        cat   = top.get("category", "unknown")
        conf  = float(top.get("confidence", 0.0)) * q
        pos   = obj.get("position", "unknown")
        sensitive = (cat == "symbols_concerning")

        claim_type = "visual_symbol" if cat in ("symbols_positive", "symbols_concerning", "marks") else "visual_object"
        claims.append(_make(
            claim_type,
            f"A {label} was detected in the {pos} area of the drawing.",
            "object_detector",
            {"label": label, "category": cat, "position": pos,
             "raw_confidence": top.get("confidence", 0.0)},
            confidence=round(conf, 3),
            raw_value=top.get("confidence", 0.0),
            sensitive=sensitive,
        ))

    for flag in concerns:
        label = flag if isinstance(flag, str) else str(flag)
        claims.append(_make(
            "safety_warning",
            f"A potentially concerning symbol was flagged: {label}.",
            "object_detector",
            {"flag": label},
            confidence=0.0,  # must be verified before any confidence is assigned
            sensitive=True,
        ))

    return claims


# ---------------------------------------------------------------------------
# OCR claims (from Part C)
# ---------------------------------------------------------------------------

def build_ocr_claims(doc: dict) -> list[dict]:
    ocr_results = doc.get("ocr_results", [])
    claims = []
    for item in ocr_results:
        text = item.get("text", "").strip()
        conf = float(item.get("confidence", 0.0))
        tag  = item.get("semantic_tag", "neutral")
        if not text:
            continue
        claims.append(_make(
            "ocr_text",
            f"The OCR reading found the text \"{text}\" (semantic tag: {tag}).",
            "ocr_extractor",
            {"text": text, "confidence": conf, "semantic_tag": tag},
            confidence=conf,
            raw_value=conf,
            sensitive=(tag == "negative"),
        ))
    return claims


# ---------------------------------------------------------------------------
# Emotion estimate claims (from emotion_heuristic)
# ---------------------------------------------------------------------------

def build_emotion_claim(doc: dict) -> list[dict]:
    heuristic = doc.get("feature_based_emotional_tendency", {})
    if not heuristic:
        return []
    emotion = heuristic.get("estimated_emotion", "neutral_or_unclear")
    conf_n  = float(heuristic.get("confidence_numeric", 0.0))
    note    = heuristic.get("note", "")
    return [_make(
        "emotion_prediction",
        (
            f"Heuristic emotional tendency estimate: {emotion} "
            f"(numeric score: {conf_n:.2f}). "
            f"Method: heuristic. Confidence: low. {note}"
        ),
        "emotion_heuristic",
        {"estimated_emotion": emotion, "confidence_numeric": conf_n,
         "method": "heuristic", "note": note},
        confidence=conf_n,
        raw_value=conf_n,
    )]


# ---------------------------------------------------------------------------
# Psychological interpretation claims (from Part E + v2 rules)
# ---------------------------------------------------------------------------

def build_psych_claims(doc: dict) -> list[dict]:
    activations = doc.get("psychological_rule_activations", [])
    # Merge v2 if present
    activations = activations + doc.get("psychological_rule_activations_v2", [])

    claims = []
    for rule in activations:
        if not rule.get("activated"):
            continue
        conf = float(rule.get("rule_confidence", 0.0))
        interp = rule.get("interpretation", "")
        caution = rule.get("caution", "")
        limitation = rule.get("limitation", "")
        claims.append(_make(
            "psychological_interpretation",
            interp,
            f"rule_engine:{rule.get('rule_id','unknown')}",
            {
                "rule_id":   rule.get("rule_id"),
                "domain":    rule.get("domain"),
                "sources":   rule.get("sources", []),
                "caution":   caution,
                "limitation": limitation,
                "is_diagnosis": False,
            },
            confidence=conf,
            raw_value=conf,
            sensitive=True,
        ))
    return claims


# ---------------------------------------------------------------------------
# Theme aggregation claims
# ---------------------------------------------------------------------------

def build_theme_claims(themes: list[dict]) -> list[dict]:
    claims = []
    for theme in themes:
        score = float(theme.get("theme_score", 0.0))
        desc  = theme.get("description", "")
        rules = theme.get("supporting_rules", [])
        claims.append(_make(
            "psychological_interpretation",
            (
                f"Multiple consistent indicators suggest a possible theme of {desc}. "
                f"This is supported by {len(rules)} consistent rule(s): {', '.join(rules)}. "
                "This is not diagnostic."
            ),
            "theme_aggregator",
            {"theme": theme.get("theme"), "supporting_rules": rules,
             "theme_score": score, "is_diagnosis": False},
            confidence=score,
            raw_value=score,
            sensitive=True,
        ))
    return claims


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_all_claims(doc: dict) -> list[dict]:
    """Build all claims from a populated doc dict. Returns list of claim dicts."""
    _claim_counter[0] = 0  # reset for each analysis
    claims = []
    claims += build_numeric_claims(doc)
    claims += build_object_claims(doc)
    claims += build_ocr_claims(doc)
    claims += build_emotion_claim(doc)
    claims += build_psych_claims(doc)
    return claims
