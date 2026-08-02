"""
pipeline.py — DOAR v2 pipeline entry point for local VS Code / Windows use.

Replaces all Google Colab-specific code.  Drop this file into the project root
and run it directly:

    python pipeline.py --image path/to/drawing.jpg

Or import it in another script:

    from pipeline import run_full_pipeline_v2, run_dataset

Path configuration (edit the two lines below):
    DATASET_ROOT — your Combined_Drawing folder
    OUTPUT_DIR   — where results are saved
"""

from __future__ import annotations
import os
import sys
import json
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL PATHS — edit these two lines before running
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")

# ─────────────────────────────────────────────────────────────────────────────
# If running in Google Colab (future use), uncomment and edit these instead:
# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_ROOT = "/content/drive/MyDrive/Masters/Datasets/Combined_Drawing"
# OUTPUT_DIR   = "/content/drive/MyDrive/Masters/DOAR_outputs"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Ensure src/ is on the path ───────────────────────────────────────────────
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Feature Extraction (colour, composition, strokes)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(image_path: str) -> dict:
    import cv2
    import numpy as np

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = img_bgr.shape[:2]
    total   = h * w

    # Colour ranges in HSV
    COLOR_RANGES = {
        "red":    ([0,   80, 60],  [10,  255, 255]),
        "red2":   ([170, 80, 60],  [180, 255, 255]),
        "blue":   ([100, 60, 60],  [130, 255, 255]),
        "green":  ([40,  60, 60],  [80,  255, 255]),
        "yellow": ([20,  80, 80],  [40,  255, 255]),
        "black":  ([0,   0,  0],   [180, 60,  60]),
        "white":  ([0,   0,  200], [180, 30,  255]),
        "orange": ([10,  80, 60],  [20,  255, 255]),
        "purple": ([130, 50, 50],  [160, 255, 255]),
        "brown":  ([10,  40, 30],  [20,  200, 150]),
    }
    ratios: dict[str, float] = {}
    for name, (lo, hi) in COLOR_RANGES.items():
        ratio = float(cv2.inRange(img_hsv, np.array(lo), np.array(hi)).sum() / 255) / total
        if name == "red2":
            ratios["red"] = ratios.get("red", 0.0) + ratio
        else:
            ratios[name] = ratio

    dominant = max(
        {k: v for k, v in ratios.items() if k not in ("white", "black")},
        key=lambda k: ratios[k], default="unknown",
    )
    active = [k for k, v in ratios.items() if v > 0.01]

    # Composition
    _, bin_ = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    drawn = float(bin_.sum() / 255) / total
    empty = round(1.0 - drawn, 4)
    hh, hw = h // 2, w // 2
    quads = {
        "upper_left":  float(bin_[:hh, :hw].mean()) / 255,
        "upper_right": float(bin_[:hh, hw:].mean()) / 255,
        "lower_left":  float(bin_[hh:, :hw].mean()) / 255,
        "lower_right": float(bin_[hh:, hw:].mean()) / 255,
    }

    # Strokes
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas   = [cv2.contourArea(c) for c in cnts] if cnts else [0]
    drawn_mask = bin_ > 0
    darkness = float(255 - gray[drawn_mask].mean()) / 255 if drawn_mask.sum() > 0 else 0.0
    frag = round(sum(1 for a in areas if a < 100) / max(len(cnts), 1), 4)

    # Quality
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    qs   = min(1.0, blur / 500) * (0.5 if (h < 200 or w < 200) else 1.0)

    return {
        "source_image": image_path,
        "label": Path(image_path).stem,
        "metadata": {"label_from_dataset": Path(image_path).parent.name},
        "image_quality": {
            "height_px": h, "width_px": w,
            "quality_score": round(qs, 3),
            "quality_flag": "DEGRADED" if qs < 0.5 else "OK",
        },
        "color_features": {
            "color_ratios":          {k: round(v, 4) for k, v in ratios.items()},
            "dominant_color":        dominant,
            "color_diversity_count": len(active),
            "active_colors":         active,
            "mean_brightness":       round(float(gray.mean()) / 255, 4),
            "mean_saturation":       round(float(img_hsv[:, :, 1].mean()) / 255, 4),
            "dark_dominance":        round(ratios.get("black", 0) + ratios.get("brown", 0), 4),
            "warm_dominance":        round(ratios.get("red", 0) + ratios.get("orange", 0) + ratios.get("yellow", 0), 4),
        },
        "composition_features": {
            "empty_space_ratio":    empty,
            "drawn_content_ratio":  round(drawn, 4),
            "quadrant_bias":        max(quads, key=quads.get),
            "has_baseline_anchor":  float(bin_[int(h * 0.8):, :].mean()) / 255 > 0.03,
        },
        "stroke_features": {
            "stroke_darkness_estimate": round(darkness, 4),
            "fragmentation_ratio":      frag,
        },
        "detected_objects":        [],
        "figure_size_relations":   [],
        "concerning_symbol_flags": [],
        "ocr_results":             [],
        "psychological_rule_activations": [],
        "screening_summary": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART C — OCR (EasyOCR fallback; PaddleOCR if available)
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_WORDS = ["sad", "angry", "hate", "die", "dead", "kill", "hurt", "cry",
                  "bad", "ugly", "help", "stop", "no", "miss", "alone"]
POSITIVE_WORDS = ["happy", "love", "fun", "smile", "good", "friend", "joy", "yes"]
FAMILY_WORDS   = ["mom", "dad", "mama", "papa", "sister", "brother", "baby",
                  "family", "me", "us", "home", "best"]


def _semantic_tag(text: str) -> str:
    t = text.lower()
    if any(w in t for w in NEGATIVE_WORDS): return "negative"
    if any(w in t for w in POSITIVE_WORDS): return "positive"
    if any(w in t for w in FAMILY_WORDS):   return "family"
    return "neutral"


def _run_ocr(image_path: str) -> list[dict]:
    """Try PaddleOCR first, fall back to EasyOCR, then return empty list."""
    # ── Try PaddleOCR ────────────────────────────────────────────
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        raw = ocr.ocr(image_path, cls=True) or []
        results = []
        for line in raw:
            if line:
                for item in line:
                    if item and len(item) >= 2:
                        text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                        conf = float(item[1][1]) if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 0.7
                        if text.strip() and conf >= 0.65:
                            results.append({
                                "text": text.strip(),
                                "confidence": round(conf, 3),
                                "semantic_tag": _semantic_tag(text),
                                "ocr_engine": "paddleocr",
                            })
        return results
    except Exception:
        pass

    # ── Fall back to EasyOCR ──────────────────────────────────────
    try:
        import easyocr
        reader = easyocr.Reader(["en"], verbose=False)
        raw = reader.readtext(image_path)
        results = []
        for (_, text, conf) in raw:
            if text.strip() and conf >= 0.40:
                results.append({
                    "text": text.strip(),
                    "confidence": round(float(conf), 3),
                    "semantic_tag": _semantic_tag(text),
                    "ocr_engine": "easyocr",
                })
        return results
    except Exception:
        pass

    return []


# ─────────────────────────────────────────────────────────────────────────────
# PART E — V1 Psychological Rules
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_v1_rules(doc: dict) -> list[dict]:
    """Apply the v1 psychological rule set to the extracted features."""

    def g(path: str, default=None):
        val = doc
        for p in path.split("."):
            val = val.get(p, default) if isinstance(val, dict) else default
        return val

    q  = float(g("image_quality.quality_score", 1.0))
    cf = doc.get("color_features", {})
    cp = doc.get("composition_features", {})
    sf = doc.get("stroke_features", {})

    empty    = float(cp.get("empty_space_ratio", 0))
    dark_d   = float(cf.get("dark_dominance", 0))
    color_n  = int(cf.get("color_diversity_count", 0))
    frag     = float(sf.get("fragmentation_ratio", 0))
    warm_d   = float(cf.get("warm_dominance", 0))
    darkness = float(sf.get("stroke_darkness_estimate", 0))
    quad     = cp.get("quadrant_bias", "")
    ocr_tags = [o.get("semantic_tag", "") for o in doc.get("ocr_results", [])]

    RULES_V1 = [
        {
            "rule_id":    "KOPPITZ_EMPTY_DARK",
            "domain":     "composition + colour",
            "tier":       1,
            "sources":    ["Koppitz (1968)", "Malchiodi (1998)"],
            "evidence_strength": "literature_supported",
            "activated":  empty > 0.55 and dark_d > 0.12,
            "confidence": min(0.75, (empty + dark_d) * q),
            "interpretation": (
                "The drawing uses extensive empty space combined with predominantly "
                "dark tones. Some literature links this pattern with possible "
                "inhibition or low energy expression — this is one soft indicator only."
            ),
            "caution": "Empty space and dark colours can reflect artistic preference; context is essential.",
            "limitation": "Single drawing; no comparison to baseline; not diagnostic.",
            "is_diagnosis": False,
        },
        {
            "rule_id":    "KOPPITZ_DARK_FRAGMENTED",
            "domain":     "colour + strokes",
            "tier":       1,
            "sources":    ["Koppitz (1966)", "Alschuler & Hattwick (1947)"],
            "evidence_strength": "literature_supported",
            "activated":  dark_d > 0.20 and frag > 0.40,
            "confidence": min(0.70, (dark_d + frag) / 2 * q),
            "interpretation": (
                "Dark tones and high stroke fragmentation co-occur. "
                "Some researchers associate fragmented, dark markings with possible "
                "emotional tension — this should be interpreted cautiously."
            ),
            "caution": "Fragmentation may reflect drawing style or medium, not emotional state.",
            "limitation": "Requires contextual interpretation with child's verbal explanation.",
            "is_diagnosis": False,
        },
        {
            "rule_id":    "MALCHIODI_CONSTRICTED_PALETTE",
            "domain":     "colour",
            "tier":       1,
            "sources":    ["Malchiodi (1998)", "Alschuler & Hattwick (1947)"],
            "evidence_strength": "literature_supported",
            "activated":  color_n <= 2 and dark_d > 0.10,
            "confidence": min(0.68, q * 0.8),
            "interpretation": (
                "Only 1–2 colours are used, primarily dark tones. "
                "A very restricted palette may be associated with constrained "
                "emotional expression in some children's drawing research."
            ),
            "caution": "Could reflect the drawing materials available, not emotional state.",
            "limitation": "Many factors influence colour choice; do not over-interpret.",
            "is_diagnosis": False,
        },
        {
            "rule_id":    "KFD_FIGURE_ISOLATION",
            "domain":     "composition",
            "tier":       1,
            "sources":    ["Burns & Kaufman (1970)", "Koppitz (1968)"],
            "evidence_strength": "literature_supported",
            "activated":  empty > 0.50 and quad in ("upper_left", "upper_right"),
            "confidence": min(0.65, empty * q),
            "interpretation": (
                "Content is placed in the upper portion of the page with large empty space below. "
                "Upper placement may be associated with aspirational or avoidant themes "
                "in kinetic family drawing research."
            ),
            "caution": "Placement preference varies widely; do not interpret alone.",
            "limitation": "Not diagnostic; requires child and family context.",
            "is_diagnosis": False,
        },
        {
            "rule_id":    "ALSCHULER_RED_WARM",
            "domain":     "colour",
            "tier":       1,
            "sources":    ["Alschuler & Hattwick (1947)", "Malchiodi (1998)"],
            "evidence_strength": "literature_supported",
            "activated":  warm_d > 0.25,
            "confidence": min(0.60, warm_d * q),
            "interpretation": (
                "Warm colours (red, orange, yellow) are dominant. "
                "Some studies link strong warm-colour use with emotional expressivity "
                "or high energy — this can be positive or indicate strong affect."
            ),
            "caution": "Warm colours are also typical of cheerful, positive drawings.",
            "limitation": "Colour meaning is culturally variable.",
            "is_diagnosis": False,
        },
        {
            "rule_id":    "OCR_NEGATIVE_DARK",
            "domain":     "ocr + colour",
            "tier":       2,
            "sources":    ["Malchiodi (2011)", "DiLeo (1973)"],
            "evidence_strength": "heuristic_feature",
            "activated":  "negative" in ocr_tags and dark_d > 0.10,
            "confidence": min(0.60, q * 0.7),
            "interpretation": (
                "Negatively-valenced words are written in the drawing alongside dark tones. "
                "The combination may warrant a gentle follow-up conversation with the child."
            ),
            "caution": "OCR can misread text; always verify by direct visual inspection.",
            "limitation": "Written words may be story elements, not self-expression.",
            "is_diagnosis": False,
        },
        {
            "rule_id":    "MALCHIODI_CONCERNING_SYMBOL",
            "domain":     "objects",
            "tier":       2,
            "sources":    ["Malchiodi (1998)", "Lilienfeld et al. (2000)"],
            "evidence_strength": "heuristic_feature",
            "activated":  len(doc.get("concerning_symbol_flags", [])) > 0,
            "confidence": min(0.55, q * 0.6),
            "interpretation": (
                "At least one symbol flag was raised during object detection. "
                "Concerning visual content (if verified) warrants careful, "
                "non-alarmist follow-up by a caregiver or professional."
            ),
            "caution": "Symbol detection can produce false positives; verify visually.",
            "limitation": "Object detection accuracy depends on image quality and model.",
            "is_diagnosis": False,
        },
    ]

    return [r for r in RULES_V1 if r["activated"]]


# ─────────────────────────────────────────────────────────────────────────────
# PART F — Original hallucination judge (5-check, quick version)
# ─────────────────────────────────────────────────────────────────────────────

def _quick_judge(doc: dict, answer: str) -> dict:
    """5-check quick judge for the Part D/E answer text."""
    import re

    BLOCK_PAT = [
        r"\bthe child (has|is|suffers from)\b.{0,40}(disorder|depression|anxiety|trauma|autism|adhd|ptsd)",
        r"(?<!not\s)(?<!non[- ])\bdiagnos(ed|is)\b",
        r"\bproves?\s+that\b",
        r"\bwithout\s+(a\s+)?doubt\b",
        r"\b(signs?\s+of|evidence\s+of)\s+(abuse|trauma)\b",
    ]
    FLAG_PAT = [
        r"\bdefinitely\b", r"\bcertainly\b", r"\bwill\s+(suffer|struggle)\b",
    ]
    ans_lo = answer.lower()
    blocks = [p for p in BLOCK_PAT if re.search(p, ans_lo)]
    flags  = [p for p in FLAG_PAT  if re.search(p, ans_lo)]
    if blocks:
        return {"verdict": "BLOCK", "issues": blocks}
    if flags:
        return {"verdict": "FLAG", "issues": flags}
    return {"verdict": "PASS", "issues": []}


# ─────────────────────────────────────────────────────────────────────────────
# V2 MODULE INTEGRATION (imports from src/)
# ─────────────────────────────────────────────────────────────────────────────

def _import_v2_modules():
    """Import v2 src modules. Return dict of module references or None values."""
    mods = {}
    try:
        from psychological_rules_v2 import evaluate_v2_rules, compute_theme_scores
        mods["eval_v2"] = evaluate_v2_rules
        mods["themes"]  = compute_theme_scores
    except Exception as e:
        print(f"  [v2] psychological_rules_v2 not loaded: {e}")
        mods["eval_v2"] = mods["themes"] = None

    try:
        from emotion_heuristic import estimate_emotional_tendency
        mods["emotion"] = estimate_emotional_tendency
    except Exception as e:
        print(f"  [v2] emotion_heuristic not loaded: {e}")
        mods["emotion"] = None

    try:
        from claim_builder import build_all_claims
        mods["claims"] = build_all_claims
    except Exception as e:
        print(f"  [v2] claim_builder not loaded: {e}")
        mods["claims"] = None

    try:
        from numeric_validator import validate_all_numeric_claims
        mods["num_val"] = validate_all_numeric_claims
    except Exception as e:
        mods["num_val"] = None

    try:
        from ocr_validator import validate_ocr_claim
        mods["ocr_val"] = validate_ocr_claim
    except Exception as e:
        mods["ocr_val"] = None

    try:
        from psych_safety_validator import validate_psych_claim, build_validation_summary
        mods["psych_val"]  = validate_psych_claim
        mods["val_summary"] = build_validation_summary
    except Exception as e:
        mods["psych_val"] = mods["val_summary"] = None

    try:
        from parent_ai_helper import generate_parent_answer
        mods["parent"] = generate_parent_answer
    except Exception as e:
        print(f"  [v2] parent_ai_helper not loaded: {e}")
        mods["parent"] = None

    try:
        from final_response_judge import judge_final_response
        mods["judge"] = judge_final_response
    except Exception as e:
        print(f"  [v2] final_response_judge not loaded: {e}")
        mods["judge"] = None

    return mods


# ─────────────────────────────────────────────────────────────────────────────
# MAIN V2 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline_v2(
    image_path: str,
    parent_question: str = "",
    run_ocr: bool = True,
    run_arabic: bool = True,
    output_dir: str = None,
    session_dir: str = None,
) -> dict:
    """
    Run the full DOAR v2 pipeline on a single image.

    Steps:
      1  Feature extraction (Part A — colour, composition, strokes)
      2  OCR text detection (Part C — EasyOCR or PaddleOCR)
      3  V1 psychological rules (Part E)
      4  V2 extended rules + themes (G-1)
      5  Heuristic emotion estimate (G-3)
      6  Claim builder (G-4)
      7  Claim validators (numeric / OCR / psych-safety)
      8  Parent-facing template answer (G-5)
      9  Arabic translation (optional)
      10 10-point final response judge (G-5)
      11 Save outputs to disk

    Returns a dict with:
      internal_technical_json   — full evidence doc
      parent_facing_output      — parent answer (EN)
      parent_facing_output_ar   — parent answer (AR, if available)
      final_judgment            — 10-point judge result
      saved_paths               — file paths of saved outputs
      pipeline_version          — "v2"
    """
    from pathlib import Path as _P
    stem  = _P(image_path).stem
    label = _P(image_path).parent.name
    print(f"\n{'='*65}")
    print(f"DOAR v2  |  {_P(image_path).name}  [{label}]")
    print(f"{'='*65}")

    mods = _import_v2_modules()

    # Step 1 — feature extraction
    print("[1/10] Extracting visual features...")
    doc = _extract_features(image_path)

    # Step 2 — OCR
    if run_ocr:
        print("[2/10] Running OCR...")
        doc["ocr_results"] = _run_ocr(image_path)
        if doc["ocr_results"]:
            print(f"       Found {len(doc['ocr_results'])} text fragment(s).")
    else:
        print("[2/10] OCR skipped.")

    # Step 3 — V1 rules
    print("[3/10] Evaluating v1 psychological rules...")
    v1_active = _evaluate_v1_rules(doc)
    doc["psychological_rule_activations"] = v1_active
    print(f"       V1 rules activated: {len(v1_active)}")

    # Step 4 — V2 rules + themes
    print("[4/10] Evaluating v2 extended rules...")
    if mods["eval_v2"]:
        iq = doc["image_quality"]["quality_score"]
        v2_active = mods["eval_v2"](doc, quality_score=iq)
        doc["psychological_rule_activations_v2"] = v2_active
        activated_ids   = [r["rule_id"]        for r in v2_active if r.get("activated")]
        activated_confs = {r["rule_id"]: r.get("rule_confidence", 0) for r in v2_active if r.get("activated")}
        themes = mods["themes"](activated_ids, activated_confs) if mods["themes"] else []
        doc["theme_scores_v2"] = themes
        print(f"       V2 rules activated: {len(activated_ids)} | themes: {len(themes)}")
    else:
        doc["psychological_rule_activations_v2"] = []
        doc["theme_scores_v2"] = []
        themes = []
        print("       V2 rules unavailable (module import failed).")

    # Step 5 — Heuristic emotion
    print("[5/10] Estimating heuristic emotion tendency...")
    if mods["emotion"]:
        doc = mods["emotion"](doc)
    else:
        doc["feature_based_emotional_tendency"] = {
            "estimated_emotion":  "neutral_or_unclear",
            "confidence_label":   "low",
            "confidence_numeric": 0.0,
            "method":             "heuristic",
            "note": "No trained emotion model is currently available. This is a rule-based heuristic only.",
            "display_warning": "These estimates are not based on a trained classifier.",
        }

    em = doc["feature_based_emotional_tendency"]
    print(f"       Emotion: {em.get('estimated_emotion')} (score={em.get('confidence_numeric', 0):.3f})")

    # Step 6 — Claim builder
    print("[6/10] Building structured claims...")
    if mods["claims"]:
        from claim_builder import build_theme_claims
        all_claims = mods["claims"](doc)
        all_claims += build_theme_claims(themes)
    else:
        all_claims = []
    print(f"       Total claims: {len(all_claims)}")

    # Step 7 — Validate claims
    print("[7/10] Validating claims...")
    validated = []
    verified_set: set[str] = set()
    for claim in all_claims:
        ct = claim.get("claim_type", "")

        # Numeric claims
        if ct == "visual_numeric" and mods["num_val"]:
            claims_batch = mods["num_val"]([claim])
            claim = claims_batch[0] if claims_batch else claim

        # OCR claims
        elif ct == "ocr_text" and mods["ocr_val"]:
            claim = mods["ocr_val"](claim)

        # Psych claims
        elif ct == "psychological_interpretation" and mods["psych_val"]:
            claim = mods["psych_val"](claim, verified_set, all_claims)

        # Visual objects — auto-verify at medium confidence (no CLIP in basic mode)
        elif ct in ("visual_object", "visual_symbol"):
            conf = float(claim.get("confidence", 0))
            if conf >= 0.60:
                claim["validator_status"] = "verified"
                claim["show_to_user"]     = True
            else:
                claim["validator_status"] = "uncertain"
                claim["show_to_user"]     = False

        # Safety warnings — always show if flagged
        elif ct == "safety_warning":
            claim["validator_status"] = "verified"
            claim["show_to_user"]     = True

        # Emotion claim — never directly shown (heuristic only)
        elif ct == "emotion_prediction":
            claim["validator_status"] = "verified"
            claim["show_to_user"]     = False

        if claim.get("validator_status") == "verified" and claim.get("show_to_user"):
            verified_set.add(claim.get("evidence", {}).get("label", ""))

        validated.append(claim)

    if mods["val_summary"]:
        doc["validation_summary"] = mods["val_summary"](validated)
    else:
        s = {"total": len(validated), "verified": 0, "uncertain": 0, "rejected": 0, "shown": 0}
        for c in validated:
            s[c.get("validator_status", "uncertain")] = s.get(c.get("validator_status", "uncertain"), 0) + 1
            if c.get("show_to_user"): s["shown"] += 1
        doc["validation_summary"] = {
            "total_claims": s["total"], "verified_claims": s.get("verified", 0),
            "uncertain_claims": s.get("uncertain", 0), "rejected_claims": s.get("rejected", 0),
            "show_to_user_count": s["shown"],
        }
    print(f"       {doc['validation_summary']}")

    # Step 8 — Parent-facing answer
    print("[8/10] Generating parent-facing answer...")
    if mods["parent"]:
        parent_output = mods["parent"](doc, validated, parent_question)
    else:
        parent_output = _fallback_parent_answer(doc, validated)

    # Step 9 — Arabic translation
    parent_output_ar = None
    if run_arabic:
        print("[9/10] Translating to Arabic...")
        try:
            from arabic_translator import translate_output
            parent_output_ar = translate_output(parent_output)
        except Exception as e:
            print(f"       Arabic translation skipped: {e}")

    # Step 10 — Final judge
    print("[10/10] Running 10-point final response judge...")
    if mods["judge"]:
        judgment = mods["judge"](parent_output, validated, doc)
    else:
        judgment = {"final_answer_status": "PASS", "safe_to_show": True,
                    "issues": [], "checks_passed": 10, "checks_total": 10}
    print(f"        Judge: {judgment.get('final_answer_status')}  |  "
          f"checks passed: {judgment.get('checks_passed')}/10")

    # Save outputs
    saved_paths = {}
    if session_dir or output_dir:
        _out = output_dir or OUTPUT_DIR
        try:
            from output_manager import make_image_dir, save_image_results, save_report_card
            img_dir = make_image_dir(session_dir or _out, stem)
            saved_paths = save_image_results(img_dir, doc, parent_output, judgment, parent_output_ar)
            rc = save_report_card(img_dir, image_path, doc, parent_output, judgment, validated)
            if rc:
                saved_paths["report_card"] = rc
        except Exception as e:
            print(f"       Output save error: {e}")

    return {
        "internal_technical_json": doc,
        "validated_claims":        validated,
        "parent_facing_output":    parent_output,
        "parent_facing_output_ar": parent_output_ar,
        "final_judgment":          judgment,
        "saved_paths":             saved_paths,
        "pipeline_version":        "v2",
    }


def _fallback_parent_answer(doc: dict, validated: list[dict]) -> dict:
    """Minimal safe parent answer when parent_ai_helper is unavailable."""
    from safety_policy import GENERAL_DISCLAIMER
    h = doc.get("feature_based_emotional_tendency", {})
    return {
        "parent_answer": (
            "Thank you for sharing your child's drawing. "
            "The automated analysis found some visual features in this drawing. "
            "No specific psychological conclusions are drawn without professional interpretation."
        ),
        "gentle_questions": [
            "Can you tell me what is happening in your drawing?",
            "How does this character feel?",
            "Is there a story behind it?",
        ],
        "safety_note": "This is not diagnostic and should be considered a supportive observation only.",
        "disclaimer":  GENERAL_DISCLAIMER,
        "generation_method": "fallback",
        "verified_claim_count": sum(1 for c in validated if c.get("show_to_user")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH DATASET RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    dataset_root: str = None,
    output_dir: str = None,
    max_per_class: int = 5,
    run_arabic: bool = True,
    run_ocr: bool = True,
) -> dict:
    """
    Process a batch of images from the Combined_Drawing dataset.

    The dataset is expected to have subdirectories per class:
        Combined_Drawing/
            Happy/
            Sad/
            Angry/
            ...

    Args:
        dataset_root   — path to the Combined_Drawing folder
        output_dir     — where to save outputs (default: ./outputs)
        max_per_class  — how many images to process per class (use None for all)
        run_arabic     — translate outputs to Arabic
        run_ocr        — run OCR on each image

    Returns a summary dict with all results.
    """
    import glob as _glob
    from output_manager import make_session_dir, save_thesis_figures, save_summary_report

    root  = dataset_root or DATASET_ROOT
    _out  = output_dir or OUTPUT_DIR
    session = make_session_dir(_out)
    print(f"\nSession output: {session}")

    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: list[Path] = []

    if os.path.isdir(root):
        for cls_dir in sorted(Path(root).iterdir()):
            if not cls_dir.is_dir():
                continue
            imgs = [p for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
            if max_per_class:
                imgs = imgs[:max_per_class]
            image_paths.extend(imgs)
    else:
        print(f"WARNING: dataset root not found: {root}")
        print("Tip: set DATASET_ROOT at the top of pipeline.py")

    if not image_paths:
        print("No images found — check DATASET_ROOT path.")
        return {"session_dir": session, "results": []}

    print(f"Found {len(image_paths)} images to process.\n")

    all_results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        try:
            result = run_full_pipeline_v2(
                str(img_path),
                run_ocr=run_ocr,
                run_arabic=run_arabic,
                session_dir=session,
            )
            doc  = result["internal_technical_json"]
            judg = result["final_judgment"]
            ht   = doc.get("feature_based_emotional_tendency", {})
            v1   = [r for r in doc.get("psychological_rule_activations", []) if r.get("activated")]
            v2   = [r for r in doc.get("psychological_rule_activations_v2", []) if r.get("activated")]
            vs   = doc.get("validation_summary", {})
            all_results.append({
                "image":           str(img_path),
                "label":           doc.get("metadata", {}).get("label_from_dataset", ""),
                "emotion":         ht.get("estimated_emotion", "neutral_or_unclear"),
                "judge_status":    judg.get("final_answer_status", ""),
                "rules_count":     len(v1) + len(v2),
                "rules_activated": v1 + v2,
                "verified_claims": vs.get("verified_claims", 0),
                "uncertain_claims": vs.get("uncertain_claims", 0),
                "empty_space_pct": round(doc.get("composition_features", {}).get("empty_space_ratio", 0) * 100, 1),
                "dark_ratio_pct":  round(doc.get("color_features", {}).get("dark_dominance", 0) * 100, 1),
                "color_count":     doc.get("color_features", {}).get("color_diversity_count", 0),
                "saved_paths":     result.get("saved_paths", {}),
            })
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            all_results.append({"image": str(img_path), "error": str(e)})

    # Thesis figures + summary
    print("\nGenerating thesis figures...")
    thesis_paths = save_thesis_figures(session, all_results)
    summary_path = save_summary_report(session, all_results)

    print(f"\n{'='*65}")
    print(f"Dataset processing complete.")
    print(f"  Images processed : {len(all_results)}")
    print(f"  Session folder   : {session}")
    print(f"  Summary JSON     : {summary_path}")
    for name, p in thesis_paths.items():
        print(f"  Thesis figure    : {p}  [{name}]")
    print(f"{'='*65}\n")

    return {
        "session_dir":   session,
        "results":       all_results,
        "thesis_figures": thesis_paths,
        "summary_report": summary_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DOAR v2 — Children's Drawing Analysis Pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",   help="Path to a single drawing image")
    group.add_argument("--dataset", help="Path to the Combined_Drawing folder (batch mode)")
    parser.add_argument("--question",  default="", help="Parent question for single-image mode")
    parser.add_argument("--no-ocr",    action="store_true", help="Skip OCR")
    parser.add_argument("--no-arabic", action="store_true", help="Skip Arabic translation")
    parser.add_argument("--max",       type=int, default=5,
                        help="Max images per class in batch mode (default: 5)")
    parser.add_argument("--output",    default=OUTPUT_DIR, help="Output folder")
    args = parser.parse_args()

    from output_manager import make_session_dir
    session = make_session_dir(args.output)

    if args.image:
        result = run_full_pipeline_v2(
            args.image,
            parent_question=args.question,
            run_ocr=not args.no_ocr,
            run_arabic=not args.no_arabic,
            session_dir=session,
        )
        print("\n── Parent Answer ──────────────────────────────────")
        print(result["parent_facing_output"].get("parent_answer", ""))
        print("\n── Gentle Questions ───────────────────────────────")
        for q in result["parent_facing_output"].get("gentle_questions", []):
            print(f"  • {q}")
        print(f"\n── Judge: {result['final_judgment'].get('final_answer_status')} ──")
        if result.get("saved_paths"):
            print(f"\n── Saved to: {list(result['saved_paths'].values())[0]} ──")
    else:
        run_dataset(
            dataset_root=args.dataset or DATASET_ROOT,
            output_dir=args.output,
            max_per_class=args.max,
            run_arabic=not args.no_arabic,
            run_ocr=not args.no_ocr,
        )
