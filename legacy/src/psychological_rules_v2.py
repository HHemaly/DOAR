"""
psychological_rules_v2.py — Extended symbolic and compositional drawing rules.

This module adds the new psychiatrist/psychologist rules as a cautious,
non-diagnostic indicator layer. Rules are separated into two tiers:

  Tier 1 — Objective/compositional features (size, empty space, placement,
            color, strokes). These have higher weights and stronger literature
            support.

  Tier 2 — Symbolic/projective features (animal types, shapes, objects,
            eye expressions). These have lower weights and are explicitly
            marked as soft, culturally-dependent, non-diagnostic indicators.

Key design rules enforced here:
- is_diagnosis is always False
- No single symbolic rule produces a psychological conclusion alone
- Stronger interpretations require a CLUSTER of consistent verified features
- All activations are passed through weighted theme aggregation before output
- All interpretation text uses cautious wording only

References checked:
  Koppitz 1966/1968, Machover 1949, Buck HTP, Burns & Kaufman 1970 (KFD),
  DiLeo 1973/1983, Malchiodi 1998/2011,
  Lilienfeld et al. 2000 (Scientific Status of Projective Techniques).

Rules whose symbolic basis is not strongly supported are marked:
  evidence_strength: "weak_symbolic_indicator" or "requires_further_validation"
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
PT = CFG.get("psychological_rules", {})
RW = CFG.get("rule_weights", {})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _w(key: str) -> float:
    return float(RW.get(key, 0.25))


def _fmt_rule(rule_id, domain, tier, evidence_strength, sources,
              check_fn, interpretation_fn, caution, limitation,
              base_weight, parent_guidance=""):
    return {
        "rule_id":          rule_id,
        "domain":           domain,
        "tier":             tier,            # 1 = objective/feature, 2 = symbolic/projective
        "evidence_strength": evidence_strength,
        "sources":          sources,
        "check":            check_fn,        # callable(doc) -> bool
        "interpretation":   interpretation_fn,  # callable(doc) -> str
        "caution":          caution,
        "limitation":       limitation,
        "base_weight":      base_weight,
        "parent_guidance":  parent_guidance,
        "is_diagnosis":     False,
    }


def _detected_objects(doc) -> list[dict]:
    return doc.get("detected_objects", [])


def _top_labels(doc) -> list[str]:
    return [
        o.get("top_match", {}).get("label", "").lower()
        for o in _detected_objects(doc)
    ]


def _has_label_containing(doc, *keywords) -> bool:
    labels = _top_labels(doc)
    for lbl in labels:
        for kw in keywords:
            if kw in lbl:
                return True
    return False


def _cf(doc) -> dict:  return doc.get("color_features",      {})
def _cp(doc) -> dict:  return doc.get("composition_features", {})
def _sf(doc) -> dict:  return doc.get("stroke_features",      {})
def _iq(doc) -> dict:  return doc.get("image_quality",        {})


# ---------------------------------------------------------------------------
# Tier 1 — Objective / Compositional Rules (new additions)
# ---------------------------------------------------------------------------

TIER1_RULES = [

    _fmt_rule(
        rule_id    = "COMP_SIZE_SMALL",
        domain     = "drawing size — possible emotional inhibition (speculative)",
        tier       = 1,
        evidence_strength = "literature_supported_feature",
        sources    = ["Koppitz (1968)", "Malchiodi (1998)", "DiLeo (1973)"],
        check_fn   = lambda d: _cp(d).get("drawn_content_ratio", 1.0) < PT.get("empty_space_low", 0.20),
        interpretation_fn = lambda d: (
            f"The drawn content covers approximately "
            f"{_cp(d).get('drawn_content_ratio',0)*100:.0f}% of the page. "
            "A very small drawing may suggest emotional inhibition, low confidence, insecurity, "
            "fear, or withdrawal. It is one of the more consistently noted compositional indicators "
            "in children's drawing literature."
        ),
        caution    = (
            "Drawing size is strongly affected by age, motor skill development, available time, "
            "task instructions, and paper size. This is not diagnostic on its own."
        ),
        limitation = (
            "Small drawing size alone cannot support any clinical inference. It should only "
            "be considered alongside other features and the child's verbal explanation."
        ),
        base_weight = _w("literature_supported_feature"),
        parent_guidance = (
            "You might gently ask: 'Would you like to add more to your drawing?' to see whether "
            "the child expands willingly, which would suggest comfort rather than inhibition."
        ),
    ),

    _fmt_rule(
        rule_id    = "COMP_SIZE_LARGE",
        domain     = "drawing size — possible confidence or expansiveness (speculative)",
        tier       = 1,
        evidence_strength = "heuristic_feature",
        sources    = ["Koppitz (1968)", "Malchiodi (1998)"],
        check_fn   = lambda d: _cp(d).get("drawn_content_ratio", 0.0) > 0.75,
        interpretation_fn = lambda d: (
            f"The drawn content covers approximately "
            f"{_cp(d).get('drawn_content_ratio',0)*100:.0f}% of the page. "
            "Drawings that fill most of the available space may suggest high self-expression, "
            "confidence, a strong desire to communicate, or an expansive emotional style."
        ),
        caution    = (
            "Large drawing size may simply reflect artistic habits, task instructions, "
            "or enthusiasm for drawing. It is not a reliable indicator on its own."
        ),
        limitation = (
            "Drawing size reflects many non-psychological factors. Interpret only as a "
            "soft, contextual observation."
        ),
        base_weight = _w("heuristic_feature"),
    ),

    _fmt_rule(
        rule_id    = "COMP_SIZE_MODERATE",
        domain     = "drawing size — balanced self-expression (observational)",
        tier       = 1,
        evidence_strength = "heuristic_feature",
        sources    = ["Koppitz (1968)"],
        check_fn   = lambda d: (
            0.40 <= _cp(d).get("drawn_content_ratio", 0.0) <= 0.60
        ),
        interpretation_fn = lambda d: (
            "The drawn content covers approximately 40–60% of the page, "
            "which may suggest balanced self-expression or moderate confidence in self-presentation."
        ),
        caution    = "Moderate page coverage is common and should not be overinterpreted.",
        limitation = "This is an observational note, not a psychological indicator.",
        base_weight = _w("heuristic_feature") * 0.5,
    ),

    _fmt_rule(
        rule_id    = "COMP_PLACEMENT_UPPER",
        domain     = "page placement — upper area (imagination/idealism, speculative)",
        tier       = 1,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Buck (1948) HTP", "DiLeo (1973)", "Malchiodi (1998)"],
        check_fn   = lambda d: _cp(d).get("center_of_mass_normalized", (0.5, 0.5))[1] < 0.35,
        interpretation_fn = lambda d: (
            "The drawing is positioned in the upper portion of the page. "
            "In projective drawing literature this placement is sometimes associated with "
            "imagination, fantasy orientation, idealism, or aspiration. "
            "However, this interpretation has weak empirical support."
        ),
        caution    = (
            "Upper placement does not mean the child 'cannot adapt to reality.' "
            "Page placement is influenced by handedness, drawing habit, and task instructions. "
            "This is a speculative symbolic indicator."
        ),
        limitation = (
            "Placement-based interpretation has limited empirical validation "
            "(see Lilienfeld et al., 2000). Use only as context, not evidence."
        ),
        base_weight = _w("weak_symbolic_indicator"),
    ),

    _fmt_rule(
        rule_id    = "COMP_PLACEMENT_LEFT",
        domain     = "page placement — left side (introversion/past-orientation, speculative)",
        tier       = 1,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Buck (1948) HTP", "Machover (1949)"],
        check_fn   = lambda d: _cp(d).get("center_of_mass_normalized", (0.5, 0.5))[0] < 0.35,
        interpretation_fn = lambda d: (
            "The drawing is concentrated on the left side of the page. "
            "Some projective drawing frameworks associate left-side placement with introversion, "
            "self-focus, past orientation, or emotional caution."
        ),
        caution    = (
            "Left-side placement is strongly influenced by handedness, writing direction, and "
            "drawing habits. This is a speculative symbolic indicator with limited validation."
        ),
        limitation = (
            "Placement interpretation in projective drawing is contested "
            "(Lilienfeld et al., 2000). Treat as a very soft, contextual observation only."
        ),
        base_weight = _w("weak_symbolic_indicator"),
    ),

    _fmt_rule(
        rule_id    = "COMP_PLACEMENT_RIGHT",
        domain     = "page placement — right side (social openness, speculative)",
        tier       = 1,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Buck (1948) HTP", "Machover (1949)"],
        check_fn   = lambda d: _cp(d).get("center_of_mass_normalized", (0.5, 0.5))[0] > 0.65,
        interpretation_fn = lambda d: (
            "The drawing is concentrated on the right side of the page. "
            "Some projective frameworks associate right-side placement with future orientation, "
            "social engagement, or outward expression."
        ),
        caution    = (
            "Right-side placement may simply reflect handedness or drawing habit. "
            "This is a speculative symbolic indicator."
        ),
        limitation = "Treat as a very soft observational note only.",
        base_weight = _w("weak_symbolic_indicator"),
    ),

    _fmt_rule(
        rule_id    = "COMP_PLACEMENT_CENTER",
        domain     = "page placement — center (balance/self-focus, observational)",
        tier       = 1,
        evidence_strength = "heuristic_feature",
        sources    = ["Buck (1948) HTP"],
        check_fn   = lambda d: (
            0.35 <= _cp(d).get("center_of_mass_normalized", (0.5, 0.5))[0] <= 0.65
            and 0.35 <= _cp(d).get("center_of_mass_normalized", (0.5, 0.5))[1] <= 0.65
        ),
        interpretation_fn = lambda d: (
            "The drawing is centered on the page. "
            "Center placement is the most common and may suggest balance, stability, "
            "or straightforward self-expression."
        ),
        caution    = "Center placement is very common and should not be overinterpreted.",
        limitation = "This is an observational note only.",
        base_weight = _w("heuristic_feature") * 0.3,
    ),
]

# ---------------------------------------------------------------------------
# Tier 2 — Symbolic / Projective Rules (new additions from psychiatrist input)
# ---------------------------------------------------------------------------

TIER2_RULES = [

    # ---- Eye features (requires visual claim validation of eye region) ----
    _fmt_rule(
        rule_id    = "SYM_EYE_WIDE_OPEN",
        domain     = "eye features — wide/open eyes (curiosity/alertness, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Machover (1949)", "Koppitz (1968)", "DiLeo (1973)"],
        check_fn   = lambda d: d.get("_validated_claims", {}).get("wide_eyes", False),
        interpretation_fn = lambda d: (
            "Wide or open eyes were noted in the drawing. "
            "This may suggest openness, curiosity, social engagement, alertness, "
            "or emotional receptiveness in some projective frameworks."
        ),
        caution    = (
            "Eye size in children's drawings is strongly affected by drawing style, "
            "cartoon influence, age, and skill level. Do not over-interpret."
        ),
        limitation = (
            "Eye feature interpretation is speculative. Requires the child's own explanation "
            "and should not be used as a standalone indicator."
        ),
        base_weight = _w("weak_symbolic_indicator"),
    ),

    _fmt_rule(
        rule_id    = "SYM_EYE_ANGRY",
        domain     = "eye features — strict/angry eyes (tension/frustration, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Machover (1949)", "DiLeo (1973)"],
        check_fn   = lambda d: d.get("_validated_claims", {}).get("angry_eyes", False),
        interpretation_fn = lambda d: (
            "Strict or downward-slanting eyes were noted in the drawing. "
            "This may suggest emotional tension, frustration, vigilance, or strong emotional "
            "intensity in some projective frameworks."
        ),
        caution    = (
            "Do not label the child as aggressive based on eye shape. "
            "This is a possible visual cue only and requires the child's explanation."
        ),
        limitation = (
            "Eye expression interpretation is speculative and subject to drawing skill. "
            "This is not diagnostic."
        ),
        base_weight = _w("weak_symbolic_indicator"),
    ),

    _fmt_rule(
        rule_id    = "SYM_EYE_CLOSED",
        domain     = "eye features — closed eyes (avoidance/withdrawal, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Machover (1949)", "Koppitz (1968)"],
        check_fn   = lambda d: d.get("_validated_claims", {}).get("closed_eyes", False),
        interpretation_fn = lambda d: (
            "Closed eyes were noted in the drawn figure. "
            "In some projective frameworks this may suggest avoidance, emotional withdrawal, "
            "reluctance to engage, or discomfort with self-reflection."
        ),
        caution    = (
            "Closed eyes are a common stylistic choice and may simply be cartoon-like "
            "or scene-appropriate. This is not diagnostic."
        ),
        limitation = (
            "Closed eye interpretation requires the child's explanation and is highly "
            "context-dependent."
        ),
        base_weight = _w("weak_symbolic_indicator"),
    ),

    # ---- Animal symbols ----
    _fmt_rule(
        rule_id    = "SYM_ANIMAL_TIGER_WOLF",
        domain     = "animal symbol — tiger/wolf (power/threat/defense, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Malchiodi (1998, 2011)", "DiLeo (1973)"],
        check_fn   = lambda d: _has_label_containing(d, "tiger", "wolf"),
        interpretation_fn = lambda d: (
            "A tiger or wolf figure was detected. "
            "Animal symbols are highly culturally and personally variable. "
            "In some projective frameworks, predator animals may be associated with themes "
            "of anger, threat perception, fear, power, defense, or protective aggressive energy."
        ),
        caution    = (
            "Animal drawings depend heavily on the child's story and cultural background. "
            "Do not characterise the child as aggressive or threatened based on this alone."
        ),
        limitation = (
            "Animal symbol interpretation is speculative and requires validation by the child's "
            "verbal explanation. This is not diagnostic."
        ),
        base_weight = _w("symbolic_with_support"),
    ),

    _fmt_rule(
        rule_id    = "SYM_ANIMAL_FOX",
        domain     = "animal symbol — fox (cleverness/caution/secrecy, speculative)",
        tier       = 2,
        evidence_strength = "requires_further_validation",
        sources    = [],  # no strong primary source found
        check_fn   = lambda d: _has_label_containing(d, "fox"),
        interpretation_fn = lambda d: (
            "A fox figure was detected. "
            "In some cultural and projective drawing contexts, fox imagery may be loosely "
            "associated with cleverness, caution, strategic thinking, or secrecy. "
            "However, this interpretation has very limited empirical support."
        ),
        caution    = (
            "Do not describe the child as malicious, deceptive, or bad based on this symbol. "
            "This interpretation is culturally variable and requires the child's story."
        ),
        limitation = (
            "Fox symbolism interpretation is speculative and lacks strong empirical validation. "
            "Treat as a very low-confidence, contextual observation only."
        ),
        base_weight = _w("symbolic_speculative"),
    ),

    _fmt_rule(
        rule_id    = "SYM_ANIMAL_SQUIRREL",
        domain     = "animal symbol — squirrel (need for safety/care, speculative)",
        tier       = 2,
        evidence_strength = "requires_further_validation",
        sources    = [],
        check_fn   = lambda d: _has_label_containing(d, "squirrel"),
        interpretation_fn = lambda d: (
            "A squirrel figure was detected. "
            "In some drawing interpretation frameworks, small, gentle animals may be loosely "
            "associated with a need for protection, care, safety, or emotional support. "
            "This interpretation is highly speculative."
        ),
        caution    = (
            "A squirrel drawing may simply reflect the child's preference for cute animals "
            "or nature themes. Do not over-interpret."
        ),
        limitation = (
            "This symbolic interpretation lacks strong empirical support. "
            "Treat as a very low-confidence, contextual note only."
        ),
        base_weight = _w("symbolic_speculative"),
    ),

    _fmt_rule(
        rule_id    = "SYM_ANIMAL_LION",
        domain     = "animal symbol — lion (strength/dominance/confidence, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Malchiodi (1998)", "DiLeo (1973)"],
        check_fn   = lambda d: _has_label_containing(d, "lion"),
        interpretation_fn = lambda d: (
            "A lion figure was detected. "
            "In some projective frameworks, lion imagery may be associated with themes of "
            "strength, confidence, pride, a desire for self-protection, or dominant identity."
        ),
        caution    = (
            "Do not automatically interpret a lion as narcissism or superiority. "
            "Lion drawings are common and may reflect cultural heroes or favorite animals."
        ),
        limitation = (
            "Animal symbol interpretation is speculative and requires the child's explanation."
        ),
        base_weight = _w("symbolic_with_support"),
    ),

    # ---- Geometric shapes and symbols ----
    _fmt_rule(
        rule_id    = "SYM_SHAPE_GEOMETRIC_FREQUENT",
        domain     = "geometric shapes — planning/control/structure (speculative)",
        tier       = 2,
        evidence_strength = "heuristic_feature",
        sources    = ["Malchiodi (1998)", "DiLeo (1973)"],
        check_fn   = lambda d: d.get("_validated_claims", {}).get("frequent_geometric_shapes", False),
        interpretation_fn = lambda d: (
            "Frequent geometric shapes were noted throughout the drawing. "
            "In some drawing interpretation frameworks, geometric patterns may suggest "
            "planning, structure, goal orientation, persistence, or a preference for control."
        ),
        caution    = (
            "Geometric shapes are very common and may reflect school habits, design preference, "
            "or a simple drawing style. This should not be overinterpreted."
        ),
        limitation = "Geometric shape interpretation is speculative and context-dependent.",
        base_weight = _w("heuristic_feature"),
    ),

    _fmt_rule(
        rule_id    = "SYM_SHAPE_STARS",
        domain     = "symbol — stars (desire for recognition/hope, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Malchiodi (1998)"],
        check_fn   = lambda d: _has_label_containing(d, "star"),
        interpretation_fn = lambda d: (
            "Star symbols were detected. "
            "In some drawing interpretation frameworks, stars may loosely suggest a desire "
            "for attention, admiration, achievement, hope, or wanting to be noticed."
        ),
        caution    = "Stars are very common decorative elements. Do not overinterpret.",
        limitation = "Star symbol interpretation has very limited empirical support.",
        base_weight = _w("symbolic_with_support"),
    ),

    _fmt_rule(
        rule_id    = "SYM_SHAPE_FLOWERS_CLOUDS_SUN",
        domain     = "symbols — flowers/clouds/sun (positivity/imagination, soft indicator)",
        tier       = 2,
        evidence_strength = "heuristic_feature",
        sources    = ["Malchiodi (1998)", "DiLeo (1973)"],
        check_fn   = lambda d: _has_label_containing(d, "flower", "cloud", "sun"),
        interpretation_fn = lambda d: (
            "Positive nature symbols (flowers, clouds, or sun) were detected. "
            "These may suggest imagination, positivity, a pleasant mood, hopefulness, "
            "or emotional warmth in the child's expression."
        ),
        caution    = (
            "These symbols are very common in children's drawings and their interpretation "
            "depends heavily on color, context, and the child's explanation."
        ),
        limitation = "Soft, contextual indicator only.",
        base_weight = _w("heuristic_feature"),
    ),

    _fmt_rule(
        rule_id    = "SYM_SHAPE_CIRCLES",
        domain     = "symbol — circles (loneliness/containment, speculative)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Machover (1949)", "Malchiodi (1998)"],
        check_fn   = lambda d: d.get("_validated_claims", {}).get("prominent_circles", False),
        interpretation_fn = lambda d: (
            "Prominent circular shapes were detected. "
            "In some projective frameworks, isolated or repeated circles may loosely suggest "
            "loneliness, a desire for closeness, containment, or safety-seeking."
        ),
        caution    = (
            "Circles are extremely common shapes in children's drawings and should not be "
            "overinterpreted without consistent supporting features."
        ),
        limitation = "Symbolic circle interpretation is speculative and context-dependent.",
        base_weight = _w("symbolic_with_support"),
    ),

    # ---- Transportation / movement symbols ----
    _fmt_rule(
        rule_id    = "SYM_TRANSPORT_VEHICLE",
        domain     = "symbol — transportation (movement/freedom/change, soft indicator)",
        tier       = 2,
        evidence_strength = "weak_symbolic_indicator",
        sources    = ["Malchiodi (1998)", "DiLeo (1973)"],
        check_fn   = lambda d: _has_label_containing(d, "car", "bus", "train", "airplane", "plane", "boat", "ship", "truck"),
        interpretation_fn = lambda d: (
            "A transportation object was detected. "
            "In some drawing contexts, vehicles may loosely suggest themes of movement, "
            "freedom, travel, adventure, social openness, or desire for change."
        ),
        caution    = (
            "Transportation drawings are extremely common and may simply reflect the child's "
            "interest in vehicles. Do not over-interpret without supporting features."
        ),
        limitation = "Soft, context-dependent symbolic indicator.",
        base_weight = _w("symbolic_with_support"),
    ),

    # ---- Heart symbol ----
    _fmt_rule(
        rule_id    = "SYM_HEART",
        domain     = "symbol — hearts (affection/connection, soft indicator)",
        tier       = 2,
        evidence_strength = "heuristic_feature",
        sources    = ["Malchiodi (1998)"],
        check_fn   = lambda d: _has_label_containing(d, "heart"),
        interpretation_fn = lambda d: (
            "Heart symbols were detected. "
            "Hearts may suggest affection, emotional warmth, love, social connection, "
            "or a desire for closeness."
        ),
        caution    = "Hearts are very common decorative symbols. May simply be ornamental.",
        limitation = "Soft, contextual indicator only.",
        base_weight = _w("heuristic_feature"),
    ),
]

# Combined list of all v2 rules
ALL_V2_RULES = TIER1_RULES + TIER2_RULES


# ---------------------------------------------------------------------------
# Theme aggregation
# ---------------------------------------------------------------------------

THEME_WEIGHTS = {
    "possible_inhibition_withdrawal":  {
        "rules": ["COMP_SIZE_SMALL", "KOPPITZ_EMPTY_DARK", "KOPPITZ_LOWER_LEFT_SPARSE",
                  "MALCHIODI_CONSTRICTED_PALETTE", "KFD_FIGURE_ISOLATION",
                  "SYM_EYE_CLOSED", "COMP_PLACEMENT_LEFT"],
        "min_cluster": 2,
        "description": "emotional inhibition or withdrawal (not diagnostic)",
    },
    "possible_anxiety_tension": {
        "rules": ["KOPPITZ_DARK_FRAGMENTED", "SYM_EYE_ANGRY",
                  "SYM_ANIMAL_TIGER_WOLF", "SYM_ANIMAL_LION"],
        "min_cluster": 2,
        "description": "anxiety or emotional tension (not diagnostic)",
    },
    "possible_positive_expression": {
        "rules": ["COMP_SIZE_LARGE", "ALSCHULER_RED_WARM", "SYM_HEART",
                  "SYM_SHAPE_FLOWERS_CLOUDS_SUN", "SYM_EYE_WIDE_OPEN"],
        "min_cluster": 2,
        "description": "positive, energetic, or warm emotional expression (not diagnostic)",
    },
    "possible_high_intensity": {
        "rules": ["ALSCHULER_RED_WARM", "KOPPITZ_DARK_FRAGMENTED",
                  "SYM_ANIMAL_TIGER_WOLF", "SYM_EYE_ANGRY"],
        "min_cluster": 2,
        "description": "high emotional intensity (not diagnostic)",
    },
    "symbolic_engagement": {
        "rules": ["SYM_SHAPE_STARS", "SYM_TRANSPORT_VEHICLE", "SYM_HEART",
                  "SYM_SHAPE_FLOWERS_CLOUDS_SUN", "SYM_ANIMAL_LION"],
        "min_cluster": 2,
        "description": "rich symbolic engagement in drawing (observational)",
    },
}


def compute_theme_scores(activated_rule_ids: list[str],
                         rule_confidences: dict[str, float]) -> list[dict]:
    """
    Aggregate activated rules into weighted theme scores.

    Returns only themes that meet min_cluster AND have weighted score >= threshold.
    No single symbolic rule generates a conclusion alone.
    """
    from safety_policy import PSYCH_THRESHOLDS
    min_score = float(PSYCH_THRESHOLDS.get("theme_min_score_for_output", 0.30))
    results = []
    for theme_id, theme in THEME_WEIGHTS.items():
        matching = [r for r in theme["rules"] if r in activated_rule_ids]
        if len(matching) < theme["min_cluster"]:
            continue
        # Weighted score = mean of confidences for matching rules
        score = sum(rule_confidences.get(r, 0.0) for r in matching) / len(matching)
        if score < min_score:
            continue
        results.append({
            "theme": theme_id,
            "description": theme["description"],
            "supporting_rules": matching,
            "theme_score": round(score, 3),
            "min_cluster_required": theme["min_cluster"],
            "cluster_met": True,
            "is_diagnosis": False,
        })
    return results


# ---------------------------------------------------------------------------
# Evaluate all v2 rules against a doc
# ---------------------------------------------------------------------------

def evaluate_v2_rules(doc: dict, quality_score: float = 1.0) -> list[dict]:
    """
    Run all Tier 1 + Tier 2 rules and return structured activation records.
    Confidence = quality_score × base_weight × cap (max 0.70).
    """
    cap = float(CFG.get("psychological_rules", {}).get("rule_base_confidence_cap", 0.70))
    activations = []
    for rule in ALL_V2_RULES:
        try:
            activated = bool(rule["check"](doc))
        except Exception:
            activated = False

        confidence = 0.0
        if activated:
            confidence = round(min(cap, quality_score * rule["base_weight"]), 3)

        interp = ""
        if activated:
            try:
                interp = rule["interpretation"](doc)
            except Exception:
                interp = rule["domain"]

        activations.append({
            "rule_id":           rule["rule_id"],
            "domain":            rule["domain"],
            "tier":              rule["tier"],
            "evidence_strength": rule["evidence_strength"],
            "sources":           rule["sources"],
            "activated":         activated,
            "visual_facts":      interp if activated else "—",
            "interpretation":    interp if activated else "",
            "caution":           rule["caution"],
            "limitation":        rule["limitation"],
            "base_weight":       rule["base_weight"],
            "rule_confidence":   confidence,
            "parent_guidance":   rule.get("parent_guidance", ""),
            "is_diagnosis":      False,
        })
    return activations
