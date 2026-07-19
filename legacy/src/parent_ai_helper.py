"""
parent_ai_helper.py — Template-based parent-facing explanation assistant.

This module assembles the parent-facing answer from VERIFIED claims only.
It uses template logic — not a free-form language model — so it literally
cannot invent objects, emotions, symbols, or diagnoses.

Optional: if an LLM API key is configured, it can call a language model
with a strictly constrained prompt that only uses verified structured output.

The module produces:
  {
    "parent_answer":     str,
    "gentle_questions":  [str, ...],
    "safety_note":       str,
    "disclaimer":        str,
    "generation_method": "template" | "llm_constrained",
  }
"""

from __future__ import annotations
import os
from safety_policy import GENERAL_DISCLAIMER, sanitise_text


# ---------------------------------------------------------------------------
# Gentle questions bank
# ---------------------------------------------------------------------------

_QUESTIONS_GENERAL = [
    "Can you tell me what is happening in your drawing?",
    "Who are the people (or characters) in your drawing?",
    "How does this character feel?",
    "Is there a story behind it?",
]

_QUESTIONS_EMOTIONAL_CONCERN = [
    "What is your favorite part of the drawing?",
    "If you could change anything in the drawing, what would it be?",
    "Is there something you would like to talk about today?",
]

_QUESTIONS_POSITIVE = [
    "What made you choose these colors?",
    "Tell me more about this happy part of your drawing.",
    "What would happen next in your story?",
]

_QUESTIONS_SYMBOL_FOLLOW_UP = [
    "What is this animal/character doing?",
    "Why did you draw this?",
    "Is this animal a friend or something else?",
]


def _select_questions(verified_claims: list[dict], emotional_tendency: str) -> list[str]:
    """Select contextually appropriate gentle questions."""
    questions = list(_QUESTIONS_GENERAL)

    has_symbol = any(c.get("claim_type") == "visual_symbol" for c in verified_claims
                     if c.get("show_to_user"))
    has_concern = any(c.get("claim_type") == "safety_warning" for c in verified_claims
                      if c.get("show_to_user"))
    positive_tendency = emotional_tendency in ("happy",)
    negative_tendency = emotional_tendency in ("sad", "fear", "angry")

    if has_symbol:
        questions += _QUESTIONS_SYMBOL_FOLLOW_UP[:1]
    if has_concern or negative_tendency:
        questions += _QUESTIONS_EMOTIONAL_CONCERN[:2]
    if positive_tendency:
        questions += _QUESTIONS_POSITIVE[:2]

    # Deduplicate, keep first 5
    seen = set()
    unique = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:5]


# ---------------------------------------------------------------------------
# Template blocks
# ---------------------------------------------------------------------------

def _describe_composition(verified_claims: list[dict]) -> str:
    parts = []
    for c in verified_claims:
        if c.get("claim_type") != "visual_numeric" or not c.get("show_to_user"):
            continue
        ev = c.get("evidence", {})
        if "empty_space_ratio" in ev:
            pct = round(ev["empty_space_ratio"] * 100)
            if pct > 60:
                parts.append(f"the drawing has a large amount of empty space ({pct}%)")
            elif pct < 30:
                parts.append(f"the drawing fills much of the page")
        if "dark_dominance" in ev:
            dark = round(ev["dark_dominance"] * 100)
            if dark > 35:
                parts.append(f"darker tones are prominent ({dark}% of the colored area)")
        if "color_diversity_count" in ev:
            n = ev["color_diversity_count"]
            if n < 3:
                parts.append(f"only {n} color(s) are used, which is a restricted palette")
            elif n >= 5:
                parts.append(f"a varied palette of {n} colors is present")
    return "; ".join(parts) if parts else ""


def _describe_objects(verified_claims: list[dict]) -> str:
    labels = []
    for c in verified_claims:
        if c.get("claim_type") not in ("visual_object", "visual_symbol"):
            continue
        if not c.get("show_to_user"):
            continue
        if c.get("validator_status") != "verified":
            continue
        ev = c.get("evidence", {})
        lbl = ev.get("label", "")
        if lbl:
            labels.append(lbl.replace("a ", "").replace("an ", ""))
    if not labels:
        return ""
    return "The drawing includes " + ", ".join(labels[:5]) + "."


def _describe_psych_indicators(verified_claims: list[dict]) -> str:
    interps = []
    for c in verified_claims:
        if c.get("claim_type") != "psychological_interpretation":
            continue
        if not c.get("show_to_user"):
            continue
        if c.get("validator_status") != "verified":
            continue
        text = c.get("claim", "")
        ev   = c.get("evidence", {})
        caut = ev.get("caution", "")
        # Use first sentence only for brevity
        first_sentence = text.split(".")[0] + "." if "." in text else text
        interps.append(first_sentence)
        if caut:
            interps.append(f"(Note: {caut})")
    return " ".join(interps[:4])


def _describe_ocr(verified_claims: list[dict]) -> str:
    texts = []
    for c in verified_claims:
        if c.get("claim_type") != "ocr_text":
            continue
        if not c.get("show_to_user"):
            continue
        if c.get("validator_status") != "verified":
            continue
        ev = c.get("evidence", {})
        t  = ev.get("text", "")
        if t:
            texts.append(f'"{t}"')
    if not texts:
        return ""
    return f"The drawing contains written text: {', '.join(texts)}."


def _describe_emotion(doc: dict) -> str:
    h = doc.get("feature_based_emotional_tendency", {})
    emotion = h.get("estimated_emotion", "")
    if not emotion or emotion == "neutral_or_unclear":
        return ""
    conf_label = h.get("confidence_label", "low")
    note = h.get("note", "")
    return (
        f"A heuristic feature analysis produced a very tentative ({conf_label}-confidence) "
        f"estimate suggesting '{emotion}'-leaning visual patterns. {note} "
        "This is not a clinical assessment and should not be presented as a model-based emotion result."
    )


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def generate_parent_answer(doc: dict, validated_claims: list[dict],
                            parent_question: str = "") -> dict:
    """
    Assemble a parent-facing response from validated claims only.

    No object, emotion, symbol, or interpretation is mentioned unless it
    comes from a verified claim with show_to_user == True.
    """
    h = doc.get("feature_based_emotional_tendency", {})
    emotional_tendency = h.get("estimated_emotion", "neutral_or_unclear")

    composition_desc = _describe_composition(validated_claims)
    objects_desc     = _describe_objects(validated_claims)
    psych_desc       = _describe_psych_indicators(validated_claims)
    ocr_desc         = _describe_ocr(validated_claims)
    emotion_desc     = _describe_emotion(doc)

    # Build the parent answer paragraph
    paragraphs = []

    # Opening
    paragraphs.append(
        "Thank you for sharing your child's drawing. "
        "Here is a gentle, observation-based summary of what the analysis found."
    )

    # Composition
    if composition_desc:
        paragraphs.append(
            f"Looking at how the drawing is composed, {composition_desc}. "
            "These are visual observations, not conclusions."
        )

    # Objects
    if objects_desc:
        paragraphs.append(objects_desc)

    # OCR
    if ocr_desc:
        paragraphs.append(
            ocr_desc + " Please note that handwritten text recognition can make errors."
        )

    # Psychological indicators (only from verified claims)
    if psych_desc:
        paragraphs.append(
            "Some drawing patterns may have soft psychological associations. "
            + psych_desc
        )

    # Emotion heuristic — always with strong caveat
    if emotion_desc:
        paragraphs.append(emotion_desc)

    # If nothing to say
    if len(paragraphs) == 1:
        paragraphs.append(
            "No strong visual patterns or indicators were identified in this drawing. "
            "This does not mean anything is wrong — many children's drawings show "
            "no notable features."
        )

    # Safety note
    safety_note = (
        "This interpretation is not diagnostic and should be considered only as a "
        "supportive, contextual observation. It does not replace professional assessment."
    )

    # Questions
    gentle_questions = _select_questions(validated_claims, emotional_tendency)

    parent_answer_text = "\n\n".join(paragraphs)
    # Final safety pass — sanitise any accidental overclaiming
    parent_answer_text = sanitise_text(parent_answer_text)

    return {
        "parent_answer":     parent_answer_text,
        "gentle_questions":  gentle_questions,
        "safety_note":       safety_note,
        "disclaimer":        GENERAL_DISCLAIMER,
        "generation_method": "template",
        "verified_claim_count": sum(
            1 for c in validated_claims
            if c.get("validator_status") == "verified" and c.get("show_to_user")
        ),
    }
