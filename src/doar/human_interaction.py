"""human_interaction.py -- DOAR Human Interaction Layer v1.

Consumes the ALREADY-GOVERNED drawing_synthesis.py / reasoning_chain.py
results; never redefines them. Nothing here adds a rule, changes a
threshold, changes evidence eligibility, or changes what
`drawing_synthesis.synthesize_drawing()` concludes -- this module only
translates that already-computed result into human language and answers
follow-up questions ABOUT it.

Layers, top to bottom:
  1. Read-only registry loader for RULE_SOURCE_REGISTER.csv (bibliography
     keyed by rule_id -- reused, not duplicated; no new registry file).
  2. Human-readable formatters (rule wording, domain wording, evidence
     family wording, "why it applies here" sentences) -- presentation
     only, built entirely from fields already in RULE_EVIDENCE_MATRIX.csv/
     CONCERN_DOMAIN_MAP.json/RULE_SOURCE_REGISTER.csv and the case's own
     `DrawingSynthesisResult`.
  3. Evidence-chain builder: one chain per already-eligible literature
     association (feature -> rule -> suggestion -> concern -> source).
  4. A 4-category question router (case / rule-source / visual /
     general) and one answerer per category.
  5. A structured-answer schema, a deterministic verification pass that
     REUSES `claim_verifier.py` unchanged (via a read-only adapter that
     projects governed evidence into the shape that module already
     checks -- no second, duplicate verifier), and a Judge layer
     (deterministic by default; an optional Gemini-backed judge/external-
     research provider behind a clean, mockable protocol, never required
     by tests, never called with a hardcoded key).

Safety: reuses `judges.py`'s own DIAGNOSTIC_PATTERNS/_ARABIC_DIAGNOSTIC
regexes UNCHANGED (imported, never edited) for the underlying forbidden-
wording list, but wraps them in a negation-aware check so a sentence that
DISCUSSES or DENIES a diagnosis ("not enough to say the child is
depressed") is correctly distinguished from an actual unsupported
diagnostic claim ("the child is depressed"). judges.py itself, and every
other frozen scientific file, is never modified.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

from . import case_interpretation as ci
from . import claim_verifier as cv
from . import reasoning_chain as rc
from .judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
from .visual_qa import extract_visual_target

ROOT = Path(__file__).resolve().parents[2]
RULE_SOURCE_REGISTER_PATH = ROOT / "RULE_SOURCE_REGISTER.csv"

FOOTER_DISCLAIMER = (
    "Drawing-based interpretations are not definitive diagnoses. Additional context or "
    "professional assessment may be needed.")

_Q_AND_A_VISUAL_RECHECK_TAG = "Q&A_VISUAL_RECHECK"
_EXTERNAL_RESEARCH_LABEL = "Additional research -- not part of the original DOAR analysis"
_VISUAL_AUDIT_LABEL = "Visual consistency audit -- AUDIT ONLY, not case evidence"


# ---------------------------------------------------------------------------
# 1. Registry loader -- read-only, cached, mirrors reasoning_chain.py's own
#    load_rule_matrix()/load_concern_domain_map() loader style exactly.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_rule_source_register() -> dict[str, dict]:
    """rule_id -> row dict, from the frozen RULE_SOURCE_REGISTER.csv (the
    project's existing bibliography: citation_title, doi_or_pmid,
    literature_support_status, ... -- no new registry invented)."""
    with open(RULE_SOURCE_REGISTER_PATH, encoding="utf-8") as f:
        return {row["rule_id"]: row for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# 2. Human-readable formatters -- pure text transforms of already-frozen
#    fields. Never invent wording not already present in the matrix/
#    domain-map/source-register.
# ---------------------------------------------------------------------------

_FEATURE_ID_LABELS = {
    "stroke.intensity_proxy": "line darkness (pressure proxy)",
    "stroke.fragmentation": "line fragmentation (shaky/broken proxy)",
    "segmentation.bounding_box_coverage": "how much of the page the drawing covers",
    "composition.centroid_x": "the drawing's horizontal position on the page",
    "composition.centroid_y": "the drawing's vertical position on the page",
}


def rule_display_name(rule_id: str, matrix_row: dict) -> str:
    """A short human title for a rule, derived from its own
    `observable_feature` column (e.g. "light_pressure" -> "Light
    pressure") -- never the bare internal rule_id as the primary label."""
    observable = (matrix_row.get("observable_feature") or rule_id).replace("_", " ").strip()
    return observable[:1].upper() + observable[1:] if observable else rule_id


def family_display(evidence_family: str) -> str:
    return (evidence_family or "").replace("_", " ").strip().capitalize()


def domain_display(concern_domain: str) -> dict:
    """{"label": short clinician-style phrase, "description": longer
    parent-safe description} -- both already-authored fields from
    CONCERN_DOMAIN_MAP.json, never generated here."""
    domains = rc.load_concern_domain_map()["domains"]
    entry = domains.get(concern_domain, {})
    return {
        "label": entry.get("example_clinician_hypothesis_wording", concern_domain.replace("_", " ")),
        "description": entry.get("description", ""),
        "parent_wording": entry.get("example_parent_wording", ""),
    }


_DISPLAY_PUNCTUATION_FIXES = {
    "—": " - ",  # em dash
    "–": "-",    # en dash
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
}


def normalize_display_text(text: str | None) -> str | None:
    """Replaces Unicode punctuation (em/en dash, curly quotes) that some
    terminals/consoles/older codepages render as mojibake (e.g. 'p.2
    <?> Line quality') with plain ASCII equivalents. The underlying CSV
    data is untouched (it already stores a correct, real em-dash
    character) -- this is a presentation-only normalization applied at
    display time, collapsing repeated whitespace left by the
    substitution."""
    if not text:
        return text
    for bad, good in _DISPLAY_PUNCTUATION_FIXES.items():
        text = text.replace(bad, good)
    return re.sub(r" {2,}", " ", text).strip()


_SECRET_LIKE_RE = re.compile(r"(api[_-]?key|apikey|key|token|secret|authorization|bearer)\s*[:=]\s*\S+",
                              re.IGNORECASE)
_LONG_OPAQUE_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")


def sanitize_error_text(exc: BaseException, *, max_length: int = 300) -> str:
    """A short, safe-to-display description of a caught provider
    exception -- NEVER the raw exception text verbatim, since a Gemini/
    HTTP error can embed the request URL (which may include the API
    key) or other request internals. Redacts anything shaped like a
    `key=...`/`token: ...` assignment AND any long opaque alphanumeric
    token (catches an API key even when not explicitly labeled),
    truncated to a bounded length so it is always safe to show in a
    'Technical error details' expander or a log line."""
    text = f"{exc.__class__.__name__}: {exc}"
    text = _SECRET_LIKE_RE.sub("[redacted]", text)
    text = _LONG_OPAQUE_TOKEN_RE.sub("[redacted]", text)
    return text[:max_length]


def source_citation(rule_id: str) -> dict:
    """Bibliographic fields for a rule. RULE_SOURCE_REGISTER.csv (the
    original PSY_AR_* rules' real academic citation_title/doi_or_pmid)
    is preferred when present; RULE_EVIDENCE_MATRIX.csv's own
    `source_pdf_page_section` (present for every one of the 41 rules,
    including the EN_COMPILED_* rules the register doesn't cover) is
    always included too. Missing fields (e.g. no separate authors/year
    column exists anywhere in either file) are reported as not-recorded,
    never fabricated."""
    register_row = load_rule_source_register().get(rule_id)
    matrix_row = rc.load_rule_matrix().get(rule_id, {})
    pdf_section = normalize_display_text(matrix_row.get("source_pdf_page_section") or None)
    if register_row is None:
        return {
            "available": bool(pdf_section), "citation_title": None, "doi_or_pmid": None,
            "literature_support_status": None, "source_pdf_page_section": pdf_section,
            "authors_and_year_note": "Not recorded for this rule (no matching entry in DOAR's source register).",
        }
    return {
        "available": True,
        "citation_title": normalize_display_text(register_row.get("citation_title") or None),
        "doi_or_pmid": register_row.get("doi_or_pmid") or None,
        "literature_support_status": register_row.get("literature_support_status") or None,
        "source_pdf_page_section": pdf_section,
        "authors_and_year_note": "Not separately recorded in DOAR's source register (only title + DOI/PMID are tracked).",
    }


def _describe_matched_evidence(matched_entity_ids: tuple[str, ...], bundle: dict) -> str:
    """A short, grounded 'why it applies here' phrase built from the
    SAME matched_entity_ids the eligible match already carries -- never a
    new inference."""
    entities_by_id = {e.entity_id: e for e in (bundle.get("entities") or [])}
    det = bundle.get("deterministic_features") or {}
    objective_features = det.get("objective_features") or {}
    evidence_id_to_feature = {fv.evidence_id: (fid, fv) for fid, fv in objective_features.items()}
    parts = []
    for mid in matched_entity_ids:
        if mid in entities_by_id:
            parts.append(f"the confirmed '{entities_by_id[mid].canonical_label}' in this drawing")
        elif mid in evidence_id_to_feature:
            fid, fv = evidence_id_to_feature[mid]
            label = _FEATURE_ID_LABELS.get(fid, fid)
            value_text = f"{fv.value:.2f}" if isinstance(fv.value, (int, float)) else str(fv.value)
            parts.append(f"the measured {label} for this drawing ({value_text})")
        else:
            parts.append(mid)
    return "; ".join(parts) if parts else "a measurement recorded for this drawing"


# ---------------------------------------------------------------------------
# 3. Evidence-chain builder -- one chain per ALREADY-ELIGIBLE literature
#    association (drawing_synthesis.py's own output, unmodified).
# ---------------------------------------------------------------------------


def build_evidence_chains(bundle: dict) -> list[dict]:
    """feature -> rule -> suggestion -> concern -> source, one entry per
    `synthesis.literature_linked_associations` row (already computed,
    already eligible -- this function only formats it for humans)."""
    synthesis = bundle.get("synthesis")
    if synthesis is None:
        return []
    matrix = rc.load_rule_matrix()
    chains = []
    for assoc in synthesis.literature_linked_associations:
        rule_id = assoc["rule_id"]
        row = matrix.get(rule_id, {})
        chains.append({
            "rule_id": rule_id,
            "rule_display_name": rule_display_name(rule_id, row),
            "matched_entity_ids": list(assoc["matched_entity_ids"]),
            "why_it_applies": _describe_matched_evidence(tuple(assoc["matched_entity_ids"]), bundle),
            "rule_suggests": assoc["possible_interpretation"],
            "source_claim": assoc["source_claim"],
            "evidence_strength": assoc["evidence_strength"],
            "evidence_direction": assoc["evidence_direction"],
            "concern_domain": assoc["concern_domain"],
            "concern": domain_display(assoc["concern_domain"]),
            "evidence_family": assoc["evidence_family"],
            "evidence_family_display": family_display(assoc["evidence_family"]),
            "alternative_explanations": list(assoc["alternative_explanations"]),
            "allowed_output_level": assoc["allowed_output_level"],
            "source": source_citation(rule_id),
        })
    return chains


def unmatched_verified_observations(bundle: dict) -> list[dict]:
    """Verified semantic observations that currently have NO eligible
    rule attached -- shown explicitly (per Part A.2's own instruction)
    rather than hidden, with the honest 'no approved interpretation'
    note."""
    synthesis = bundle.get("synthesis")
    if synthesis is None:
        return []
    out = []
    for ue in synthesis.unified_evidence:
        item = ue.item
        if item.category == "semantic_objects" and item.status == "available" and not ue.rule_eligible:
            out.append({"label": item.value, "reason": "No approved DOAR psychological interpretation is currently attached to this observation."})
    return out


def grouped_unmatched_observation_labels(bundle: dict) -> list[str]:
    """Deduplicated, sorted labels for Parent/Clinician View's compact
    'no current psychological rule' summary (final-correction-pass Part
    5 fix) -- the same underlying observations as
    `unmatched_verified_observations`, grouped by label instead of one
    entry per individual entity, since 5 separate butterflies do not
    need 5 separate 'no rule' cards. Nothing is hidden -- every distinct
    label still appears, just once."""
    return sorted({item["label"] for item in unmatched_verified_observations(bundle)})


def _pluralize(label: str) -> str:
    if label.endswith("y") and len(label) > 1 and label[-2].casefold() not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith(("s", "x", "ch", "sh")):
        return label + "es"
    return label + "s"


def observation_bullets(bundle: dict) -> dict:
    """Confirmed / possible / not-confirmed object bullets for Part A.1
    (Parent) / Part B.2 (Clinician) -- grouped by canonical label and
    verification status, built entirely from this case's own cached
    Observer/Verifier rows. Colour/line/composition bullets are built
    separately (clinician_review_app.build_parent_friendly_profile),
    since those come from deterministic features, not entities."""
    from collections import Counter

    entities = bundle.get("entities") or []
    verified_counts = Counter(e.canonical_label for e in entities if e.case_verification_status == "verified")
    uncertain_counts = Counter(e.canonical_label for e in entities if e.case_verification_status == "uncertain")
    not_confirmed_labels = sorted({e.canonical_label for e in entities
                                    if e.case_verification_status in ("rejected", "unreviewed")})

    def _title(label: str) -> str:
        return label[:1].upper() + label[1:] if label else label

    confirmed = []
    for label, count in sorted(verified_counts.items()):
        if count == 1:
            confirmed.append(f"{_title(label)} — confirmed")
        elif count == 2:
            confirmed.append(f"Two {_pluralize(label)} — confirmed")
        else:
            confirmed.append(f"Several {_pluralize(label)} — confirmed")

    possible = [f"{_title(label)} — possible / not confirmed" for label in sorted(uncertain_counts)]

    return {
        "confirmed": confirmed, "possible": possible,
        "not_confirmed_labels": not_confirmed_labels,
        "not_confirmed_note": (f"Other observations not confirmed: {', '.join(not_confirmed_labels)}."
                                if not_confirmed_labels else None),
    }


def case_specific_questions(bundle: dict) -> list[str]:
    """Non-leading conversation starters grounded ONLY in this drawing's
    own noticed content -- e.g. "Tell me about the car.", never a
    leading/presumptive question like "Why are the people in the car
    sad?". Kept separate from Ask DOAR (Part A.5's own instruction).
    Falls back to generic prompts when nothing at all was observed."""
    entities = bundle.get("entities") or []
    labels, seen = [], set()
    for e in entities:
        if e.canonical_label not in seen and e.case_verification_status in ("verified", "uncertain", "unreviewed"):
            seen.add(e.canonical_label)
            labels.append(e.canonical_label)
    questions = [f"Tell me about the {label}." for label in labels[:4]]
    questions.append("What is happening in this picture?")
    if len(questions) < 3:
        questions.append("Is there anything you'd like to add to this drawing?")
    return questions


_DOMAIN_NOUN_PHRASE = {
    "anxiety_or_stress_related": "stress", "depressive_or_low_mood_related": "low-mood",
    "aggression_or_threat_related": "aggression", "social_withdrawal_related": "withdrawal",
    "developmental_or_attention_related": "attention",
    "positive_affect_or_social_engagement": "positive-engagement",
    "neutral_descriptive_only": "descriptive", "neutral_descriptive_only_no_construct_proposed": "descriptive",
}


def overall_interpretation_text(bundle: dict) -> str:
    """Section 3 (Parent) / Section 4 (Clinician) headline sentence --
    built entirely from `synthesis.overall_synthesis`'s own already-
    computed level/domains/families, phrased in plain language. Never
    replaces a real, specific association with a generic template when
    one exists (Part A.3's own instruction) -- a single-chain
    limited_association case names that one chain explicitly."""
    synthesis = bundle["synthesis"]
    overall = synthesis.overall_synthesis
    level = overall["level"]
    chains = build_evidence_chains(bundle)

    if level == "insufficient_interpretable_evidence":
        return "DOAR did not find enough evidence in this drawing to support any literature-linked pattern."
    if level == "descriptive_only":
        return ("The measurable features of this drawing are descriptive only (for example composition or "
                "size) and are not linked by DOAR to a specific feeling or concern.")
    if level == "limited_association":
        if len(chains) == 1:
            c = chains[0]
            noun = _DOMAIN_NOUN_PHRASE.get(c["concern_domain"], c["concern_domain"].replace("_", " "))
            feature_phrase = c["rule_display_name"][:1].lower() + c["rule_display_name"][1:]
            return (f"The main psychological association found in this drawing comes from {feature_phrase}. "
                    f"The drawing literature currently used by DOAR links this feature with possible "
                    f"{noun}-related indicators. No other independent feature in this drawing currently "
                    f"supports the same pattern strongly enough to make it a broader finding.")
        domains = sorted({c["concern_domain"] for c in chains})
        phrase = ", ".join(_DOMAIN_NOUN_PHRASE.get(d, d.replace("_", " ")) for d in domains)
        feature_names = ", ".join(sorted({c["rule_display_name"] for c in chains}))
        return (f"This drawing includes {len(chains)} separate features ({feature_names}), each individually "
                f"linked to {phrase}-related themes in the literature DOAR uses, but they do not converge into "
                f"one consistent overall pattern.")
    if level in ("convergent_positive_pattern", "convergent_concern_pattern"):
        domain = overall.get("convergent_positive_domain") or overall.get("convergent_concern_domain")
        families = overall["positive_evidence_families"] or overall["concern_evidence_families"]
        noun = _DOMAIN_NOUN_PHRASE.get(domain, (domain or "").replace("_", " "))
        family_text = ", ".join(f.replace("_", " ") for f in families)
        return (f"Multiple independent features in this drawing ({family_text}) point in the same direction, "
                f"converging on a {noun}-related pattern. This is a stronger signal than any single feature "
                f"alone, but it is still not a diagnosis.")
    if level == "mixed_evidence":
        return ("This drawing shows convergent patterns pointing in different directions, so DOAR does not "
                "identify one single overall interpretation.")
    return overall["summary"]


# ---------------------------------------------------------------------------
# 4. Question router + per-category answerers.
# ---------------------------------------------------------------------------

_RULE_SOURCE_MARKERS = (
    "source for", "reference for", "citation", " doi", "where does this rule",
    "what is the source", "which reference", "what reference", "where is this from",
)
_VISUAL_QUESTION_MARKERS = (
    "is there", "are there", "how many", "do you see", "is the", "are the",
    "in the drawing", "in the picture", "does the drawing show", "does the drawing have",
)
_CASE_MARKERS = (
    "why did you", "why do you", "why does doar", "is my child", "is the child",
    "does this drawing", "does the drawing suggest", "in this drawing", "this case",
    "overall", "convergence", "hypothesis", "what did you find", "what does this mean",
)
_CONCERN_KEYWORDS = (
    "stress", "anxi", "worry", "depress", "low mood", "sad", "aggress", "anger", "angry",
    "withdraw", "isolat", "attention", "adhd", "positive", "engage",
)
_CONDITION_TO_DOMAIN = {
    "depress": "depressive_or_low_mood_related",
    "sad": "depressive_or_low_mood_related",
    "low mood": "depressive_or_low_mood_related",
    "anxi": "anxiety_or_stress_related",
    "stress": "anxiety_or_stress_related",
    "worry": "anxiety_or_stress_related",
    "aggress": "aggression_or_threat_related",
    "anger": "aggression_or_threat_related",
    "angry": "aggression_or_threat_related",
    "withdraw": "social_withdrawal_related",
    "isolat": "social_withdrawal_related",
    "attention": "developmental_or_attention_related",
    "adhd": "developmental_or_attention_related",
    "positive": "positive_affect_or_social_engagement",
    "engage": "positive_affect_or_social_engagement",
}
# Adjective/phrase form for "...to say that the child is {X}" -- a small,
# purely grammatical mapping (never a new clinical claim).
_CONDITION_ADJECTIVE = {
    "depress": "depressed", "sad": "sad", "low mood": "experiencing low mood",
    "anxi": "anxious", "stress": "stressed", "worry": "worried",
    "aggress": "aggressive", "anger": "angry", "angry": "angry",
    "withdraw": "withdrawn", "isolat": "isolated",
    "attention": "experiencing attention difficulties", "adhd": "experiencing ADHD-related difficulties",
    "positive": "showing positive engagement", "engage": "engaged",
}
# Noun-phrase stem for "...-related themes" contexts (also purely
# grammatical, not a new clinical claim).
_CONDITION_NOUN = {
    "depress": "depression", "sad": "low-mood", "low mood": "low-mood",
    "anxi": "anxiety", "stress": "stress", "worry": "worry",
    "aggress": "aggression", "anger": "anger", "angry": "anger",
    "withdraw": "withdrawal", "isolat": "isolation",
    "attention": "attention", "adhd": "ADHD", "positive": "positive-engagement", "engage": "engagement",
}


def route_question(question: str) -> str:
    """Returns one of "rule_source" / "visual" / "case" / "general".
    Deliberately simple, ordered keyword heuristics -- not an ML
    classifier -- so behaviour stays auditable and testable."""
    q = (question or "").casefold()
    if any(m in q for m in _RULE_SOURCE_MARKERS):
        return "rule_source"
    target = extract_visual_target(question)
    if target is not None and any(m in q for m in _VISUAL_QUESTION_MARKERS):
        return "visual"
    if any(m in q for m in _CASE_MARKERS) or any(k in q for k in _CONCERN_KEYWORDS):
        return "case"
    return "general"


def _extract_condition_keyword(question: str) -> str | None:
    q = (question or "").casefold()
    for kw in _CONDITION_TO_DOMAIN:
        if kw in q:
            return kw
    return None


def _chains_for_domain(chains: list[dict], domain: str) -> list[dict]:
    return [c for c in chains if c["concern_domain"] == domain]


def _answer_case_question(question: str, bundle: dict) -> dict:
    synthesis = bundle["synthesis"]
    overall = synthesis.overall_synthesis
    chains = build_evidence_chains(bundle)
    condition_kw = _extract_condition_keyword(question)

    if condition_kw:
        domain = _CONDITION_TO_DOMAIN[condition_kw]
        domain_chains = _chains_for_domain(chains, domain)
        convergent = overall.get("convergent_concern_domain") == domain or overall.get("convergent_positive_domain") == domain
        if not domain_chains:
            answer = (f"DOAR does not currently have any literature-linked evidence in this drawing "
                       f"associated with {_CONDITION_NOUN.get(condition_kw, condition_kw.strip())}-related themes.")
        elif convergent:
            families = sorted({c["evidence_family_display"] for c in domain_chains})
            answer = (f"Several independent features in this drawing ({', '.join(families)}) converge on a "
                       f"pattern the literature DOAR uses sometimes associates with {_CONDITION_NOUN.get(condition_kw, condition_kw.strip())}-related "
                       f"themes. This is still not a diagnosis.")
        else:
            rule_names = sorted({c["rule_display_name"] for c in domain_chains})
            answer = (f"This drawing includes {len(domain_chains)} feature(s) ({', '.join(rule_names)}) "
                       f"associated with {_CONDITION_NOUN.get(condition_kw, condition_kw.strip())}-related themes in the literature DOAR uses, but "
                       f"this does not provide enough evidence to say that the child is "
                       f"{_CONDITION_ADJECTIVE.get(condition_kw, condition_kw.strip())}.")
        relevant_chains = domain_chains
    else:
        # "why did you mention X" without a recognized condition keyword,
        # or a general "what did you find" case question -- summarize
        # using the already-computed overall synthesis.
        answer = overall["summary"]
        relevant_chains = chains

    return {
        "answer": answer, "category": "case", "chains": relevant_chains,
        "evidence_ids": sorted({eid for c in relevant_chains for eid in c["matched_entity_ids"]}),
        "rule_ids": sorted({c["rule_id"] for c in relevant_chains}),
        "source_ids": sorted({c["rule_id"] for c in relevant_chains}),
        "used_external_research": False, "used_visual_recheck": False,
    }


def _find_chain_by_keyword(chains: list[dict], question: str) -> dict | None:
    q = (question or "").casefold()
    for c in chains:
        if c["rule_display_name"].casefold() in q or any(w in q for w in c["rule_display_name"].casefold().split()):
            return c
    return None


def _find_any_rule_by_keyword(question: str) -> tuple[str, dict] | None:
    q = (question or "").casefold()
    matrix = rc.load_rule_matrix()
    for rule_id, row in matrix.items():
        observable = (row.get("observable_feature") or "").replace("_", " ").casefold()
        if observable and observable in q:
            return rule_id, row
    return None


_BARE_RULE_ANAPHORS = ("that rule", "this rule", "the rule")


def _answer_rule_source_question(question: str, bundle: dict) -> dict:
    chains = build_evidence_chains(bundle)
    hit = _find_chain_by_keyword(chains, question)
    if hit is None and len(chains) == 1 and any(a in (question or "").casefold() for a in _BARE_RULE_ANAPHORS):
        # "What is the source for THAT rule?" with no named rule -- when
        # this case has exactly one matched rule, the anaphor
        # unambiguously refers to it (never guessed when more than one
        # rule matched, to avoid picking the wrong one).
        hit = chains[0]
    if hit is not None:
        src = hit["source"]
        if src["available"]:
            title = src["citation_title"] or src.get("source_pdf_page_section") or "untitled source"
            answer = (f"The '{hit['rule_display_name']}' rule is sourced from: {title}"
                       + (f" ({src['doi_or_pmid']})" if src["doi_or_pmid"] else "")
                       + f". Literature support status: {src['literature_support_status'] or 'not recorded'}.")
        else:
            answer = f"DOAR does not have a recorded source for the '{hit['rule_display_name']}' rule."
        return {"answer": answer, "category": "rule_source", "chains": [hit],
                "evidence_ids": [], "rule_ids": [hit["rule_id"]], "source_ids": [hit["rule_id"]],
                "used_external_research": False, "used_visual_recheck": False}

    found = _find_any_rule_by_keyword(question)
    if found is not None:
        rule_id, row = found
        src = source_citation(rule_id)
        if src["available"]:
            title = src["citation_title"] or src.get("source_pdf_page_section") or "untitled source"
            answer = (f"The '{rule_display_name(rule_id, row)}' rule (not currently matched in this drawing) is "
                       f"sourced from: {title}"
                       + (f" ({src['doi_or_pmid']})" if src["doi_or_pmid"] else "") + ".")
        else:
            answer = "DOAR has this rule but no recorded source for it."
        return {"answer": answer, "category": "rule_source", "chains": [],
                "evidence_ids": [], "rule_ids": [rule_id], "source_ids": [rule_id],
                "used_external_research": False, "used_visual_recheck": False}

    return {"answer": "I couldn't identify which DOAR rule that question refers to, so I can't return a source for it.",
            "category": "rule_source", "chains": [], "evidence_ids": [], "rule_ids": [], "source_ids": [],
            "used_external_research": False, "used_visual_recheck": False}


# ---------------------------------------------------------------------------
# 4a. Governed, image-path-based on-demand visual re-check (Part H/4 fix).
#
# The governed dev-set bundle (scripts/clinician_review_app.py::
# load_case_bundle) has no `case_dir`/legacy detections.json layout, so
# `visual_evidence.search_visual` (which needs one) can never run against
# it -- see outputs/human_interaction_v1/h38_visual_qa_demonstration.json
# for the original diagnosis of that gap. This path instead operates
# DIRECTLY on the bundle's own already-known image path (read-only,
# never writes a detections file, never touches the bundle/cache). It is
# a SEPARATE, narrow, single-question class from the frozen, broad
# `visual_observer.GeminiVisualObserver` -- never imports or calls that
# class, never changes its prompt/model -- and reuses only the low-level
# HTTP-request plumbing `visual_observer.py`'s own real providers already
# share (`_post_gemini_generate_content`), not its prompt.
# ---------------------------------------------------------------------------

_VISUAL_RECHECK_MODEL_ENV_VAR = "DOAR_VISUAL_RECHECK_MODEL"

_VISUAL_RECHECK_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "present": {"type": "BOOLEAN"},
        "count": {"type": "INTEGER"},
        "description": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["present", "description", "confidence"],
    "propertyOrdering": ["present", "count", "description", "confidence"],
}

_VISUAL_RECHECK_SYSTEM_PROMPT = (
    "You are answering ONE targeted yes/no visual question about a child's drawing, for "
    "DOAR's Ask-DOAR on-demand re-check path. This is a separate, narrow Q&A check -- NOT "
    "DOAR's main broad visual analysis, and its result is never added to that analysis. "
    "Look only at what is actually visible in the image. Report honestly if something is "
    "not present or not clear; never guess just to be helpful."
)


@dataclass(frozen=True)
class GeminiVisualRecheckProvider:
    """Optional real, targeted, single-question visual re-check. Reads
    GEMINI_API_KEY from the environment (never hardcoded); model
    configurable via DOAR_VISUAL_RECHECK_MODEL, defaulting to
    `gemini-3.6-flash` -- the SAME model ID as the frozen broad Observer,
    but a completely separate class/prompt/call (never imports or calls
    `GeminiVisualObserver`, never changes its configuration). `request_fn`
    is injectable (same pattern
    as `visual_observer.GeminiVisualObserver`/`GeminiVisualVerifier`) so
    tests never make a real network call. Raises RuntimeError at
    construction if the key or `google-genai` package is missing."""
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 30.0
    request_fn: Callable[[str, str, dict, float], bytes] | None = None

    def __post_init__(self):
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set -- GeminiVisualRecheckProvider is unavailable.")
        object.__setattr__(self, "api_key", key)
        object.__setattr__(
            self, "model", self.model or os.environ.get(_VISUAL_RECHECK_MODEL_ENV_VAR, "gemini-3.6-flash"))
        if self.request_fn is None:
            try:
                from google import genai  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai package is not installed -- GeminiVisualRecheckProvider is unavailable. "
                    "Install the 'interaction' extra (pip install -e '.[interaction]') to enable it."
                ) from exc

    def recheck(self, image_path: str, target: str) -> dict:
        """ONE call against `image_path` -- never persists anything,
        never writes to `image_path`'s directory or any cache file."""
        from .visual_observer import _guess_image_mime, _pil_image_to_png_b64, _post_gemini_generate_content
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        try:
            image_b64 = _pil_image_to_png_b64(img)
        finally:
            img.close()
        payload = {
            "systemInstruction": {"parts": [{"text": _VISUAL_RECHECK_SYSTEM_PROMPT}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": f"Question: is/are '{target}' visible anywhere in this drawing? Answer with "
                              "whether present, an approximate count if relevant, a one-sentence description "
                              "of what you actually see (or don't see), and your confidence from 0 to 1."},
                    {"inlineData": {"mimeType": _guess_image_mime(image_path), "data": image_b64}},
                ],
            }],
            "generationConfig": {
                "responseMimeType": "application/json", "responseSchema": _VISUAL_RECHECK_RESPONSE_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        send = self.request_fn or _post_gemini_generate_content
        raw = send(self.api_key, self.model, payload, self.timeout_seconds)
        response = json.loads(raw)
        if "error" in response:
            raise RuntimeError(f"Gemini API returned an error: {response['error']!r}")
        content_text = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(content_text)
        return {
            "present": bool(parsed.get("present")), "count": parsed.get("count"),
            "description": parsed.get("description", ""), "confidence": parsed.get("confidence"),
            "provider": f"gemini:{self.model}",
        }


def governed_visual_recheck(target: str, bundle: dict, *, provider: "GeminiVisualRecheckProvider | None" = None) -> dict:
    """READ-ONLY, image-path-based on-demand visual re-check -- the Part
    4 fix. Never mutates the bundle or any cached case file. Returns
    {"status": "unavailable"|"error"|"ok", "reason", "candidate",
    "verification"}. When `candidate["present"]` is True, makes ONE
    additional, independently-worded call as a lightweight cross-check
    ("verification" -- 'independent confirmation where possible', per
    this task's own wording); never invents a bbox or a formal Verifier
    result that doesn't exist for this narrow Q&A path."""
    image_rel_path = (bundle.get("image") or {}).get("relative_path")
    if not image_rel_path:
        return {"status": "unavailable", "reason": "bundle has no image path", "candidate": None, "verification": None}
    image_path = str(ROOT / image_rel_path)
    if provider is None:
        return {"status": "unavailable",
                "reason": "no visual re-check provider configured (GEMINI_API_KEY/google-genai not available)",
                "candidate": None, "verification": None}
    try:
        candidate = provider.recheck(image_path, target)
    except Exception as exc:
        return {"status": "error", "reason": sanitize_error_text(exc), "candidate": None, "verification": None}

    verification = None
    if candidate.get("present"):
        try:
            second = provider.recheck(image_path, target)
            verification = {"independently_confirmed": bool(second.get("present")), "second_call": second}
        except Exception as exc:  # pragma: no cover -- defensive, first call already succeeded
            verification = {"independently_confirmed": None, "error": sanitize_error_text(exc)}
    return {"status": "ok", "reason": None, "candidate": candidate, "verification": verification}


# ---------------------------------------------------------------------------
# Visual Consistency Judge -- an independent VLM audit of the original
# drawing against DOAR's own structured observations. AUDIT ONLY: this is
# a read-only cross-check for a human reviewer, NEVER wired into
# answer_question/deterministic_verify/build_evidence_package, and its
# output is never written back into any case file (detections.json/
# analysis.json/structured_analysis.json are untouched). It can flag a
# "likely missed" or "disputed" observation, but that finding can never by
# itself add an entity, activate a rule, or become case evidence -- only a
# human (or a future, separately-governed pipeline change) could act on it.
# ---------------------------------------------------------------------------

_VISUAL_CONSISTENCY_MODEL_ENV_VAR = "DOAR_VISUAL_CONSISTENCY_MODEL"

_VISUAL_CONSISTENCY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "supported_observations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "likely_missed_observations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "disputed_observations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "unavailable_checks": {"type": "ARRAY", "items": {"type": "STRING"}},
        "overall_agreement": {"type": "STRING", "enum": ["high", "partial", "low"]},
    },
    "required": ["supported_observations", "likely_missed_observations", "disputed_observations",
                 "unavailable_checks", "overall_agreement"],
}

_VISUAL_CONSISTENCY_SYSTEM_PROMPT = (
    "You are an independent visual-consistency AUDITOR for DOAR, a children's-drawing research "
    "tool. You will see the original drawing and a list of DOAR's own structured observations "
    "about it (confirmed entities, uncertain/rejected entities, and measured objective "
    "features). Your ONLY job is to compare what you actually see in the image against that "
    "list, and report:\n"
    "- supported_observations: DOAR observations you independently agree are visible.\n"
    "- likely_missed_observations: things plainly visible in the drawing that DOAR's list does "
    "not mention.\n"
    "- disputed_observations: DOAR observations you do NOT think match what is actually in the "
    "image.\n"
    "- unavailable_checks: anything you cannot judge from the image (too small, ambiguous, out "
    "of frame).\n"
    "- overall_agreement: 'high'/'partial'/'low'.\n"
    "This is an AUDIT ONLY. Never state a diagnosis, a psychological interpretation, or a cause. "
    "Never invent an item not actually visible. Respond with the required JSON object only."
)


@dataclass(frozen=True)
class GeminiVisualConsistencyJudge:
    """Optional real VLM audit, same construction/call pattern as
    `GeminiVisualRecheckProvider` (separate class, separate prompt, never
    shares state with the frozen broad Observer/Verifier or the Q&A visual
    re-check path). Raises RuntimeError at construction if the key or
    `google-genai` package is missing, exactly like every other optional
    Gemini class in this module."""
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 30.0
    request_fn: Callable[[str, str, dict, float], bytes] | None = None

    def __post_init__(self):
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set -- GeminiVisualConsistencyJudge is unavailable.")
        object.__setattr__(self, "api_key", key)
        object.__setattr__(
            self, "model", self.model or os.environ.get(_VISUAL_CONSISTENCY_MODEL_ENV_VAR, "gemini-3.6-flash"))
        if self.request_fn is None:
            try:
                from google import genai  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai package is not installed -- GeminiVisualConsistencyJudge is unavailable. "
                    "Install the 'interaction' extra (pip install -e '.[interaction]') to enable it."
                ) from exc

    def audit(self, image_path: str, structured_observations: dict) -> dict:
        """ONE call against `image_path`. Never persists anything."""
        from .visual_observer import _guess_image_mime, _pil_image_to_png_b64, _post_gemini_generate_content
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        try:
            image_b64 = _pil_image_to_png_b64(img)
        finally:
            img.close()
        payload = {
            "systemInstruction": {"parts": [{"text": _VISUAL_CONSISTENCY_SYSTEM_PROMPT}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "DOAR's structured observations for this drawing (JSON): "
                              + json.dumps(structured_observations, default=str)},
                    {"inlineData": {"mimeType": _guess_image_mime(image_path), "data": image_b64}},
                ],
            }],
            "generationConfig": {
                "responseMimeType": "application/json", "responseSchema": _VISUAL_CONSISTENCY_RESPONSE_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        send = self.request_fn or _post_gemini_generate_content
        raw = send(self.api_key, self.model, payload, self.timeout_seconds)
        response = json.loads(raw)
        if "error" in response:
            raise RuntimeError(f"Gemini API returned an error: {response['error']!r}")
        content_text = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(content_text)
        return {
            "supported_observations": list(parsed.get("supported_observations") or []),
            "likely_missed_observations": list(parsed.get("likely_missed_observations") or []),
            "disputed_observations": list(parsed.get("disputed_observations") or []),
            "unavailable_checks": list(parsed.get("unavailable_checks") or []),
            "overall_agreement": parsed.get("overall_agreement", "low"),
        }


def summarize_bundle_for_visual_audit(bundle: dict) -> dict:
    """Compact, human-readable summary of THIS case's structured
    observations -- the only thing the Visual Consistency Judge is shown
    besides the raw image. Reuses the same compact-summary helpers
    `build_evidence_package` already uses; never a raw dump of internal
    IDs/thresholds."""
    entities = _entity_observation_summaries(bundle)
    objective = _compact_objective_feature_summary(bundle)
    chains = build_evidence_chains(bundle)
    return {
        "confirmed_entities": entities["verified"],
        "uncertain_or_unreviewed_entities": entities["uncertain_or_unreviewed"],
        "rejected_entities": entities["rejected"],
        "objective_features": objective,
        "literature_linked_observations": sorted({c["rule_display_name"] for c in chains}),
    }


def run_visual_consistency_audit(bundle: dict, *, judge: "GeminiVisualConsistencyJudge | None" = None) -> dict:
    """READ-ONLY entry point the UI calls on explicit user request (never
    automatically, never on every rerun). Returns {"status": "unavailable"
    |"error"|"ok", "reason", "audit"}. `audit`, when present, is ALWAYS
    AUDIT_ONLY -- the caller must never feed it back into
    answer_question/deterministic_verify/detections.json/any evidence
    file."""
    image_rel_path = (bundle.get("image") or {}).get("relative_path")
    if not image_rel_path:
        return {"status": "unavailable", "reason": "bundle has no image path", "audit": None}
    if judge is None:
        return {"status": "unavailable",
                "reason": "no visual consistency judge configured (GEMINI_API_KEY/google-genai not available)",
                "audit": None}
    image_path = str(ROOT / image_rel_path)
    observations = summarize_bundle_for_visual_audit(bundle)
    try:
        audit = judge.audit(image_path, observations)
    except Exception as exc:
        return {"status": "error", "reason": sanitize_error_text(exc), "audit": None}
    return {"status": "ok", "reason": None, "audit": audit}


def resolve_default_visual_consistency_judge() -> "GeminiVisualConsistencyJudge | None":
    try:
        return GeminiVisualConsistencyJudge()
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Gemini Global Observer -- a whole-image CANDIDATE HYPOTHESIS GENERATOR,
# used by the "Run Full Analysis" workflow. Same construction/call pattern
# as GeminiVisualConsistencyJudge (separate class, separate prompt, no
# shared state). NOT authoritative: its `candidate_concern_hypotheses` are
# verified by case_interpretation.verify_gemini_concern_candidates against
# DOAR's own governed, Gemini-blind concern-domain results before they are
# ever shown -- Gemini's own text can never raise a concern domain's
# support level. `overall_scene`/`overall_visual_tone`/etc. are surfaced
# as a separate, clearly-labeled Gemini note alongside (never merged into)
# the deterministic GlobalImpression.
# ---------------------------------------------------------------------------

_GLOBAL_OBSERVER_MODEL_ENV_VAR = "DOAR_GLOBAL_OBSERVER_MODEL"

_GLOBAL_OBSERVER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall_scene": {"type": "STRING"},
        "overall_visual_tone": {"type": "STRING"},
        "expressive_descriptors": {"type": "ARRAY", "items": {"type": "STRING"}},
        "salient_relationships": {"type": "ARRAY", "items": {"type": "STRING"}},
        "important_visual_observations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "candidate_concern_hypotheses": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "domain": {"type": "STRING"}, "reason": {"type": "STRING"},
                    "observations": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["domain", "reason", "observations"],
            },
        },
        "uncertainty": {"type": "STRING"},
    },
    "required": ["overall_scene", "overall_visual_tone", "expressive_descriptors", "salient_relationships",
                 "important_visual_observations", "candidate_concern_hypotheses", "uncertainty"],
}

# The exact domain vocabulary Gemini is asked to use for
# candidate_concern_hypotheses.domain -- every REAL clinical domain DOAR's
# registry defines, plus the same explicitly-unmapped extras case_
# interpretation.py already recognizes (so an unsupported-but-named guess
# still resolves to NOT_CURRENTLY_ASSESSED instead of an unrecognized
# free-text string). maltreatment_or_safety_concern is deliberately
# INCLUDED in the vocabulary (so Gemini uses the real key rather than
# inventing one) but is always forced to NOT_CURRENTLY_ASSESSED at
# verification time regardless of what Gemini says -- see
# case_interpretation._FORBIDDEN_DOMAINS.
_GLOBAL_OBSERVER_DOMAIN_VOCABULARY = (
    "positive_affect_or_social_engagement", "anxiety_or_stress_related", "depressive_or_low_mood_related",
    "aggression_or_threat_related", "social_withdrawal_related", "developmental_or_attention_related",
    "maltreatment_or_safety_concern", "trauma_related", "bullying_related", "autism_related",
)

_GLOBAL_OBSERVER_SYSTEM_PROMPT = (
    "You are DOAR's whole-image observer for a children's-drawing research tool. Look at the "
    "COMPLETE drawing (not just one detail) and describe it as a candidate-hypothesis generator, "
    "never as a final authority:\n"
    "- overall_scene: one or two sentences describing the whole image.\n"
    "- overall_visual_tone: a short phrase (e.g. 'mostly cheerful', 'tense and dark', 'mixed').\n"
    "- expressive_descriptors: short adjectives for the overall expressive impression.\n"
    "- salient_relationships: how elements relate (grouping, isolation, proximity, scale) -- "
    "whole-drawing observations, not single-object descriptions.\n"
    "- important_visual_observations: plainly visible things worth noting.\n"
    "- candidate_concern_hypotheses: a list of {domain, reason, observations}. `domain` MUST be "
    "one of exactly this vocabulary (use the closest match, or omit if none fit -- never invent a "
    "new domain string): " + ", ".join(_GLOBAL_OBSERVER_DOMAIN_VOCABULARY) + ". These are CANDIDATES "
    "ONLY -- DOAR will independently verify each one against its own governed evidence; you are not "
    "deciding anything.\n"
    "- uncertainty: one short sentence on what is unclear or hard to judge from this image.\n"
    "Never state a diagnosis. Never claim certainty a single drawing cannot support. Respond with "
    "the required JSON object only."
)


@dataclass(frozen=True)
class GeminiGlobalObserver:
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 30.0
    request_fn: Callable[[str, str, dict, float], bytes] | None = None

    def __post_init__(self):
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set -- GeminiGlobalObserver is unavailable.")
        object.__setattr__(self, "api_key", key)
        object.__setattr__(
            self, "model", self.model or os.environ.get(_GLOBAL_OBSERVER_MODEL_ENV_VAR, "gemini-3.6-flash"))
        if self.request_fn is None:
            try:
                from google import genai  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai package is not installed -- GeminiGlobalObserver is unavailable. "
                    "Install the 'interaction' extra (pip install -e '.[interaction]') to enable it."
                ) from exc

    def observe(self, image_path: str) -> dict:
        """ONE call against `image_path`. Never persists anything, never
        touches any case file."""
        from .visual_observer import _guess_image_mime, _pil_image_to_png_b64, _post_gemini_generate_content
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        try:
            image_b64 = _pil_image_to_png_b64(img)
        finally:
            img.close()
        payload = {
            "systemInstruction": {"parts": [{"text": _GLOBAL_OBSERVER_SYSTEM_PROMPT}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe this complete drawing."},
                    {"inlineData": {"mimeType": _guess_image_mime(image_path), "data": image_b64}},
                ],
            }],
            "generationConfig": {
                "responseMimeType": "application/json", "responseSchema": _GLOBAL_OBSERVER_RESPONSE_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        send = self.request_fn or _post_gemini_generate_content
        raw = send(self.api_key, self.model, payload, self.timeout_seconds)
        response = json.loads(raw)
        if "error" in response:
            raise RuntimeError(f"Gemini API returned an error: {response['error']!r}")
        content_text = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(content_text)
        return {
            "overall_scene": parsed.get("overall_scene", ""),
            "overall_visual_tone": parsed.get("overall_visual_tone", ""),
            "expressive_descriptors": list(parsed.get("expressive_descriptors") or []),
            "salient_relationships": list(parsed.get("salient_relationships") or []),
            "important_visual_observations": list(parsed.get("important_visual_observations") or []),
            "candidate_concern_hypotheses": [
                {"domain": h.get("domain"), "reason": h.get("reason", ""), "observations": list(h.get("observations") or [])}
                for h in (parsed.get("candidate_concern_hypotheses") or [])
            ],
            "uncertainty": parsed.get("uncertainty", ""),
        }


def run_gemini_global_observation(bundle: dict, *, observer: "GeminiGlobalObserver | None" = None) -> dict:
    """READ-ONLY entry point -- called only from the explicit "Run Full
    Analysis" action, never automatically. Returns {"status":
    "unavailable"|"error"|"ok", "reason", "observation"}."""
    image_rel_path = (bundle.get("image") or {}).get("relative_path")
    if not image_rel_path:
        return {"status": "unavailable", "reason": "bundle has no image path", "observation": None}
    if observer is None:
        return {"status": "unavailable",
                "reason": "no Gemini global observer configured (GEMINI_API_KEY/google-genai not available)",
                "observation": None}
    image_path = str(ROOT / image_rel_path)
    try:
        observation = observer.observe(image_path)
    except Exception as exc:
        return {"status": "error", "reason": sanitize_error_text(exc), "observation": None}
    return {"status": "ok", "reason": None, "observation": observation}


def resolve_default_global_observer() -> "GeminiGlobalObserver | None":
    try:
        return GeminiGlobalObserver()
    except RuntimeError:
        return None


def _answer_visual_question(question: str, bundle: dict, *, open_vocab_predict_fn=None,
                             visual_recheck_provider: "GeminiVisualRecheckProvider | None" = None) -> dict:
    target = extract_visual_target(question)
    if target is None:
        return {"answer": "I couldn't tell which object this question is about.", "category": "visual",
                "chains": [], "evidence_ids": [], "rule_ids": [], "source_ids": [],
                "used_external_research": False, "used_visual_recheck": False}

    entities = bundle.get("entities") or []
    matches = [e for e in entities if e.matches_search_term(target)]
    if matches:
        verified = [e for e in matches if e.case_verification_status == "verified"]
        other = [e for e in matches if e.case_verification_status != "verified"]
        if verified:
            answer = (f"Yes -- DOAR's saved evidence for this drawing includes {len(verified)} confirmed "
                       f"'{target}' observation(s).")
        elif other:
            statuses = sorted({e.case_verification_status for e in other})
            answer = (f"A '{target}' was noticed by the first visual pass on this drawing, but it was not "
                       f"independently confirmed (status: {', '.join(statuses)}).")
        return {"answer": answer, "category": "visual", "chains": [],
                # "ev_semantic_" prefix matches drawing_synthesis.normalize_semantic_evidence's own
                # evidence_id scheme exactly -- a bare entity_id is never a known evidence_id in
                # _governed_case_as_legacy_view and would make every genuinely confirmed visual
                # answer fail deterministic_verify's evidence_id check.
                "evidence_ids": [f"ev_semantic_{e.entity_id}" for e in matches], "rule_ids": [], "source_ids": [],
                "used_external_research": False, "used_visual_recheck": False}

    # Nothing in saved evidence -- this is the ORIGINAL_SAVED_EVIDENCE ->
    # insufficient -> Q&A visual re-check branch (Part H). Prefer the
    # legacy case_dir/detections.json path only for callers still using
    # that (non-governed) format; the governed dev-set bundles used by
    # this app never have a case_dir, so they always take the new
    # image-path-based path below.
    if open_vocab_predict_fn is not None and bundle.get("case_dir"):
        try:
            from .registry_v2_build import build_registry_v2
            from .visual_evidence import search_visual
            finding = search_visual(bundle.get("case_dir"), target, registry_v2=build_registry_v2(),
                                     open_vocab_predict_fn=open_vocab_predict_fn)
        except Exception as exc:  # pragma: no cover -- defensive, never silently modifies the case
            return {"answer": f"On-demand visual re-check could not be run for this drawing ({sanitize_error_text(exc)}).",
                    "category": "visual", "chains": [], "evidence_ids": [], "rule_ids": [], "source_ids": [],
                    "used_external_research": False, "used_visual_recheck": True,
                    "visual_recheck_tag": _Q_AND_A_VISUAL_RECHECK_TAG, "visual_recheck_status": "error"}
        if finding is not None:
            answer = (f"No prior saved evidence existed for '{target}'; an on-demand visual re-check found it "
                       f"(confidence {finding.confidence:.2f}, status: {finding.validation_status.lower()}). This is "
                       f"experimental re-check evidence, not part of the original DOAR analysis.")
            status = "found"
        else:
            answer = (f"No prior saved evidence existed for '{target}', and an on-demand visual re-check found none "
                       f"either. This does not prove it is absent from the drawing.")
            status = "not_found"
        return {"answer": answer, "category": "visual", "chains": [],
                "evidence_ids": [], "rule_ids": [], "source_ids": [],
                "used_external_research": False, "used_visual_recheck": True,
                "visual_recheck_tag": _Q_AND_A_VISUAL_RECHECK_TAG, "visual_recheck_status": status}

    recheck = governed_visual_recheck(target, bundle, provider=visual_recheck_provider)
    if recheck["status"] == "unavailable":
        answer = (f"DOAR's saved evidence for this drawing does not include '{target}', and on-demand visual "
                   f"re-checking is not available in this environment ({recheck['reason']}), so I can't verify "
                   f"this reliably. This does not prove '{target}' is absent from the drawing.")
        return {"answer": answer, "category": "visual", "chains": [],
                "evidence_ids": [], "rule_ids": [], "source_ids": [],
                "used_external_research": False, "used_visual_recheck": True,
                "visual_recheck_tag": _Q_AND_A_VISUAL_RECHECK_TAG, "visual_recheck_status": "detector_unavailable"}
    if recheck["status"] == "error":
        return {"answer": f"On-demand visual re-check could not be run for this drawing ({recheck['reason']}).",
                "category": "visual", "chains": [], "evidence_ids": [], "rule_ids": [], "source_ids": [],
                "used_external_research": False, "used_visual_recheck": True,
                "visual_recheck_tag": _Q_AND_A_VISUAL_RECHECK_TAG, "visual_recheck_status": "error"}

    candidate = recheck["candidate"]
    verification = recheck["verification"]
    conf = candidate.get("confidence")
    conf_text = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
    if candidate["present"]:
        confirm_note = ""
        if verification is not None and verification.get("independently_confirmed") is True:
            confirm_note = " A second, independent re-check call confirmed this."
        elif verification is not None and verification.get("independently_confirmed") is False:
            confirm_note = " A second, independent re-check call did not confirm this, so treat it as uncertain."
        answer = (f"No prior saved evidence existed for '{target}'. An on-demand visual re-check of the original "
                   f"image found: {candidate['description']}{conf_text}.{confirm_note} This is experimental Q&A "
                   f"re-check evidence, not part of the original DOAR analysis.")
        status = "found"
    else:
        answer = (f"No prior saved evidence existed for '{target}', and an on-demand visual re-check of the "
                   f"original image did not find it{conf_text}: {candidate['description']} This does not "
                   f"definitively prove it is absent from the drawing.")
        status = "not_found"
    return {"answer": answer, "category": "visual", "chains": [],
            "evidence_ids": [], "rule_ids": [], "source_ids": [],
            "used_external_research": False, "used_visual_recheck": True,
            "visual_recheck_tag": _Q_AND_A_VISUAL_RECHECK_TAG, "visual_recheck_status": status,
            "visual_recheck_candidate": candidate, "visual_recheck_verification": verification}


def _search_internal_corpus(question: str) -> dict | None:
    """Searches the FROZEN rule matrix + source register text for the
    question's own keywords -- e.g. "what does red mean" searches for
    "red" in observable_feature/source_claim/possible_interpretation.
    Returns None (never a guess) if nothing textually matches."""
    q = (question or "").casefold()
    stop = {"what", "does", "do", "mean", "means", "is", "are", "the", "a", "an", "in", "of", "to", "for",
            "using", "use", "used", "usually", "children's", "children", "child", "drawings", "drawing"}
    words = [w.strip("?.,!") for w in q.split() if w.strip("?.,!") not in stop and len(w) > 2]
    if not words:
        return None
    matrix = rc.load_rule_matrix()
    for rule_id, row in matrix.items():
        haystack = " ".join([
            row.get("observable_feature", ""), row.get("source_claim", ""),
            row.get("possible_interpretation_as_written", ""),
        ]).casefold()
        # Word-boundary matching -- a plain substring check would let
        # "red" false-match inside "stressed"/"already", etc.
        if any(re.search(rf"\b{re.escape(w)}\b", haystack) for w in words):
            return {"rule_id": rule_id, "row": row}
    return None


# ---------------------------------------------------------------------------
# 4b. Structured evidence package + AI answer helper (Part 1 fix).
#
# The package is the ONLY thing an answer helper or Judge ever sees --
# never the raw bundle/synthesis object, never unrestricted hidden case
# state. It is built entirely from fields this module already computes
# (build_evidence_chains, unmatched_verified_observations, the category
# answerer's own already-scoped `result["chains"]`) -- nothing new is
# derived from the case for this purpose.
# ---------------------------------------------------------------------------


def _entity_observation_summaries(bundle: dict) -> dict:
    entities = bundle.get("entities") or []

    def _group(statuses):
        return sorted({e.canonical_label for e in entities if e.case_verification_status in statuses})

    return {
        "verified": _group({"verified"}),
        "uncertain_or_unreviewed": _group({"uncertain", "unreviewed"}),
        "rejected": _group({"rejected"}),
    }


def _compact_objective_feature_summary(bundle: dict) -> dict:
    det = bundle.get("deterministic_features") or {}
    colour = det.get("colour") or {}
    comp = det.get("composition") or {}
    obj = det.get("objective_features") or {}
    intensity = obj.get("stroke.intensity_proxy")
    frag = obj.get("stroke.fragmentation")
    return {
        "dominant_colour": colour.get("dominant_colour"),
        "colour_diversity": colour.get("colour_diversity"),
        "line_intensity_proxy": (round(intensity.value, 3) if intensity is not None and not intensity.missing else None),
        "line_fragmentation_proxy": (round(frag.value, 3) if frag is not None and not frag.missing else None),
        "composition_bounding_box_coverage": comp.get("bounding_box_coverage"),
        "composition_placement": comp.get("placement"),
    }


def _compact_emotion_summary(bundle: dict) -> dict | None:
    """Real, calibrated expressive-content model output for this case, if
    one ran -- included as raw grounding data only (Milestone 1: live-case
    adapter). Never phrased as a psychological claim here; the answer/
    judge prompts already forbid that. Returns None (never fabricated)
    when no model ran or it produced no probabilities."""
    emotion = bundle.get("emotion")
    if not emotion or emotion.get("status") != "available":
        return None
    probabilities = emotion.get("probabilities") or {}
    if not probabilities:
        return None
    top_class = max(probabilities, key=probabilities.get)
    return {"top_class": top_class, "top_probability": round(probabilities[top_class], 3),
            "calibration_status": emotion.get("calibration_status")}


def build_evidence_package(question: str, audience: str, result: dict, bundle: dict) -> dict:
    """The ONLY input an AI answer helper / AI Judge ever receives.
    `relevant_matched_rules` is deliberately just `result["chains"]`
    (every category answerer already returns exactly the chains it
    scoped its own deterministic answer to) -- not every rule in the
    case -- so the AI is grounded in the SAME material the deterministic
    path already selected as relevant to this specific question."""
    synthesis = bundle.get("synthesis")
    observations = _entity_observation_summaries(bundle)
    provenance_labels = []
    if result.get("used_visual_recheck"):
        provenance_labels.append(result.get("visual_recheck_tag", _Q_AND_A_VISUAL_RECHECK_TAG))
    if result.get("used_external_research"):
        provenance_labels.append("external_research")
    if not provenance_labels:
        provenance_labels.append("original_saved_evidence")
    return {
        "question": question,
        "audience": audience,
        "verified_observations": observations["verified"],
        "uncertain_or_unreviewed_observations": observations["uncertain_or_unreviewed"],
        "rejected_observations": observations["rejected"],
        "objective_features": _compact_objective_feature_summary(bundle),
        "relevant_matched_rules": [
            {
                "rule_display_name": c["rule_display_name"], "rule_suggests": c["rule_suggests"],
                "why_it_applies": c["why_it_applies"], "concern_label": c["concern"]["label"],
                "evidence_family": c["evidence_family_display"], "evidence_strength": c["evidence_strength"],
                "source_pdf_page_section": c["source"]["source_pdf_page_section"],
                "citation_title": c["source"]["citation_title"], "doi_or_pmid": c["source"]["doi_or_pmid"],
            }
            for c in result.get("chains", [])
        ],
        "observations_without_an_approved_rule": unmatched_verified_observations(bundle),
        "overall_synthesis": ({"level": synthesis.overall_synthesis["level"],
                                "summary": synthesis.overall_synthesis["summary"]}
                               if synthesis is not None else None),
        "expressive_model": _compact_emotion_summary(bundle),
        "case_interpretation": ci.summarize_for_grounding(ci.build_case_interpretation(bundle)),
        "external_research": (
            {"sources": result.get("external_sources", [])} if result.get("used_external_research") else None),
        "deterministic_draft_answer": result.get("answer"),
        "provenance_labels": provenance_labels,
    }


class AnswerProvider(Protocol):
    def generate(self, *, question: str, audience: str, package: dict,
                  revision_instructions: list[str] | None = None,
                  conversation_history: list[dict] | None = None) -> str | None:
        """Returns a natural-language answer string grounded ONLY in
        `package`, or None to signal the caller should fall back to
        `package["deterministic_draft_answer"]`. MUST NEVER invent an
        evidence item, rule, source, object, or cause not already
        present in `package`. `conversation_history` (a bounded list of
        recent {"role": "user"|"assistant", "content": str} turns) is
        CONTEXT ONLY, for resolving follow-ups like "what about the
        second feature?" -- it is never a source of factual grounding; a
        prior assistant turn's own text can never itself become
        evidence."""
        ...


class DeterministicAnswerProvider:
    """The offline, always-available answer helper -- no network call.
    Simply returns the already-computed deterministic template answer
    verbatim; this is the fallback every optional Gemini answer provider
    degrades to when unavailable."""

    def generate(self, *, question: str, audience: str, package: dict,
                  revision_instructions: list[str] | None = None,
                  conversation_history: list[dict] | None = None) -> str | None:
        return package.get("deterministic_draft_answer")


_ANSWER_MODEL_ENV_VAR = "DOAR_ANSWER_MODEL"

_ANSWER_SYSTEM_PROMPT = (
    "You are DOAR's Ask-DOAR answer helper. You are given a QUESTION and a structured "
    "EVIDENCE PACKAGE containing everything DOAR currently knows about this drawing: "
    "verified/uncertain/rejected observations, objective features, matched literature-"
    "linked rules (with their own wording and sources), observations without an approved "
    "rule, the overall synthesis, and any external research already retrieved. You may also "
    "receive recent CONVERSATION HISTORY.\n\n"
    "Write like a thoughtful human explaining this to the reader, NOT like database output. "
    "The first sentence should normally answer the question directly. Weave the evidence into "
    "natural, flowing prose.\n\n"
    "Do NOT:\n"
    "- say things like 'this drawing includes 1 feature(s)' or 'N feature(s)' -- name the "
    "feature(s) in plain words instead\n"
    "- expose internal IDs, rule IDs, evidence IDs, or field names from the package\n"
    "- repeat a generic disclaimer after every sentence (one brief caveat, if any, is enough)\n"
    "- sound like you are reading a structured record aloud\n\n"
    "Example of the desired style, for 'Why did you mention stress?':\n"
    "\"The main reason is the relatively light line quality in this drawing. One of the rules "
    "used by DOAR links this feature with stress- or anxiety-related indicators. In this "
    "drawing, no second independent feature supports the same pattern, so it is treated as one "
    "possible indicator rather than a strong overall stress pattern.\"\n\n"
    "Use ONLY the material in the evidence package for facts. NEVER invent an object, rule, "
    "source, cause, or feeling that is not already present in the package. NEVER describe an "
    "uncertain/unreviewed/rejected observation as confirmed. NEVER state a diagnosis (e.g. 'the "
    "child has depression') -- only describe associations the evidence/literature supports, and "
    "clearly say when evidence is insufficient. If external research is present in the package, "
    "keep it clearly distinguishable from DOAR's own original evidence.\n\n"
    "CONVERSATION HISTORY, if provided, is CONTEXT ONLY -- it helps you understand what a "
    "follow-up question like 'what about the second feature?' or 'why?' refers to. A prior "
    "assistant turn's own text is NEVER evidence and must never be treated as a fact source; "
    "every factual claim in your answer must still be grounded in the evidence package above.\n\n"
    "Audience style: for 'parent', use plain, short, warm, non-technical language. For "
    "'clinician', use professional language and may mention evidence families, concern "
    "domains, and convergence -- but still write in readable sentences, not a report dump."
)


def _format_conversation_history(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return ""
    lines = [f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in conversation_history]
    return "\n\nRECENT CONVERSATION HISTORY (context only, NOT evidence):\n" + "\n".join(lines)


@dataclass(frozen=True)
class GeminiAnswerProvider:
    """Optional real answer helper -- Gemini, given ONLY the structured
    evidence package `build_evidence_package()` produces (plus, if
    supplied, a bounded recent conversation history for follow-up
    context). Reads GEMINI_API_KEY from the environment (never
    hardcoded); model configurable via DOAR_ANSWER_MODEL, defaulting to
    `gemini-3.6-flash`. Raises RuntimeError at construction if the key or
    `google-genai` package is missing, so callers fail fast and fall
    back to `DeterministicAnswerProvider` rather than degrading silently
    mid-question. `timeout_seconds` (default 30.0, matching every other
    Gemini-backed class in this module) bounds the network call itself --
    without it, a stalled/slow request to Gemini hangs the entire
    Streamlit script run with no visible error, which is exactly what
    made Ask DOAR look like "Send does nothing" during live testing."""
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self):
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set -- GeminiAnswerProvider is unavailable.")
        object.__setattr__(self, "api_key", key)
        object.__setattr__(self, "model", self.model or os.environ.get(_ANSWER_MODEL_ENV_VAR, "gemini-3.6-flash"))
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is not installed -- GeminiAnswerProvider is unavailable. "
                "Install the 'interaction' extra (pip install -e '.[interaction]') to enable it."
            ) from exc

    def generate(self, *, question: str, audience: str, package: dict,
                  revision_instructions: list[str] | None = None,
                  conversation_history: list[dict] | None = None) -> str | None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt_parts = [f"QUESTION: {question}", f"AUDIENCE: {audience}",
                         f"EVIDENCE PACKAGE (JSON): {json.dumps(package, default=str)}"]
        history_text = _format_conversation_history(conversation_history)
        if history_text:
            prompt_parts.append(history_text.strip())
        if revision_instructions:
            prompt_parts.append(
                "REVISION INSTRUCTIONS from DOAR's Judge (your previous answer failed review -- fix these "
                "specific issues, still using ONLY the evidence package above): " + "; ".join(revision_instructions))
        response = client.models.generate_content(
            model=self.model, contents="\n\n".join(prompt_parts),
            config=types.GenerateContentConfig(
                system_instruction=_ANSWER_SYSTEM_PROMPT,
                http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
            ),
        )
        text = getattr(response, "text", None)
        return text.strip() if text else None


def _safe_generate(provider: AnswerProvider, *, question: str, audience: str, package: dict,
                    revision_instructions: list[str] | None = None,
                    conversation_history: list[dict] | None = None) -> tuple[str | None, str | None]:
    """Never lets a provider's network/parsing failure crash the Q&A
    flow. Returns (text, error): `text` is the generated answer or None
    (either the provider had nothing to say, e.g. `DeterministicAnswerProvider`
    re-echoing an unchanged draft, or it failed); `error` is a sanitized
    reason ONLY when an exception was actually caught, so the caller can
    tell "provider had nothing new to say" apart from "provider crashed"
    (needed for accurate answer_provider/answer_provider_error provenance
    -- Part 3)."""
    try:
        text = provider.generate(question=question, audience=audience, package=package,
                                  revision_instructions=revision_instructions,
                                  conversation_history=conversation_history)
    except Exception as exc:
        return None, sanitize_error_text(exc)
    return (text.strip() if isinstance(text, str) and text.strip() else None), None


def _describe_answer_provider(provider: AnswerProvider, *, succeeded: bool) -> str:
    """`succeeded` must reflect whether THIS provider's text is what is
    actually being shown as the final answer right now -- never just
    which provider object was configured (Part 3's own fix: reporting
    "gemini:..." while displaying deterministic fallback text is exactly
    the bug this guards against)."""
    if isinstance(provider, DeterministicAnswerProvider):
        return "deterministic"
    if succeeded:
        model = getattr(provider, "model", None)
        return f"gemini:{model}" if model else provider.__class__.__name__
    return "deterministic_fallback"


class ExternalResearchProvider(Protocol):
    def research(self, question: str) -> dict | None:
        """Return {"summary": str, "sources": [{"title":..., "url":...}]}
        or None if no reliable result was found. MUST NEVER modify any
        case's saved evidence and MUST NEVER be presented as part of the
        original DOAR analysis by the caller."""
        ...


class DisabledExternalResearchProvider:
    """The default, no-network provider -- always returns None (honest
    'not enabled' rather than a fabricated result)."""

    def research(self, question: str) -> dict | None:
        return None


class GeminiGroundedResearchProvider:
    """Optional real provider: Gemini with Google Search grounding, using
    the project's existing GEMINI_API_KEY (never hardcoded). Only
    constructible when both the API key and the `google-genai` package
    are present; otherwise raises RuntimeError at construction time so
    the caller fails fast and falls back to
    DisabledExternalResearchProvider rather than silently degrading
    mid-question. Model ID is configurable, defaults to a current Gemini
    model; never touches the frozen Observer/Verifier model settings."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 timeout_seconds: float = 30.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("DOAR_RESEARCH_MODEL", "gemini-3.6-flash")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set -- GeminiGroundedResearchProvider is unavailable.")
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is not installed -- GeminiGroundedResearchProvider is unavailable. "
                "See https://ai.google.dev/gemini-api/docs/google-search for the grounding tool this provider uses."
            ) from exc

    def research(self, question: str) -> dict | None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        response = client.models.generate_content(
            model=self.model,
            contents=(f"Answer this question about children's drawing psychology using reliable "
                      f"scientific/academic/clinical sources only. Be concise and honest about "
                      f"uncertainty. Question: {question}"),
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            return None
        sources = []
        try:
            candidate = response.candidates[0]
            chunks = candidate.grounding_metadata.grounding_chunks or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web is not None:
                    sources.append({"title": getattr(web, "title", "") or "", "url": getattr(web, "uri", "") or ""})
        except Exception:
            sources = []
        if not sources:
            return None
        return {"summary": text, "sources": sources}


def _answer_general_question(question: str, bundle: dict, *,
                              external_research_provider: ExternalResearchProvider | None = None) -> dict:
    hit = _search_internal_corpus(question)
    if hit is not None:
        rule_id, row = hit["rule_id"], hit["row"]
        src = source_citation(rule_id)
        answer = (f"DOAR's approved rule corpus has a related entry -- '{rule_display_name(rule_id, row)}': "
                   f"{row.get('possible_interpretation_as_written', '')}"
                   + (f" (source: {src['citation_title']})" if src.get("citation_title") else "") + ".")
        return {"answer": answer, "category": "general", "chains": [],
                "evidence_ids": [], "rule_ids": [rule_id], "source_ids": [rule_id],
                "used_external_research": False, "used_visual_recheck": False}

    provider = external_research_provider or DisabledExternalResearchProvider()
    try:
        result = provider.research(question)
        research_error = None
    except Exception as exc:
        # The provider RAISED (network/API/model error) -- this is
        # DIFFERENT from a clean "no reliable evidence found" result and
        # must be reported as such (Part 2's own A vs. B distinction);
        # never let this propagate up and crash the caller.
        result = None
        research_error = sanitize_error_text(exc)

    if research_error is not None:
        return {"answer": ("I couldn't complete the external literature search right now, so I can't verify a "
                            "broader interpretation beyond DOAR's current sources."),
                "category": "general", "chains": [], "evidence_ids": [], "rule_ids": [], "source_ids": [],
                "used_external_research": False, "used_visual_recheck": False,
                "research_status": "error", "research_error": research_error}

    if result and result.get("sources"):
        source_lines = "; ".join(f"{s.get('title', 'source')} ({s.get('url', '')})" for s in result["sources"])
        answer = f"{_EXTERNAL_RESEARCH_LABEL}: {result['summary']} Sources: {source_lines}"
        return {"answer": answer, "category": "general", "chains": [],
                "evidence_ids": [], "rule_ids": [], "source_ids": [],
                "used_external_research": True, "used_visual_recheck": False,
                "external_sources": result["sources"], "research_status": "success"}

    return {"answer": "I couldn't find reliable evidence for a specific interpretation of this.",
            "category": "general", "chains": [], "evidence_ids": [], "rule_ids": [], "source_ids": [],
            "used_external_research": False, "used_visual_recheck": False, "research_status": "no_result"}


# ---------------------------------------------------------------------------
# 5a. Negation-aware diagnostic-language check -- ADDITIVE wrapper around
#     judges.py's own, unmodified DIAGNOSTIC_PATTERNS/_ARABIC_DIAGNOSTIC.
# ---------------------------------------------------------------------------

# Short, explicit negation/hedge cues that, when present in the same
# sentence BEFORE a diagnostic-pattern match, mean the sentence is
# DISCUSSING or DENYING a diagnosis rather than asserting one. Kept small
# and literal (not a general negation parser) so it never accidentally
# absorbs a genuine positive diagnostic claim.
_NEGATION_CUES_EN = (
    "not enough", "not sufficient", "insufficient evidence", "does not mean", "doesn't mean",
    "is not evidence", "isn't evidence", "cannot say", "can't say", "cannot conclude", "can't conclude",
    "not saying", "doesn't prove", "does not prove", "not confirm", "does not confirm", "no evidence that",
    "not enough evidence", "not a diagnosis", "not diagnose", "does not diagnose", "not enough to say",
    "associated with", "sometimes associated", "may be associated", "possible", "not enough evidence to say",
)
_NEGATION_CUES_AR = ("لا يكفي", "لا يعني", "ليس دليلاً على", "لا يثبت", "لا يمكن الجزم")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")


def _sentence_containing(text: str, match_start: int) -> str:
    """Returns the sentence (period-delimited) containing character
    offset `match_start`, for a local negation check -- never scans the
    whole answer, only the clause the diagnostic term actually appears
    in, so a hedge earlier/later in a long answer can't mask a genuine
    unsupported claim elsewhere."""
    boundaries = [0] + [m.start() for m in re.finditer(r"[.!?\n]", text)] + [len(text)]
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= match_start <= boundaries[i + 1]:
            return text[boundaries[i]:boundaries[i + 1]]
    return text


def is_unsupported_diagnostic_claim(text: str) -> bool:
    """True only for a genuine positive/unsupported diagnostic assertion
    -- reuses judges.py's own pattern list verbatim, then requires the
    containing sentence to NOT also contain a negation/association/hedge
    cue. "This drawing does not provide enough evidence to say the child
    is depressed" and "associated with anxiety/stress" are both False
    here; "the child is depressed" alone is True."""
    if not text:
        return False
    for m in _ARABIC_DIAGNOSTIC.finditer(text):
        sentence = _sentence_containing(text, m.start())
        if not any(cue in sentence for cue in _NEGATION_CUES_AR):
            return True
    for pattern in DIAGNOSTIC_PATTERNS:
        for m in pattern.finditer(text):
            sentence = _sentence_containing(text, m.start()).casefold()
            if not any(cue in sentence for cue in _NEGATION_CUES_EN):
                return True
    return False


# ---------------------------------------------------------------------------
# 5b. Structured answer + deterministic verification (reuses claim_verifier.py
#     unchanged, via a read-only adapter) + Judge.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredAnswer:
    answer: str
    claims: list[dict]
    provenance: list[str]
    used_external_research: bool
    used_visual_recheck: bool
    category: str
    judge_verdict: str
    judge_reasons: list[str] = field(default_factory=list)
    judge_mode: str = "deterministic_fallback"
    judge_details: dict = field(default_factory=dict)
    answer_provider: str = "deterministic"
    answer_provider_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer, "claims": self.claims, "provenance": self.provenance,
            "used_external_research": self.used_external_research,
            "used_visual_recheck": self.used_visual_recheck, "category": self.category,
            "judge_verdict": self.judge_verdict, "judge_reasons": self.judge_reasons,
            "judge_mode": self.judge_mode, "judge_details": self.judge_details,
            "answer_provider": self.answer_provider, "answer_provider_error": self.answer_provider_error,
        }


def _governed_case_as_legacy_view(bundle: dict) -> tuple[dict, dict]:
    """Read-only adapter: projects the governed `DrawingSynthesisResult`
    into the (analysis_dict, registry_dict) shape `claim_verifier.py`
    already knows how to check (evidence / rule_evaluations /
    objective_features // references) -- so the EXISTING verifier
    functions run unmodified against governed evidence. Never written
    back anywhere; a throwaway view built fresh per question."""
    synthesis = bundle.get("synthesis")
    if synthesis is None:
        return {"evidence": [], "rule_evaluations": [], "objective_features": {}}, {"references": {}}

    evidence = [{"evidence_id": ue.item.evidence_id, "value": ue.item.value, "status": ue.item.status}
                for ue in synthesis.unified_evidence]
    matched_by_rule: dict[str, list[str]] = {}
    for a in synthesis.literature_linked_associations:
        matched_by_rule.setdefault(a["rule_id"], []).extend(a["matched_entity_ids"])

    checks = bundle.get("checks") or []
    det = bundle.get("deterministic_features")
    det_status_by_rule: dict[str, str] = {}
    if det is not None:
        from . import drawing_synthesis as ds
        det_checks = ds.check_deterministic_preconditions(det["composition"], det["objective_features"],
                                                            det.get("page_reference"))
        det_status_by_rule = {c.rule_id: c.status for c in det_checks}

    rule_evaluations = []
    for c in checks:
        if c.rule_id in det_status_by_rule:
            real_status = det_status_by_rule[c.rule_id]
            status = "weak_support" if real_status == "satisfied" else (
                "missing_detector" if real_status == "not_assessable" else "not_matched")
        else:
            status = "weak_support" if c.status == "satisfied" else (
                "missing_detector" if c.status in ("blocked_structural", "blocked_needs_unbuilt_feature")
                else "not_matched")
        rule_evaluations.append({"rule_id": c.rule_id, "status": status,
                                  "matched_evidence_ids": matched_by_rule.get(c.rule_id, [])})

    matrix = rc.load_rule_matrix()
    references = {rid: True for rid in load_rule_source_register()} | {rid: True for rid in matrix}
    return {"evidence": evidence, "rule_evaluations": rule_evaluations, "objective_features": {}}, {"references": references}


def _claims_from_result(result: dict) -> list[cv.Claim]:
    return [cv.Claim(text=result["answer"], evidence_ids=list(result.get("evidence_ids", [])),
                      rule_ids=list(result.get("rule_ids", [])), source_ids=list(result.get("source_ids", [])))]


_CERTAINTY_UPGRADE_WORDS = ("confirmed", "verified", "definitely", "certainly", "clearly shows", "clearly present")
_CERTAINTY_UPGRADE_NEGATION_CUES = (
    "not confirmed", "not verified", "not independently confirmed", "not clearly", "unconfirmed",
    "possible", "possibly", "uncertain", "not confirm", "may not", "isn't confirmed", "is not confirmed",
)


def _narrator_upgrades_uncertain_evidence(answer_text: str, bundle: dict) -> list[str]:
    """Catches an answer text that describes an entity this CASE only
    has as uncertain/unreviewed/rejected as if it were confirmed --
    complementary to `claim_verifier.verify_unavailable_wording` (which
    only checks evidence_ids actually CITED on the claim); this instead
    scans the free answer text itself, since a narrator could describe
    an unverified entity by its plain label without citing any ID at
    all. Skips a sentence that also contains a hedging/negation cue, so
    "noticed but not independently confirmed" is correctly allowed."""
    if not answer_text:
        return []
    entities = bundle.get("entities") or []
    not_verified_labels = sorted({e.canonical_label for e in entities
                                   if e.case_verification_status in ("uncertain", "unreviewed", "rejected")})
    if not not_verified_labels:
        return []
    offenses = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer_text):
        s_lower = sentence.casefold()
        if not any(w in s_lower for w in _CERTAINTY_UPGRADE_WORDS):
            continue
        if any(cue in s_lower for cue in _CERTAINTY_UPGRADE_NEGATION_CUES):
            continue
        for label in not_verified_labels:
            if label.casefold() in s_lower:
                offenses.append(f"'{label}' is described as confirmed/certain in the answer text but is not "
                                 f"verified in this case's saved evidence")
    return offenses


def deterministic_verify(result: dict, bundle: dict) -> dict:
    """Reuses claim_verifier.verify_claims() UNCHANGED against the
    governed-case adapter view. This is the FIRST verification layer."""
    analysis_view, registry_view = _governed_case_as_legacy_view(bundle)
    claims = _claims_from_result(result)
    report = cv.verify_claims(claims, analysis_view, registry_view)
    # Extra, human-interaction-specific checks not covered by the legacy
    # verifier's own vocabulary (which never saw "used_external_research"
    # or a free-text visual-recheck answer):
    extra_failed = []
    if is_unsupported_diagnostic_claim(result["answer"]):
        extra_failed.append("unsupported diagnostic language in answer text")
    if result.get("used_external_research") and _EXTERNAL_RESEARCH_LABEL not in result["answer"]:
        extra_failed.append("external research used but not labeled as separate from the original DOAR analysis")
    if result.get("used_visual_audit") and _VISUAL_AUDIT_LABEL not in result["answer"]:
        # Defensive/forward-looking: no current answerer sets used_visual_audit (the Visual
        # Consistency Judge is never wired into answer_question), but this check exists so a
        # future integration mistake can't silently blend an AUDIT_ONLY finding into case evidence.
        extra_failed.append("visual consistency audit used but not labeled as a separate, audit-only finding")
    extra_failed.extend(_narrator_upgrades_uncertain_evidence(result["answer"], bundle))
    return {**report, "all_passed": report["all_passed"] and not extra_failed, "extra_checks_failed": extra_failed}


class Judge(Protocol):
    def judge(self, *, question: str, result: dict, verification: dict, package: dict,
              conversation_history: list[dict] | None = None) -> dict:
        """Returns a structured dict with at least "verdict" (PASS/
        REVISE/FAIL) and "revision_instructions" (list[str]);
        `DeterministicJudge`/`GeminiJudge` both also fill the full Part-3
        schema (answers_question/case_grounded/rule_fidelity/
        source_support/status_fidelity/external_provenance_correct/
        unsupported_claims) plus "judge_mode". `conversation_history` is
        CONTEXT ONLY -- e.g. to confirm a follow-up like "why?" was
        correctly resolved against the RIGHT prior topic -- never a
        substitute for checking the answer against `package` itself."""
        ...


def _answers_the_question(question: str, answer_text: str) -> bool:
    """Coarse but real relevance check: at least one non-trivial word
    from the question should appear in the answer, OR the answer is a
    recognized honest-refusal template (which DOES answer -- it answers
    'no reliable evidence exists')."""
    refusal_markers = ("couldn't find", "couldn't identify", "does not currently have", "not available",
                        "I can't verify", "could not be run")
    if any(m in answer_text for m in refusal_markers):
        return True
    q_words = {w.strip("?.,!").casefold() for w in (question or "").split() if len(w) > 3}
    a_lower = (answer_text or "").casefold()
    # Substring containment (not exact-token match) so a hyphenated/
    # inflected answer word ("stress-related", "stressed") still counts
    # as addressing a question word ("stress").
    return any(w in a_lower for w in q_words) or not q_words


def _judge_dict_defaults() -> dict:
    return {
        "answers_question": True, "case_grounded": True, "rule_fidelity": True, "source_support": True,
        "status_fidelity": True, "external_provenance_correct": True, "unsupported_claims": [],
        "revision_instructions": [],
    }


def _safe_judge(judge: "Judge", *, question: str, result: dict, verification: dict, package: dict,
                 conversation_history: list[dict] | None = None) -> dict:
    """Never lets a Judge implementation's own failure crash the Q&A
    flow -- `GeminiJudge` already catches its own network/parse errors
    internally, but this is the Human Interaction boundary's own
    defense-in-depth (Part 2) for any Judge, current or future. On an
    uncaught exception, falls back to the already-computed deterministic
    `verification` result alone rather than blocking the answer."""
    try:
        return judge.judge(question=question, result=result, verification=verification, package=package,
                            conversation_history=conversation_history)
    except Exception as exc:
        out = _judge_dict_defaults()
        out["verdict"] = "PASS" if verification.get("all_passed") else "REVISE"
        out["judge_mode"] = "judge_error_fallback"
        out["revision_instructions"] = [f"Judge call failed ({sanitize_error_text(exc)}); used the deterministic "
                                         f"verification result only."]
        return out


class DeterministicJudge:
    """No LLM, no network call -- the default, always-available Judge,
    and the FIRST layer that ALWAYS runs (see `GeminiJudge`, which
    delegates to this one before ever calling out). PASS/REVISE/FAIL
    from: (1) deterministic verification result, (2) the negation-aware
    diagnostic-language check, (3) a coarse question-relevance check.
    `judge_mode` is always "deterministic_fallback" -- the exact string
    this task's Part 3 asks to expose when no semantic Judge ran."""

    def judge(self, *, question: str, result: dict, verification: dict, package: dict | None = None,
              conversation_history: list[dict] | None = None) -> dict:
        reasons = []
        if not verification["all_passed"]:
            for pc in verification.get("per_claim", []):
                if not pc["passed"]:
                    reasons.extend(c["reason"] for c in pc["checks"] if not c["passed"])
            reasons.extend(verification.get("extra_checks_failed", []))
        if not _answers_the_question(question, result["answer"]):
            reasons.append("answer does not appear to address the question asked")

        out = _judge_dict_defaults()
        out["judge_mode"] = "deterministic_fallback"
        if not reasons:
            out["verdict"] = "PASS"
            return out
        out["unsupported_claims"] = reasons
        out["revision_instructions"] = reasons
        out["case_grounded"] = not any("unknown evidence_id" in r for r in reasons)
        out["rule_fidelity"] = not any("unknown rule_id" in r for r in reasons)
        out["source_support"] = not any("unknown source_id" in r for r in reasons)
        out["external_provenance_correct"] = not any(
            "external research used but not labeled" in r for r in reasons)
        out["answers_question"] = not any("does not appear to address" in r for r in reasons)
        severe = any("diagnostic" in r or "unknown evidence_id" in r or "unknown rule_id" in r for r in reasons)
        out["verdict"] = "FAIL" if severe else "REVISE"
        return out


_JUDGE_MODEL_ENV_VAR = "DOAR_JUDGE_MODEL"

_JUDGE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["PASS", "REVISE", "FAIL"]},
        "answers_question": {"type": "BOOLEAN"},
        "case_grounded": {"type": "BOOLEAN"},
        "rule_fidelity": {"type": "BOOLEAN"},
        "source_support": {"type": "BOOLEAN"},
        "status_fidelity": {"type": "BOOLEAN"},
        "external_provenance_correct": {"type": "BOOLEAN"},
        "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
        "revision_instructions": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["verdict", "answers_question", "case_grounded", "rule_fidelity", "source_support",
                 "status_fidelity", "external_provenance_correct", "unsupported_claims", "revision_instructions"],
}

_JUDGE_SYSTEM_PROMPT = (
    "You are DOAR's independent semantic Judge for Ask-DOAR answers. You receive the "
    "user's exact question, a candidate answer, and the EXACT structured evidence package "
    "the answer was supposed to be grounded in (case evidence, rules, references, evidence "
    "statuses, external research if any). Check against this FIXED checklist, strictly:\n"
    "1. Evidence grounding -- is every factual claim about THIS drawing traceable to this "
    "case's own saved evidence in the package?\n"
    "2. Verified/uncertain/rejected fidelity -- is UNCERTAIN/UNREVIEWED/REJECTED evidence ever "
    "presented as certain/confirmed?\n"
    "3. Missing detector != absence -- does the answer ever treat 'not detected' or 'not "
    "assessed' as proof something is absent from the drawing?\n"
    "4. Page-assessability gates -- if the page/placement could not be confirmed assessable, "
    "does the answer still make a page-relative claim as if it could?\n"
    "5. Rule eligibility/prerequisites -- are descriptions of DOAR rules faithful to the rule "
    "wording and status given (never a rule presented as satisfied when its status says "
    "otherwise)?\n"
    "6. Source validity -- do cited references actually support the statement attributed to "
    "them?\n"
    "7. No diagnosis/causal invention -- did the answer state or imply a diagnosis, or invent a "
    "cause, reason, object, or emotion not in the package?\n"
    "8. No single weak observation escalated to a strong combined conclusion -- is one "
    "individual, unconverged heuristic described with the certainty of a combined, multi-family "
    "pattern?\n"
    "9. External research separated -- was external/outside research ever confused with or "
    "presented as original DOAR case evidence?\n"
    "10. Visual-audit findings separated -- was a visual-consistency AUDIT finding (if any) ever "
    "presented as original DOAR case evidence rather than a separate, audit-only observation?\n"
    "11. Alternatives/limitations preserved -- if the package lists alternative explanations or "
    "limitations for a cited observation, does the answer suppress them in a way that overstates "
    "certainty?\n"
    "12. Did the response directly answer the question asked?\n"
    "You may also receive recent CONVERSATION HISTORY -- this is CONTEXT ONLY, to help you tell "
    "whether a follow-up question was resolved against the right earlier topic. A prior "
    "assistant turn's own text is NEVER itself evidence; every factual claim in the candidate "
    "answer must still be checked against the EVIDENCE PACKAGE, not against what a prior turn "
    "said.\n"
    "Return PASS only if the answer is fully faithful. Return REVISE with specific, actionable "
    "revision_instructions if it is fixable. Return FAIL only if it is fundamentally "
    "unsupportable from the package. Respond with the required JSON object only."
)


@dataclass(frozen=True)
class GeminiJudge:
    """Optional real second-layer Judge -- SEMANTIC checks the
    deterministic layer cannot make (rule fidelity, source support,
    invented causes/emotions, over-strong associations). The
    deterministic layer ALWAYS runs first (`DeterministicJudge`,
    delegated to here) and its verdict is never overridden when it
    already failed -- this class only ADDS a semantic check on top of an
    already-deterministically-clean candidate, it never replaces the
    hard ID/status/diagnostic-language gate. Configurable model via
    DOAR_JUDGE_MODEL, separate by default from the answer-generation
    model (DOAR_ANSWER_MODEL). Same fail-fast-at-construction contract as
    the other optional Gemini providers in this module."""
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self):
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set -- GeminiJudge is unavailable.")
        object.__setattr__(self, "api_key", key)
        object.__setattr__(
            self, "model", self.model or os.environ.get(_JUDGE_MODEL_ENV_VAR, "gemini-3.5-flash-lite"))
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is not installed -- GeminiJudge is unavailable. "
                "Install the 'interaction' extra (pip install -e '.[interaction]') to enable it."
            ) from exc

    def judge(self, *, question: str, result: dict, verification: dict, package: dict,
              conversation_history: list[dict] | None = None) -> dict:
        deterministic_out = DeterministicJudge().judge(
            question=question, result=result, verification=verification, package=package,
            conversation_history=conversation_history)
        if deterministic_out["verdict"] != "PASS":
            # Defense in depth: a hard deterministic failure (fabricated
            # ID, unsupported diagnostic language) is NEVER overridden by
            # the semantic layer -- no Gemini call is even made.
            deterministic_out["judge_mode"] = "gemini+deterministic_gate"
            return deterministic_out

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            contents = (f"QUESTION: {question}\n\nCANDIDATE ANSWER: {result['answer']}\n\n"
                        f"EVIDENCE PACKAGE (JSON): {json.dumps(package, default=str)}"
                        + _format_conversation_history(conversation_history))
            response = client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_JUDGE_SYSTEM_PROMPT,
                    response_mime_type="application/json", response_schema=_JUDGE_RESPONSE_SCHEMA,
                    http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
                ),
            )
            parsed = json.loads(response.text)
            if parsed.get("verdict") not in ("PASS", "REVISE", "FAIL"):
                raise ValueError(f"Judge returned an invalid verdict: {parsed.get('verdict')!r}")
        except Exception as exc:
            # Network/parse/malformed-verdict failure -- deterministic
            # checks already passed above, so this is a safe PASS, just
            # not a semantic one; never silently invents a verdict.
            out = _judge_dict_defaults()
            out["verdict"] = "PASS"
            out["judge_mode"] = "gemini_error_fallback"
            out["revision_instructions"] = [f"Gemini Judge call failed ({sanitize_error_text(exc)}); "
                                             f"deterministic checks passed."]
            return out
        parsed.setdefault("unsupported_claims", [])
        parsed.setdefault("revision_instructions", [])
        parsed["judge_mode"] = "gemini"
        return parsed


_FALLBACK_ANSWER = "I couldn't verify a reliable answer to that question from the available evidence and sources."


def build_structured_answer(question: str, category: str, result: dict, bundle: dict, *,
                             judge: Judge | None = None, answer_provider: AnswerProvider | None = None,
                             audience: str = "clinician",
                             conversation_history: list[dict] | None = None) -> StructuredAnswer:
    """Answer helper generates -> deterministic verify -> Judge ->
    (REVISE: send Judge's revision_instructions back to the answer
    helper and regenerate ONCE, then verify+Judge again) -> still not
    PASS: safe fallback with every citation cleared. Never loops more
    than twice regardless of which Judge/answer-provider is used.
    `conversation_history` is bounded to the last 8 turns here (never
    trusted as evidence -- see `AnswerProvider`/`Judge`'s own docs) and
    passed through to whichever provider/Judge is actually used.
    `answer_provider`/`judge_mode` in the returned `StructuredAnswer`
    ALWAYS reflect what actually produced the displayed answer/verdict,
    never just which provider object was configured (Part 3)."""
    judge = judge or DeterministicJudge()
    answer_provider = answer_provider or DeterministicAnswerProvider()
    history = (conversation_history or [])[-8:]
    package = build_evidence_package(question, audience, result, bundle)

    answer_provider_error = None
    provider_succeeded = False

    generated, gen_error = _safe_generate(answer_provider, question=question, audience=audience, package=package,
                                           conversation_history=history)
    if generated:
        result = {**result, "answer": generated}
        provider_succeeded = True
    elif gen_error:
        answer_provider_error = gen_error

    verification = deterministic_verify(result, bundle)
    judge_out = _safe_judge(judge, question=question, result=result, verification=verification, package=package,
                             conversation_history=history)
    verdict = judge_out.get("verdict", "FAIL")
    reasons = judge_out.get("revision_instructions") or judge_out.get("unsupported_claims") or []

    if verdict != "PASS":
        # One bounded retry: ask the SAME answer provider to regenerate
        # using the Judge's own revision_instructions. If that yields
        # nothing new (e.g. DeterministicAnswerProvider, which cannot
        # "revise" free text, or the Gemini provider fails again), fall
        # back to the honest fallback text with EVERY citation cleared --
        # a claim that cites nothing always passes verify_evidence_ids/
        # verify_rule_ids/verify_source_ids/verify_unavailable_wording,
        # so this is a genuine, verifiable re-attempt either way, never a
        # cosmetic one.
        revised_text, revise_error = _safe_generate(
            answer_provider, question=question, audience=audience, package=package,
            conversation_history=history,
            revision_instructions=reasons or ["Revise to be fully grounded in the evidence package and directly "
                                               "answer the question."])
        if revise_error and answer_provider_error is None:
            answer_provider_error = revise_error
        if revised_text and revised_text != result["answer"]:
            result = {**result, "answer": revised_text}
            provider_succeeded = True
        else:
            result = {**result, "answer": _FALLBACK_ANSWER, "evidence_ids": [], "rule_ids": [], "source_ids": []}
            provider_succeeded = False
        verification = deterministic_verify(result, bundle)
        judge_out = _safe_judge(judge, question=question, result=result, verification=verification, package=package,
                                 conversation_history=history)
        verdict = judge_out.get("verdict", "FAIL")
        reasons = judge_out.get("revision_instructions") or judge_out.get("unsupported_claims") or []

    if verdict != "PASS":
        # Still not clean after the one retry -- never loop again.
        result = {**result, "answer": _FALLBACK_ANSWER, "evidence_ids": [], "rule_ids": [], "source_ids": []}
        provider_succeeded = False
        verdict = "PASS"
        reasons = list(reasons) + ["fallback used after Judge did not pass on retry"]

    provider_label = _describe_answer_provider(answer_provider, succeeded=provider_succeeded)

    claims = [{
        "text": result["answer"], "claim_type": category,
        "evidence_ids": result.get("evidence_ids", []), "rule_ids": result.get("rule_ids", []),
        "source_ids": result.get("source_ids", []),
        "external_source_ids": [s.get("url") for s in result.get("external_sources", [])],
    }]
    provenance = list(result.get("rule_ids", [])) + list(result.get("evidence_ids", []))
    if result.get("used_visual_recheck"):
        provenance.append(result.get("visual_recheck_tag", _Q_AND_A_VISUAL_RECHECK_TAG))
    if result.get("used_external_research"):
        provenance.append("external_research")
    return StructuredAnswer(
        answer=result["answer"], claims=claims, provenance=provenance,
        used_external_research=bool(result.get("used_external_research")),
        used_visual_recheck=bool(result.get("used_visual_recheck")),
        category=category, judge_verdict=verdict, judge_reasons=reasons,
        judge_mode=judge_out.get("judge_mode", "deterministic_fallback"), judge_details=judge_out,
        answer_provider=provider_label, answer_provider_error=answer_provider_error,
    )


# ---------------------------------------------------------------------------
# Optional-provider resolution -- tries the real Gemini-backed provider,
# catches RuntimeError (missing key/package), and falls back to the
# always-available deterministic/disabled default. answer_question()
# itself NEVER calls these automatically (its own defaults stay fully
# offline/deterministic, so it is always safe to call from a test with no
# network) -- only the UI layer (clinician_review_app.py) calls these to
# decide what to inject.
# ---------------------------------------------------------------------------


def resolve_default_answer_provider() -> AnswerProvider:
    try:
        return GeminiAnswerProvider()
    except RuntimeError:
        return DeterministicAnswerProvider()


def resolve_default_research_provider() -> ExternalResearchProvider:
    try:
        return GeminiGroundedResearchProvider()
    except RuntimeError:
        return DisabledExternalResearchProvider()


def resolve_default_judge() -> Judge:
    try:
        return GeminiJudge()
    except RuntimeError:
        return DeterministicJudge()


def resolve_default_visual_recheck_provider() -> "GeminiVisualRecheckProvider | None":
    try:
        return GeminiVisualRecheckProvider()
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Top-level entry point -- the ONE function the UI (Parent/Clinician Ask
# DOAR box) calls. Same evidence for both audiences; only rendering of
# the returned StructuredAnswer (and the `audience` string, which only
# affects answer-helper phrasing style) differs by view. Every optional
# provider defaults to the always-available offline behaviour -- callers
# (the UI layer) opt into the real Gemini-backed versions explicitly via
# `resolve_default_*()`.
# ---------------------------------------------------------------------------


def answer_question(question: str, bundle: dict, *, audience: str = "clinician", open_vocab_predict_fn=None,
                     visual_recheck_provider: "GeminiVisualRecheckProvider | None" = None,
                     external_research_provider: ExternalResearchProvider | None = None,
                     answer_provider: AnswerProvider | None = None,
                     judge: Judge | None = None,
                     conversation_history: list[dict] | None = None) -> StructuredAnswer:
    """`conversation_history` (optional): recent {"role", "content"}
    turns from THIS chat, for resolving follow-ups ("what about the
    second feature?", "why?"). Bounded to the last 8 turns internally.
    Context only -- never a source of factual grounding (a prior
    assistant turn can never itself become evidence; see
    `AnswerProvider`/`Judge`)."""
    category = route_question(question)
    if category == "case":
        result = _answer_case_question(question, bundle)
    elif category == "rule_source":
        result = _answer_rule_source_question(question, bundle)
    elif category == "visual":
        result = _answer_visual_question(question, bundle, open_vocab_predict_fn=open_vocab_predict_fn,
                                          visual_recheck_provider=visual_recheck_provider)
    else:
        result = _answer_general_question(question, bundle, external_research_provider=external_research_provider)
    return build_structured_answer(question, category, result, bundle, judge=judge,
                                    answer_provider=answer_provider, audience=audience,
                                    conversation_history=conversation_history)
