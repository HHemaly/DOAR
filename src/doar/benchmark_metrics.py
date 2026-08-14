"""DOAR development benchmark metrics (V1.7). Computes the visual and
reasoning metrics `BENCHMARK_SCHEMA.md` defines, plus inter-annotator
visual agreement (normalized concept-set pairwise F1/Jaccard +
adjudication rate).

**Human annotation is ALWAYS the reference.** No function here ever
compares one LLM/DOAR output against another and calls the result
"ground truth" -- every visual metric takes `HumanAnnotationItem`s as an
explicit, required argument. **The two annotators are never silently
merged**: `visual_precision_recall_f1`/`salient_recall`/`hallucination_
rate` all take ONE annotator's items at a time -- a caller wanting both
annotators' numbers calls these twice and reports both, never a union.
The only function that looks at both annotators together is
`inter_annotator_agreement` itself, whose whole purpose is comparing them
to EACH OTHER, not producing a merged ground truth.

Reuses `visual_observer._labels_plausibly_match` (the same deterministic,
token-overlap label matcher already used throughout the Observer/Verifier
pipeline) for every "does this label match that label" decision --
"matched" means the same thing here as it already does everywhere else in
DOAR, not a new, second definition of matching.
"""
from __future__ import annotations

from dataclasses import dataclass

from .visual_observer import _labels_plausibly_match

# ---------------------------------------------------------------------------
# Human ground truth shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanAnnotationItem:
    label: str
    location: str = ""
    confidence: str | None = None  # "clear" | "ambiguous" -- Pass 2 only, None for Pass 1


def _matches_any(label: str, items: list[HumanAnnotationItem]) -> bool:
    return any(_labels_plausibly_match(label, item.label) for item in items)


def _item_matched_by_any(item: HumanAnnotationItem, labels: list[str]) -> bool:
    return any(_labels_plausibly_match(label, item.label) for label in labels)


# ---------------------------------------------------------------------------
# Visual metrics (BENCHMARK_SCHEMA.md Section 3) -- one annotator at a time.
# ---------------------------------------------------------------------------


def visual_precision_recall_f1(candidate_labels: list[str], human_items_pass2: list[HumanAnnotationItem]) -> dict:
    """Precision/recall/F1 of `candidate_labels` (one condition's output on
    one image) against ONE annotator's Pass-2 exhaustive list. Returns
    None for any ratio whose denominator is zero -- never a fabricated
    0.0 or 1.0 for an undefined case."""
    total_candidates = len(candidate_labels)
    total_human_items = len(human_items_pass2)
    matched_candidates = sum(1 for c in candidate_labels if _matches_any(c, human_items_pass2))
    matched_human_items = sum(1 for h in human_items_pass2 if _item_matched_by_any(h, candidate_labels))
    precision = (matched_candidates / total_candidates) if total_candidates else None
    recall = (matched_human_items / total_human_items) if total_human_items else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "matched_candidates": matched_candidates, "total_candidates": total_candidates,
        "matched_human_items": matched_human_items, "total_human_items": total_human_items,
    }


def salient_recall(candidate_labels: list[str], human_items_pass1: list[HumanAnnotationItem]) -> dict:
    """What fraction of a Pass-1 item list did the candidates cover?
    Takes whatever item list the caller passes (one annotator's own
    Pass-1 items, the union `salient` set, or the intersection
    `core_salient` set -- see `normalized_pass1_salience` below); this
    function itself has no opinion about which one is "the" reference."""
    total = len(human_items_pass1)
    matched = sum(1 for h in human_items_pass1 if _item_matched_by_any(h, candidate_labels))
    return {"salient_recall": (matched / total) if total else None, "matched": matched, "total": total}


def normalized_pass1_salience(
        items_a_pass1: list[HumanAnnotationItem], items_b_pass1: list[HumanAnnotationItem],
) -> dict:
    """The frozen salience definitions, computed from two annotators'
    independent Pass-1 lists:

        salient      = normalized Pass-1 concept UNION (present for >=1 annotator)
        core_salient = normalized Pass-1 concept INTERSECTION (present for BOTH)

    "Normalized" means concept identity is decided by the same
    deterministic `_labels_plausibly_match` matcher every other label
    comparison in this module uses (greedy one-to-one, matching
    `inter_annotator_agreement`'s own matching so the two metrics agree
    about which items are "the same concept") -- it does NOT rewrite or
    canonicalize any label text. Neither list is copied into or mutated
    by this function; `items_a_pass1`/`items_b_pass1` (and the original
    annotation files they were loaded from) are left exactly as they
    were -- this only reads them to build two NEW lists."""
    remaining_b = list(items_b_pass1)
    core: list[HumanAnnotationItem] = []
    unmatched_a: list[HumanAnnotationItem] = []
    for item_a in items_a_pass1:
        match_index = None
        for i, item_b in enumerate(remaining_b):
            if _labels_plausibly_match(item_a.label, item_b.label):
                match_index = i
                break
        if match_index is not None:
            core.append(item_a)
            del remaining_b[match_index]
        else:
            unmatched_a.append(item_a)
    unmatched_b = remaining_b
    salient = core + unmatched_a + unmatched_b
    return {"salient": salient, "core_salient": core}


def primary_and_sensitivity_salient_recall(
        candidate_labels: list[str],
        items_a_pass1: list[HumanAnnotationItem], items_b_pass1: list[HumanAnnotationItem],
) -> dict:
    """The two salient-recall numbers this phase's parent task requires:

        primary_salient_recall     -- recall against `salient` (the union;
                                       the more lenient, "did DOAR notice
                                       ANYTHING either annotator flagged"
                                       reading)
        sensitivity_salient_recall -- recall against `core_salient` (the
                                       intersection; the stricter reading
                                       -- only items BOTH annotators
                                       independently flagged as salient)

    Per-annotator `salient_recall(candidate_labels, items_a_pass1)` /
    `salient_recall(candidate_labels, items_b_pass1)` remain separately
    available to callers -- this function does not replace them, and
    does not merge the two annotators into anything treated as sole
    ground truth."""
    salience = normalized_pass1_salience(items_a_pass1, items_b_pass1)
    return {
        "primary_salient_recall": salient_recall(candidate_labels, salience["salient"]),
        "sensitivity_salient_recall": salient_recall(candidate_labels, salience["core_salient"]),
        "salient_count": len(salience["salient"]),
        "core_salient_count": len(salience["core_salient"]),
    }


def hallucination_rate(candidate_labels: list[str], human_items_pass2_any_annotator: list[HumanAnnotationItem]) -> dict:
    """Fraction of candidates matching NO human-listed item from EITHER
    annotator's Pass-2 list (caller passes the union of both annotators'
    items here specifically -- a candidate is only a hallucination if
    NEITHER annotator saw it, the more charitable and correct reading of
    "matching no human-listed item" than requiring both annotators to
    agree)."""
    total = len(candidate_labels)
    unmatched = sum(1 for c in candidate_labels if not _matches_any(c, human_items_pass2_any_annotator))
    return {"hallucination_rate": (unmatched / total) if total else None, "unmatched": unmatched, "total": total}


_ABSTENTION_ENTITY_TYPES = frozenset({"unknown", "scribble", "abstract_mark"})
_ABSTENTION_VERIFICATION_STATUSES = frozenset({"uncertain", "unreviewed"})


def abstention_rate(candidates: list[dict]) -> dict:
    """Fraction of candidates that HONESTLY abstained (unknown/scribble/
    abstract_mark entity_type, or an uncertain/unreviewed verifier status)
    rather than committing to a specific, possibly-wrong label. Each dict
    needs `entity_type` and, if verified, `case_verification_status`."""
    total = len(candidates)
    abstained = sum(
        1 for c in candidates
        if c.get("entity_type") in _ABSTENTION_ENTITY_TYPES
        or c.get("case_verification_status") in _ABSTENTION_VERIFICATION_STATUSES
    )
    return {"abstention_rate": (abstained / total) if total else None, "abstained": abstained, "total": total}


def verifier_correction_rate(candidates_with_verification: list[dict]) -> dict:
    """Fraction of bbox-bearing observer candidates the independent
    verifier corrected away from a confident observer read (observer
    confidence >= 0.5) to `rejected`/`uncertain`. Each dict needs
    `confidence`, `bbox`, and `case_verification_status`."""
    eligible = [c for c in candidates_with_verification if c.get("bbox") is not None]
    total = len(eligible)
    corrected = sum(
        1 for c in eligible
        if (c.get("confidence") or 0.0) >= 0.5 and c.get("case_verification_status") in ("rejected", "uncertain")
    )
    return {"verifier_correction_rate": (corrected / total) if total else None, "corrected": corrected, "total": total}


# ---------------------------------------------------------------------------
# Reasoning metrics (BENCHMARK_SCHEMA.md Section 4) -- condition B only.
# ---------------------------------------------------------------------------


def evidence_backed_claim_rate(matches: list) -> dict:
    """EligibleAtomicRuleMatch objects (reasoning_chain.py) -- every one
    is, BY CONSTRUCTION, backed by >=1 matched_entity_id (see
    reasoning_chain.build_eligible_matches's own docstring). Expected
    1.0 always; a value below 1.0 is a regression to investigate, not a
    benchmark finding about drawing content."""
    total = len(matches)
    backed = sum(1 for m in matches if getattr(m, "matched_entity_ids", None))
    return {"evidence_backed_claim_rate": (backed / total) if total else None, "backed": backed, "total": total}


def unsupported_claim_rate(hypotheses: list, rule_matrix: dict) -> dict:
    """CandidateHypothesis objects -- a claim is "unsupported" if any of
    its supporting_rule_ids is not a real row in RULE_EVIDENCE_MATRIX.csv.
    Expected 0.0 always, by reasoning_chain.py's own construction (it only
    ever reads rule_ids out of that same matrix)."""
    total = sum(len(h.supporting_rule_ids) for h in hypotheses)
    unsupported = sum(1 for h in hypotheses for rid in h.supporting_rule_ids if rid not in rule_matrix)
    return {"unsupported_claim_rate": (unsupported / total) if total else None, "unsupported": unsupported, "total": total}


def rule_reference_traceability(hypotheses: list, rule_matrix: dict) -> dict:
    total = sum(len(h.supporting_rule_ids) for h in hypotheses)
    traceable = sum(1 for h in hypotheses for rid in h.supporting_rule_ids if rid in rule_matrix)
    return {"traceability_rate": (traceable / total) if total else None, "traceable": traceable, "total": total}


def hypothesis_derivability(hypotheses_by_image: dict[str, list]) -> dict:
    """image_id -> list[CandidateHypothesis]. Reports how many images
    produced >=1 hypothesis, broken down by support_level -- never a
    correctness claim, purely a "how often does the chain reach each
    strength level on real drawings" count."""
    from collections import Counter
    images_with_hypothesis = sum(1 for hs in hypotheses_by_image.values() if hs)
    level_counts: Counter = Counter()
    for hs in hypotheses_by_image.values():
        for h in hs:
            level_counts[h.support_level] += 1
    return {
        "images_with_at_least_one_hypothesis": images_with_hypothesis,
        "total_images": len(hypotheses_by_image),
        "support_level_counts": dict(level_counts),
    }


# ---------------------------------------------------------------------------
# Inter-annotator visual agreement -- "normalized concept-set pairwise
# F1/Jaccard + adjudication rate" (the frozen plan this phase implements).
# ---------------------------------------------------------------------------


def inter_annotator_agreement(items_a: list[HumanAnnotationItem], items_b: list[HumanAnnotationItem]) -> dict:
    """Greedy one-to-one concept matching between two annotators' item
    lists (each item matches at most one counterpart, avoiding double-
    counting when several B items would otherwise fuzzy-match the same A
    item), using the SAME `_labels_plausibly_match` normalization as
    every other matching decision in this module.

    - matched_pairs: count of A items with a found, not-yet-claimed match in B.
    - jaccard = matched_pairs / (len(A) + len(B) - matched_pairs)
    - f1 = 2 * matched_pairs / (len(A) + len(B))
    - adjudication_rate = unmatched items (either annotator) / (len(A) + len(B))
      -- the fraction of ALL recorded items that have no counterpart and
      would need a human adjudicator to resolve before being trusted as
      agreed-upon ground truth. This function does NOT resolve that
      disagreement or produce a merged list -- see module docstring."""
    remaining_b = list(items_b)
    matched_pairs = 0
    for item_a in items_a:
        for i, item_b in enumerate(remaining_b):
            if _labels_plausibly_match(item_a.label, item_b.label):
                matched_pairs += 1
                del remaining_b[i]
                break
    total = len(items_a) + len(items_b)
    unmatched = total - 2 * matched_pairs
    jaccard = matched_pairs / (total - matched_pairs) if (total - matched_pairs) else None
    f1 = (2 * matched_pairs / total) if total else None
    adjudication_rate = (unmatched / total) if total else None
    return {
        "matched_pairs": matched_pairs, "total_items_a": len(items_a), "total_items_b": len(items_b),
        "jaccard": jaccard, "f1": f1, "adjudication_rate": adjudication_rate, "unmatched_items": unmatched,
    }
