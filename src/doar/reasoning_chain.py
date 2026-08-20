"""DOAR reasoning chain (V1.9): verified visual observation -> eligible
atomic rule -> evidence family -> concern domain -> candidate clinical
hypothesis.

**Semantic matching fixes (15-image development rehearsal, V1.9):** four
real precondition-matching bugs found by inspecting real trace output,
none of which touch `RULE_EVIDENCE_MATRIX.csv`, `CONCERN_DOMAIN_MAP.
json`, or any rule's claim/interpretation/strength -- only which
`VisualEntity` tokens are allowed to satisfy which rule's PRECONDITION:
1. Generic "eye"/"eyes" presence no longer satisfies all three mutually
   incompatible eye-style rules (wide/stern/closed) at once -- each now
   requires an explicit style qualifier on the SAME entity
   (`STYLE_QUALIFIED_PRESENCE_TERMS`).
2. A person-like entity being verified while a part (mouth/hands) was
   merely never independently verified no longer satisfies the
   corresponding missing-part rule -- "not detected" is not "confirmed
   absent"; these rules are now intentionally near-unsatisfiable pending
   a real omission detector (`ABSENCE_REQUIRES_EXPLICIT_OMISSION_
   EVIDENCE`).
3. A face/head's own circular OUTLINE (e.g. "green face circle") no
   longer satisfies the standalone circle-symbolism rule
   (`CONTEXT_EXCLUDED_PRESENCE_TERMS`).
4. Generic face/eyes/hair presence no longer satisfies the face-
   expression rule -- it now requires the rule's own source_claim
   expression vocabulary (smiling/sad/frown/tense/frightened/crying,
   etc.), not mere face-shaped presence (`POSITIVE_PRESENCE_TERMS["EN_
   COMPILED_FACE_EXPRESSION_021"]`).

**SHADOW MODE, unconditionally.** This module is entirely NEW and
ADDITIVE: it reads the frozen `RULE_EVIDENCE_MATRIX.csv` / `RULE_
RELATIONSHIP_GRAPH.json` / `CONCERN_DOMAIN_MAP.json` (all read-only, never
written here) and a case's `VisualEntity` list, and produces
`CandidateHypothesis` objects. It NEVER calls `rules.py::evaluate_rules`,
NEVER calls `concerns.py::derive_concerns`, and NEVER writes into a case's
real `analysis.json` `rule_evaluations`/`concerns` fields. Nothing this
module produces can reach the psychological rule engine, exactly the same
structural guarantee `visual_entity.merge_observer_candidates_into_
entities` already provides for observer-only entities (no backing
`VisualFinding` => invisible to `integrate_visual_findings_into_case`).

**VISUALLY VERIFIED != PSYCHOLOGICALLY VALIDATED**, enforced two ways:
1. A rule's visual PRECONDITION can only be satisfied by a `VisualEntity`
   with `case_verification_status == "verified"` (the Observer/Verifier
   pipeline's own strongest status) -- but satisfying the precondition
   only produces an `EligibleAtomicRuleMatch`, carrying that rule's own
   frozen `allowed_output_level` (`disabled` or `individual_heuristic_
   only` for every one of the 41 rules -- never escalated here).
2. Aggregation reuses `concerns.py`'s own tested `_aggregation_strength`
   unchanged. Every rule-sourced piece of evidence is tagged source_type
   `clinician_symbolic` (same as `concerns.py` already tags every
   clinician-authored rule) REGARDLESS of whether its visual precondition
   was Gemini-verified -- verifying that an object is PRESENT is not
   independent confirmation of the PSYCHOLOGICAL CLAIM attached to it by
   the source PDF. Multiple such rules firing together therefore still
   cap out at `WEAK_HYPOTHESIS`, exactly as they would with zero visual
   verification at all -- see `test_reasoning_chain.py`'s dedicated proof
   of this property. Only a genuinely independent second source (e.g. the
   emotion model's own prediction, passed in as `model_evidence`, exactly
   as `rules.py::evaluate_rules` already does for `concerns.py`) can ever
   raise a hypothesis past that ceiling.

No rule is "arbitrarily promoted": every one of the 41 rules keeps
exactly its `RULE_EVIDENCE_MATRIX.csv`-frozen `allowed_output_level`,
`evidence_direction`, and `context_transfer_justification`.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .concerns import _aggregation_strength

if TYPE_CHECKING:
    from .visual_entity import VisualEntity

ROOT = Path(__file__).resolve().parents[2]
RULE_EVIDENCE_MATRIX_PATH = ROOT / "RULE_EVIDENCE_MATRIX.csv"
RULE_RELATIONSHIP_GRAPH_PATH = ROOT / "RULE_RELATIONSHIP_GRAPH.json"
CONCERN_DOMAIN_MAP_PATH = ROOT / "CONCERN_DOMAIN_MAP.json"

# The only allowed_output_level value a rule may carry to become usable
# scientific evidence (an EligibleAtomicRuleMatch). "disabled" (31 of 41
# rules today) means the rule's own detector is documented as unbuilt/
# unvalidated -- governance eligibility gate, added after a registry audit
# found that a satisfied visual precondition alone (with no check on this
# field) was enough to promote a disabled rule into synthesis; see
# RULE_ELIGIBILITY_GATE_FIX.md. Reused verbatim by drawing_synthesis.py's
# deterministic match builder so there is exactly one gate definition.
ELIGIBLE_OUTPUT_LEVEL = "individual_heuristic_only"

DISCLAIMER = (
    "This is a decision-support hypothesis, not a final diagnosis. Visual verification "
    "confirms that something was drawn -- it does not validate any psychological "
    "interpretation attached to it. See SCIENTIFIC_LIMITATIONS.md and RULE_EVIDENCE_AUDIT.md."
)

# ---------------------------------------------------------------------------
# Frozen-file loaders -- read-only, cached at module scope. Never mutate the
# loaded structures; never write back to any of these three files.
# ---------------------------------------------------------------------------


def load_rule_matrix() -> dict[str, dict]:
    """rule_id -> row dict, from the frozen RULE_EVIDENCE_MATRIX.csv."""
    with open(RULE_EVIDENCE_MATRIX_PATH, encoding="utf-8") as f:
        return {row["rule_id"]: row for row in csv.DictReader(f)}


def load_relationship_graph() -> dict:
    return json.loads(RULE_RELATIONSHIP_GRAPH_PATH.read_text(encoding="utf-8"))


def load_concern_domain_map() -> dict:
    return json.loads(CONCERN_DOMAIN_MAP_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Visual precondition -> search-term mapping. Curated by hand from each
# rule's `observable_feature` (RULE_EVIDENCE_MATRIX.csv) and REQUIRES
# precondition text (RULE_RELATIONSHIP_GRAPH.json) -- NOT auto-derived from
# the observable string, so an odd observable name never silently produces
# a wrong search term. Four disjoint categories, each with an honest,
# stated reason when a rule cannot be checked:
#
#   POSITIVE_PRESENCE_TERMS: satisfied by >=1 VERIFIED VisualEntity whose
#     canonical_label/aliases match one of the listed terms.
#   STYLE_QUALIFIED_PRESENCE_TERMS: satisfied by >=1 VERIFIED VisualEntity
#     whose canonical_label/aliases match BOTH an object term (e.g. "eye")
#     AND a style-qualifier term (e.g. "wide") -- see the dict's own
#     docstring for why this exists (Bug 1 fix).
#   CONTEXT_EXCLUDED_PRESENCE_TERMS: like POSITIVE_PRESENCE_TERMS, but an
#     entity whose tokens ALSO include a disqualifying context term is
#     never counted (Bug 3 fix).
#   ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE: intentionally near-
#     unsatisfiable today -- see the dict's own docstring (Bug 2 fix).
#   Everything else in the 41-rule matrix is `blocked_structural` with a
#   reason drawn directly from RULE_EVIDENCE_MATRIX.csv's own
#   requires_process_data/requires_longitudinal_data/requires_absolute_
#   scale_or_context columns, or -- for the handful of static-detectable
#   rules that still cannot be checked by simple presence/absence matching
#   (a quality/proportion/density judgment, not an identity check) -- an
#   explicit `blocked_needs_unbuilt_feature` reason.
# ---------------------------------------------------------------------------

POSITIVE_PRESENCE_TERMS: dict[str, tuple[str, ...]] = {
    "PSY_AR_ANIMAL_TIGER_WOLF_004": ("tiger", "wolf"),
    "PSY_AR_ANIMAL_FOX_005": ("fox",),
    "PSY_AR_ANIMAL_SQUIRREL_006": ("squirrel",),
    "PSY_AR_ANIMAL_LION_007": ("lion",),
    "PSY_AR_GEOMETRY_008": ("geometric shape", "shape", "square", "triangle", "rectangle"),
    "PSY_AR_STARS_009": ("star",),
    "PSY_AR_TRANSPORT_012": ("car", "vehicle", "truck", "bicycle", "bus", "train", "airplane", "boat"),
    "PSY_AR_HEARTS_013": ("heart",),
    # Bug 4 fix: `observable_feature` for this rule is literally
    # "face_expression" and its own `source_claim` names two concrete
    # readings -- "smiling" (positive tone) vs "sad, tense, or frightened"
    # (distress) -- yet the precondition used to be just the bare term
    # "face", so ANY verified face-shaped entity (e.g. a plain "green face
    # circle") wrongly satisfied a rule about a specific EXPRESSION. Fixed
    # by requiring one of the source_claim's own expression terms (plus
    # "crying"/"tears"/"tearful" and "frown/frowning/downturned", the
    # ordinary visual vocabulary for the same "distress" reading, and
    # explicitly named in this task's own bug report as legitimate
    # expression evidence) -- never mere face/eyes/hair presence.
    "EN_COMPILED_FACE_EXPRESSION_021": (
        "smiling", "smile", "sad", "frown", "frowning", "downturned", "tense", "frightened",
        "crying", "tears", "tearful",
    ),
    "EN_COMPILED_ANIMAL_CHOICE_GENERAL_022": ("animal", "tiger", "wolf", "fox", "squirrel", "lion"),
    "EN_COMPILED_HOUSE_023": ("house",),
    "EN_COMPILED_TREE_024": ("tree",),
}

# Bug 1 fix: a generic "eye"/"eyes" entity only proves eyes are PRESENT --
# it says nothing about their STYLE, so it must never satisfy all three
# mutually incompatible style rules (wide vs. stern vs. closed) at once, a
# real bug found during the 15-image development rehearsal (any "eye"/
# "eyes"/"left eye"/"right eye" entity satisfied all three simultaneously).
# Each rule here now requires BOTH the generic object term (an eye is
# present) AND a style-qualifier term drawn from that rule's OWN name/
# `observable_feature` -- "wide"/"open"/"exaggerated" for wide_eyes (this
# task's own wording), "closed"/"shut" for closed_eyes (ditto), and
# "stern"/"narrowed"/"glaring" for stern_eyes (the closest ordinary visual
# -- not psychological -- synonyms for "stern" as an eye's drawn
# appearance; deliberately NOT the broader "angry/tense/suspicious"
# vocabulary from the rule's `possible_interpretation_as_written`, since
# those are the PSYCHOLOGICAL reading this precondition must not smuggle
# in as if it were visual evidence). A style-qualifier term must appear on
# the SAME entity as the object term -- a separate entity being "wide" or
# "stern" elsewhere in the drawing does not count.
STYLE_QUALIFIED_PRESENCE_TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "PSY_AR_EYES_WIDE_001": (("eye", "eyes"), ("wide", "open", "exaggerated")),
    "PSY_AR_EYES_STERN_002": (("eye", "eyes"), ("stern", "narrowed", "glaring")),
    "PSY_AR_EYES_CLOSED_003": (("eye", "eyes"), ("closed", "shut")),
}

# Bug 3 fix: "circle" as a bare term matched a face/head's own OUTLINE
# shape (e.g. observer candidate "green face circle", aliases including
# "head"/"face") just as readily as a genuine standalone circle
# symbol/shape drawn in the scene -- a real bug found during the 15-image
# rehearsal. `PSY_AR_CIRCLES_011` is a shape-SYMBOLISM rule (`evidence_
# family=shape_symbolism`) about a circle drawn AS a symbol, not about
# anatomy happening to be round -- so an entity whose tokens also include
# a face/head context term is excluded even if "circle" is also present.
CONTEXT_EXCLUDED_PRESENCE_TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "PSY_AR_CIRCLES_011": (("circle",), ("face", "head")),
}

# Bug 2 fix: "not detected" is NOT "visually confirmed absent". The prior
# implementation satisfied a missing-part rule whenever a person-like
# entity was VERIFIED but no part entity was independently VERIFIED --
# but a part can go unverified for reasons that have nothing to do with
# it being absent (the Observer never proposed it at all, or the
# label-blind Verifier rejected/left it "uncertain" for an unrelated
# labelling disagreement, e.g. p2b_0003's own "frowning mouth" candidate,
# independently re-described by the Verifier as "unknown" and left
# `uncertain` -- clearly a real mouth the child drew, not a missing one).
# This module has no explicit, reliable OMISSION detector (a calibrated
# whole-person completeness assessment that positively confirms a part
# was never drawn, as opposed to merely never being independently
# confirmed) -- until one exists, these two rules are intentionally never
# satisfied by absence-of-verification alone; they stay `not_satisfied`,
# abstaining rather than guessing. `omission_evidence_terms` is the
# extension point for that future detector: once a real one exists and
# tags its output with vocabulary listed here, this same branch in
# `check_visual_preconditions` will already know how to use it -- no
# redesign needed. It is deliberately empty for both rules today, so
# neither can ever be satisfied by the current pipeline.
ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "EN_COMPILED_MISSING_HANDS_035": (
        ("person", "girl", "boy", "man", "woman", "child", "figure"), ("hand", "hands"), ()),
    "EN_COMPILED_MISSING_MOUTH_036": (
        ("person", "girl", "boy", "man", "woman", "child", "figure", "face"), ("mouth",), ()),
}

# Rules this module explicitly cannot check via presence/absence matching,
# with an honest, specific reason -- never silently omitted.
BLOCKED_NEEDS_UNBUILT_FEATURE: dict[str, str] = {
    "EN_COMPILED_EYES_MISSING_DETAIL_020": (
        "Source claim mixes an ABSENCE condition (missing eyes) with a QUALITY judgment "
        "(little detail) -- neither is a simple presence/absence-of-a-labelled-entity check; "
        "would need a per-region detail-density feature this module does not have."),
    "EN_COMPILED_LINE_ZIGZAG_033": (
        "Observable is a stroke-GEOMETRY property, not an object identity -- unlike the other "
        "line_* rules (heavy/light pressure, shaky/broken), no classical stroke-feature proxy "
        "exists for zigzag-ness, and treating an open-vocabulary Gemini label as a validated "
        "geometry classifier would not be an honest precondition check."),
    "EN_COMPILED_EXAGGERATED_BODY_PARTS_037": (
        "Requires a body-proportion norm (what counts as 'exaggerated') DOAR has no calibrated "
        "reference for -- a bare presence check cannot express a relative-size judgment."),
    "EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038": (
        "Compound construct requiring a colour-darkness feature (features.py's colour_* outputs, "
        "not a VisualEntity) combined with co-occurring face/isolation observations -- outside "
        "this module's visual-entity-presence scope; the face_expression sub-component alone is "
        "checkable via EN_COMPILED_FACE_EXPRESSION_021, but the colour sub-component is not."),
    "EN_COMPILED_EXCESSIVE_DETAIL_040": (
        "Requires a quantitative detail/complexity-density feature this module does not have -- "
        "not expressible as presence or absence of a single labelled entity."),
    "EN_COMPILED_NEGLECT_BACKGROUND_041": (
        "Requires a foreground/background segmentation-coverage comparison, not an entity-presence "
        "check."),
}


@dataclass(frozen=True)
class VisualPreconditionCheck:
    rule_id: str
    status: str  # "satisfied" | "not_satisfied" | "blocked_structural" | "blocked_needs_unbuilt_feature"
    reason: str
    matched_entity_ids: tuple[str, ...] = ()


def _verified_entities(entities: list["VisualEntity"]) -> list["VisualEntity"]:
    return [e for e in entities if e.case_verification_status == "verified"]


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", text.lower()))


def _entity_tokens(entity: "VisualEntity") -> frozenset[str]:
    tokens: set[str] = set()
    for label in (entity.canonical_label, *entity.aliases_en):
        tokens |= _tokenize(label)
    return frozenset(tokens)


def _find_matches(entities: list["VisualEntity"], terms: tuple[str, ...]) -> tuple[str, ...]:
    """WHOLE-WORD term matching, deliberately NOT `VisualEntity.
    matches_search_term`'s liberal substring-both-ways check: that check
    is tuned for open-ended search recall, where "eye" matching inside
    "eyebrow" or "car" matching inside "cartoon" is an acceptable false
    positive. A rule PRECONDITION needs the opposite bias -- a term only
    counts as present if it appears as an actual whole word (or, for a
    multi-word term like "geometric shape", all of its words) among the
    entity's own canonical label/aliases. Found live during this phase's
    own smoke test (an "eyebrow" entity was wrongly satisfying the
    wide_eyes precondition, and a verifier-supplied "cartoon eye"
    alternative label was wrongly satisfying the vehicles precondition
    via "car" inside "cartoon") -- a real implementation bug, fixed here,
    not a scientific-policy change."""
    matched = []
    for entity in entities:
        entity_tokens = _entity_tokens(entity)
        if _any_term_present(entity_tokens, terms):
            matched.append(entity.entity_id)
    return tuple(matched)


def _any_term_present(entity_tokens: frozenset[str], terms: tuple[str, ...]) -> bool:
    """True if any `terms` entry's own tokens are a whole-word subset of
    `entity_tokens` -- shared by every matcher in this module so "whole
    word, not substring" means the same thing everywhere."""
    return any(_tokenize(term) and _tokenize(term) <= entity_tokens for term in terms)


def _find_style_qualified_matches(
        entities: list["VisualEntity"], object_terms: tuple[str, ...], qualifier_terms: tuple[str, ...],
) -> tuple[str, ...]:
    """Bug 1 fix: satisfied only by an entity whose own tokens contain
    BOTH an object term (e.g. "eye"/"eyes" -- proves eyes are present)
    AND a style-qualifier term (e.g. "wide"/"closed" -- proves which
    STYLE) -- so a generic "eye"/"eyes" entity with no style qualifier at
    all never satisfies a style-specific rule, and mutually incompatible
    styles (wide vs. stern vs. closed) can never all activate from the
    same generic entity."""
    matched = []
    for entity in entities:
        entity_tokens = _entity_tokens(entity)
        if _any_term_present(entity_tokens, object_terms) and _any_term_present(entity_tokens, qualifier_terms):
            matched.append(entity.entity_id)
    return tuple(matched)


def _find_matches_excluding_context(
        entities: list["VisualEntity"], terms: tuple[str, ...], exclude_context_terms: tuple[str, ...],
) -> tuple[str, ...]:
    """Bug 3 fix: like `_find_matches`, but an entity whose tokens ALSO
    include a disqualifying context term is never counted, even if a
    `terms` entry also matches -- e.g. "circle" describing a face/head's
    own outline shape ("green face circle") is context, not a standalone
    circle symbol; only an entity with NO such context term can satisfy
    the precondition."""
    matched = []
    for entity in entities:
        entity_tokens = _entity_tokens(entity)
        if _any_term_present(entity_tokens, exclude_context_terms):
            continue
        if _any_term_present(entity_tokens, terms):
            matched.append(entity.entity_id)
    return tuple(matched)


def check_visual_preconditions(entities: list["VisualEntity"]) -> list[VisualPreconditionCheck]:
    """One check per rule in the frozen 41-rule matrix. Never invents a
    4th category outside the two structural-block reasons + satisfied/
    not_satisfied -- every rule_id in RULE_EVIDENCE_MATRIX.csv gets
    exactly one result here."""
    matrix = load_rule_matrix()
    verified = _verified_entities(entities)
    results = []
    for rule_id, row in matrix.items():
        if rule_id in STYLE_QUALIFIED_PRESENCE_TERMS:
            object_terms, qualifier_terms = STYLE_QUALIFIED_PRESENCE_TERMS[rule_id]
            matched = _find_style_qualified_matches(verified, object_terms, qualifier_terms)
            if matched:
                results.append(VisualPreconditionCheck(
                    rule_id, "satisfied",
                    f"{len(matched)} verified entity(ies) matched {object_terms!r} WITH style qualifier "
                    f"{qualifier_terms!r} on the same entity.",
                    matched))
            else:
                results.append(VisualPreconditionCheck(
                    rule_id, "not_satisfied",
                    f"No verified entity carries both an eye term {object_terms!r} and an explicit style "
                    f"qualifier {qualifier_terms!r} -- generic eye presence alone does not imply this style."))
            continue
        if rule_id in CONTEXT_EXCLUDED_PRESENCE_TERMS:
            terms, exclude_terms = CONTEXT_EXCLUDED_PRESENCE_TERMS[rule_id]
            matched = _find_matches_excluding_context(verified, terms, exclude_terms)
            if matched:
                results.append(VisualPreconditionCheck(
                    rule_id, "satisfied",
                    f"{len(matched)} verified entity(ies) matched {terms!r} without any of the excluded "
                    f"context term(s) {exclude_terms!r}.",
                    matched))
            else:
                results.append(VisualPreconditionCheck(
                    rule_id, "not_satisfied",
                    f"No standalone verified entity matched {terms!r} -- any match was describing "
                    f"another object's shape (context term(s) {exclude_terms!r} present on the same "
                    f"entity), not a standalone symbolic instance."))
            continue
        if rule_id in POSITIVE_PRESENCE_TERMS:
            matched = _find_matches(verified, POSITIVE_PRESENCE_TERMS[rule_id])
            if matched:
                results.append(VisualPreconditionCheck(
                    rule_id, "satisfied",
                    f"{len(matched)} verified entity(ies) matched {POSITIVE_PRESENCE_TERMS[rule_id]!r}",
                    matched))
            else:
                results.append(VisualPreconditionCheck(
                    rule_id, "not_satisfied", "No verified entity in this case matched the required term(s)."))
            continue
        if rule_id in ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE:
            person_terms, part_terms, omission_terms = ABSENCE_REQUIRES_EXPLICIT_OMISSION_EVIDENCE[rule_id]
            person_matches = _find_matches(verified, person_terms)
            part_matches = _find_matches(verified, part_terms)
            omission_matches = _find_matches(verified, omission_terms) if omission_terms else ()
            if omission_matches:
                # Extension point for a future, real omission detector --
                # unreachable today since every rule's omission_terms is
                # empty (see the dict's own docstring).
                results.append(VisualPreconditionCheck(
                    rule_id, "satisfied",
                    f"Explicit omission evidence {omission_terms!r} confirms this part was verified "
                    f"absent (not merely undetected).",
                    omission_matches))
            elif part_matches:
                results.append(VisualPreconditionCheck(
                    rule_id, "not_satisfied", "The part was itself verified as present -- not missing."))
            elif not person_matches:
                results.append(VisualPreconditionCheck(
                    rule_id, "not_satisfied",
                    "No verified person-like entity in this case -- absence of a part is only "
                    "meaningful once a person is confirmed present."))
            else:
                results.append(VisualPreconditionCheck(
                    rule_id, "not_satisfied",
                    "A person is present and the part was not independently verified, but "
                    "non-detection is not proof of visual absence -- this module has no explicit, "
                    "reliable omission detector (e.g. a calibrated whole-person completeness "
                    "assessment). Abstaining rather than guessing absence."))
            continue
        if row["requires_process_data"] == "yes":
            results.append(VisualPreconditionCheck(
                rule_id, "blocked_structural", "requires_process_data=yes (drawing-time behaviour, not recoverable from a finished image)."))
            continue
        if row["requires_longitudinal_data"] == "yes":
            results.append(VisualPreconditionCheck(
                rule_id, "blocked_structural", "requires_longitudinal_data=yes (needs multiple dated drawings from the same case)."))
            continue
        if row["requires_absolute_scale_or_context"] == "yes":
            results.append(VisualPreconditionCheck(
                rule_id, "blocked_structural", "requires_absolute_scale_or_context=yes (physical-scale calibration DOAR does not have)."))
            continue
        if rule_id in BLOCKED_NEEDS_UNBUILT_FEATURE:
            results.append(VisualPreconditionCheck(
                rule_id, "blocked_needs_unbuilt_feature", BLOCKED_NEEDS_UNBUILT_FEATURE[rule_id]))
            continue
        # Tier-1 composition rules (size_composition/spatial_placement families) and the
        # stroke-proxy line rules are evaluated by rules.py's own existing Tier-1 dispatch
        # against composition/colour features, not VisualEntity presence -- out of scope
        # for this entity-presence-based module, not "blocked" (they already work).
        results.append(VisualPreconditionCheck(
            rule_id, "not_applicable_to_this_module",
            "Tier-1 composition/line-proxy rule already evaluated by rules.py against "
            "composition/stroke features directly -- not a VisualEntity-presence precondition."))
    return results


# ---------------------------------------------------------------------------
# Eligible atomic rule match -> evidence family -> concern domain -> hypothesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EligibleAtomicRuleMatch:
    rule_id: str
    evidence_family: str
    concern_domain: str
    allowed_output_level: str
    source_claim: str
    possible_interpretation: str
    alternative_explanations: tuple[str, ...]
    matched_entity_ids: tuple[str, ...]
    evidence_direction: str
    context_transfer_justification: str


def build_eligible_matches(entities: list["VisualEntity"]) -> list[EligibleAtomicRuleMatch]:
    """The ONLY function in this module that turns a satisfied visual
    precondition into a rule-level object. Every field is copied verbatim
    from the frozen RULE_EVIDENCE_MATRIX.csv row -- this function invents
    nothing, promotes nothing beyond what that row already says.

    ELIGIBILITY GATE: a satisfied precondition is necessary but not
    sufficient -- the rule's own allowed_output_level must also equal
    ELIGIBLE_OUTPUT_LEVEL. A `disabled` rule (its detector is documented
    as unbuilt/unvalidated) can visually satisfy its precondition text
    exactly like an enabled rule can (the precondition check has no
    concept of allowed_output_level), but must never be promoted here --
    this is the ONLY choke point that turns a check into evidence, so
    gating here is sufficient to keep disabled rules out of every
    downstream consumer (literature associations, overall synthesis,
    candidate hypotheses). `check_visual_preconditions` itself is
    UNCHANGED and still reports `satisfied` for a disabled rule's matched
    precondition -- traceability for the Technical/debug view (which reads
    those raw checks directly, e.g. clinician_review_app.py's
    bundle["checks"]) is unaffected by this gate."""
    matrix = load_rule_matrix()
    checks = check_visual_preconditions(entities)
    matches = []
    for check in checks:
        if check.status != "satisfied":
            continue
        row = matrix[check.rule_id]
        if row["allowed_output_level"] != ELIGIBLE_OUTPUT_LEVEL:
            continue
        matches.append(EligibleAtomicRuleMatch(
            rule_id=check.rule_id,
            evidence_family=row["evidence_family"],
            concern_domain=row["concern_domain"],
            allowed_output_level=row["allowed_output_level"],
            source_claim=row["source_claim"],
            possible_interpretation=row["possible_interpretation_as_written"],
            alternative_explanations=tuple(row["alternative_explanations"].split(" | ")) if row["alternative_explanations"] else (),
            matched_entity_ids=check.matched_entity_ids,
            evidence_direction=row["evidence_direction"],
            context_transfer_justification=row["context_transfer_justification"],
        ))
    return matches


def deduplicate_by_evidence_family(matches: list[EligibleAtomicRuleMatch]) -> dict[str, list[EligibleAtomicRuleMatch]]:
    """Groups matches by evidence_family -- RULE_RELATIONSHIP_GRAPH.json's
    same_evidence_family_edges exist precisely so multiple rules in one
    family (e.g. wide_eyes + stern_eyes, both facial_feature_style) are
    counted as ONE family's worth of evidence, never N independent ones."""
    grouped: dict[str, list[EligibleAtomicRuleMatch]] = {}
    for m in matches:
        grouped.setdefault(m.evidence_family, []).append(m)
    return grouped


def aggregate_by_concern_domain(matches: list[EligibleAtomicRuleMatch]) -> dict[str, list[EligibleAtomicRuleMatch]]:
    grouped: dict[str, list[EligibleAtomicRuleMatch]] = {}
    for m in matches:
        grouped.setdefault(m.concern_domain, []).append(m)
    return grouped


@dataclass(frozen=True)
class CandidateHypothesis:
    concern_domain: str
    hypothesis_label: str
    support_level: str
    evidence_families_represented: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]
    supporting_entity_ids: tuple[str, ...]
    contradicting_reference_ids: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    missing_clinical_information: tuple[str, ...]
    disclaimer: str = DISCLAIMER


# Concern domains that never produce a clinical hypothesis regardless of
# evidence -- descriptive-only by design (CONCERN_DOMAIN_MAP.json), and the
# permanently-empty abuse/maltreatment domain (defensive: it has zero
# candidate rules today, so this never fires in practice, but the domain
# is excluded explicitly rather than relying on that alone).
_NON_HYPOTHESIS_DOMAINS = frozenset({
    "neutral_descriptive_only", "neutral_descriptive_only_no_construct_proposed",
    "maltreatment_or_safety_concern",
})


def build_candidate_hypotheses(
        matches: list[EligibleAtomicRuleMatch], *, model_evidence: list | None = None,
) -> list[CandidateHypothesis]:
    """Reuses concerns.py's OWN _aggregation_strength unchanged -- every
    rule-sourced evidence id is tagged "clinician_symbolic" (never a
    second source type just because its visual precondition was Gemini-
    verified; see module docstring point 2), so N rules from the SAME
    concern domain, however many, cap out at WEAK_HYPOTHESIS unless
    `model_evidence` (a genuinely independent source, e.g. the emotion
    model's own prediction -- same object rules.py already threads through
    to concerns.py) is supplied and contributes."""
    domain_map = load_concern_domain_map()["domains"]
    by_domain = aggregate_by_concern_domain(matches)
    hypotheses = []
    for domain, domain_matches in by_domain.items():
        if domain not in domain_map or domain in _NON_HYPOTHESIS_DOMAINS:
            continue
        evidence_ids = {f"rule_evidence:{m.rule_id}" for m in domain_matches}
        source_types = {"clinician_symbolic"}
        contradicting = []
        for m in domain_matches:
            for edge in load_relationship_graph()["context_limited_by_edges"]:
                if m.rule_id in edge["rule_ids"]:
                    contradicting.append(edge["reference_id"])
        for item in (model_evidence or []):
            evidence_ids.add(item.evidence_id)
            source_types.add("model_prediction")
        strength = _aggregation_strength(evidence_ids, source_types)
        if strength == "INSUFFICIENT":
            continue
        families = tuple(sorted({m.evidence_family for m in domain_matches}))
        alt_explanations = tuple(sorted({e for m in domain_matches for e in m.alternative_explanations}))
        entity_ids = tuple(sorted({eid for m in domain_matches for eid in m.matched_entity_ids}))
        domain_defn = domain_map[domain]
        hypotheses.append(CandidateHypothesis(
            concern_domain=domain,
            hypothesis_label=domain_defn["example_clinician_hypothesis_wording"],
            support_level=strength,
            evidence_families_represented=families,
            supporting_rule_ids=tuple(sorted({m.rule_id for m in domain_matches})),
            supporting_entity_ids=entity_ids,
            contradicting_reference_ids=tuple(sorted(set(contradicting))),
            alternative_explanations=alt_explanations,
            missing_clinical_information=(
                "Child's own explanation of the drawing (not collected)",
                "Whether this theme recurs across more than the available drawings",
                "Any reported context from parent/caregiver/teacher",
            ),
        ))
    return hypotheses


# ---------------------------------------------------------------------------
# Clinician / parent packages + LLM writer payload (schemas from
# CLINICIAN_OUTPUT_SCHEMA.md / PARENT_OUTPUT_SCHEMA.md / LLM_SYNTHESIS_POLICY.md)
# ---------------------------------------------------------------------------


def build_clinician_package(hypothesis: CandidateHypothesis) -> dict:
    matrix = load_rule_matrix()
    return {
        "concern_domain": hypothesis.concern_domain,
        "candidate_hypothesis_label": hypothesis.hypothesis_label,
        "support_level": hypothesis.support_level,
        "why": {
            "triggered_rule_ids": list(hypothesis.supporting_rule_ids),
            "evidence_families": list(hypothesis.evidence_families_represented),
            "supporting_entity_ids": list(hypothesis.supporting_entity_ids),
            "source_claims": [matrix[r]["source_claim"] for r in hypothesis.supporting_rule_ids],
            "evidence_strength_as_written": [matrix[r]["evidence_strength_as_written"] for r in hypothesis.supporting_rule_ids],
        },
        "counterevidence": list(hypothesis.contradicting_reference_ids),
        "alternative_explanations": list(hypothesis.alternative_explanations),
        "missing_clinical_information": list(hypothesis.missing_clinical_information),
        "disclaimer": hypothesis.disclaimer,
        "allowed_output_level": "LEVEL_2_clinical_hypothesis_review_flag",
        "hypothesis_worth_considering": None,
        "final_clinical_assessment": None,
    }


_PARENT_WORDING_FIELD = "example_parent_wording"


def build_parent_package(hypothesis: CandidateHypothesis) -> dict:
    domain_defn = load_concern_domain_map()["domains"][hypothesis.concern_domain]
    matrix = load_rule_matrix()
    questions = [matrix[r]["clinical_question_to_ask_child_or_caregiver"] for r in hypothesis.supporting_rule_ids
                 if matrix[r]["clinical_question_to_ask_child_or_caregiver"]]
    return {
        "what_was_observed": domain_defn[_PARENT_WORDING_FIELD],
        "simple_explanation": domain_defn["description"],
        "uncertainty": ("This is not a diagnosis, and the same feature has other common, "
                         "non-clinical explanations."),
        "gentle_questions_to_ask": questions,
        "what_to_monitor": "Whether this comes up again in future drawings, or in how your child talks about their day.",
        "when_professional_review_may_help": (
            "If this pattern keeps repeating, or if you notice it alongside other changes in mood "
            "or behaviour, mentioning it to your child's pediatrician or a counselor is a reasonable "
            "next step -- not because anything is confirmed, but because a professional can look at "
            "the fuller picture."),
    }


def build_llm_writer_payload(hypothesis: CandidateHypothesis, *, audience: str, case_id: str = "") -> dict:
    """The structured evidence package LLM_SYNTHESIS_POLICY.md Section 1
    specifies -- the ONLY input a future LLM writer step would receive.
    `audience` selects which allowed_claim_level/wording register applies;
    it never changes the underlying evidence."""
    if audience not in ("clinician", "parent"):
        raise ValueError(f"audience must be 'clinician' or 'parent', got {audience!r}")
    clinician = build_clinician_package(hypothesis)
    return {
        "case_id": case_id,
        "verified_observations": clinician["why"]["supporting_entity_ids"],
        "measurements": {},
        "eligible_rules": [{"rule_id": r, "source_claim": c}
                            for r, c in zip(hypothesis.supporting_rule_ids, clinician["why"]["source_claims"])],
        "evidence_strength": dict(zip(hypothesis.supporting_rule_ids, clinician["why"]["evidence_strength_as_written"])),
        "candidate_hypotheses": [clinician["candidate_hypothesis_label"]],
        "contradictions": clinician["counterevidence"],
        "alternative_explanations": clinician["alternative_explanations"],
        "missing_information": clinician["missing_clinical_information"],
        "references": clinician["counterevidence"],
        "allowed_claim_level": "LEVEL_2",
        "audience": audience,
    }
