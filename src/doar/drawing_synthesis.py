"""DOAR unified multimodal evidence + always-on drawing synthesis
(realignment milestone).

**The problem this module fixes**: the Gemini Observer/Verifier/
`reasoning_chain.py` semantic pipeline was, until now, the ONLY evidence
stream reaching a case's analysis -- color, line/stroke, composition, and
relationship evidence (all real, already implemented in
`analysis.py`/`features.py`, just never combined with the semantic
stream) were invisible to it. This module is a thin, ADDITIVE bridge, not
a rewrite: it normalizes every stream into one evidence representation,
lets the EXISTING, frozen 10 composition/line rules (already wired
elsewhere, in `rules.py`/`rule_engine_v2.py`) participate in
`reasoning_chain.py`'s own aggregation, and produces a structured,
always-on synthesis for every image regardless of whether a candidate
hypothesis ever clears the aggregation threshold.

**What is NOT touched**: `visual_observer.py` (Observer/Verifier prompts/
models), `reasoning_chain.py`'s semantic precondition tables (the
corrected eyes/missing-part/circle/expression logic) and its aggregation
functions (`deduplicate_by_evidence_family`, `aggregate_by_concern_
domain`, `build_candidate_hypotheses`, `build_clinician_package`,
`build_parent_package` -- all called here UNCHANGED, not reimplemented),
`RULE_EVIDENCE_MATRIX.csv`/`CONCERN_DOMAIN_MAP.json` (read-only, as
always), and the LEGACY `analyze_image`/`rules.py`/`concerns.py`
production pipeline (a separate, working system -- this module reuses
its pure sub-functions `_segment`/`_composition`/`_colour`/
`objective_feature_row` and its frozen threshold CONSTANTS from
`rule_engine_v2.py`, but never calls `evaluate_rules`/`analyze_image`
itself, so nothing here can write into a real case's `analysis.json`).

**Evidence schema**: reuses `evidence_schema.EvidenceItem` (the most
complete of the four pre-existing, previously-disconnected evidence
schemas found in the architecture audit) as the canonical per-feature
shape, wrapped in a small `UnifiedEvidenceItem` that adds the one field
`EvidenceItem` doesn't carry: `rule_eligible` (+ which rule_ids it fed).

**Architecture distinction, enforced structurally, not just in the
UI**: ALL measured evidence (`objective_profile`) is descriptive. ONLY
evidence that satisfied an existing, frozen rule's precondition
(`literature_linked_associations`) ever contributes to
`overall_synthesis` or `candidate_hypotheses`. VISUALLY VERIFIED !=
PSYCHOLOGICALLY VALIDATED still holds: every rule-sourced item, semantic
or deterministic, is still tagged `clinician_symbolic` by
`reasoning_chain.py`'s own unchanged `build_candidate_hypotheses`, so
multiple rules from one concern domain still cap at `WEAK_HYPOTHESIS`
regardless of which stream produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import reasoning_chain as rc
from .analysis import _colour, _composition, _save_artifacts, _segment
from .concerns import MIN_EVIDENCE
from .evidence_schema import CONCLUSIVE_STATUSES, EvidenceItem, EvidenceLocation
from .features import FeatureValue, objective_feature_row, serialize_feature_row
from .page_frame import assess_page_frame
from .page_reference import resolve_page_reference
from .relationship_features import compute_relationship_features
from .rule_engine_v2 import (
    ALL_PAGE_GATED_RULE_IDS,
    COVERAGE_FULL_MARGIN_THRESHOLD,
    FRAGMENTATION_SHAKY_THRESHOLD,
    INTENSITY_PROXY_HEAVY_THRESHOLD,
    INTENSITY_PROXY_LIGHT_THRESHOLD,
    PLACEMENT_CENTER_BAND,
)

if TYPE_CHECKING:
    from .visual_entity import VisualEntity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETERMINISTIC_CACHE_DIR = ROOT / "outputs" / "prototype_cases" / "development_deterministic_features_cache"

# ---------------------------------------------------------------------------
# Deterministic feature computation -- reuses analysis.py's own pure
# sub-functions verbatim (no formula duplicated) and features.py's own
# aggregator. Pure numpy/PIL: no network call, no model weights, no
# dependency on the Gemini pipeline at all.
# ---------------------------------------------------------------------------


def compute_deterministic_features(image_path: str | Path, artifacts_dir: str | Path) -> dict:
    """Runs ONLY the deterministic (non-Gemini, non-emotion, non-legacy-
    rule) portion of `analysis.py::analyze_image`: segmentation ->
    page-frame assessment -> page-reference resolution -> composition ->
    colour -> `features.objective_feature_row`. Writes the same small set
    of artifact PNGs `analyze_image` always writes (`_save_artifacts`,
    reused unchanged) -- needed because `objective_feature_row` reads the
    foreground mask back from disk, exactly as it already does in the
    legacy pipeline; nothing new is invented here.

    `page_reference` (`page_frame.assess_page_frame` + `page_reference.
    resolve_page_reference`, both reused unchanged, no user declaration
    since this is the same batch/no-interactive-user context
    `analyze_image` itself runs in) is threaded through into
    `analysis_context` exactly like `analyze_image` does, so `features.
    objective_feature_row`'s own existing `segmentation.page_relative_
    bounding_box_coverage` feature is real (not permanently NaN) and the
    deterministic rule bridge below can gate page-relative rules on real
    assessability -- never assume the image frame IS the page."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    mask, background, seg_conf, candidates, seg_diagnostics = _segment(rgb)
    page_frame = assess_page_frame(rgb, mask, seg_diagnostics.get("background_stability", 0.0)).to_dict()
    page_reference = resolve_page_reference(
        page_frame, image_width=image.width, image_height=image.height).to_dict()
    composition = _composition(mask)
    colour = _colour(rgb, mask, background)
    artifacts = _save_artifacts(image, mask, composition, candidates, Path(artifacts_dir))
    analysis_context = {
        "artifacts": artifacts,
        "segmentation": {
            "confidence": seg_conf, "background_rgb": background.round().astype(int).tolist(),
            **seg_diagnostics,
        },
        "composition": composition,
        "colour": colour,
        "page_reference": page_reference,
    }
    objective_features = objective_feature_row(image_path, analysis_context)
    return {
        "composition": composition, "colour": colour, "segmentation": analysis_context["segmentation"],
        "objective_features": objective_features, "page_reference": page_reference,
    }


def deterministic_cache_path(image_id: str, cache_dir: str | Path = DEFAULT_DETERMINISTIC_CACHE_DIR) -> Path:
    return Path(cache_dir) / image_id / "deterministic_features.json"


def _serialize_deterministic(result: dict) -> dict:
    return {
        "composition": result["composition"], "colour": result["colour"], "segmentation": result["segmentation"],
        "objective_features": serialize_feature_row(result["objective_features"]),
        "page_reference": result.get("page_reference"),
    }


def _deserialize_deterministic(data: dict) -> dict:
    objective_features = {
        name: FeatureValue(**fv) for name, fv in data["objective_features"].items()
    }
    return {
        "composition": data["composition"], "colour": data["colour"], "segmentation": data["segmentation"],
        "objective_features": objective_features,
        # .get(): tolerates cache files written before page_reference existed --
        # treated as "not assessable" (the safe default) by the rule bridge below,
        # never silently upgraded to assessable.
        "page_reference": data.get("page_reference"),
    }


def load_or_compute_deterministic_features(
        image_id: str, image_path: str | Path, cache_dir: str | Path = DEFAULT_DETERMINISTIC_CACHE_DIR,
) -> dict:
    """Cache-first, exactly like `run_development_benchmark.find_saved_
    verification_rows`'s own pattern for the Gemini cache -- deterministic
    features are cheap to compute (pure local CV, sub-second) but caching
    keeps the clinician app instant and every run byte-reproducible."""
    import json

    cache_file = deterministic_cache_path(image_id, cache_dir)
    if cache_file.exists():
        return _deserialize_deterministic(json.loads(cache_file.read_text(encoding="utf-8")))
    result = compute_deterministic_features(image_path, cache_file.parent / "artifacts")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(_serialize_deterministic(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Deterministic rule bridge -- ONLY the 10 rules already real and wired
# elsewhere (rules.py::_TIER1_DISPATCH + rule_engine_v2.py::V2_RULE_IDS).
# Thresholds are the exact frozen constants those modules already use --
# nothing re-derived, nothing new.
#
# Page-frame safety: `ALL_PAGE_GATED_RULE_IDS` (imported from rule_engine_
# v2.py -- the SAME set `apply_page_frame_gating`/`evaluate_v2_rules`
# already gate) covers the 7 rules here whose observable is only
# meaningful relative to the full page (the 3 size/coverage rules, the 3
# PSY_AR placement rules, EN_COMPILED_PLACEMENT_CENTER_029). When
# `page_reference.page_relative_features_assessable` is False, those 7
# become `not_assessable` -- never silently computed from the raw image
# frame as if it were the page. The 3 line-pressure/fragmentation rules
# are not page-relative (a stroke-darkness proxy does not depend on how
# much of the page is visible) and are never gated. `PSY_AR_SIZE_FULL_015`
# additionally reuses `rule_engine_v2.redefine_coverage_full`'s own
# margin-based "approaches all four page margins" definition
# (`COVERAGE_FULL_MARGIN_THRESHOLD`) instead of a raw image-relative
# `bounding_box_coverage >= 0.90` -- the same formula/threshold that
# function already uses, not a new one.
# ---------------------------------------------------------------------------

DETERMINISTIC_RULE_IDS = frozenset({
    "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_FULL_015", "PSY_AR_SIZE_SMALL_016",
    "PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019",
    "EN_COMPILED_PLACEMENT_CENTER_029",
    "EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031",
    "EN_COMPILED_LINE_SHAKY_BROKEN_032",
})

# The 7 of the 10 rules above that are page-relative -- must be EXACTLY
# rule_engine_v2.ALL_PAGE_GATED_RULE_IDS (the same set that module's own
# apply_page_frame_gating/evaluate_v2_rules already gate), not a
# separately-maintained list -- asserted, not just claimed in prose.
_PAGE_GATED_DETERMINISTIC_RULE_IDS = DETERMINISTIC_RULE_IDS - {
    "EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "EN_COMPILED_LINE_SHAKY_BROKEN_032",
}
assert _PAGE_GATED_DETERMINISTIC_RULE_IDS == ALL_PAGE_GATED_RULE_IDS

# rules.py::_tier1_status's own literal thresholds, reused verbatim for
# HALF/SMALL (coverage_full is redefined below via the page-margin
# formula instead).
_COVERAGE_HALF_BAND = (0.40, 0.60)
_COVERAGE_SMALL_MAX = 0.20

_PAGE_NOT_ASSESSABLE_REASON = (
    "page_reference.page_relative_features_assessable is False (no confirmed/detected page polygon) -- "
    "this page-relative measurement is withheld rather than computed against the raw image frame.")


def _page_assessable(page_reference: dict | None) -> bool:
    return bool(page_reference and page_reference.get("page_relative_features_assessable"))


def check_deterministic_preconditions(
        composition: dict, objective_features: dict[str, FeatureValue], page_reference: dict | None = None,
) -> list[rc.VisualPreconditionCheck]:
    """Same shape/spirit as `reasoning_chain.check_visual_preconditions`,
    for the 10 rules whose observable is a composition/line measurement
    instead of a semantic entity. Returns exactly one check per rule in
    `DETERMINISTIC_RULE_IDS`. `page_reference` defaults to None (treated
    as NOT assessable, the safe default -- see module docstring) for
    callers that haven't resolved one; real callers should always pass
    `compute_deterministic_features`'s own `page_reference` result."""
    checks: list[rc.VisualPreconditionCheck] = []
    page_assessable = _page_assessable(page_reference)
    bbox_fv = objective_features.get("segmentation.bounding_box_coverage")

    if not page_assessable:
        for rule_id in ("PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_SMALL_016"):
            checks.append(rc.VisualPreconditionCheck(rule_id, "not_assessable", _PAGE_NOT_ASSESSABLE_REASON))
    elif bbox_fv is None or bbox_fv.missing:
        for rule_id in ("PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_SMALL_016"):
            checks.append(rc.VisualPreconditionCheck(rule_id, "not_satisfied", "No page-coverage measurement available for this image."))
    else:
        value = bbox_fv.value
        half_ok = _COVERAGE_HALF_BAND[0] <= value <= _COVERAGE_HALF_BAND[1]
        checks.append(rc.VisualPreconditionCheck(
            "PSY_AR_SIZE_HALF_014", "satisfied" if half_ok else "not_satisfied",
            f"bounding_box_coverage={value:.3f} vs band {_COVERAGE_HALF_BAND}",
            (bbox_fv.evidence_id,) if half_ok else ()))
        small_ok = 0 < value <= _COVERAGE_SMALL_MAX
        checks.append(rc.VisualPreconditionCheck(
            "PSY_AR_SIZE_SMALL_016", "satisfied" if small_ok else "not_satisfied",
            f"bounding_box_coverage={value:.3f} vs maximum {_COVERAGE_SMALL_MAX}",
            (bbox_fv.evidence_id,) if small_ok else ()))

    # PSY_AR_SIZE_FULL_015 -- rule_engine_v2.redefine_coverage_full's own
    # margin-based definition, reused verbatim (not the raw >=0.90 image-
    # relative area check): content must approach all four PAGE margins.
    margins = composition.get("margins_normalized")
    if not page_assessable:
        checks.append(rc.VisualPreconditionCheck("PSY_AR_SIZE_FULL_015", "not_assessable", _PAGE_NOT_ASSESSABLE_REASON))
    elif margins is None or bbox_fv is None:
        checks.append(rc.VisualPreconditionCheck("PSY_AR_SIZE_FULL_015", "not_satisfied", "No page margins available for this image."))
    else:
        full_ok = all(m <= COVERAGE_FULL_MARGIN_THRESHOLD for m in margins)
        checks.append(rc.VisualPreconditionCheck(
            "PSY_AR_SIZE_FULL_015", "satisfied" if full_ok else "not_satisfied",
            f"margins_normalized={margins} vs page-margin threshold {COVERAGE_FULL_MARGIN_THRESHOLD} on every side",
            (bbox_fv.evidence_id,) if full_ok else ()))

    placement = composition.get("placement", "unavailable")
    cx_fv = objective_features.get("composition.centroid_x")
    cy_fv = objective_features.get("composition.centroid_y")
    placement_evidence_ids = tuple(fv.evidence_id for fv in (cx_fv, cy_fv) if fv is not None)

    if not page_assessable:
        for rule_id in ("PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019"):
            checks.append(rc.VisualPreconditionCheck(rule_id, "not_assessable", _PAGE_NOT_ASSESSABLE_REASON))
    elif placement == "unavailable" or not placement_evidence_ids:
        for rule_id in ("PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019"):
            checks.append(rc.VisualPreconditionCheck(rule_id, "not_satisfied", "No placement measurement available for this image."))
    else:
        for rule_id, wanted in (
            ("PSY_AR_PLACE_TOP_017", "top"), ("PSY_AR_PLACE_LEFT_018", "left"), ("PSY_AR_PLACE_RIGHT_019", "right"),
        ):
            matched = wanted in placement
            checks.append(rc.VisualPreconditionCheck(
                rule_id, "satisfied" if matched else "not_satisfied",
                f"placement={placement!r} vs required {wanted!r}",
                placement_evidence_ids if matched else ()))

    if not page_assessable:
        checks.append(rc.VisualPreconditionCheck("EN_COMPILED_PLACEMENT_CENTER_029", "not_assessable", _PAGE_NOT_ASSESSABLE_REASON))
    elif cx_fv is None or cy_fv is None or cx_fv.missing or cy_fv.missing:
        checks.append(rc.VisualPreconditionCheck("EN_COMPILED_PLACEMENT_CENTER_029", "not_satisfied", "Centroid unavailable."))
    else:
        lo, hi = PLACEMENT_CENTER_BAND
        matched = lo <= cx_fv.value <= hi and lo <= cy_fv.value <= hi
        checks.append(rc.VisualPreconditionCheck(
            "EN_COMPILED_PLACEMENT_CENTER_029", "satisfied" if matched else "not_satisfied",
            f"centroid=({cx_fv.value:.3f},{cy_fv.value:.3f}) vs band [{lo},{hi}]",
            placement_evidence_ids if matched else ()))

    intensity_fv = objective_features.get("stroke.intensity_proxy")
    if intensity_fv is None or intensity_fv.missing:
        for rule_id in ("EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031"):
            checks.append(rc.VisualPreconditionCheck(rule_id, "not_satisfied", "stroke.intensity_proxy unavailable."))
    else:
        heavy = intensity_fv.value >= INTENSITY_PROXY_HEAVY_THRESHOLD
        checks.append(rc.VisualPreconditionCheck(
            "EN_COMPILED_LINE_HEAVY_PRESSURE_030", "satisfied" if heavy else "not_satisfied",
            f"stroke.intensity_proxy={intensity_fv.value:.4f} vs top-quartile threshold {INTENSITY_PROXY_HEAVY_THRESHOLD}",
            (intensity_fv.evidence_id,) if heavy else ()))
        light = intensity_fv.value <= INTENSITY_PROXY_LIGHT_THRESHOLD
        checks.append(rc.VisualPreconditionCheck(
            "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "satisfied" if light else "not_satisfied",
            f"stroke.intensity_proxy={intensity_fv.value:.4f} vs bottom-quartile threshold {INTENSITY_PROXY_LIGHT_THRESHOLD}",
            (intensity_fv.evidence_id,) if light else ()))

    frag_fv = objective_features.get("stroke.fragmentation")
    if frag_fv is None or frag_fv.missing:
        checks.append(rc.VisualPreconditionCheck("EN_COMPILED_LINE_SHAKY_BROKEN_032", "not_satisfied", "stroke.fragmentation unavailable."))
    else:
        shaky = frag_fv.value >= FRAGMENTATION_SHAKY_THRESHOLD
        checks.append(rc.VisualPreconditionCheck(
            "EN_COMPILED_LINE_SHAKY_BROKEN_032", "satisfied" if shaky else "not_satisfied",
            f"stroke.fragmentation={frag_fv.value:.4f} vs top-quartile threshold {FRAGMENTATION_SHAKY_THRESHOLD}",
            (frag_fv.evidence_id,) if shaky else ()))

    assert {c.rule_id for c in checks} == DETERMINISTIC_RULE_IDS
    return checks


def build_deterministic_eligible_matches(
        composition: dict, objective_features: dict[str, FeatureValue], page_reference: dict | None = None,
) -> list[rc.EligibleAtomicRuleMatch]:
    """Same construction discipline as `reasoning_chain.build_eligible_
    matches`: every field copied verbatim from the frozen matrix row --
    invents nothing, promotes nothing. Only `satisfied` checks become
    matches -- `not_assessable` (page not confirmed) is excluded exactly
    like `not_satisfied`, never treated as a match.

    ELIGIBILITY GATE: same `rc.ELIGIBLE_OUTPUT_LEVEL` check
    `build_eligible_matches` applies, reused verbatim here for the same
    reason -- `DETERMINISTIC_RULE_IDS` happens to equal today's enabled-
    rule set, so this gate is currently a no-op in practice, but nothing
    in this module's own logic enforces that coincidence; the explicit
    check makes it a real invariant rather than an accident of the
    registry's current state, defended against a future registry/
    DETERMINISTIC_RULE_IDS drift."""
    matrix = rc.load_rule_matrix()
    checks = check_deterministic_preconditions(composition, objective_features, page_reference)
    matches = []
    for check in checks:
        if check.status != "satisfied":
            continue
        row = matrix[check.rule_id]
        if row["allowed_output_level"] != rc.ELIGIBLE_OUTPUT_LEVEL:
            continue
        matches.append(rc.EligibleAtomicRuleMatch(
            rule_id=check.rule_id, evidence_family=row["evidence_family"], concern_domain=row["concern_domain"],
            allowed_output_level=row["allowed_output_level"], source_claim=row["source_claim"],
            possible_interpretation=row["possible_interpretation_as_written"],
            alternative_explanations=tuple(row["alternative_explanations"].split(" | ")) if row["alternative_explanations"] else (),
            matched_entity_ids=check.matched_entity_ids, evidence_direction=row["evidence_direction"],
            context_transfer_justification=row["context_transfer_justification"],
        ))
    return matches


# ---------------------------------------------------------------------------
# Unified evidence normalization -- reuses evidence_schema.EvidenceItem.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedEvidenceItem:
    """Wraps a reused `EvidenceItem` with the one field it doesn't carry:
    whether this piece of evidence actually fed an eligible (satisfied)
    rule, and which rule_id(s). Not a competing schema -- `item` IS the
    canonical `evidence_schema.EvidenceItem`."""
    item: EvidenceItem
    rule_eligible: bool
    matched_rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"item": self.item.to_dict(), "rule_eligible": self.rule_eligible, "matched_rule_ids": list(self.matched_rule_ids)}


# Gemini Observer/Verifier's own 4-way status vocabulary mapped onto
# evidence_schema's 6-way EVIDENCE_STATUSES. "verified" is the only
# status allowed a real `value` (EvidenceItem's own invariant) --
# uncertain/unreviewed become `insufficient_evidence` (an extractor ran
# but did not produce a reliable, independently-confirmed result);
# rejected becomes `requires_manual_review` (an active mismatch between
# Observer and Verifier, not simply "unknown"). The raw candidate label
# is never discarded even when value must be None -- it is preserved in
# `reason`, so nothing is hidden from the Technical/Clinician views.
_SEMANTIC_STATUS_MAP = {
    "verified": "available", "uncertain": "insufficient_evidence",
    "unreviewed": "insufficient_evidence", "rejected": "requires_manual_review",
}

_DETERMINISTIC_CATEGORY_BY_PREFIX = (
    ("colour.", "colour"),
    ("stroke.", "strokes_shading_repetition_overwriting"),
    ("shape.", "strokes_shading_repetition_overwriting"),
    ("composition.", "global_composition"),
    ("segmentation.", "foreground_page_segmentation"),
    ("quality.", "image_page_quality"),
)


def _rule_ids_by_matched_id(matches: list[rc.EligibleAtomicRuleMatch]) -> dict[str, tuple[str, ...]]:
    by_id: dict[str, list[str]] = {}
    for m in matches:
        for matched_id in m.matched_entity_ids:
            by_id.setdefault(matched_id, []).append(m.rule_id)
    return {k: tuple(v) for k, v in by_id.items()}


def normalize_semantic_evidence(
        entities: list["VisualEntity"], rule_ids_by_id: dict[str, tuple[str, ...]],
) -> list[UnifiedEvidenceItem]:
    items = []
    for e in entities:
        status = _SEMANTIC_STATUS_MAP.get(e.case_verification_status, "requires_manual_review")
        is_conclusive = status in CONCLUSIVE_STATUSES
        location = EvidenceLocation(region="bounding_box", bbox_xywh=tuple(e.bbox)) if e.bbox else None
        matched_rule_ids = rule_ids_by_id.get(e.entity_id, ())
        evidence_item = EvidenceItem(
            evidence_id=f"ev_semantic_{e.entity_id}", feature_id="semantic.object_presence",
            value=e.canonical_label if is_conclusive else None, unit=None, status=status,
            confidence=e.confidence if is_conclusive else None, extractor=e.detector or "visual_observer",
            extractor_version="gemini_observer_verifier_v1", validation_status=e.case_verification_status,
            location=location, visualization_reference=None,
            reason=(f"Observer candidate {e.canonical_label!r} (confidence={e.confidence}); "
                    f"Verifier status={e.case_verification_status!r}."),
            limitations=["Semantic label from an open-vocabulary vision-language model; independently "
                         "re-checked by a separate Verifier model, never assumed correct from the Observer alone."],
            category="semantic_objects",
        )
        items.append(UnifiedEvidenceItem(evidence_item, rule_eligible=bool(matched_rule_ids), matched_rule_ids=matched_rule_ids))
    return items


def normalize_deterministic_evidence(
        objective_features: dict[str, FeatureValue], rule_ids_by_id: dict[str, tuple[str, ...]],
) -> list[UnifiedEvidenceItem]:
    items = []
    for name, fv in objective_features.items():
        category = next((cat for prefix, cat in _DETERMINISTIC_CATEGORY_BY_PREFIX if name.startswith(prefix)), None)
        status = "unavailable" if fv.missing else "available"
        matched_rule_ids = rule_ids_by_id.get(fv.evidence_id, ())
        evidence_item = EvidenceItem(
            evidence_id=fv.evidence_id, feature_id=name, value=None if fv.missing else fv.value,
            unit=None, status=status, confidence=None if fv.missing else fv.confidence,
            extractor=fv.method, extractor_version=fv.version,
            validation_status="not_applicable" if fv.missing else "measured",
            location=None, visualization_reference=None,
            reason="No extractor produces this feature yet." if fv.missing else f"Measured via {fv.method}.",
            limitations=[], category=category,
        )
        items.append(UnifiedEvidenceItem(evidence_item, rule_eligible=bool(matched_rule_ids), matched_rule_ids=matched_rule_ids))
    return items


def normalize_relationship_evidence(relationship_features: dict) -> list[UnifiedEvidenceItem]:
    """Descriptive-only by design (module docstring, relationship_
    features.py) -- rule_eligible is always False; no rule in the frozen
    matrix currently has a relationship-shaped observable."""
    items = []
    evidence_item = EvidenceItem(
        evidence_id="ev_relationship_entity_count", feature_id="relationships.entity_count",
        value=relationship_features["entity_count"], unit="count", status="available", confidence=None,
        extractor="relationship_features.compute_relationship_features", extractor_version="1.0.0",
        validation_status="measured", location=None, visualization_reference=None,
        reason="Count of verified, bbox-bearing entities.", limitations=[], category="spatial_relationships",
    )
    items.append(UnifiedEvidenceItem(evidence_item, rule_eligible=False))
    if relationship_features["repeated_labels"]:
        evidence_item = EvidenceItem(
            evidence_id="ev_relationship_repeated_labels", feature_id="relationships.repeated_labels",
            value=relationship_features["repeated_labels"], unit=None, status="available", confidence=None,
            extractor="relationship_features.compute_relationship_features", extractor_version="1.0.0",
            validation_status="measured", location=None, visualization_reference=None,
            reason="Canonical labels appearing more than once among verified entities.", limitations=[],
            category="spatial_relationships",
        )
        items.append(UnifiedEvidenceItem(evidence_item, rule_eligible=False))
    return items


def normalize_emotion_evidence(model_evidence: list | None) -> list[UnifiedEvidenceItem]:
    """Explicit placeholder for global-emotion evidence (realignment task,
    Section 8): the classifier pipeline exists in this repo but no trained
    checkpoint is present for this milestone, so nothing is invoked here.
    Rather than silently omitting the category, this emits one explicit
    `unavailable` `EvidenceItem` so its absence is auditable in every
    case's evidence set, exactly like `evidence_schema.unavailable_item`'s
    own rationale. `model_evidence` is the pre-existing plug-in point
    (`reasoning_chain.build_candidate_hypotheses` already accepts it) --
    when a real classifier is wired in later, the caller passes real
    per-image `model_evidence` and this placeholder is simply skipped."""
    if model_evidence:
        return []
    evidence_item = EvidenceItem(
        evidence_id="ev_global_emotion_model", feature_id="semantic.global_emotion_classification",
        value=None, unit=None, status="unavailable", confidence=None,
        extractor="none", extractor_version="n/a", validation_status="not_applicable",
        location=None, visualization_reference=None,
        reason=("No trained checkpoint is available in this repository for the global drawing-emotion "
                "classifier as of this milestone; the extractor/model pipeline exists but was not trained "
                "or validated here. This is an explicit placeholder, not a silent omission."),
        limitations=["No trained checkpoint present in this repository as of this milestone."],
        category=None,
    )
    return [UnifiedEvidenceItem(evidence_item, rule_eligible=False)]


# ---------------------------------------------------------------------------
# Always-on drawing-level synthesis.
# ---------------------------------------------------------------------------

# CONCERN_DOMAIN_MAP.json's own domain names bucketed for the synthesis
# summary -- no new construct invented, just grouping existing frozen
# domain names for a readable sentence. maltreatment_or_safety_concern is
# never populated by any rule today (reasoning_chain.py's own
# _NON_HYPOTHESIS_DOMAINS) so it is intentionally absent from every bucket.
_POSITIVE_DOMAINS = frozenset({"positive_affect_or_social_engagement"})
_DISTRESS_DOMAINS = frozenset({
    "depressive_or_low_mood_related", "anxiety_or_stress_related", "aggression_or_threat_related",
    "social_withdrawal_related", "developmental_or_attention_related",
})
_NEUTRAL_DOMAINS = frozenset({"neutral_descriptive_only", "neutral_descriptive_only_no_construct_proposed"})


def _domain_scoped_convergence(direction_matches: list[rc.EligibleAtomicRuleMatch]) -> tuple[dict[str, int], dict[str, dict], str | None]:
    """Domain-scoped family convergence for ONE direction (positive or
    concern), reusing `reasoning_chain.aggregate_by_concern_domain`
    verbatim -- the SAME strict per-domain partition `build_candidate_
    hypotheses` already applies before ever evaluating convergence
    (acceptance-audit fix: evidence from different concern domains must
    never be pooled together just because both happen to point in the
    same direction). Returns (family_count_by_domain, families_by_domain,
    convergent_domain) -- `convergent_domain` is the lexicographically
    first domain reaching `MIN_EVIDENCE` independent families within
    itself, or None if no single domain does."""
    by_domain = rc.aggregate_by_concern_domain(direction_matches)
    families_by_domain = {domain: rc.deduplicate_by_evidence_family(dom_matches) for domain, dom_matches in by_domain.items()}
    family_count_by_domain = {domain: len(fams) for domain, fams in families_by_domain.items()}
    convergent_domain = next(
        (domain for domain in sorted(family_count_by_domain) if family_count_by_domain[domain] >= MIN_EVIDENCE), None)
    return family_count_by_domain, families_by_domain, convergent_domain


def _scattered_direction_phrase(direction_label: str, families: dict, domain_family_counts: dict[str, int], rule_ids: list[str]) -> str | None:
    """Describes ONE direction's evidence for the `limited_association`
    summary -- either a genuine single piece of evidence, or (acceptance-
    audit fix) an explicit statement that multiple associations exist but
    are spread thinly across unrelated domains, never phrased as if they
    formed one pattern."""
    if not families:
        return None
    if len(families) == 1:
        family = next(iter(families))
        return (f"a single literature-linked association toward a {direction_label} theme, from one evidence "
                f"family ({family}, rule(s): {', '.join(sorted(rule_ids))})")
    domains_desc = ", ".join(f"{domain} ({count} family)" if count == 1 else f"{domain} ({count} families)"
                              for domain, count in sorted(domain_family_counts.items()))
    return (f"{len(families)} literature-linked evidence families toward {direction_label} themes, but spread "
            f"across {len(domain_family_counts)} different, unrelated concern domains ({domains_desc}) with no "
            f"single domain reaching independent convergence -- these are multiple individual associations "
            f"across different domains, not one coherent convergent pattern")


def build_overall_synthesis(matches: list[rc.EligibleAtomicRuleMatch]) -> dict:
    """Level C of the always-on synthesis: the drawing-LEVEL convergent
    pattern, distinct from (and never a replacement for) level B, the
    per-rule literature-linked associations (`build_literature_linked_
    associations`) -- every individual association stays visible there
    regardless of what this function concludes.

    A single individual_heuristic_only rule must NEVER by itself produce
    a "this drawing is positive/concern-associated" reading -- that was
    the original synthesis's over-claiming bug. This function requires
    INDEPENDENT EVIDENCE-FAMILY CONVERGENCE WITHIN ONE CONCERN DOMAIN
    (>=2 distinct evidence families inside the SAME domain -- acceptance-
    audit fix: evidence from different, unrelated concern domains, e.g.
    aggression_or_threat_related and anxiety_or_stress_related, must
    never be pooled together to manufacture a "convergent" reading just
    because both happen to be concern-direction; `_domain_scoped_
    convergence` reuses `reasoning_chain.aggregate_by_concern_domain` --
    the SAME strict per-domain partition `build_candidate_hypotheses`
    itself already applies -- before ever counting families, via
    `reasoning_chain.deduplicate_by_evidence_family`, the SAME function
    `build_candidate_hypotheses` relies on to avoid counting two rules
    that are really "the same evidence" twice) before calling a direction
    "convergent". The >=2 threshold is `concerns.py::MIN_EVIDENCE`,
    imported and reused verbatim, not a new number invented from the
    15-case validation set -- the same "at least two independent items"
    bar concerns.py's own `_aggregation_strength` already applies to
    clinical-hypothesis convergence, applied here one level up (family-
    count instead of rule-count) and WITHIN one domain, mirroring exactly
    how `build_candidate_hypotheses` itself never mixes domains.

    Evidence families that exist but never converge within any single
    domain remain real evidence -- reported as `limited_association`,
    with an explicit note when they span multiple unrelated domains
    (never silently dropped, never overclaimed as a whole-drawing
    pattern). `mixed_evidence` is reserved for the case where BOTH
    directions independently achieve real domain-level convergence -- not
    merely because individual associations of both directions exist."""
    positive_matches = [m for m in matches if m.concern_domain in _POSITIVE_DOMAINS]
    concern_matches = [m for m in matches if m.concern_domain in _DISTRESS_DOMAINS]
    neutral_matches = [m for m in matches if m.concern_domain in _NEUTRAL_DOMAINS]

    # Pooled (cross-domain) family/rule listings -- transparency fields only
    # ("what evidence exists at all"), never themselves the basis for a
    # convergence decision; the decision uses the domain-scoped values below.
    positive_families = rc.deduplicate_by_evidence_family(positive_matches)
    concern_families = rc.deduplicate_by_evidence_family(concern_matches)

    positive_domain_family_counts, positive_families_by_domain, convergent_positive_domain = _domain_scoped_convergence(positive_matches)
    concern_domain_family_counts, concern_families_by_domain, convergent_concern_domain = _domain_scoped_convergence(concern_matches)

    domains_touched = sorted({m.concern_domain for m in matches})
    supporting = {
        "positive_evidence_families": sorted(positive_families),
        "positive_rule_ids": sorted({m.rule_id for m in positive_matches}),
        "concern_evidence_families": sorted(concern_families),
        "concern_rule_ids": sorted({m.rule_id for m in concern_matches}),
        "positive_domain_family_counts": positive_domain_family_counts,
        "concern_domain_family_counts": concern_domain_family_counts,
        "convergent_positive_domain": convergent_positive_domain,
        "convergent_concern_domain": convergent_concern_domain,
    }

    if not matches:
        level = "insufficient_interpretable_evidence"
        summary = ("No literature-linked evidence was eligible for this drawing -- either too little was "
                   "confidently identified, or nothing identified matches an existing rule's required pattern. "
                   "This reflects the evidence available, not a conclusion about the child.")
    elif convergent_positive_domain and convergent_concern_domain:
        level = "mixed_evidence"
        summary = (
            f"This drawing shows independently convergent literature-linked evidence in two different directions: "
            f"within the '{convergent_positive_domain}' concern domain "
            f"({positive_domain_family_counts[convergent_positive_domain]} independent evidence families, "
            f"{', '.join(sorted(positive_families_by_domain[convergent_positive_domain]))}) associated with "
            f"positive/engaged presentation, and within the '{convergent_concern_domain}' concern domain "
            f"({concern_domain_family_counts[convergent_concern_domain]} independent evidence families, "
            f"{', '.join(sorted(concern_families_by_domain[convergent_concern_domain]))}) associated with "
            f"concern-related themes. No single reading is favored by this evidence alone.")
    elif convergent_positive_domain:
        level = "convergent_positive_pattern"
        families_in_domain = sorted(positive_families_by_domain[convergent_positive_domain])
        summary = (
            f"This drawing shows literature-linked indicators from {len(families_in_domain)} independent evidence "
            f"families ({', '.join(families_in_domain)}) within the '{convergent_positive_domain}' concern domain "
            f"that converge on a positive/engaged presentation. This is descriptive pattern-matching, not a "
            f"clinical finding.")
    elif convergent_concern_domain:
        level = "convergent_concern_pattern"
        families_in_domain = sorted(concern_families_by_domain[convergent_concern_domain])
        summary = (
            f"This drawing shows literature-linked indicators from {len(families_in_domain)} independent evidence "
            f"families ({', '.join(families_in_domain)}) within the '{convergent_concern_domain}' concern domain "
            f"that converge on themes sometimes associated with low mood, anxiety, withdrawal, or tension in the "
            f"drawing literature. This is descriptive pattern-matching, not a clinical finding.")
    elif positive_families or concern_families:
        level = "limited_association"
        phrases = [p for p in (
            _scattered_direction_phrase("positive/engaged", positive_families, positive_domain_family_counts, supporting["positive_rule_ids"]),
            _scattered_direction_phrase("concern-related", concern_families, concern_domain_family_counts, supporting["concern_rule_ids"]),
        ) if p]
        summary = (
            "This drawing shows " + "; and it separately shows ".join(phrases) + ". A single individual heuristic, "
            "or multiple individual associations that never converge within one domain, is not sufficient to "
            "characterize the whole drawing. See each rule's own alternative explanations.")
    elif neutral_matches:
        level = "descriptive_only"
        summary = ("This drawing's eligible evidence is descriptive only (e.g. composition/size facts) with no "
                   "rule-linked positive- or concern-associated indicator identified.")
    else:
        level = "insufficient_interpretable_evidence"
        summary = "The eligible evidence for this drawing does not clearly fall into an interpretable positive, concern, or descriptive pattern."

    return {
        "level": level, "summary": summary, "domains_touched": domains_touched,
        **supporting,
        "uncertainty_note": ("Every item behind this summary is visually-confirmed OR deterministically-measured "
                              "evidence that happened to satisfy an existing literature rule's precondition -- it is "
                              "NOT independent confirmation of any psychological interpretation attached to that rule. "
                              "Convergent levels require evidence from >=2 independent evidence families WITHIN ONE "
                              f"concern domain (MIN_EVIDENCE={MIN_EVIDENCE}, reused from concerns.py, same per-domain "
                              "partition build_candidate_hypotheses uses); evidence spread across different, "
                              "unrelated domains is reported as limited_association, never pooled to manufacture "
                              "convergence. See each rule's own alternative explanations and counterevidence."),
    }


@dataclass(frozen=True)
class DrawingSynthesisResult:
    image_id: str
    unified_evidence: list[UnifiedEvidenceItem]
    objective_profile: dict[str, list[dict]]
    literature_linked_associations: list[dict]
    overall_synthesis: dict
    candidate_hypotheses: list[rc.CandidateHypothesis] = field(default_factory=list)


def _matrix_row_for(rule_id: str, matrix: dict[str, dict]) -> dict:
    return matrix[rule_id]


def build_literature_linked_associations(matches: list[rc.EligibleAtomicRuleMatch]) -> list[dict]:
    matrix = rc.load_rule_matrix()
    associations = []
    for m in matches:
        row = _matrix_row_for(m.rule_id, matrix)
        associations.append({
            "rule_id": m.rule_id, "evidence_family": m.evidence_family, "concern_domain": m.concern_domain,
            "source_claim": m.source_claim, "possible_interpretation": m.possible_interpretation,
            "evidence_strength": row["evidence_strength_as_written"], "evidence_direction": m.evidence_direction,
            "alternative_explanations": list(m.alternative_explanations),
            "matched_entity_ids": list(m.matched_entity_ids), "allowed_output_level": m.allowed_output_level,
        })
    return associations


def build_objective_profile(unified_evidence: list[UnifiedEvidenceItem]) -> dict[str, list[dict]]:
    """Groups EVERY measured/observed evidence item (whether or not it is
    rule-eligible) by category -- the descriptive visual profile every
    drawing gets, independent of whether any rule ever fires."""
    grouped: dict[str, list[dict]] = {}
    for ue in unified_evidence:
        category = ue.item.category or "other"
        grouped.setdefault(category, []).append(ue.to_dict())
    return grouped


def synthesize_drawing(
        image_id: str, entities: list["VisualEntity"], deterministic_features: dict,
        *, model_evidence: list | None = None,
) -> DrawingSynthesisResult:
    """Top-level orchestrator. `entities` = the Gemini Observer/Verifier
    entity list (same shape `run_development_benchmark.entities_from_
    verification_rows` already produces). `deterministic_features` = the
    dict `compute_deterministic_features`/`load_or_compute_deterministic_
    features` returns."""
    semantic_matches = rc.build_eligible_matches(entities)
    deterministic_matches = build_deterministic_eligible_matches(
        deterministic_features["composition"], deterministic_features["objective_features"],
        deterministic_features.get("page_reference"))
    all_matches = semantic_matches + deterministic_matches

    rule_ids_by_id = _rule_ids_by_matched_id(all_matches)
    relationship_features = compute_relationship_features(entities)

    unified_evidence = (
        normalize_semantic_evidence(entities, rule_ids_by_id)
        + normalize_deterministic_evidence(deterministic_features["objective_features"], rule_ids_by_id)
        + normalize_relationship_evidence(relationship_features)
        + normalize_emotion_evidence(model_evidence)
    )

    objective_profile = build_objective_profile(unified_evidence)
    literature_linked_associations = build_literature_linked_associations(all_matches)
    overall_synthesis = build_overall_synthesis(all_matches)
    candidate_hypotheses = rc.build_candidate_hypotheses(all_matches, model_evidence=model_evidence)

    return DrawingSynthesisResult(
        image_id=image_id, unified_evidence=unified_evidence, objective_profile=objective_profile,
        literature_linked_associations=literature_linked_associations, overall_synthesis=overall_synthesis,
        candidate_hypotheses=candidate_hypotheses,
    )
