#!/usr/bin/env python
"""DOAR Psychologist/Clinician Review App (Streamlit) -- first clinician
usability/design session prototype, NOT clinical validation.

Browses the 15 cached development cases end to end: original drawing ->
raw Observer candidates -> Verifier result -> eligible/ineligible
evidence -> atomic rule -> evidence family -> concern domain -> candidate
hypothesis -> clinician package -> parent package. Every pipeline step is
reused as-is from `reasoning_chain.py` and `scripts/run_development_
benchmark.py` -- this file adds NO new reasoning logic, NO new rule
matching, and never weakens the existing safety gate ("only verified
evidence may trigger a rule" -- `reasoning_chain.check_visual_
preconditions` already enforces this and is called unmodified here).

**No Gemini calls when browsing cases** -- this file never imports or
calls `run_live_observer_and_verifier`, `GeminiVisualObserver`, or
`GeminiVisualVerifier`. It only ever reads already-cached verification
JSON via `run_development_benchmark.find_saved_saved_verification_rows`
(live cache first, then legacy dev-check dirs -- same search order the
benchmark runner uses).

Three views:
  A. Clinician view (default) -- clean drawing by default (optional bbox
     overlay), observations grouped VERIFIED/UNCERTAIN/REJECTED-UNREVIEWED
     (nothing hidden), unified evidence sections (Objects/People,
     Expression/Pose, Colors, Lines/Strokes, Composition, Relationships,
     Global/Overall Drawing Profile), literature-linked evidence/reasoning
     trace, an ALWAYS-shown Overall Drawing Synthesis, a separate OPTIONAL
     candidate hypothesis section, and feedback controls.
  B. Parent view preview -- the SAME unified evidence, parent-safe wording
     only. Never empty: even with zero hypotheses it shows what was
     observed (objects/colours/lines/composition in plain language) and
     an always-on overall interpretation
     (`drawing_synthesis.build_overall_synthesis`'s own plain-language
     summary -- no rule IDs, no internal model details).
  C. Technical/audit view -- structured `feature | value | source | status
     | rule eligible` tables first, then the feature -> rule -> evidence
     family -> concern domain -> output trace; raw JSON only inside
     optional expanders.

Every new section reuses `drawing_synthesis.synthesize_drawing()` (unified
multimodal evidence + always-on synthesis, DOAR realignment milestone) and
`drawing_synthesis.load_or_compute_deterministic_features()` (cached
colour/line/composition features) -- no live Gemini call, no new reasoning
logic added here; this file only renders what those modules already
produce.

Launch:
    streamlit run scripts/clinician_review_app.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import drawing_synthesis as ds  # noqa: E402
from doar import human_interaction as hi  # noqa: E402
from doar import reasoning_chain as rc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_development_benchmark", ROOT / "scripts" / "run_development_benchmark.py")
rdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdb)

# DOAR_CLINICIAN_FEEDBACK_DIR lets tests redirect saves to a throwaway
# directory -- unset in normal use, where this is exactly
# ROOT / "outputs" / "clinician_feedback".
FEEDBACK_DIR = Path(os.environ.get("DOAR_CLINICIAN_FEEDBACK_DIR", str(ROOT / "outputs" / "clinician_feedback")))

STATUS_COLORS = {"verified": "#1a9850", "uncertain": "#e08214", "rejected": "#d73027", "unreviewed": "#999999"}
STATUS_GROUP_TITLES = {"verified": "VERIFIED", "uncertain": "UNCERTAIN", "other": "REJECTED / UNREVIEWED"}

CLINICIAN_NO_HYPOTHESIS_MESSAGE = (
    "No candidate clinical hypothesis -- current evidence does not meet the required convergence/strength "
    "threshold. This does not indicate absence of a condition.")
PARENT_NO_HYPOTHESIS_MESSAGE = (
    "No specific pattern was flagged by this automated review. This does not mean there is nothing "
    "to discuss -- a professional can look at the fuller picture regardless of what an automated "
    "review does or doesn't flag.")
CLINICIAN_ASSESSMENT_DISCLAIMER = "Candidate for professional assessment -- not a diagnosis."
PARENT_FOOTER_DISCLAIMER = "This tool supports observation and discussion. It does not provide a diagnosis."
PARENT_UNCONFIRMED_ELEMENTS_NOTE = "Other possible elements were noticed but could not be independently confirmed."
PARENT_PAGE_NOT_ASSESSABLE_NOTE = (
    "The full page boundary could not be confirmed, so page-relative size and placement were not interpreted.")

OBSERVATION_FEEDBACK_OPTIONS = ("(no feedback yet)", "Confirm visually", "Reject", "Cannot determine")
RELEVANCE_VERDICT_OPTIONS = ("(no feedback yet)", "Agree -- worth investigating", "Disagree", "Insufficient evidence")


# ---------------------------------------------------------------------------
# Pure data-loading/preparation -- no Streamlit calls below this point until
# the render_* functions, so this half is directly unit-testable.
# ---------------------------------------------------------------------------


def list_case_ids() -> list[str]:
    return [im["image_id"] for im in rdb.load_development_set()]


def load_case_bundle(image_id: str) -> dict | None:
    """Everything one case needs, purely from cache. Returns None if no
    saved Observer/Verifier data exists for this image_id (never makes a
    live call to find out more).

    `deterministic_features`/`synthesis` are the unified multimodal
    evidence + always-on drawing synthesis (`drawing_synthesis.py`,
    realignment milestone) -- computed here from the SAME cached entities
    already loaded for `conditions`/`checks`, with deterministic (colour/
    line/composition) features served from
    `outputs/prototype_cases/development_deterministic_features_cache`
    (see `scripts/precompute_deterministic_features.py`) so no per-request
    image processing is needed once that cache is warm."""
    images = rdb.load_development_set()
    image = next((im for im in images if im["image_id"] == image_id), None)
    if image is None:
        return None
    rows, source_path = rdb.find_saved_verification_rows(image_id)
    if rows is None:
        return {"image": image, "rows": None, "source_path": None, "entities": None,
                "conditions": None, "checks": None, "deterministic_features": None, "synthesis": None}
    entities = rdb.entities_from_verification_rows(rows)
    conditions = rdb.run_conditions_for_image(image_id, rows)
    checks = rc.check_visual_preconditions(entities)
    image_path = ROOT / image["relative_path"]
    deterministic_features = ds.load_or_compute_deterministic_features(image_id, image_path)
    synthesis = ds.synthesize_drawing(image_id, entities, deterministic_features)
    return {"image": image, "rows": rows, "source_path": source_path, "entities": entities,
            "conditions": conditions, "checks": checks, "deterministic_features": deterministic_features,
            "synthesis": synthesis}


def group_rows_by_status(rows: list[dict]) -> dict[str, list[dict]]:
    """VERIFIED / UNCERTAIN / REJECTED-UNREVIEWED -- nothing is dropped,
    every candidate lands in exactly one group."""
    groups: dict[str, list[dict]] = {"verified": [], "uncertain": [], "other": []}
    for row in rows:
        status = row["verification_status"]
        if status == "verified":
            groups["verified"].append(row)
        elif status == "uncertain":
            groups["uncertain"].append(row)
        else:
            groups["other"].append(row)
    return groups


def draw_annotated_image(image_path: Path, rows: list[dict]) -> Image.Image:
    """Bbox overlay color-coded by verification status -- a fresh
    in-memory copy, never written to disk. A candidate with no bbox is
    simply not drawn (nothing hidden from the text listing, just not
    drawable)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for row in rows:
        bbox = row["observer_candidate"].get("bbox")
        if not bbox:
            continue
        x, y, bw, bh = bbox
        left, top, right, bottom = x * w, y * h, (x + bw) * w, (y + bh) * h
        color = STATUS_COLORS.get(row["verification_status"], "#000000")
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left + 2, max(0, top - 14)), row["observation_id"], fill=color)
    return img


def build_parent_view_sections(hypotheses: list) -> list[dict]:
    """One parent-safe section per hypothesis, built ENTIRELY from
    `reasoning_chain.build_parent_package` (already code/data-constrained
    -- no rule IDs, no disorder labels, no LLM-invented wording). Empty
    list means the caller should show PARENT_NO_HYPOTHESIS_MESSAGE."""
    return [rc.build_parent_package(h) for h in hypotheses]


# ---------------------------------------------------------------------------
# Unified-evidence display sections (Clinician View). Purely a UI-layer
# grouping of `drawing_synthesis.py`'s own `objective_profile` categories
# -- no new evidence, no new rule logic. `semantic_objects` is split into
# "Objects / People" vs "Expression / Pose" using each entity's EXISTING
# `broader_categories` ("body_part" -- see `visual_entity.py`), never a
# label guess invented here.
# ---------------------------------------------------------------------------

DISPLAY_SECTION_TITLES = (
    "Objects / People", "Expression / Pose", "Colors", "Lines / Strokes",
    "Composition", "Relationships", "Global / Overall Drawing Profile",
)

_CATEGORY_TO_SECTION = {
    "colour": "Colors",
    "strokes_shading_repetition_overwriting": "Lines / Strokes",
    "global_composition": "Composition",
    "foreground_page_segmentation": "Composition",
    "spatial_relationships": "Relationships",
    "image_page_quality": "Global / Overall Drawing Profile",
}


def build_display_sections(bundle: dict) -> dict[str, list[dict]]:
    """Regroups `synthesis.objective_profile` (already every measured/
    observed evidence item, descriptive-or-not) into the 7 sections the
    realignment task specifies. Every item lands in exactly one section --
    nothing dropped, nothing duplicated."""
    sections: dict[str, list[dict]] = {title: [] for title in DISPLAY_SECTION_TITLES}
    synthesis = bundle["synthesis"]
    if synthesis is None:
        return sections
    entities_by_id = {e.entity_id: e for e in (bundle["entities"] or [])}
    for category, items in synthesis.objective_profile.items():
        for entry in items:
            item = entry["item"]
            if category == "semantic_objects":
                entity_id = item["evidence_id"].removeprefix("ev_semantic_")
                entity = entities_by_id.get(entity_id)
                is_body_part = bool(entity and "body_part" in entity.broader_categories)
                section = "Expression / Pose" if is_body_part else "Objects / People"
            else:
                section = _CATEGORY_TO_SECTION.get(category, "Global / Overall Drawing Profile")
            sections[section].append(entry)
    return sections


def build_technical_feature_rows(bundle: dict) -> list[dict]:
    """One row per unified evidence item: `feature | value | source |
    status | rule eligible` -- the structured table the Technical View
    shows BEFORE any raw JSON, per the realignment task's Section 7."""
    synthesis = bundle["synthesis"]
    if synthesis is None:
        return []
    rows = []
    for ue in synthesis.unified_evidence:
        item = ue.item
        rows.append({
            "feature": item.feature_id, "value": item.value if item.value is not None else "(unavailable)",
            "source": item.extractor, "status": item.status,
            "rule_eligible": "yes" if ue.rule_eligible else "no",
            "matched_rule_ids": ", ".join(ue.matched_rule_ids) or "-",
            "category": item.category or "other",
        })
    return rows


def _natural_join(items: list[str]) -> str:
    """'a' / 'a and b' / 'a, b and c' -- ordinary prose listing, not a
    raw comma-separated dump. Purely a text-formatting helper; the items
    themselves are never altered."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


# drawing_synthesis.build_overall_synthesis's own `summary` text is
# clinician/technical-facing -- per the realignment task's Section 1 fix
# it deliberately embeds the exact supporting rule_id(s)/evidence
# family(ies) so the Clinician/Technical views can "show the exact
# supporting rules/families behind every synthesis". That text must NEVER
# reach the Parent View (no rule IDs, no evidence-family names, no
# concern-domain identifiers there -- ParentViewSafetyTests, already
# established). Parent-facing wording is instead selected purely from
# `level` (+ whether the total family count is exactly one, to distinguish
# "one detail" from "several scattered details") -- never from which
# specific rule/family/domain matched.
_PARENT_NO_PATTERN_TEXT = "No clear psychological pattern was identified from the evidence available in this drawing."
_PARENT_ONE_ASSOCIATION_TEXT = (
    "One visual feature has been discussed in some drawing literature, but it is not enough on its own to "
    "support a broader interpretation.")
_PARENT_SCATTERED_ASSOCIATIONS_TEXT = (
    "Several individual features have been discussed in the literature, but they do not point to one "
    "consistent pattern.")
_PARENT_CONVERGENT_WORDING = {
    "convergent_positive_pattern": (
        "A consistent pattern was identified across several independent features in this drawing, of a kind "
        "sometimes associated with positive engagement. This is not a diagnosis."),
    "convergent_concern_pattern": (
        "A consistent pattern was identified across several independent features in this drawing, of a kind "
        "sometimes discussed in the drawing literature alongside low mood, worry, or tension. This is not a "
        "diagnosis."),
    "mixed_evidence": (
        "This drawing showed consistent patterns pointing in different directions -- some features of a kind "
        "linked to positive engagement, and others linked to worry or low mood. Taken together, they don't "
        "point clearly in one direction. This is not a diagnosis."),
}
_PARENT_SAFE_UNCERTAINTY_NOTE = (
    "Every detail behind this summary is something we could actually see or measure in the drawing -- it is not "
    "independent confirmation of any feeling or experience. The same detail commonly has other, non-clinical "
    "explanations too.")

_PARENT_CONVERSATION_QUESTIONS_WITH_CONTENT = (
    "What is happening in the picture?",
    "Who are the people (or characters) shown?",
    "Which part is the most important to you?",
    "How did you feel while drawing this?",
    "What do you think would happen next?",
)
_PARENT_CONVERSATION_QUESTIONS_NO_CONTENT = (
    "What is happening in the picture?",
    "What were you thinking about while drawing this?",
    "Is there anything you'd like to add to this drawing?",
)


def _parent_evidence_text(overall_synthesis: dict) -> list[str]:
    """Section C ("What the evidence suggests") -- very direct language,
    selected purely from `level` (+ total pooled family count for the
    one-vs-several distinction within `limited_association`). Never
    describes a pattern confidently unless the level itself is a genuine
    convergent one; never mentions a rule ID, evidence-family name, or
    concern-domain identifier.

    A genuine convergent level is described cautiously, alone. Otherwise
    the framing sentence ("no clear psychological pattern") always leads,
    followed by whatever individual-association note applies -- matching
    the acceptance example's own two-line shape (h38: "No clear
    psychological pattern..." + "One visual feature has been
    discussed...")."""
    level = overall_synthesis["level"]
    if level in _PARENT_CONVERGENT_WORDING:
        return [_PARENT_CONVERGENT_WORDING[level]]
    lines = [_PARENT_NO_PATTERN_TEXT]
    total_families = len(overall_synthesis["positive_evidence_families"]) + len(overall_synthesis["concern_evidence_families"])
    if total_families == 1:
        lines.append(_PARENT_ONE_ASSOCIATION_TEXT)
    elif total_families >= 2:
        lines.append(_PARENT_SCATTERED_ASSOCIATIONS_TEXT)
    return lines


def build_parent_conversation_questions(bundle: dict) -> list[str]:
    """Section D -- neutral, drawing-content-grounded CONVERSATION
    prompts, not diagnostic questions, and never tied to any specific
    rule/hypothesis (that would leak evidence-family-specific wording
    into the Parent View). A fixed, deterministic set; the only thing
    that varies is whether anything was confirmed in the drawing at all,
    never WHICH rule fired."""
    entities = bundle["entities"] or []
    has_verified_entity = any(e.case_verification_status == "verified" for e in entities)
    return list(_PARENT_CONVERSATION_QUESTIONS_WITH_CONTENT if has_verified_entity
                else _PARENT_CONVERSATION_QUESTIONS_NO_CONTENT)


def build_parent_friendly_profile(bundle: dict) -> dict:
    """Plain-language, always-populated content for the redesigned Parent
    View (sections A-E) -- built ENTIRELY from real measured/observed
    values (deterministic composition/colour/repetition + verified
    semantic labels) plus a parent-safe phrasing of the overall synthesis
    LEVEL (never its raw, rule-ID-bearing `summary` string). Non-empty
    even when zero hypotheses were raised -- Parent View must never be
    empty. Composition/placement is described ONLY when
    `page_relative_features_assessable` is True; raw image bounds are
    NEVER described as "the page" otherwise (page-frame safety, unchanged
    this round)."""
    det = bundle["deterministic_features"]
    synthesis = bundle["synthesis"]
    entities = bundle["entities"] or []
    rows = bundle["rows"] or []

    verified_labels = sorted({e.canonical_label for e in entities if e.case_verification_status == "verified"})
    has_unconfirmed = any(r["verification_status"] != "verified" for r in rows)

    comp = det["composition"]
    colour = det["colour"]
    objective_features = det["objective_features"]
    page_reference = det.get("page_reference") or {}

    # --- B. Visual style -----------------------------------------------
    colour_note = (
        f"Bright, varied colours are used, with {colour['dominant_colour'].replace('_', ' ')} particularly "
        f"noticeable." if colour.get("colour_diversity", 0) and colour.get("colour_diversity", 0) >= 2
        else (f"The most noticeable colour used is {colour['dominant_colour'].replace('_', ' ')}."
              if colour.get("dominant_colour") not in (None, "none_or_neutral")
              else "No single colour stands out strongly in this drawing."))

    intensity_fv = objective_features.get("stroke.intensity_proxy")
    if intensity_fv is not None and not intensity_fv.missing:
        if intensity_fv.value >= ds.INTENSITY_PROXY_HEAVY_THRESHOLD:
            lines_note = "The lines appear relatively heavy/dark."
        elif intensity_fv.value <= ds.INTENSITY_PROXY_LIGHT_THRESHOLD:
            lines_note = "The lines appear relatively light."
        else:
            lines_note = "The line weight is in a typical range."
    else:
        lines_note = "Line-weight information is not available for this drawing."

    repeated_labels = next(
        (ue.item.value for ue in synthesis.unified_evidence if ue.item.feature_id == "relationships.repeated_labels"), None)
    repetition_note = (
        f"Some elements (e.g. {_natural_join(sorted(repeated_labels)[:3])}) appear more than once in the drawing."
        if repeated_labels else None)

    page_assessable = bool(page_reference.get("page_relative_features_assessable"))
    if not page_assessable:
        composition_note = PARENT_PAGE_NOT_ASSESSABLE_NOTE
    elif comp["bounding_box"] is None:
        composition_note = "No clear drawing content was detected."
    else:
        page_relative_fv = objective_features.get("segmentation.page_relative_bounding_box_coverage")
        coverage = (page_relative_fv.value if page_relative_fv is not None and not page_relative_fv.missing
                    else comp["bounding_box_coverage"])
        placement = (comp["placement"] or "unavailable").replace("_", " ")
        composition_note = f"The drawing takes up about {coverage * 100:.0f}% of the page, positioned toward the {placement}."

    visual_style_notes = [note for note in (colour_note, lines_note, repetition_note, composition_note) if note]

    return {
        "objects_seen": _natural_join(verified_labels) if verified_labels else "",
        "has_unconfirmed_elements": has_unconfirmed,
        "visual_style_notes": visual_style_notes,
        "evidence_text": _parent_evidence_text(synthesis.overall_synthesis),
        "uncertainty_note": _PARENT_SAFE_UNCERTAINTY_NOTE,
        "questions": build_parent_conversation_questions(bundle),
    }


# ---------------------------------------------------------------------------
# Clinician View helpers -- a compact case summary + readable association
# cards, replacing the old debugging-oriented full-dump layout. All pure
# regrouping/lookup of `synthesis`'s own already-computed fields (plus
# read-only lookups into the frozen `rc.load_rule_matrix()` for display-
# only fields like `source_pdf_page_section`) -- no new evidence, no new
# eligibility/convergence logic.
# ---------------------------------------------------------------------------


def build_clinician_case_summary(bundle: dict) -> dict:
    """The compact top-of-page numbers a clinician needs before deciding
    whether to read further: how many entities were detected/confirmed/
    uncertain, how many literature associations exist, which domain (if
    any) reached real convergence, and how many candidate hypotheses."""
    rows = bundle["rows"] or []
    synthesis = bundle["synthesis"]
    overall = synthesis.overall_synthesis
    groups = group_rows_by_status(rows)
    convergent_domains = [d for d in (overall["convergent_positive_domain"], overall["convergent_concern_domain"]) if d]
    return {
        "entities_detected": len(rows),
        "confirmed": len(groups["verified"]),
        "uncertain_unreviewed": len(groups["uncertain"]) + len(groups["other"]),
        "associations": len(synthesis.literature_linked_associations),
        "synthesis_level": overall["level"],
        "convergent_domains": convergent_domains,
        "candidate_hypotheses": len(synthesis.candidate_hypotheses),
    }


def build_entity_table_rows(bundle: dict) -> list[dict]:
    """Compact `label | verification status | confidence` rows for
    Clinician View section 1 -- replaces the old per-entity bordered-
    container layout. Nothing is dropped or reclassified; every row's
    `status` is copied verbatim from the Verifier's own result."""
    rows = bundle["rows"] or []
    return [
        {"label": r["observer_candidate"]["label"], "status": r["verification_status"],
         "observer_confidence": r["observer_candidate"].get("confidence"),
         "verifier_confidence": r.get("verifier_confidence")}
        for r in rows
    ]


def build_association_cards(bundle: dict) -> list[dict]:
    """One readable card per literature-linked association (Clinician
    View section 3) -- `synthesis.literature_linked_associations`'s own
    fields plus `source_pdf_page_section` looked up read-only from
    `rc.load_rule_matrix()` (a presentation-only enrichment of an
    already-frozen matrix column; no new evidence, no new rule)."""
    synthesis = bundle["synthesis"]
    if synthesis is None:
        return []
    matrix = rc.load_rule_matrix()
    cards = []
    for a in synthesis.literature_linked_associations:
        row = matrix.get(a["rule_id"], {})
        cards.append({
            "observation": ", ".join(a["matched_entity_ids"]),
            "literature_interpretation": a["possible_interpretation"],
            "evidence_strength": a["evidence_strength"],
            "domain": a["concern_domain"],
            "alternative_explanations": a["alternative_explanations"],
            "source": row.get("source_pdf_page_section") or a["source_claim"],
            "rule_id": a["rule_id"],
            "evidence_family": a["evidence_family"],
        })
    return cards


_CLINICIAN_SHORT_SYNTHESIS_LABEL = {
    "insufficient_interpretable_evidence": "No interpretable literature-linked evidence.",
    "descriptive_only": "Descriptive evidence only -- no positive- or concern-associated indicator.",
    "limited_association": "Individual association(s) present; none reach within-domain convergence.",
    "convergent_positive_pattern": "Convergent positive-direction pattern (within-domain convergence reached).",
    "convergent_concern_pattern": "Convergent concern-direction pattern (within-domain convergence reached).",
    "mixed_evidence": "Independently convergent evidence in both directions.",
}


def build_synthesis_summary_view(bundle: dict) -> dict:
    """Clinician View section 4 -- a short, scannable synthesis summary
    (level + supporting families + which domain, if any, converges),
    with the full machine-generated paragraph available separately for
    anyone who wants it (never hidden, just not the primary read)."""
    overall = bundle["synthesis"].overall_synthesis
    return {
        "level": overall["level"],
        "short_label": _CLINICIAN_SHORT_SYNTHESIS_LABEL.get(overall["level"], overall["level"]),
        "positive_evidence_families": overall["positive_evidence_families"],
        "concern_evidence_families": overall["concern_evidence_families"],
        "convergent_positive_domain": overall["convergent_positive_domain"],
        "convergent_concern_domain": overall["convergent_concern_domain"],
        "domains_touched": overall["domains_touched"],
        "full_explanation": overall["summary"],
    }


def build_missing_evidence_notes(bundle: dict) -> list[str]:
    """Clinician View section 6 -- limitations surfaced prominently
    rather than buried in a JSON dump. Grounded entirely in fields the
    pipeline already computes: page-reference assessability, unverified/
    uncertain entities, and the semantic rule engine's own `blocked_*`
    reasons (`requires_process_data`/`requires_longitudinal_data`/
    `requires_absolute_scale_or_context`/`blocked_needs_unbuilt_feature`
    -- real, frozen RULE_EVIDENCE_MATRIX.csv flags, not invented text)."""
    notes = []
    page_reference = (bundle["deterministic_features"] or {}).get("page_reference") or {}
    if not page_reference.get("page_relative_features_assessable"):
        notes.append("Page-relative size/placement interpretation: unavailable (the full page boundary could not be confirmed).")

    rows = bundle["rows"] or []
    unverified = [r["observer_candidate"]["label"] for r in rows if r["verification_status"] != "verified"]
    if unverified:
        notes.append(f"Unverified/uncertain candidates (not used as evidence): {', '.join(unverified)}.")

    checks = bundle["checks"] or []
    blocked_reasons: dict[str, int] = {}
    for c in checks:
        if c.status in ("blocked_structural", "blocked_needs_unbuilt_feature"):
            blocked_reasons[c.reason] = blocked_reasons.get(c.reason, 0) + 1
    for reason, count in sorted(blocked_reasons.items()):
        notes.append(f"{count} rule(s) structurally not assessable from this pipeline: {reason}")

    notes.append("This review is based on the image alone -- the child's age, the drawing prompt given, and any "
                 "verbal context were not captured or used.")
    return notes


def git_commit_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Clinician feedback storage -- one JSON per (session, case), plus a
# rebuilt-every-save CSV summary. Never stores API keys; never trains
# anything from this data.
# ---------------------------------------------------------------------------


def feedback_path(session_id: str, case_id: str) -> Path:
    safe_session = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    out_dir = FEEDBACK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_session}__{case_id}.json"


def save_feedback(session_id: str, case_id: str, payload: dict) -> Path:
    """Deterministic, idempotent: saving the same (session_id, case_id)
    again overwrites the prior file rather than appending or duplicating.
    Uses the same atomic write already validated for the live cache."""
    record = dict(payload)
    record["session_id"] = session_id
    record["case_id"] = case_id
    record["timestamp"] = record.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S")
    record["git_commit"] = git_commit_sha()
    path = feedback_path(session_id, case_id)
    rdb._atomic_write_json(path, record)
    rebuild_feedback_summary_csv()
    return path


def load_feedback(session_id: str, case_id: str) -> dict | None:
    path = feedback_path(session_id, case_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_feedback_summary_csv() -> Path:
    """Rebuilt from scratch from every saved JSON file every time --
    always consistent with what's on disk, never an append-only log that
    could drift or duplicate rows across repeated saves."""
    import csv

    out_dir = FEEDBACK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    rows = []
    for json_path in sorted(out_dir.glob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({
            "session_id": record.get("session_id", ""), "case_id": record.get("case_id", ""),
            "timestamp": record.get("timestamp", ""), "git_commit": record.get("git_commit", ""),
            "num_observations_reviewed": len(record.get("observation_feedback", {})),
            "num_hypotheses_reviewed": len(record.get("hypothesis_feedback", [])),
            "general_comments": record.get("general_comments", ""),
            "missing_information": record.get("missing_information", ""),
            "what_should_be_removed": record.get("what_should_be_removed", ""),
            "what_should_be_added": record.get("what_should_be_added", ""),
            "wording_too_strong": record.get("wording_too_strong", ""),
            "references_useful": record.get("references_useful", ""),
            "evidence_trace_understandable": record.get("evidence_trace_understandable", ""),
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["session_id", "case_id", "timestamp", "git_commit", "num_observations_reviewed",
                      "num_hypotheses_reviewed", "general_comments", "missing_information",
                      "what_should_be_removed", "what_should_be_added", "wording_too_strong",
                      "references_useful", "evidence_trace_understandable"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    return f"anon_{uuid.uuid4().hex[:8]}"


@st.cache_resource(show_spinner=False)
def _cached_answer_provider():
    """Resolved once per Streamlit process (not per rerun/question) --
    construction only reads env vars / imports google-genai, it never
    depends on the case being viewed. Degrades to the deterministic
    provider automatically when GEMINI_API_KEY/google-genai are absent."""
    return hi.resolve_default_answer_provider()


@st.cache_resource(show_spinner=False)
def _cached_research_provider():
    return hi.resolve_default_research_provider()


@st.cache_resource(show_spinner=False)
def _cached_judge():
    return hi.resolve_default_judge()


@st.cache_resource(show_spinner=False)
def _cached_visual_recheck_provider():
    return hi.resolve_default_visual_recheck_provider()


_ASK_DOAR_GENERIC_ERROR = "Ask DOAR could not complete that request. Please try again."


def _recent_conversation_history(turns: list[dict], *, max_turns: int = 8) -> list[dict]:
    """Last `max_turns` {"role", "content"} pairs only -- never the
    `structured`/`error_detail` bookkeeping fields, and never more than a
    bounded window (final-correction-pass Part 9)."""
    return [{"role": t["role"], "content": t["content"]} for t in turns[-max_turns:]]


def render_ask_doar_chat(bundle: dict, case_id: str, session_id: str, *, mode: str) -> None:
    """Shared 'Ask DOAR' chat panel (Human Interaction Layer v1, Part C).
    SAME underlying evidence/answer for Parent and Clinician -- both call
    `human_interaction.answer_question()` with the SAME resolved
    providers; only `audience` (phrasing style) and the technical-detail
    expander (clinician only) differ. Every answer has already passed
    the deterministic verifier + Judge (optionally a real Gemini-backed
    answer helper/Judge/external-research/visual-recheck provider, each
    independently falling back to its offline default when
    GEMINI_API_KEY/google-genai are unavailable) before it reaches here
    -- this function never renders raw, unchecked model text. ANY
    exception from the interaction layer is caught here (final-
    correction-pass Part 12) -- Parent/Clinician never see a raw Python
    traceback; the sanitized detail is only ever shown, collapsed, in
    Clinician mode, and the full traceback is printed to the console for
    development."""
    st.subheader("Ask DOAR")
    st.caption("Ask anything about this drawing, the observations, rules, references or interpretation.")
    history_key = f"chat_history_{mode}_{session_id}_{case_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for turn in st.session_state[history_key]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            sa = turn.get("structured")
            error_detail = turn.get("error_detail")
            if turn["role"] == "assistant" and mode == "clinician" and sa:
                with st.expander("Technical details"):
                    st.caption(f"Category: `{sa['category']}`  ·  Judge verdict: `{sa['judge_verdict']}`  ·  "
                               f"Judge mode: `{sa['judge_mode']}`  ·  Answer provider: `{sa['answer_provider']}`")
                    if sa.get("answer_provider_error"):
                        st.caption(f"Answer-provider error (sanitized): {sa['answer_provider_error']}")
                    if sa["judge_reasons"]:
                        st.caption("Judge notes: " + "; ".join(sa["judge_reasons"]))
                    if sa["used_external_research"]:
                        st.caption("This answer used additional research outside DOAR's frozen corpus -- "
                                   "labelled separately in the answer text above, not part of the original analysis.")
                    if sa["used_visual_recheck"]:
                        st.caption("This answer used an on-demand visual re-check (tagged Q&A_VISUAL_RECHECK), "
                                   "separate from the original saved analysis.")
                    for claim in sa["claims"]:
                        ids = ", ".join(claim.get("evidence_ids", []) + claim.get("rule_ids", []) +
                                        claim.get("source_ids", []) + claim.get("external_source_ids", []))
                        st.write(f"- ({claim['claim_type']}) {claim['text']}" + (f"  ·  refs: `{ids}`" if ids else ""))
                    if sa["provenance"]:
                        st.caption("Provenance: " + " -> ".join(sa["provenance"]))
            elif turn["role"] == "assistant" and mode == "clinician" and error_detail:
                with st.expander("Technical error details"):
                    st.caption(error_detail)

    question = st.chat_input("Ask DOAR -- ask anything about this drawing, the observations, rules, "
                              "references or interpretation.", key=f"chat_input_{mode}_{session_id}_{case_id}")
    if question:
        history = _recent_conversation_history(st.session_state[history_key])
        try:
            answer = hi.answer_question(
                question, bundle, audience=mode,
                answer_provider=_cached_answer_provider(),
                external_research_provider=_cached_research_provider(),
                judge=_cached_judge(),
                visual_recheck_provider=_cached_visual_recheck_provider(),
                conversation_history=history,
            )
            st.session_state[history_key].append({"role": "user", "content": question})
            st.session_state[history_key].append(
                {"role": "assistant", "content": answer.answer, "structured": answer.to_dict()})
        except Exception as exc:  # noqa: BLE001 -- last-resort UI boundary, never a raw traceback to the user
            import traceback
            traceback.print_exc()  # dev console only
            st.session_state[history_key].append({"role": "user", "content": question})
            st.session_state[history_key].append({
                "role": "assistant", "content": _ASK_DOAR_GENERIC_ERROR, "structured": None,
                "error_detail": f"{hi.sanitize_error_text(exc)}",
            })
        st.rerun()


def render_clinician_view(bundle: dict, case_id: str, session_id: str) -> None:
    image_path = ROOT / bundle["image"]["relative_path"]
    rows = bundle["rows"]

    if rows is None:
        st.warning("No cached Observer/Verifier data available for this case.")
        st.image(str(image_path), width='stretch')
        return

    synthesis = bundle["synthesis"]
    groups = group_rows_by_status(rows)
    prior_feedback = load_feedback(session_id, case_id) or {}
    observation_feedback = dict(prior_feedback.get("observation_feedback", {}))
    hypothesis_feedback = list(prior_feedback.get("hypothesis_feedback", []))

    # --- CASE SUMMARY ---------------------------------------------------
    summary = build_clinician_case_summary(bundle)
    st.subheader("Case summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Entities detected", summary["entities_detected"])
    s2.metric("Confirmed", summary["confirmed"])
    s3.metric("Uncertain / unreviewed", summary["uncertain_unreviewed"])
    s4.metric("Associations", summary["associations"])
    s5.metric("Candidate hypotheses", summary["candidate_hypotheses"])
    st.caption(
        f"Synthesis level: `{summary['synthesis_level']}`  |  Convergent domain(s): "
        f"{', '.join(summary['convergent_domains']) or '(none)'}")

    st.markdown("---")
    st.subheader("Drawing")
    show_overlay = st.checkbox("Show bbox overlay (color-coded by Verifier status)",
                                value=False, key=f"overlay_{session_id}_{case_id}")
    if show_overlay:
        st.image(draw_annotated_image(image_path, rows), width='stretch')
        st.caption("🟩 Verified   🟧 Uncertain   🟥 Rejected   ⬜ Unreviewed")
    else:
        st.image(str(image_path), width='stretch')

    st.markdown("---")
    # --- 2. Key observations (default-visible, compact) -------------------
    # Confirmed / possible-uncertain / not-confirmed, in the SAME plain
    # grouping Parent View uses (final-correction-pass Part 8) -- the
    # full per-row table + feedback controls move under a collapsed
    # expander so the first screen stays scannable in ~1-2 minutes.
    st.subheader("2. Key observations")
    obs = hi.observation_bullets(bundle)
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown("**Confirmed**")
        for line in obs["confirmed"]:
            st.write(f"- {line}")
        if not obs["confirmed"]:
            st.caption("(none)")
    with oc2:
        st.markdown("**Possible / uncertain**")
        for line in obs["possible"]:
            st.write(f"- {line}")
        if not obs["possible"]:
            st.caption("(none)")
    with oc3:
        st.markdown("**Not confirmed**")
        if obs["not_confirmed_labels"]:
            for label in obs["not_confirmed_labels"]:
                st.write(f"- {label}")
        else:
            st.caption("(none)")

    with st.expander("All visual observations (full detail)"):
        st.caption("Every candidate is shown -- confirmed, uncertain, or unreviewed. Only VERIFIED evidence can trigger a rule.")
        st.dataframe(build_entity_table_rows(bundle), width='stretch', hide_index=True)
        with st.expander("Give feedback on individual observations"):
            for group_key in ("verified", "uncertain", "other"):
                for row in groups[group_key]:
                    oc = row["observer_candidate"]
                    key = f"obs_{session_id}_{case_id}_{row['observation_id']}"
                    current = observation_feedback.get(row["observation_id"], {}).get("verdict", OBSERVATION_FEEDBACK_OPTIONS[0])
                    st.caption(f"`{row['observation_id']}` **{oc['label']}** -- verifier status: {row['verification_status']}")
                    verdict = st.selectbox("Clinician verdict", OBSERVATION_FEEDBACK_OPTIONS,
                                            index=OBSERVATION_FEEDBACK_OPTIONS.index(current) if current in OBSERVATION_FEEDBACK_OPTIONS else 0,
                                            key=key, label_visibility="collapsed")
                    observation_feedback[row["observation_id"]] = {"observer_label": oc["label"], "verdict": verdict}

    with st.expander("Detailed objective measurements"):
        det = bundle["deterministic_features"]
        page_reference = det.get("page_reference") or {}
        repeated_labels = next(
            (ue.item.value for ue in synthesis.unified_evidence if ue.item.feature_id == "relationships.repeated_labels"), {})
        p1, p2, p3 = st.columns(3)
        p1.markdown(f"**Colour**  \n{det['colour']['dominant_colour']}, {det['colour']['colour_diversity']} meaningful colour(s)")
        intensity_fv = det["objective_features"].get("stroke.intensity_proxy")
        frag_fv = det["objective_features"].get("stroke.fragmentation")
        p2.markdown(
            f"**Lines / strokes**  \nintensity_proxy={intensity_fv.value:.3f}" if intensity_fv and not intensity_fv.missing else "**Lines / strokes**  \n(unavailable)")
        if frag_fv and not frag_fv.missing:
            p2.caption(f"fragmentation={frag_fv.value:.3f}")
        p3.markdown(f"**Relationships**  \n{repeated_labels or '(no repeated elements)'}")
        if page_reference.get("page_relative_features_assessable"):
            st.markdown(
                f"**Composition** -- page-relative: bounding_box_coverage={det['composition']['bounding_box_coverage']:.0%}, "
                f"placement={det['composition']['placement']}")
        else:
            st.markdown("**Composition** -- page-relative interpretation unavailable (page boundary not confirmed).")
        st.caption(f"page_reference_mode: `{page_reference.get('page_reference_mode')}`  |  "
                   f"page_relative_features_assessable: `{page_reference.get('page_relative_features_assessable')}`")
        with st.expander("View all measured objective features"):
            display_sections = build_display_sections(bundle)
            for section_title in DISPLAY_SECTION_TITLES:
                items = display_sections[section_title]
                with st.expander(f"{section_title} ({len(items)})", expanded=False):
                    if not items:
                        st.caption("(no evidence in this section for this drawing)")
                        continue
                    for entry in items:
                        item = entry["item"]
                        flag = "✅ rule-eligible" if entry["rule_eligible"] else "descriptive only"
                        value_str = item["value"] if item["value"] is not None else "(unavailable)"
                        st.write(f"- **{item['feature_id']}** = {value_str}  ·  status: {item['status']}  ·  {flag}")
                        if item.get("reason"):
                            st.caption(item["reason"])

    st.markdown("---")
    st.subheader("3. Evidence -> rule -> association")
    st.caption("Each chain is readable without any internal ID -- IDs are still available in the expanders below.")
    chains = hi.build_evidence_chains(bundle)
    if not chains:
        st.caption("(no literature-linked chains for this drawing)")
    for c in chains:
        domain_label = c["concern"]["label"]
        st.write(f"**{c['evidence_family_display']}**  ↓  {c['rule_display_name']} rule  ↓  "
                 f"{domain_label} association  (strength: {c['evidence_strength']})")

    with st.expander("Full evidence/rule details"):
        cards = build_association_cards(bundle)
        if not cards:
            st.caption("(none -- no verified/measured evidence currently satisfies an existing rule's precondition)")
        for card in cards:
            with st.container(border=True):
                st.markdown(f"**{card['observation']}**  ·  {card['domain']}  ·  strength: {card['evidence_strength']}")
                st.write(card["literature_interpretation"])
                if card["alternative_explanations"]:
                    st.caption("Alternative explanations: " + "; ".join(card["alternative_explanations"]))
                st.caption(f"Source: {card['source']}")
                with st.expander("Rule ID / evidence family (audit reference)"):
                    st.write(f"`{card['rule_id']}`  ·  family: {card['evidence_family']}")

    st.markdown("---")
    st.subheader("4. Overall synthesis")
    synth_view = build_synthesis_summary_view(bundle)
    st.info(f"**{synth_view['short_label']}**")
    st.write(f"Supporting families -- positive: {', '.join(synth_view['positive_evidence_families']) or '(none)'}; "
             f"concern: {', '.join(synth_view['concern_evidence_families']) or '(none)'}")
    convergent = synth_view["convergent_positive_domain"] or synth_view["convergent_concern_domain"]
    st.write(f"Within-domain convergence: {'yes, in **' + convergent + '**' if convergent else 'no'}")
    with st.expander("Full synthesis explanation"):
        st.write(synth_view["full_explanation"])

    st.markdown("---")
    st.subheader("5. Candidate clinical hypothesis")
    hypotheses = synthesis.candidate_hypotheses
    if not hypotheses:
        st.info(CLINICIAN_NO_HYPOTHESIS_MESSAGE)
    for i, h in enumerate(hypotheses):
        cp = rc.build_clinician_package(h)
        with st.container(border=True):
            st.markdown(f"### {cp['candidate_hypothesis_label']}")
            st.markdown(f"**Support level:** {cp['support_level']}")
            st.markdown("**Why:**")
            st.write(f"- Triggered rules: {', '.join(cp['why']['triggered_rule_ids'])}")
            st.write(f"- Evidence families: {', '.join(cp['why']['evidence_families'])}")
            st.write(f"- Supporting entities: {', '.join(cp['why']['supporting_entity_ids'])}")
            for src, strength in zip(cp["why"]["source_claims"], cp["why"]["evidence_strength_as_written"]):
                st.caption(f"  • {src} (strength: {strength})")
            st.markdown(f"**Contradictions/counterevidence:** {', '.join(cp['counterevidence']) or '(none on file)'}")
            st.markdown(f"**Alternative explanations:** {', '.join(cp['alternative_explanations']) or '(none listed)'}")
            st.markdown("**Missing diagnostic information:**")
            for m in cp["missing_clinical_information"]:
                st.write(f"- {m}")
            st.warning(f"{CLINICIAN_ASSESSMENT_DISCLAIMER} {cp['disclaimer']}")

            existing = next((hf for hf in hypothesis_feedback if hf.get("hypothesis_label") == cp["candidate_hypothesis_label"]), {})
            key = f"hyp_{session_id}_{case_id}_{i}"
            current = existing.get("clinician_relevance_verdict", RELEVANCE_VERDICT_OPTIONS[0])
            verdict = st.selectbox("clinician_relevance_verdict", RELEVANCE_VERDICT_OPTIONS,
                                    index=RELEVANCE_VERDICT_OPTIONS.index(current) if current in RELEVANCE_VERDICT_OPTIONS else 0,
                                    key=key)
            st.caption("clinical_diagnostic_status: assessment_pending (not used in this usability study)")
            comment = st.text_input("Comment on this hypothesis", value=existing.get("comment", ""), key=f"{key}_comment")
            new_entry = {
                "concern_domain": h.concern_domain, "hypothesis_label": cp["candidate_hypothesis_label"],
                "clinician_relevance_verdict": verdict, "clinical_diagnostic_status": "assessment_pending",
                "comment": comment,
            }
            hypothesis_feedback = [hf for hf in hypothesis_feedback if hf.get("hypothesis_label") != cp["candidate_hypothesis_label"]]
            hypothesis_feedback.append(new_entry)

    st.markdown("---")
    st.subheader("6. References")
    if not chains:
        st.caption("(no literature-linked rules contributed to this drawing's interpretation)")
    for c in chains:
        src = c["source"]
        if src["citation_title"]:
            citation = src["citation_title"] + (f" ({src['doi_or_pmid']})" if src["doi_or_pmid"] else "")
        elif src["source_pdf_page_section"]:
            citation = (f"Primary literature reference not yet recorded in the DOAR source registry. "
                        f"Internal source on file: {src['source_pdf_page_section']}")
        else:
            citation = "No source currently recorded."
        st.write(f"- **{c['rule_display_name']}**  ·  family: {c['evidence_family_display']}  ·  "
                 f"domain: `{c['concern_domain']}`  ·  literature support: "
                 f"{src['literature_support_status'] or 'not recorded'}")
        st.caption(citation)
        with st.expander(f"`{c['rule_id']}` (audit reference)"):
            st.write(f"Evidence family: `{c['evidence_family']}`  ·  Allowed output level: `{c['allowed_output_level']}`")

    st.markdown("---")
    render_ask_doar_chat(bundle, case_id, session_id, mode="clinician")

    st.markdown("---")
    with st.expander("Full technical/audit trace"):
        st.markdown("**Missing / not-assessable evidence**")
        for note in build_missing_evidence_notes(bundle):
            st.write(f"- {note}")
        st.caption("See the Technical / Audit View tab for complete traceability (raw Observer/Verifier JSON, "
                   "deterministic preconditions, blocked rules, candidate-hypothesis packages).")
        st.json([rdb._entity_to_dict(e) for e in bundle["entities"]])

    st.markdown("---")
    st.subheader("Reviewer feedback")
    general_comments = st.text_area("General comments", value=prior_feedback.get("general_comments", ""), key=f"gc_{session_id}_{case_id}")
    missing_information = st.text_area("What information is missing?", value=prior_feedback.get("missing_information", ""), key=f"mi_{session_id}_{case_id}")
    what_removed = st.text_area("What should be removed?", value=prior_feedback.get("what_should_be_removed", ""), key=f"wr_{session_id}_{case_id}")
    what_added = st.text_area("What should be added?", value=prior_feedback.get("what_should_be_added", ""), key=f"wa_{session_id}_{case_id}")
    wording_too_strong = st.radio("Is wording too strong?", ("unsure", "yes", "no"),
                                   index=("unsure", "yes", "no").index(prior_feedback.get("wording_too_strong", "unsure")),
                                   key=f"wts_{session_id}_{case_id}", horizontal=True)
    references_useful = st.radio("Are references useful?", ("unsure", "yes", "no"),
                                  index=("unsure", "yes", "no").index(prior_feedback.get("references_useful", "unsure")),
                                  key=f"ru_{session_id}_{case_id}", horizontal=True)
    trace_understandable = st.radio("Is the evidence trace understandable?", ("unsure", "yes", "no"),
                                     index=("unsure", "yes", "no").index(prior_feedback.get("evidence_trace_understandable", "unsure")),
                                     key=f"tu_{session_id}_{case_id}", horizontal=True)

    if st.button("Save feedback for this case", key=f"save_{session_id}_{case_id}", type="primary"):
        payload = {
            "observation_feedback": observation_feedback, "hypothesis_feedback": hypothesis_feedback,
            "general_comments": general_comments, "missing_information": missing_information,
            "what_should_be_removed": what_removed, "what_should_be_added": what_added,
            "wording_too_strong": wording_too_strong, "references_useful": references_useful,
            "evidence_trace_understandable": trace_understandable,
        }
        path = save_feedback(session_id, case_id, payload)
        st.success(f"Saved -> {path}")


def render_parent_view(bundle: dict, case_id: str, session_id: str) -> None:
    """Redesigned Parent View (Human Interaction Layer v1, Part A) -- six
    short, case-specific sections, understandable without any rule ID:
    (1) What we noticed, (2) What these features may suggest -- one card
    per literature-linked feature plus unmatched-but-verified
    observations shown explicitly (never hidden), (3) Overall
    interpretation, (4) Sources and rules, (5) Suggested questions to ask
    the child, (6) one footer disclaimer. Every value comes from
    `human_interaction.py`'s read-only, case-specific builders (which in
    turn read only `bundle["synthesis"]`/`bundle["entities"]` -- no new
    evidence, no new rule matching, nothing invented for this drawing)."""
    image_path = ROOT / bundle["image"]["relative_path"]
    rows = bundle["rows"]
    st.image(str(image_path), width='stretch')
    if rows is None:
        st.warning("No cached data available for this case.")
        return

    profile = build_parent_friendly_profile(bundle)
    obs = hi.observation_bullets(bundle)
    chains = hi.build_evidence_chains(bundle)
    unmatched_labels = hi.grouped_unmatched_observation_labels(bundle)

    # --- 1. What we noticed ---------------------------------------------
    st.subheader("What we noticed")
    for line in obs["confirmed"]:
        st.write(f"- {line}")
    for line in obs["possible"]:
        st.write(f"- {line}")
    for note in profile["visual_style_notes"]:
        st.write(f"- {note}")
    if not obs["confirmed"] and not obs["possible"] and not profile["visual_style_notes"]:
        st.write("Nothing could be confidently identified in this drawing.")
    if obs["not_confirmed_note"]:
        st.caption(obs["not_confirmed_note"])

    st.markdown("---")
    # --- 2. What these features may suggest ------------------------------
    # Individual cards ONLY for matched/eligible rules (final-correction-
    # pass Part 5) -- observations with no current rule are grouped into
    # ONE compact, deduplicated section below instead of one large card
    # each (the prior design repeated e.g. 5 separate "butterfly" cards).
    st.subheader("What these features may suggest")
    if not chains and not unmatched_labels:
        st.caption("No literature-linked features were found for this drawing.")
    for c in chains:
        with st.container(border=True):
            st.markdown(f"**{c['rule_display_name']}**")
            st.write(c["rule_suggests"])
            st.caption(f"Why it applies here: {c['why_it_applies']}.")
            with st.expander("View reference"):
                src = c["source"]
                if src["citation_title"]:
                    st.markdown(f"**{src['citation_title']}**")
                    if src["doi_or_pmid"]:
                        st.caption(f"DOI/PMID: {src['doi_or_pmid']}")
                    st.caption(src["authors_and_year_note"])
                elif src["source_pdf_page_section"]:
                    st.caption("Primary literature reference not yet recorded in the DOAR source registry. "
                               "Internal source on file:")
                    st.write(src["source_pdf_page_section"])
                else:
                    st.caption("No source is currently recorded for this rule.")
            with st.expander("Technical details"):
                st.caption(f"Rule ID: `{c['rule_id']}`  ·  Evidence family: {c['evidence_family_display']}  ·  "
                           f"Concern domain: `{c['concern_domain']}`")
    if unmatched_labels:
        st.markdown("**Other things noticed without a current psychological rule**")
        for label in unmatched_labels:
            st.write(f"- {label}")

    st.markdown("---")
    # --- 3. Overall interpretation ---------------------------------------
    st.subheader("Overall interpretation")
    st.info(hi.overall_interpretation_text(bundle))

    st.markdown("---")
    # --- 4. Sources and rules --------------------------------------------
    st.subheader("Sources and rules")
    if not chains:
        st.caption("(no literature-linked rules contributed to this drawing's interpretation)")
    for c in chains:
        src = c["source"]
        if src["citation_title"]:
            citation = src["citation_title"]
        elif src["source_pdf_page_section"]:
            citation = (f"Primary literature reference not yet recorded in the DOAR source registry. "
                        f"Internal source on file: {src['source_pdf_page_section']}")
        else:
            citation = "No source currently recorded."
        st.write(f"- **{c['rule_display_name']}** -- matched: {c['why_it_applies']}; concern area: {c['concern']['label']}")
        st.caption(citation + (f"  ·  DOI/PMID: {src['doi_or_pmid']}" if src["doi_or_pmid"] else ""))

    st.markdown("---")
    # --- 5. Suggested questions to ask the child -------------------------
    st.subheader("Suggested questions to ask the child")
    for q in hi.case_specific_questions(bundle):
        st.write(f"- {q}")

    st.markdown("---")
    render_ask_doar_chat(bundle, case_id, session_id, mode="parent")

    st.markdown("---")
    st.caption(hi.FOOTER_DISCLAIMER)


def render_technical_view(bundle: dict, case_id: str) -> None:
    rows = bundle["rows"]
    if rows is None:
        st.warning("No cached Observer/Verifier data for this case.")
        return

    st.subheader("Provenance")
    st.write(f"Source file: `{bundle['source_path']}`")
    if rows:
        st.write(f"Observer model: `{rows[0].get('observer_model')}`  |  Verifier model: `{rows[0].get('verifier_model')}`")

    st.subheader("Unified evidence table (feature | value | source | status | rule eligible)")
    st.caption(
        "Every feature the unified pipeline produced for this case -- semantic (Gemini), deterministic "
        "(colour/line/composition), and relationship evidence, normalized to one shape "
        "(`evidence_schema.EvidenceItem`). Structured table first, per this view's own convention; raw JSON "
        "is only in the expanders below.")
    feature_rows = build_technical_feature_rows(bundle)
    st.dataframe(feature_rows, width='stretch', hide_index=True)

    st.subheader("Feature -> rule -> evidence family -> concern domain -> output trace")
    synthesis = bundle["synthesis"]
    associations = synthesis.literature_linked_associations if synthesis else []
    if associations:
        trace_rows = [
            {"matched_entity_ids": ", ".join(a["matched_entity_ids"]), "rule_id": a["rule_id"],
             "evidence_family": a["evidence_family"], "concern_domain": a["concern_domain"],
             "allowed_output_level": a["allowed_output_level"], "evidence_strength": a["evidence_strength"]}
            for a in associations
        ]
        st.dataframe(trace_rows, width='stretch', hide_index=True)
    else:
        st.caption("(no rule-eligible evidence for this case)")

    st.subheader("Overall Drawing Synthesis (raw)")
    if synthesis:
        with st.expander("overall_synthesis JSON"):
            st.json(synthesis.overall_synthesis)

    st.markdown("---")
    st.subheader("Raw traces (JSON)")

    with st.expander("Raw Observer candidates -> Verifier result"):
        st.json([
            {"observation_id": r["observation_id"], "observer_candidate": r["observer_candidate"],
             "verifier_independent_label": r.get("verifier_independent_label"),
             "verifier_independent_alternative_labels": r.get("verifier_independent_alternative_labels"),
             "verifier_confidence": r.get("verifier_confidence"), "verification_status": r["verification_status"],
             "verifier_notes": r.get("verifier_notes"), "verifier_source_note": r.get("verifier_source_note")}
            for r in rows
        ])

    with st.expander("Entities (post Observer+Verifier merge)"):
        st.json([rdb._entity_to_dict(e) for e in bundle["entities"]])

    with st.expander("Unified evidence (full, incl. deterministic + relationship items)"):
        st.json([ue.to_dict() for ue in synthesis.unified_evidence] if synthesis else [])

    st.subheader("Visual precondition results (semantic rules, from reasoning_chain.py)")
    checks = bundle["checks"]
    by_status: dict[str, list] = {}
    for c in checks:
        by_status.setdefault(c.status, []).append(c)
    for status, group in sorted(by_status.items()):
        with st.expander(f"{status} ({len(group)})"):
            st.json([{"rule_id": c.rule_id, "reason": c.reason, "matched_entity_ids": list(c.matched_entity_ids)}
                      for c in group])

    if bundle["deterministic_features"] is not None:
        with st.expander("Deterministic precondition results (10 composition/line rules, from drawing_synthesis.py)"):
            page_reference = bundle["deterministic_features"].get("page_reference")
            st.caption(
                f"page_reference_mode: `{(page_reference or {}).get('page_reference_mode')}`  |  "
                f"page_relative_features_assessable: `{(page_reference or {}).get('page_relative_features_assessable')}` "
                "-- page-relative rules (size/placement) report `not_assessable`, never a guess, when this is False.")
            det_checks = ds.check_deterministic_preconditions(
                bundle["deterministic_features"]["composition"], bundle["deterministic_features"]["objective_features"],
                page_reference)
            st.json([{"rule_id": c.rule_id, "status": c.status, "reason": c.reason,
                      "matched_entity_ids": list(c.matched_entity_ids)} for c in det_checks])

    with st.expander("Eligible atomic rule matches (semantic only, legacy trace)"):
        doar = bundle["conditions"]["doar_full_pipeline"]
        st.json([rdb._match_to_dict(m) for m in doar["eligible_matches"]])

    st.subheader("Candidate hypotheses + full packages")
    doar = bundle["conditions"]["doar_full_pipeline"]
    if doar["packages"]:
        for pkg in doar["packages"]:
            with st.expander(rdb._hypothesis_to_dict(pkg["hypothesis"]).get("concern_domain", "hypothesis")):
                st.json({
                    "hypothesis": rdb._hypothesis_to_dict(pkg["hypothesis"]),
                    "clinician_package": pkg["clinician_package"],
                    "parent_package": pkg["parent_package"],
                })
    else:
        st.caption("(no hypotheses -- zero candidate hypotheses is a valid, accepted outcome)")

    st.caption(f"case_id={case_id}  |  git_commit={git_commit_sha()}")


def main() -> None:
    st.set_page_config(page_title="DOAR Clinician Review (prototype)", layout="wide")
    st.title("DOAR -- Clinician Review Prototype")
    st.caption(
        "First clinician usability/design session -- NOT clinical validation. Uses cached development-set "
        "results only; no Gemini calls are made while browsing.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = _new_session_id()
    st.sidebar.text_input("Reviewer/session ID (anonymous)", key="session_id")
    session_id = st.session_state.session_id or _new_session_id()

    case_ids = list_case_ids()
    case_id = st.sidebar.selectbox("Case", case_ids, key="selected_case_id")
    st.sidebar.caption(f"{len(case_ids)} development cases available.")

    bundle = load_case_bundle(case_id)
    if bundle is None:
        st.error(f"Unknown case_id: {case_id!r}")
        return

    tab_clinician, tab_parent, tab_technical = st.tabs(
        ["Clinician View", "Parent View Preview", "Technical / Audit View"])
    with tab_clinician:
        render_clinician_view(bundle, case_id, session_id)
    with tab_parent:
        render_parent_view(bundle, case_id, session_id)
    with tab_technical:
        render_technical_view(bundle, case_id)


if __name__ == "__main__":
    main()
