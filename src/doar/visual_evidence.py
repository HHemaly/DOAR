"""DOAR MVP: broad visual-object evidence store.

Extends Phase 2C.7's `validation_status`/`evidence_status` vocabulary
(never redefines it) with two new, additive concepts this phase's
architecture explicitly requires:

- `UNKNOWN` validation_status: a detection for a target that has NEVER
  been individually evaluated against human ground truth (the broad-scan
  "common child-drawing content" extras, or any on-demand search result)
  -- distinct from `EXPERIMENTAL` (a Phase 2C.7 frozen-policy target that
  WAS evaluated, just not well enough to validate).
- `rule_mapping_status` (`MAPPED`/`UNMAPPED`): whether any rule in the
  registry currently references this label at all -- ORTHOGONAL to
  validation. An `UNMAPPED` finding is never discarded; it is preserved
  as evidence for the Technical View and Q&A, exactly per instruction.

`DISABLED_FOR_RULES` corresponds to Phase 2C.7's `DISABLED` (circle,
vehicle -- measured at or below chance; the detector is never even run
for these two, exactly as Phase 2C.7 established, not re-litigated here).

Nothing in this module ever assigns a `related_rule_ids` match to
`evidence_status="validated_evidence"` -- only a target whose Phase 2C.7
`validation_status` is `VALIDATED_AUTOMATIC` can ever produce
`validated_evidence`. Rule mapping never upgrades trust.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .broad_vocabulary import UNVALIDATED_EXTRA_TARGETS
from .case_output import refresh_module_availability, write_versioned
from .phase2c7.detector_policy import DISABLED, EXPERIMENTAL_AUTOMATIC, VALIDATED_AUTOMATIC, full_policy
from .phase2c7.visual_detector import analyze_image
from .schemas import Evidence

UNKNOWN = "UNKNOWN"
DISABLED_FOR_RULES = "DISABLED_FOR_RULES"
VALIDATED = "VALIDATED"
EXPERIMENTAL = "EXPERIMENTAL"

_VALIDATION_STATUS_MAP = {
    VALIDATED_AUTOMATIC: VALIDATED,
    EXPERIMENTAL_AUTOMATIC: EXPERIMENTAL,
    DISABLED: DISABLED_FOR_RULES,
}
_EVIDENCE_STATUS_FOR = {
    VALIDATED: "validated_evidence",
    EXPERIMENTAL: "experimental_evidence_technical_view_only",
    UNKNOWN: "experimental_evidence_technical_view_only",
    DISABLED_FOR_RULES: "not_used",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_finding_id(*, label: str, source: str, query: str | None, detector: str,
                         timestamp: str, bbox, confidence: float) -> str:
    """Deterministic, stable "source finding ID" -- reused by `from_dict`
    when loading an OLDER detections.json written before this field
    existed (so old cases reload without crashing; see
    `test_visual_evidence.py`'s backward-compatibility test), and used at
    construction time by `build_finding`. Not a database primary key --
    just a stable string the canonical Evidence adapter and rule engine
    can cite as provenance back to the exact detection that produced it."""
    raw = f"{label}|{source}|{query or ''}|{detector}|{timestamp}|{bbox}|{confidence}"
    return "vf_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class VisualFinding:
    label: str
    finding_id: str                 # stable provenance id -- see _compute_finding_id
    free_form_label: str | None
    bbox: tuple[float, float, float, float] | None
    confidence: float
    detector: str
    checkpoint: str
    prompt: str
    validation_status: str          # VALIDATED | EXPERIMENTAL | UNKNOWN | DISABLED_FOR_RULES
    evidence_status: str            # validated_evidence | experimental_evidence_technical_view_only | not_used
    rule_mapping_status: str        # MAPPED | UNMAPPED
    related_rule_ids: tuple[str, ...]
    source: str                     # "initial_scan" | "on_demand_search"
    query: str | None               # the exact on-demand question/query text, if source == "on_demand_search"
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "label": self.label, "finding_id": self.finding_id,
            "free_form_label": self.free_form_label, "bbox": self.bbox,
            "confidence": self.confidence, "detector": self.detector, "checkpoint": self.checkpoint,
            "prompt": self.prompt, "validation_status": self.validation_status,
            "evidence_status": self.evidence_status, "rule_mapping_status": self.rule_mapping_status,
            "related_rule_ids": list(self.related_rule_ids), "source": self.source, "query": self.query,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "VisualFinding":
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        source = d.get("source", "initial_scan")
        finding_id = d.get("finding_id") or _compute_finding_id(
            label=d["label"], source=source, query=d.get("query"), detector=d["detector"],
            timestamp=d.get("timestamp", ""), bbox=bbox, confidence=d["confidence"])
        return VisualFinding(
            label=d["label"], finding_id=finding_id, free_form_label=d.get("free_form_label"),
            bbox=bbox, confidence=d["confidence"],
            detector=d["detector"], checkpoint=d["checkpoint"], prompt=d["prompt"],
            validation_status=d["validation_status"], evidence_status=d["evidence_status"],
            rule_mapping_status=d["rule_mapping_status"],
            related_rule_ids=tuple(d.get("related_rule_ids") or ()),
            source=source, query=d.get("query"),
            timestamp=d.get("timestamp", ""),
        )


_WORD_RE = re.compile(r"[a-z]{4,}")


def compute_rule_mapping(label: str, registry_v2: dict, policy: dict) -> tuple[str, tuple[str, ...]]:
    """MAPPED/UNMAPPED + which rule_ids. Frozen-policy targets use their
    own curated `related_rule_ids` (exact, from Phase 2C.7). Any other
    label (broad-scan extras, on-demand queries) falls back to a
    documented HEURISTIC: does the label's own significant words appear
    in a rule's `observable` or `faithful_source_quote` text? This is a
    coarse text match, not an authoritative mapping -- it never upgrades
    a finding's trustworthiness (see module docstring), only whether it
    is worth surfacing as "a rule mentions this" in the Technical View."""
    if label in policy:
        ids = policy[label].related_rule_ids
        return ("MAPPED" if ids else "UNMAPPED"), ids
    label_words = set(_WORD_RE.findall(label.lower()))
    if not label_words:
        return "UNMAPPED", ()
    matched = []
    for rule in registry_v2.get("rules", []):
        haystack = f"{rule.get('observable', '')} {rule.get('faithful_source_quote', '')}".lower()
        haystack_words = set(_WORD_RE.findall(haystack))
        if label_words & haystack_words:
            matched.append(rule["rule_id"])
    return ("MAPPED" if matched else "UNMAPPED"), tuple(sorted(set(matched)))


def build_finding(label: str, *, detected: bool, confidence: float, bbox, model: str, checkpoint: str,
                   prompt: str, policy: dict, registry_v2: dict, source: str = "initial_scan",
                   query: str | None = None) -> VisualFinding | None:
    """Returns None for a non-detection (nothing to store as a finding --
    absence is not itself evidence in this schema, matching the rest of
    the project's convention)."""
    if not detected:
        return None
    if label in policy:
        entry = policy[label]
        validation_status = _VALIDATION_STATUS_MAP[entry.status]
    else:
        validation_status = UNKNOWN
    rule_status, rule_ids = compute_rule_mapping(label, registry_v2, policy)
    timestamp = utc_now_iso()
    finding_id = _compute_finding_id(label=label, source=source, query=query, detector=model,
                                      timestamp=timestamp, bbox=bbox, confidence=confidence)
    return VisualFinding(
        label=label, finding_id=finding_id, free_form_label=(label if label not in policy else None),
        bbox=bbox, confidence=confidence, detector=model, checkpoint=checkpoint, prompt=prompt,
        validation_status=validation_status, evidence_status=_EVIDENCE_STATUS_FOR[validation_status],
        rule_mapping_status=rule_status, related_rule_ids=rule_ids, source=source, query=query,
        timestamp=timestamp,
    )


def run_initial_visual_scan(image_path: str, *, eye_entry, registry_v2: dict,
                             model_predict_fns: dict[str, Callable]) -> list[VisualFinding]:
    """The broad initial scan: every Phase 2C.7 frozen-policy target
    (DISABLED targets are silently skipped -- never run, per Phase 2C.7's
    own finding that they carry no usable signal) PLUS the unvalidated
    common-content extras, all attempted, all preserved."""
    policy = full_policy(eye_entry)
    records = analyze_image(image_path, policy, model_predict_fns=model_predict_fns)
    findings = []
    for r in records:
        f = build_finding(r.target, detected=r.present, confidence=r.confidence, bbox=r.bbox,
                           model=r.model, checkpoint=r.checkpoint, prompt=(policy[r.target].prompt or ""),
                           policy=policy, registry_v2=registry_v2, source="initial_scan")
        if f is not None:
            findings.append(f)

    extra_predict_fn = model_predict_fns.get("open_vocab_query")
    if extra_predict_fn is not None:
        for target in UNVALIDATED_EXTRA_TARGETS:
            hits = extra_predict_fn(image_path, target)
            for detected, confidence, bbox, model_name, checkpoint, prompt in hits:
                f = build_finding(target, detected=detected, confidence=confidence, bbox=bbox,
                                   model=model_name, checkpoint=checkpoint, prompt=prompt,
                                   policy=policy, registry_v2=registry_v2, source="initial_scan")
                if f is not None:
                    findings.append(f)
    return findings


def load_detections(case_dir: str | Path) -> list[VisualFinding]:
    path = Path(case_dir) / "detections.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [VisualFinding.from_dict(d) for d in doc.get("findings", [])]


def load_entities(case_dir: str | Path) -> list:
    """The Visual Knowledge V2 counterpart to `load_detections` -- reads
    the `"entities"` key `save_detections` adds when given `entities=...`.
    An older case (or one whose scan predates this phase) simply has no
    `"entities"` key -- returns `[]`, never an error; `load_detections`
    (unchanged) remains the source of truth for any older case."""
    from .visual_entity import VisualEntity
    path = Path(case_dir) / "detections.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [VisualEntity.from_dict(d) for d in doc.get("entities", [])]


def save_detections(case_dir: str | Path, findings: list[VisualFinding], *, status: str = "available",
                     entities: list | None = None) -> None:
    """`entities` is optional and purely additive: omitted (the default),
    the written document is byte-for-byte identical to before this phase
    -- every existing caller (`search_visual`, and any test written before
    Visual Knowledge V2) is unaffected. When provided, adds `"entities"`/
    `"n_entities"` alongside the unchanged `"findings"`/`"n_findings"`."""
    doc = {"status": status, "findings": [f.to_dict() for f in findings],
           "n_findings": len(findings), "generated_at": utc_now_iso()}
    if entities is not None:
        doc["entities"] = [e.to_dict() for e in entities]
        doc["n_entities"] = len(entities)
    write_versioned(Path(case_dir) / "detections.json", doc)


def update_entities(case_dir: str | Path, entities: list) -> None:
    """Re-persists just the `"entities"`/`"n_entities"` keys of an
    already-written `detections.json`, leaving `"findings"`/`"status"`/
    `"generated_at"` untouched -- for a later re-verification pass (e.g.
    a new `VisualVerifier` run) that doesn't need to redo the scan
    itself. No-op if `detections.json` doesn't exist yet."""
    path = Path(case_dir) / "detections.json"
    if not path.exists():
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["entities"] = [e.to_dict() for e in entities]
    doc["n_entities"] = len(entities)
    write_versioned(path, doc)


def run_and_persist_initial_scan(case_dir: str | Path, image_path: str, *, eye_entry, registry_v2: dict,
                                  model_predict_fns: dict[str, Callable], observer=None,
                                  verifier=None) -> list[VisualFinding]:
    """The one function the app calls after `analyze_image_with_timing` to
    replace the honest `{"status": "unavailable"}` stub finalize_case
    writes by default with real findings. Never called by
    `case_output.finalize_case` itself (kept untouched, zero risk to its
    existing tests) -- purely additive, a separate step.

    Also the ONLY caller of `integrate_visual_findings_into_case` -- the
    real rule-engine connection runs automatically right after every
    initial scan, never after an on-demand search.

    Also builds and persists the richer `VisualEntity` records (Visual
    Knowledge V2) alongside the unchanged `VisualFinding` list -- the
    rule engine/Q&A/`find_matching` all keep reading `findings` exactly
    as before; `entities` is additive, read by Technical View and the
    new alias-aware search only.

    `observer`/`verifier` (Visual Resolver phase) are OPTIONAL and default
    to `None` -- omitted, behavior is identical to before this phase.
    When supplied (a `visual_observer.VisualObserver`/`VisualVerifier`,
    e.g. `CallableVisualObserver`/`CallableVisualVerifier` in tests, or a
    real provider once one exists), the pipeline additionally: (1) merges
    the observer's structured candidates into `entities` -- new,
    observer-only entities get `model_validation_status="UNKNOWN"` and
    `source="visual_observer"`, and have NO backing `VisualFinding`, so
    they are structurally invisible to `integrate_visual_findings_into_
    case`/the rule engine, exactly like every other UNKNOWN-status
    entity; (2) saves a real crop image for every entity with a bbox
    (`crop_ref`); (3) runs the verifier per entity, updating ONLY
    `case_verification_status`. Expert review (read from
    `clinician_review.json`, if any) is applied LAST, so a human
    correction always overrides an automatic verifier's verdict."""
    findings = run_initial_visual_scan(image_path, eye_entry=eye_entry, registry_v2=registry_v2,
                                        model_predict_fns=model_predict_fns)

    from .expert_review import load_review
    from .visual_entity import (
        apply_expert_review_to_entities, apply_verifier_to_entities, build_entities_from_findings,
        merge_observer_candidates_into_entities, populate_crop_refs,
    )
    entities = build_entities_from_findings(findings, image_path=image_path)

    if observer is not None:
        candidates = observer.analyze(image_path)
        entities = merge_observer_candidates_into_entities(entities, candidates, registry_v2=registry_v2)

    entities = populate_crop_refs(entities, image_path, case_dir)

    if verifier is not None:
        entities = apply_verifier_to_entities(entities, image_path, verifier)

    review_history = load_review(case_dir).get("history", [])
    if review_history:
        entities = apply_expert_review_to_entities(entities, review_history)

    save_detections(case_dir, findings, entities=entities)
    refresh_module_availability(Path(case_dir), detection="available", visual_detection="available",
                                 open_world_search="available")
    integrate_visual_findings_into_case(case_dir, findings, registry_v2=registry_v2)
    return findings


def build_rule_evidence_trace(findings: list[VisualFinding]) -> list[dict]:
    """Groups findings by every `related_rule_ids` entry they carry --
    ONE row per (rule_id, finding) pair. `can_activate` is True only for
    `evidence_status == "validated_evidence"` findings; experimental/
    unknown findings always appear with `can_activate=False` so the
    Technical View can show "a rule mentions this" without implying the
    rule was, or could be, triggered by it. Never triggers a rule itself --
    purely descriptive, for Technical View / expert review."""
    rows = []
    for f in findings:
        for rule_id in f.related_rule_ids:
            rows.append({
                "rule_id": rule_id, "label": f.label, "confidence": f.confidence,
                "validation_status": f.validation_status, "evidence_status": f.evidence_status,
                "can_activate": f.evidence_status == "validated_evidence",
                "source": f.source, "detector": f.detector,
            })
    return rows


def visual_finding_to_evidence(finding: VisualFinding) -> Evidence:
    """The canonical-evidence adapter: every VisualFinding becomes exactly
    one `schemas.Evidence` record (the SAME dataclass every other part of
    DOAR uses -- composition/colour/emotion evidence, no third schema
    invented here). Full technical traceability regardless of validation
    status (Technical View/Q&A need to reference ANY finding's evidence
    record, not just validated ones) -- but `rule_eligible` (baked into
    `value`, never a separate trust channel a caller could miss) is True
    ONLY when `evidence_status == "validated_evidence"`. This is the ONE
    field `rule_engine_v2.evaluate_visual_object_presence_rules` checks
    before ever using a visual evidence record -- experimental/unknown/
    disabled findings always get `rule_eligible=False` and can never
    satisfy a rule, no matter how confident the detection."""
    rule_eligible = finding.evidence_status == "validated_evidence"
    limitations = []
    if finding.validation_status == "EXPERIMENTAL":
        limitations.append("Experimental visual evidence -- must not activate a psychological rule.")
    elif finding.validation_status == "UNKNOWN":
        limitations.append("Unvalidated visual evidence (never individually measured against human "
                            "ground truth) -- must not activate a psychological rule.")
    elif finding.validation_status == "DISABLED_FOR_RULES":
        limitations.append("Detector disabled for this target -- must not enter rule reasoning.")
    return Evidence(
        evidence_id=f"ev_visual_{finding.finding_id}",
        kind="visual_detection",
        value={
            "label": finding.label, "free_form_label": finding.free_form_label, "bbox": finding.bbox,
            "validation_status": finding.validation_status, "rule_mapping_status": finding.rule_mapping_status,
            "related_rule_ids": list(finding.related_rule_ids), "source": finding.source,
            "query": finding.query, "prompt": finding.prompt, "source_finding_id": finding.finding_id,
            "rule_eligible": rule_eligible,
        },
        method=f"{finding.detector}:{finding.checkpoint}" if finding.checkpoint else finding.detector,
        confidence=finding.confidence,
        limitations=limitations,
    )


def visual_findings_to_evidence(findings: list[VisualFinding]) -> list[Evidence]:
    return [visual_finding_to_evidence(f) for f in findings]


def rule_eligible_visual_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """The one function a rule evaluator should call before using any
    canonical evidence for a psychological rule -- filters to only
    visual-detection records whose source finding was VALIDATED."""
    return [e for e in evidence if e.kind == "visual_detection" and e.value.get("rule_eligible")]


def integrate_visual_findings_into_case(case_dir: str | Path, findings: list[VisualFinding], *,
                                         registry_v2: dict) -> list[dict]:
    """The missing connection: converts this scan's findings into canonical
    Evidence, lets the REAL rule engine (rule_engine_v2.py) see the
    rule-eligible ones, merges any resulting rule_evaluations into
    analysis.json, and re-runs the existing synthesis tail (judges,
    structured_analysis, judges_v2, generated_claims, verification_report)
    via case_output.resynthesize_case_with_visual_evidence -- so a
    legitimately triggered visual rule reaches aggregation/Parent
    View/Technical View exactly like any other rule already does, no
    special-casing downstream.

    Called ONLY from `run_and_persist_initial_scan` (the initial scan),
    NEVER from `search_visual` (on-demand search) -- an on-demand finding
    must not immediately activate a rule, per instruction, even if it
    happens to match a VALIDATED target.

    Returns the new visual-derived rule_evaluations (usually empty: as of
    this session every static_detector rule in rules_registry_v2.json has
    allowed_output_level=disabled, so this correctly produces nothing to
    merge today -- see rule_engine_v2.evaluate_visual_object_presence_rules's
    own docstring for the registry-gate reasoning). No-op if analysis.json
    doesn't exist yet (case not finalized)."""
    from .case_output import resynthesize_case_with_visual_evidence
    from .rule_engine_v2 import evaluate_visual_object_presence_rules

    case_dir = Path(case_dir)
    analysis_path = case_dir / "analysis.json"
    if not analysis_path.exists():
        return []
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    visual_evidence_records = visual_findings_to_evidence(findings)
    rules_v2_by_id = {r["rule_id"]: r for r in registry_v2["rules"]}
    new_rule_evaluations = evaluate_visual_object_presence_rules(rules_v2_by_id, visual_evidence_records)

    analysis["evidence"] = analysis.get("evidence", []) + [asdict(e) for e in visual_evidence_records]
    if new_rule_evaluations:
        analysis["rule_evaluations"] = analysis.get("rule_evaluations", []) + new_rule_evaluations

    write_versioned(analysis_path, analysis)
    resynthesize_case_with_visual_evidence(analysis, case_dir, registry_v2=registry_v2)
    return new_rule_evaluations


def find_matching(findings: list[VisualFinding], query_target: str) -> list[VisualFinding]:
    """Case-insensitive exact-or-substring match against `label` and
    `free_form_label` -- used both by Q&A's "check existing evidence
    first" step and by tests."""
    q = query_target.strip().lower()
    return [f for f in findings
            if q == f.label.lower() or q in f.label.lower() or f.label.lower() in q
            or (f.free_form_label and (q == f.free_form_label.lower() or q in f.free_form_label.lower()))]


def search_visual(case_dir: str | Path, query: str, *, registry_v2: dict,
                   open_vocab_predict_fn: Callable) -> VisualFinding | None:
    """On-demand visual search: runs ONE live open-vocabulary query
    against the case's already-uploaded image for `query`, appends
    whatever is found (or nothing, if absent) to the case's persisted
    detections -- traceable case evidence from then on, exactly like an
    initial-scan finding, always `source='on_demand_search'`. Never
    upgrades an on-demand hit to VALIDATED -- these are always UNKNOWN
    unless `query` happens to exactly name an already-frozen-policy
    target (in which case that target's own real validation_status
    applies, since it IS the same measured detector/target)."""
    case_dir = Path(case_dir)
    image_candidates = list(case_dir.glob("*.png")) + list(case_dir.glob("*.jpg")) + list(case_dir.glob("*.jpeg"))
    if not image_candidates:
        return None
    image_path = str(image_candidates[0])

    existing = load_detections(case_dir)
    hits = open_vocab_predict_fn(image_path, query)
    from .phase2c7.detector_policy import OBJECT_CLASS_POLICY
    policy = dict(OBJECT_CLASS_POLICY)
    new_finding = None
    for detected, confidence, bbox, model_name, checkpoint, prompt in hits:
        f = build_finding(query.strip().lower(), detected=detected, confidence=confidence, bbox=bbox,
                           model=model_name, checkpoint=checkpoint, prompt=prompt, policy=policy,
                           registry_v2=registry_v2, source="on_demand_search", query=query)
        if f is not None:
            new_finding = f
            break
    if new_finding is not None:
        existing.append(new_finding)
        save_detections(case_dir, existing)
    return new_finding
