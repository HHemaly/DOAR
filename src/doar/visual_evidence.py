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

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .broad_vocabulary import UNVALIDATED_EXTRA_TARGETS
from .case_output import refresh_module_availability, write_versioned
from .phase2c7.detector_policy import DISABLED, EXPERIMENTAL_AUTOMATIC, VALIDATED_AUTOMATIC, full_policy
from .phase2c7.visual_detector import analyze_image

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


@dataclass(frozen=True)
class VisualFinding:
    label: str
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
            "label": self.label, "free_form_label": self.free_form_label, "bbox": self.bbox,
            "confidence": self.confidence, "detector": self.detector, "checkpoint": self.checkpoint,
            "prompt": self.prompt, "validation_status": self.validation_status,
            "evidence_status": self.evidence_status, "rule_mapping_status": self.rule_mapping_status,
            "related_rule_ids": list(self.related_rule_ids), "source": self.source, "query": self.query,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "VisualFinding":
        return VisualFinding(
            label=d["label"], free_form_label=d.get("free_form_label"),
            bbox=tuple(d["bbox"]) if d.get("bbox") else None, confidence=d["confidence"],
            detector=d["detector"], checkpoint=d["checkpoint"], prompt=d["prompt"],
            validation_status=d["validation_status"], evidence_status=d["evidence_status"],
            rule_mapping_status=d["rule_mapping_status"],
            related_rule_ids=tuple(d.get("related_rule_ids") or ()),
            source=d.get("source", "initial_scan"), query=d.get("query"),
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
    return VisualFinding(
        label=label, free_form_label=(label if label not in policy else None), bbox=bbox,
        confidence=confidence, detector=model, checkpoint=checkpoint, prompt=prompt,
        validation_status=validation_status, evidence_status=_EVIDENCE_STATUS_FOR[validation_status],
        rule_mapping_status=rule_status, related_rule_ids=rule_ids, source=source, query=query,
        timestamp=utc_now_iso(),
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


def save_detections(case_dir: str | Path, findings: list[VisualFinding], *, status: str = "available") -> None:
    doc = {"status": status, "findings": [f.to_dict() for f in findings],
           "n_findings": len(findings), "generated_at": utc_now_iso()}
    write_versioned(Path(case_dir) / "detections.json", doc)


def run_and_persist_initial_scan(case_dir: str | Path, image_path: str, *, eye_entry, registry_v2: dict,
                                  model_predict_fns: dict[str, Callable]) -> list[VisualFinding]:
    """The one function the app calls after `analyze_image_with_timing` to
    replace the honest `{"status": "unavailable"}` stub finalize_case
    writes by default with real findings. Never called by
    `case_output.finalize_case` itself (kept untouched, zero risk to its
    existing tests) -- purely additive, a separate step."""
    findings = run_initial_visual_scan(image_path, eye_entry=eye_entry, registry_v2=registry_v2,
                                        model_predict_fns=model_predict_fns)
    save_detections(case_dir, findings)
    refresh_module_availability(Path(case_dir), detection="available")
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
