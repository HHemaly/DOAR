"""Evidence schema v2 (DOAR-TRACE Phase 1, Section 4A).

An additive schema, separate from `schemas.py::Evidence` (the original,
simple dataclass every existing output already uses) and from
`evidence_schema.py::EvidenceItem` (this session's earlier
extractor-availability schema). Nothing in this module changes what
either of those write or how any existing reader consumes them — see
`CURRENT_TO_TARGET_GAP_V2.md` for why a third schema exists rather than a
rename/merge of the other two.

`EvidenceRecordV2.status` is a **judge-style verdict on the evidence
itself** — was this specific measurement trustworthy enough to build a
claim on — not "was it ever computed." The four values:

    PASS    -- the evidence is trustworthy as measured
    WARN    -- usable, but with a caveat the caller must not omit
    FAIL    -- an extractor ran and produced a value that should not be trusted
    ABSTAIN -- no defensible measurement exists (no extractor, or the
               question the evidence would answer isn't observable here)

`source_evidence_ids` lets one v2 record be *derived from* others (e.g. a
theme-level evidence record built by aggregating rule-level records) while
keeping every derivation traceable back to its inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

STATUSES = frozenset({"PASS", "WARN", "FAIL", "ABSTAIN"})


@dataclass(frozen=True)
class Provenance:
    """Where a v2 evidence record came from -- always present, never optional,
    so a record can never claim a value without saying how it was produced."""

    pipeline_stage: str          # e.g. "objective_feature_extraction", "rule_engine_v2"
    schema_version: str          # e.g. "3.1.0" (features.py), "doar_trace_evidence_v2"
    generated_at: str            # ISO-8601 UTC timestamp
    input_hashes: dict[str, str] = field(default_factory=dict)  # e.g. {"image_sha256": "..."}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecordV2:
    evidence_id: str
    evidence_type: str            # e.g. "objective_feature", "rule_evaluation", "model_prediction", "aggregated_theme"
    producer: str                 # module/function name that produced this record
    producer_version: str
    value: Any
    unit: str | None
    confidence: float | None
    method: str
    status: str
    limitations: list[str]
    provenance: Provenance
    source_evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"EvidenceRecordV2 {self.evidence_id!r}: status {self.status!r} "
                f"not in {sorted(STATUSES)}"
            )
        if self.status == "ABSTAIN" and self.value is not None:
            raise ValueError(
                f"EvidenceRecordV2 {self.evidence_id!r} has status ABSTAIN but a "
                f"non-None value ({self.value!r}) -- ABSTAIN means no defensible "
                "measurement exists and must never carry a value."
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"EvidenceRecordV2 {self.evidence_id!r}: confidence {self.confidence!r} out of [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance"] = self.provenance.to_dict()
        return data


def make_provenance(pipeline_stage: str, schema_version: str, input_hashes: dict[str, str] | None = None) -> Provenance:
    return Provenance(
        pipeline_stage=pipeline_stage,
        schema_version=schema_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        input_hashes=input_hashes or {},
    )


@dataclass
class EvidenceSetV2:
    """A validated, indexed collection of v2 evidence records."""

    records: list[EvidenceRecordV2]

    def __post_init__(self) -> None:
        ids = [r.evidence_id for r in self.records]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate evidence_id(s): {sorted(duplicates)}")
        known = set(ids)
        for record in self.records:
            unknown = [sid for sid in record.source_evidence_ids if sid not in known]
            if unknown:
                raise ValueError(
                    f"EvidenceRecordV2 {record.evidence_id!r} references unknown "
                    f"source_evidence_ids: {unknown}"
                )

    def by_id(self, evidence_id: str) -> EvidenceRecordV2 | None:
        for record in self.records:
            if record.evidence_id == evidence_id:
                return record
        return None

    def by_type(self, evidence_type: str) -> list[EvidenceRecordV2]:
        return [r for r in self.records if r.evidence_type == evidence_type]

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]
