"""Converters from existing, real evidence outputs into evidence-v2 records
(`trace_evidence.py`). Purely additive: nothing here changes what
`schemas.py::Evidence` or `features.py::FeatureValue` write; these
functions only build a second, parallel v2 view over the same real data.
"""

from __future__ import annotations

from .features import FeatureValue
from .schemas import Evidence
from .trace_evidence import EvidenceRecordV2, make_provenance

FEATURES_SCHEMA_VERSION = "3.1.0"  # matches FeatureValue.version default


def feature_value_to_v2(
    feature_id: str, feature: FeatureValue, *, image_sha256: str | None = None,
) -> EvidenceRecordV2:
    """One objective feature (features.py) -> one EvidenceRecordV2.

    Status mapping (judge-style verdict, not availability):
      - honestly-missing features (no detector) -> ABSTAIN, value=None
      - a computed-but-non-finite value (should not happen post-features.py,
        but defensively checked) -> FAIL
      - a real, computed value -> PASS if confidence >= 0.5, else WARN
        (mirrors the same 0.5 threshold `evidence_adapter.py` uses for
        available-vs-experimental, kept consistent across both schemas)
    """
    limitations: list[str] = []
    if feature.missing:
        status = "ABSTAIN"
        value: float | None = None
        limitations.append("No extractor/detector exists for this feature in this release.")
    else:
        status = "PASS" if feature.confidence >= 0.5 else "WARN"
        value = feature.value
        if status == "WARN":
            limitations.append("Low upstream confidence (segmentation/quality) for this measurement.")
    provenance = make_provenance(
        pipeline_stage="objective_feature_extraction",
        schema_version=FEATURES_SCHEMA_VERSION,
        input_hashes={"image_sha256": image_sha256} if image_sha256 else {},
    )
    return EvidenceRecordV2(
        evidence_id=feature.evidence_id,
        evidence_type="objective_feature",
        producer="features.py::objective_feature_row",
        producer_version=feature.version,
        value=value,
        unit=None,  # features.py does not track physical units today -- honestly None, not invented
        confidence=None if feature.missing else feature.confidence,
        method=feature.method,
        status=status,
        limitations=limitations,
        provenance=provenance,
        source_evidence_ids=[],
    )


def legacy_evidence_to_v2(item: Evidence, *, image_sha256: str | None = None) -> EvidenceRecordV2:
    """One legacy `schemas.py::Evidence` record -> one EvidenceRecordV2.
    Legacy Evidence has no explicit availability flag, so status is always
    PASS (it was successfully computed by definition -- Evidence is only
    ever constructed with a real value in the existing pipeline)."""
    provenance = make_provenance(
        pipeline_stage="analysis_pipeline",
        schema_version="schemas.Evidence",
        input_hashes={"image_sha256": image_sha256} if image_sha256 else {},
    )
    return EvidenceRecordV2(
        evidence_id=item.evidence_id,
        evidence_type=item.kind,
        producer="analysis.py",
        producer_version="3.0.0",
        value=item.value,
        unit=None,
        confidence=item.confidence,
        method=item.method,
        status="PASS",
        limitations=list(item.limitations),
        provenance=provenance,
        source_evidence_ids=[],
    )
