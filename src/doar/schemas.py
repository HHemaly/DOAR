from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: str
    value: Any
    method: str
    confidence: float
    limitations: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    schema_version: str
    image_path: str
    quality: dict[str, Any]
    segmentation: dict[str, Any]
    composition: dict[str, Any]
    colour: dict[str, Any]
    emotion: dict[str, Any]
    evidence: list[Evidence]
    rule_evaluations: list[dict[str, Any]]
    concerns: list[dict[str, Any]]
    safety_disclaimer: str
    artifacts: dict[str, str]
    module_execution: dict[str, Any] = field(default_factory=dict)
    label_provenance: dict[str, Any] = field(default_factory=dict)
    # DOAR-TRACE 4B: objective_feature_row's output, persisted for EVERY
    # analyze_image run (previously computed only inside emotion.py's
    # fusion-checkpoint branch -- see CURRENT_TO_TARGET_GAP_V2.md). Keys are
    # feature_id strings, values are features.py::FeatureValue instances;
    # kept as dict[str, Any] here (not FeatureValue) to avoid a schemas.py
    # -> features.py import for a dataclass this module never constructs.
    objective_features: dict[str, Any] = field(default_factory=dict)
    # DOAR-TRACE Phase 2A Section 3: page_frame.py's assessment, computed
    # for EVERY analyze_image run. Gates page-relative rule evaluation in
    # structured_report.py -- see docs/PAGE_FRAME_ASSESSABILITY.md.
    page_frame: dict[str, Any] = field(default_factory=dict)
    # DOAR-TRACE Phase 2A.1 Section 3: page_reference.py's resolved
    # PageReference (mode, polygon, confidence, provenance) -- the
    # explicit "what counts as the page" decision that page_frame.py's
    # status alone never captured. See docs/PAGE_REFERENCE_MODEL.md.
    page_reference: dict[str, Any] = field(default_factory=dict)
    # DOAR-TRACE Phase 2A.1 Section 6: canonical_input.py's
    # resolution-normalized counterparts for the specific features Phase
    # 2A found resize-sensitive -- SEPARATE from objective_features
    # (which always reflects the original, unmodified upload). See
    # docs/INPUT_NORMALIZATION_POLICY.md.
    canonical_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [asdict(item) for item in self.evidence]
        return data
