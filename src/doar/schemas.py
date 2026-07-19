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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [asdict(item) for item in self.evidence]
        return data
