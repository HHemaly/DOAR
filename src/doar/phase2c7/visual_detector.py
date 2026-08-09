"""Phase 2C.7 Stage 6: the one unified, image-only automatic visual
detector API.

    records = analyze_image(image_path, policy, model_predict_fns={...})

The caller supplies only a path to an image (no annotation, no pilot ID,
no ground truth) and a `{model_name: predict_fn}` mapping of already-
loaded, already-frozen prediction functions (see
`scripts/phase2c7_run_sanity_check.py` for the real-weight wiring --
mirrors every other phase's injectable-backend pattern: a real
`load_real_*` loader is the only code that touches real model weights,
never called by the test suite). The caller never chooses which
underlying model (Grounding DINO vs OWLv2) answers for a given target --
each target's own frozen `detector_policy.TargetPolicyEntry.best_model`
decides that internally. Routing is BY MODEL, not by a hardcoded
object/part split: `heart` freezes to OWLv2 while every other object
class freezes to Grounding DINO (see `detector_policy.py`), and this
function calls each distinct model at most once per image regardless of
how many targets route to it.

A `DISABLED` target is never sent to a model at all (no wasted inference,
no chance of its output leaking downstream). A `VALIDATED_AUTOMATIC`
target's record carries `evidence_status="validated_evidence"`; an
`EXPERIMENTAL_AUTOMATIC` target's record carries
`evidence_status="experimental_evidence_technical_view_only"` -- the
single field every downstream consumer (a future rule-integration layer)
must check before treating a record as usable for a psychological
conclusion. This module does not itself have a notion of "psychological
conclusion" at all -- it stops at structured visual evidence, exactly the
existing project-wide boundary between detection and interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .detector_policy import DISABLED, TargetPolicyEntry

# Documentation-only policy rows -- never independently dispatched to a
# model (their evidence is the same underlying detection as another
# target, used differently downstream; see detector_policy.py's own
# docstring for "person_part_reference").
NON_DISPATCHABLE_TARGETS = frozenset({"person_part_reference"})

EVIDENCE_STATUS_BY_VALIDATION = {
    "VALIDATED_AUTOMATIC": "validated_evidence",
    "EXPERIMENTAL_AUTOMATIC": "experimental_evidence_technical_view_only",
    "DISABLED": "not_used",
}

# image_path -> {target: (detected: bool, score: float, bbox_or_None)}
PredictFn = Callable[[str], dict]


@dataclass(frozen=True)
class VisualEvidenceRecord:
    target: str
    present: bool
    bbox: tuple[float, float, float, float] | None
    confidence: float
    model: str
    checkpoint: str
    threshold: float | None
    validation_status: str
    evidence_status: str

    def to_dict(self) -> dict:
        return {
            "target": self.target, "present": self.present, "bbox": self.bbox,
            "confidence": self.confidence, "model": self.model, "checkpoint": self.checkpoint,
            "threshold": self.threshold, "validation_status": self.validation_status,
            "evidence_status": self.evidence_status,
        }


def analyze_image(image_path: str, policy: dict[str, TargetPolicyEntry], *,
                   model_predict_fns: dict[str, PredictFn]) -> list[VisualEvidenceRecord]:
    """`image_path`: the ONLY thing this function needs about the image --
    no pilot ID, no store, no annotation lookup. Returns one
    `VisualEvidenceRecord` per non-`DISABLED`, dispatchable target in
    `policy`. A `DISABLED` target contributes nothing (no record at all,
    not even a `present=False` one) -- it was never sent to a model.
    Each distinct model in `model_predict_fns` is called at most once,
    regardless of how many targets route to it (results cached per model
    for the duration of this single image)."""
    dispatchable = {t: e for t, e in policy.items()
                     if e.status != DISABLED and t not in NON_DISPATCHABLE_TARGETS}

    predictions_by_model: dict[str, dict] = {}
    records = []
    for target, entry in dispatchable.items():
        model_name = entry.best_model
        if model_name not in predictions_by_model:
            predict_fn = model_predict_fns.get(model_name)
            predictions_by_model[model_name] = predict_fn(image_path) if predict_fn else {}
        detected, score, bbox = predictions_by_model[model_name].get(target, (False, 0.0, None))
        records.append(VisualEvidenceRecord(
            target=target, present=bool(detected), bbox=bbox, confidence=float(score),
            model=entry.best_model or "", checkpoint=entry.best_model_checkpoint or "",
            threshold=entry.threshold, validation_status=entry.status,
            evidence_status=EVIDENCE_STATUS_BY_VALIDATION[entry.status],
        ))
    return records


def validated_records_only(records: list[VisualEvidenceRecord]) -> list[VisualEvidenceRecord]:
    """The one function a future rule-integration layer should call before
    treating anything here as usable evidence for a psychological
    conclusion -- filters out every experimental/disabled record."""
    return [r for r in records if r.evidence_status == "validated_evidence"]
