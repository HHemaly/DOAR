"""Milestone 1 -- live-case -> Human Interaction Layer bundle adapter.

Builds the SAME bundle shape `scripts/clinician_review_app.py::
load_case_bundle` produces for cached dev-set cases, but from a REAL,
arbitrary live case directory written by `doar_prototype_app.py`'s normal
`analyze_image`/`run_and_persist_initial_scan` path.

Reuses `drawing_synthesis.synthesize_drawing()` and
`VisualEntity.from_dict()` verbatim -- this module derives nothing new: it
only reads already-persisted case files and reconstructs the exact objects
those already-tested functions expect. It never promotes unreviewed
evidence to verified, never treats a missing detector as negative
evidence, never turns a model probability into a psychological claim, and
never invents evidence where the case has none.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import drawing_synthesis as ds
from . import reasoning_chain as rc
from .visual_entity import VisualEntity

ROOT = Path(__file__).resolve().parents[2]


def build_live_case_bundle(case_dir: str | Path) -> dict:
    """Returns a bundle usable directly by `human_interaction.answer_question`.

    Fields intentionally match `clinician_review_app.load_case_bundle`'s
    own shape (`image`, `entities`, `deterministic_features`, `synthesis`)
    plus `case_dir` (real for a live case, unlike the cached dev-set path)
    and `emotion` (the real expressive-model result, if this case has one)."""
    case_dir = Path(case_dir)
    analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))

    detections_path = case_dir / "detections.json"
    entity_dicts = []
    if detections_path.exists():
        detections = json.loads(detections_path.read_text(encoding="utf-8"))
        entity_dicts = detections.get("entities") or []
    # Verbatim reconstruction -- case_verification_status/model_validation_status
    # are copied exactly as persisted, never upgraded/downgraded here.
    entities = [VisualEntity.from_dict(e) for e in entity_dicts]

    image_path = case_dir / Path(analysis["image_path"]).name
    image_id = case_dir.name
    # One check per rule in the frozen matrix (rc.check_visual_preconditions) --
    # mirrors clinician_review_app.load_case_bundle exactly. Without this,
    # _governed_case_as_legacy_view's rule_evaluations stays empty and every
    # case-question answer citing a rule_id fails deterministic verification.
    checks = rc.check_visual_preconditions(entities)
    deterministic_features = ds.load_or_compute_deterministic_features(image_id, image_path)
    synthesis = ds.synthesize_drawing(image_id, entities, deterministic_features)

    try:
        relative_path = str(image_path.resolve().relative_to(ROOT))
    except ValueError:
        # Case lives outside the repo root (e.g. a temp dir in tests) --
        # governed_visual_recheck degrades to "unavailable" for this case,
        # exactly as it already does when bundle["image"] has no path.
        relative_path = None

    return {
        "image": {"image_id": image_id, "relative_path": relative_path},
        "entities": entities,
        "checks": checks,
        "deterministic_features": deterministic_features,
        "synthesis": synthesis,
        "case_dir": str(case_dir),
        "emotion": analysis.get("emotion"),
    }
