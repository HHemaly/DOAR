"""label_provenance.py -- non-destructive, post-hoc dataset-label auditing.

Implements the isolation contract from the working spec: a dataset ImageFolder
folder name (e.g. `.../train/Angry/foo.png`) is preserved as
`original_source_label`, but must never be exposed to preprocessing, quality
assessment, feature extraction, object detection, emotion inference, rule
execution, evidence aggregation, report generation, Q&A, or external AI
auditing. It is compared against the independent model prediction only AFTER
inference has already produced its own result, and the comparison is recorded
as a label-quality signal -- never fed back into inference, and never used to
delete, overwrite, or silently correct an original image or label.

Both functions here take only an already-resolved emotion result and a raw
path string; neither is called from, nor able to influence, any stage of
`analysis.py` upstream of report writing. See CURRENT_STATE_AUDIT.md Section 4
and SCIENTIFIC_LIMITATIONS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import CLASSES


def extract_original_source_label(image_path: str | Path) -> str | None:
    """Best-effort ImageFolder label recovery from the path string only
    (`.../<split>/<Class>/filename`). Returns None if the immediate parent
    directory name is not one of the fixed DOAR classes -- this is the normal
    case for a single uploaded image with no dataset provenance, and is not an
    error."""
    parent_name = Path(image_path).parent.name
    return parent_name if parent_name in CLASSES else None


def compute_label_audit_status(original_source_label: str | None, emotion: dict[str, Any]) -> dict[str, Any]:
    """Pure, read-only comparison. Allowed audit_status values:
    CONSISTENT, POSSIBLE_CONFLICT, UNCERTAIN, REVIEWED_CONFIRMED,
    REVIEWED_CORRECTED, ADJUDICATION_REQUIRED. This function only ever
    produces CONSISTENT / POSSIBLE_CONFLICT / UNCERTAIN -- the REVIEWED_* and
    ADJUDICATION_REQUIRED states are set later by a human reviewer action
    (not yet implemented; see IMPLEMENTATION_PLAN.md Phase 6), never here."""
    if original_source_label is None:
        return {
            "original_source_label": None,
            "model_predicted_label": None,
            "audit_status": "UNCERTAIN",
            "note": "Image path does not match a recognizable dataset split/class "
                    "folder; no original label to compare against.",
        }
    status = emotion.get("status")
    if status != "available":
        return {
            "original_source_label": original_source_label,
            "model_predicted_label": None,
            "audit_status": "UNCERTAIN",
            "note": f"No independent emotion prediction is available (status={status!r}); "
                    "cannot compare against the original source label.",
        }
    predicted = emotion.get("top_class")
    consistent = predicted == original_source_label
    return {
        "original_source_label": original_source_label,
        "model_predicted_label": predicted,
        "audit_status": "CONSISTENT" if consistent else "POSSIBLE_CONFLICT",
        "note": (
            "The independent model prediction matches the dataset folder label."
            if consistent else
            "The independent model prediction differs from the dataset folder "
            "label. This does not by itself mean either is wrong -- dataset "
            "folder labels may be inaccurate (see SCIENTIFIC_LIMITATIONS.md "
            "Section 6), and model predictions are not ground truth either. "
            "Route to professional review for REVIEWED_CONFIRMED / "
            "REVIEWED_CORRECTED / ADJUDICATION_REQUIRED."
        ),
    }
