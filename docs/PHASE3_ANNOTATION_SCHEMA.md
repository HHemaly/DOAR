# Phase 3 — Detector Validation Annotation Schema

Companion to `PHASE3_DETECTOR_EVALUATION_PLAN.md`. Defines the structure a
human annotator (or annotation tool) fills in for the held-out validation
sample. No annotation has been collected yet — this is the schema only.

These are **descriptive** labels ("is there a circle here," "is a face
present") — not psychological judgments. Any careful annotator can produce
them; no clinician is required for this step (clinician time is reserved for
Phase 6's confidence-ceiling validation study).

## File-level record (one JSON object per annotated image)

```json
{
  "schema_version": "phase3_annotation_v1",
  "image_id": "sha256 or stable manifest image_id, NOT the dataset folder path",
  "annotator_id": "string, pseudonymous is fine",
  "annotated_at_utc": "ISO 8601 timestamp",
  "face": {
    "present": true,
    "bbox": [x_min, y_min, x_max, y_max],
    "eyes": [
      {"side": "left", "state": "open|stern|closed|not_assessable"},
      {"side": "right", "state": "open|stern|closed|not_assessable"}
    ]
  },
  "shapes": [
    {
      "component_id": "matches a connected-component ID from DOAR's existing "
                       "features.py extraction, so classifier crops and human "
                       "labels can be joined by ID",
      "bbox": [x_min, y_min, x_max, y_max],
      "class": "star|circle|heart|flower|cloud|sun|vehicle|geometric_shape|other|none"
    }
  ],
  "animals": [
    {
      "bbox": [x_min, y_min, x_max, y_max],
      "species": "tiger|wolf|fox|squirrel|lion|other_animal|ambiguous"
    }
  ],
  "notes": "free text, optional -- e.g. image quality issues, ambiguous cases"
}
```

## Rules for annotators

- `"ambiguous"` / `"not_assessable"` / `"other"` are valid, expected answers —
  do not force a best guess. Per `SCIENTIFIC_LIMITATIONS.md`, `NOT_DETECTED`,
  `NOT_ASSESSABLE`, and genuine ambiguity must be distinguishable in the data
  used to validate any detector, not collapsed into false certainty.
- `tiger` and `wolf` are recorded **separately** even though the rule
  (`PSY_AR_ANIMAL_TIGER_WOLF_004`) merges them — the merge is a modeling
  decision for later, not something to bake into ground truth.
- Every image gets a second independent annotation for at least a 30–50 image
  subsample, so Cohen's/Fleiss' kappa can be computed via the existing
  `review.py::compute_agreement` machinery (no new agreement code needed).
- `component_id` must be traceable back to the exact DOAR pipeline run that
  produced it (manifest + extraction-config hash), so a future re-run with
  different segmentation parameters doesn't silently invalidate old labels.

## What this schema deliberately does NOT include

- Any emotion, personality, or psychological judgment.
- Any confidence score from the annotator (precision/recall against these
  labels is computed later, from the detector's own confidence, not
  solicited from the human).
- Any field that would let this annotation set double as training data for
  the *emotion* classifier — it is strictly for detector validation.
