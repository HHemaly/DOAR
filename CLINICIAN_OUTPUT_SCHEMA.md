# DOAR — Clinician Output Schema

Machine-readable structure for future clinician-facing output. **Schema
only — no UI is built by this audit.** Reuses existing DOAR types
(`schemas.py::Evidence`, `visual_entity.py::VisualEntity`,
`rules_registry_v2.json`'s per-rule fields, `concerns.py`'s aggregation
output) rather than inventing parallel ones.

## 1. Clinician evidence card (per observation)

One card per verified observation feeding a rule evaluation:

```json
{
  "observation_id": "obs_<case>_<n>",
  "observed_feature": "sad_face_expression",
  "bbox_crop_provenance": {
    "bbox": [0.31, 0.34, 0.12, 0.17],
    "crop_ref": "artifacts/crops/ve_....png",
    "source": "visual_observer | detector:<name> | expert_review",
    "verification_status": "verified | uncertain | rejected | unreviewed"
  },
  "rule_id": "EN_COMPILED_FACE_EXPRESSION_021",
  "source_interpretation": "positive tone (smiling) or distress (sad/tense/frightened)",
  "concern_domain": "depressive_or_low_mood_related",
  "evidence_strength": "Moderate for basic emotion only",
  "evidence_direction": "insufficient",
  "supporting_references": [],
  "contradicting_references": ["REF_CRAWFORD_COLOUR_CAUTION_2012 (if colour co-cited)"],
  "alternative_explanations": [
    "May reflect an intentionally depicted expression, not the child's own state.",
    "May reflect copying a reference image, drawing style, motor/fine-skill limitations, or artistic convention."
  ],
  "clinical_question": "Would you be willing to ask your child about the face expression in this drawing?",
  "allowed_output_level": "disabled",
  "final_cautious_interpretation": "Observable 'face_expression' matched. This is not a diagnosis. See professional_wording in rules_registry_v2.json."
}
```

Every field above already exists somewhere in DOAR's data model
(`RULE_EVIDENCE_MATRIX.csv`, `visual_entity.py`, `rules_registry_v2.json`,
`external_literature_register.json`) — this schema is the **assembly
shape**, not new data.

## 2. Candidate clinical hypothesis card

```json
{
  "hypothesis_id": "hyp_<case>_<n>",
  "concern_domain": "depressive_or_low_mood_related",
  "candidate_hypothesis_label": "Possible depressive/low-mood presentation",
  "support_level": "INSUFFICIENT | WEAK_HYPOTHESIS | POSSIBLE_FOR_EXPLORATION | PROFESSIONAL_REVIEW_SUGGESTED",
  "why": {
    "observed_features": ["obs_...", "obs_..."],
    "crops_bboxes": ["see referenced observation cards"],
    "triggered_rule_ids": ["EN_COMPILED_FACE_EXPRESSION_021", "..."],
    "evidence_families": ["facial_feature_style"],
    "compound_rules_applied": [],
    "recurrence": {"drawings_with_finding": 1, "drawings_in_case": 3, "trend": "first_occurrence"},
    "literature": ["REF_..."],
    "evidence_strength_breakdown": {"strong": 0, "moderate": 1, "limited": 1}
  },
  "counterevidence": ["REF_CRAWFORD_COLOUR_CAUTION_2012 (if applicable)"],
  "alternative_explanations": ["..."],
  "missing_clinical_information": [
    "Child's own explanation of the drawing (not collected)",
    "Whether this theme recurs across more than the available drawings",
    "Any reported context from parent/caregiver/teacher"
  ],
  "disclaimer": "This is a decision-support hypothesis, not a final diagnosis. See SCIENTIFIC_LIMITATIONS.md.",
  "allowed_output_level": "LEVEL_2_clinical_hypothesis_review_flag"
}
```

**LEVEL 3 (final diagnosis) is never emitted by DOAR itself** — per
`RULE_EVIDENCE_AUDIT.md` Section 9, no rule in the current 41-rule corpus
qualifies, and the AAP ADHD guideline / Allen & Tussey abuse review are
retained specifically to justify why no future rule should either, absent
exceptional, directly-applicable validated evidence this audit did not find.

## 3. Clinician feedback (two SEPARATE fields, never conflated)

```json
{
  "hypothesis_worth_considering": "AGREE | DISAGREE | INSUFFICIENT_INFORMATION",
  "hypothesis_feedback_by": "<clinician_id>",
  "hypothesis_feedback_at": "<ISO timestamp>",
  "final_clinical_assessment": {
    "status": "confirmed_after_assessment | ruled_out | alternative_diagnosis | further_assessment_required",
    "alternative_diagnosis_note": "<free text, only if status=alternative_diagnosis>",
    "comment": "<free text>",
    "assessed_by": "<clinician_id>",
    "assessed_at": "<ISO timestamp>"
  }
}
```

**Non-negotiable**: `hypothesis_worth_considering` and
`final_clinical_assessment` are two **independent** fields, stored
separately, exactly per the task's instruction — one clinician's `AGREE`/
`DISAGREE` click is feedback data for future calibration, never an
automatic re-weighting of any rule's `confidence_ceiling` or `allowed_
output_level`. No code path exists (or is proposed) that reads clinician
feedback and mutates the registry. A future calibration phase would
require its own explicit governance step (`DECISION_LOG.md` process),
never an automatic loop.

## 4. Storage note

Both cards and the feedback structure are additive to the existing
per-case JSON documents (`analysis.json`/`detections.json`-adjacent) — no
existing field is renamed or removed. Exact file/module placement is
implementation work for a future phase, out of this audit's scope
(`RULE_EVIDENCE_AUDIT.md` §11).
