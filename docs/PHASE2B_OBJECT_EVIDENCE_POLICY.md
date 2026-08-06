# Phase 2B Object Evidence Policy

## What a Phase 2B detection means

An object detection from `src/doar/phase2b/inference.py` or
`schemas.py::DetectorEvidenceRecord` means only: **"the drawing contains
visual evidence resembling this depicted class."** It is raw visual
evidence, evaluated on a 20-image pilot, from a zero-shot model never
trained on this dataset. Nothing more.

## Forbidden inferences (hard constraint, mechanically checked where possible)

A `DetectorEvidenceRecord` must never be described, in any surface this
pipeline generates, as evidence of: abuse, depression, anxiety, fear,
aggression, intelligence, superiority, family relationships, or any
emotional state. No species-level animal interpretation is made — the
`animal` class is general-purpose only. `tests/test_phase2b_safety.py`
scans every string field Phase 2B produces (`limitations`, `notes`,
Technical View captions) for the same diagnostic-language patterns
`judges.py::safety_judge` already checks for the rest of the pipeline,
reusing that existing pattern rather than inventing a second one.

## No rule activation, under any confidence level

No rule in `resources/psychology_sources/rules_registry_v2.json` is
activated, gated, or given a new `allowed_output_level` by this phase.
This is verified two ways: (1) `rule_engine_v2.py`/`evidence_rule_engine.py`
are never imported by anything under `src/doar/phase2b/`, confirmed by
grep in `tests/test_phase2b_safety.py`; (2) no Phase 2B evidence file is
ever passed as an argument to any rule-evaluation function in this
codebase.

## No Level C combined hypotheses, no serious warnings, no diagnostic wording, no clinical probabilities

Phase 2B evidence is never merged into `structured_report.py`'s
combined-hypothesis aggregation (`CombinedHypothesesTests` in
`test_structured_report.py` already covers that aggregation's real
inputs; Phase 2B evidence is not one of them, confirmed by the same grep
check above). No Phase 2B output includes a warning level, a diagnostic
label, or a percentage/probability framed as clinical.

## Missing detection != absence

Per `DETECTOR_STATUSES`, a `not_detected` result is a statement about
what this pilot's specific baseline found, not a claim that the object is
truly absent from the drawing — `docs/OBJECT_EVIDENCE_ONTOLOGY.md` and
`technical_view.py`'s own caption both restate this explicitly wherever a
result is shown, since it is the single most consequential thing a reader
could misread.

## Parent View

**Parent View shows nothing from Phase 2B.** The task allows Parent View
to state that object detection is "experimental" but forbids displaying
unvalidated object-based interpretations — given this pilot's own honest
results (`docs/PHASE2B_EVALUATION_PROTOCOL.md`: one class performs worse
than random, none exceed a 20-image sample), there is no result here a
parent should see at all yet. `doar_prototype_app.py`'s Parent View
(`render_parent_view`) is unmodified by this phase — confirmed by
`git diff` touching only the Technical View render function and its
imports.

## What consumes this evidence today

Only `technical_view.py::render_phase2b_pilot_summary` — a read-only
summary of `artifacts/phase2b/*.csv`, gated behind Technical View,
labeled experimental in its own header and caption. No other code path
in the repository reads Phase 2B evidence.
