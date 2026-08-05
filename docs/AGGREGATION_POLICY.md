# Aggregation Policy (DOAR-TRACE Phase 1.5, Section 6)

**Status: implemented, unit-tested, real-image tested.** This document
specifies the policy `src/doar/structured_report.py` implements, so a
reviewer can check the code against a written rule rather than reverse-
engineering it from source.

## The three levels

| Level | Name | What it is | Where it appears |
|---|---|---|---|
| A | Observation | A raw extracted feature, quality metric, or model output. No interpretation attached. | `structured_analysis.json`'s `quality`/`segmentation`/`objective_features`/`model_output` |
| B | Individual rule suggestion | One matched registry rule's own cautious reading, shown alone. | `individual_rule_suggestions` |
| C | Combined drawing-level hypothesis | A construct-level pattern, only when the construct's own policy (below) is satisfied by >=2 independent contributors. | `combined_drawing_level_hypotheses` |

**A rule can never skip from A to C.** Level B is mandatory and
unconditional for every triggered rule; Level C is an *additional*,
separately-computed output that some triggered rules' constructs may
also reach, never a replacement for Level B, and never reachable by a
single rule alone.

## Ordinal levels (0-4)

| Level | Label | Meaning |
|---|---|---|
| 0 | no evidence | Construct not evaluated / nothing triggered |
| 1 | isolated clue | Exactly one contributing family -- **stays at Level B only, never shown as a combined hypothesis** |
| 2 | possible pattern | Exactly two independent contributing families |
| 3 | converging drawing pattern | Three or more independent contributing families |
| 4 | drawing evidence + external context | Requires parent/child/professional-supplied context in addition to drawing evidence -- **not reachable in this phase**, no such context input exists in the pipeline yet. Defined for schema completeness only. |

Only ordinal levels 2-4 may ever appear in `combined_drawing_level_hypotheses`.

## The construct policy (per `construct_registry.json`)

A construct is promoted to Level C only when **all** of the following hold:

1. **>= construct's `minimum_independent_evidence_families`** (currently
   2 for every one of the 12 constructs) distinct `evidence_family`
   values contribute.
2. **>= 2 distinct evidence IDs** across all contributors (dependency-
   group-aware: the same `evidence_id` cited by two different rules
   counts once, never twice).
3. **No upstream failure.** A contributing rule must have
   `status == "weak_support"` (rules.py's real "triggered" state); the
   expressive-content model must have `status == "available"`.
4. **Low-confidence evidence is suppressed.** The expressive-content
   model's contribution is dropped entirely below
   `MODEL_MIN_CONFIDENCE_FOR_AGGREGATION = 0.50` calibrated confidence
   (mirrors `emotion.py`'s own "moderate" uncertainty band).
5. **For the 3 "serious" constructs** (`low_mood_or_emotional_distress_pattern`,
   `fear_or_insecurity_pattern`, `tension_or_anger_pattern`) **at least
   one contributor must be above purely-`Speculative` evidence grade**
   (or ungraded, i.e. from the Arabic source, which never assigned
   itself a grade at all -- treated as clearing the bar, since
   "Speculative" is a specific downgrade the English source explicitly
   assigned, not a default absence of information). Every other
   construct has no such requirement.
6. **Contradictions are always shown, never omitted.** If two Level-C
   hypotheses' constructs are registered opposites
   (`OPPOSING_CONSTRUCTS`) and both are present, both are kept and the
   contradiction is recorded in `cross_theme_contradictions` -- no
   automatic winner is ever chosen.

`aggregation_judge` (`judge_schemas.py`) independently re-verifies every
one of these six conditions per case, as a second check distinct from
`structured_report.py`'s own unit tests.

## The expressive-content model as an evidence family

The calibrated Angry/Fear/Happy/Sad classifier
(`outputs/phase5/.../best.pt`) is wired in as its own family,
`global_expressive_content_model`, contributing to exactly the construct
its top class maps to:

| Top class | Construct |
|---|---|
| Angry | `tension_or_anger_pattern` |
| Fear | `fear_or_insecurity_pattern` |
| Happy | `positive_affective_tone` |
| Sad | `low_mood_or_emotional_distress_pattern` |

**Mandatory wording** (mechanically enforced by test, not just
convention): *"The visual model found the drawing most similar to the
dataset's `<X>` expressive-content category."* Never *"the child is
`<X>`."* This is the model's own classification confidence over a
dataset of drawings, not a validated psychological instrument -- it
carries exactly the same caveats as the emotion-classification track
documented in `THESIS_EXPERIMENT_RESULTS.md`, and is never presented as
validating any PDF-sourced rule.

## What is genuinely reachable on real data today

With the current, unexpanded detector inventory (only the original 6
tier-1 composition/placement rules are individually executable -- see
`docs/RULE_PDF_COVERAGE_AUDIT_V2.md`), the only real, demonstrated
convergence path to Level C is:

**`PSY_AR_SIZE_SMALL_016` (coverage_small, triggered) + a confident
Fear-class expressive-model prediction** → Level 2,
`fear_or_insecurity_pattern`.

This is exactly the task's own worked example ("small page use +
frightened/sad expression + Fear/Sad expressive model may contribute to
a fear/insecurity or low-mood pattern") minus the facial-expression
contributor (no face detector exists yet) -- 2 families is enough to
clear the threshold, so this is genuinely reachable, not theoretical.
Verified with a real image + real checkpoint in
`PHASE1_5_IMPLEMENTATION_REPORT.md`'s smoke-test log.

**No other construct can reach Level C on real data today** -- every
other executable tier-1 rule (`coverage_full`, `coverage_about_half`,
`placement_top/left/right`) maps to a construct the expressive model
cannot also reach (or maps to no construct at all), and no two
executable rules ever map to the same construct. This is a real,
current ceiling, not a hidden limitation.
