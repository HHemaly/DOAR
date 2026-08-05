# Current-to-Target Gap (v5) — DOAR-TRACE Phase 2A.1

**Status: post-implementation record.** Supersedes `CURRENT_TO_TARGET_GAP_V4.md`
(Phase 2A) for the areas Phase 2A.1 touched.

## What Phase 2A.1's audit found (Section 2, presented to the user before any rule-semantics change)

1. **The exact `border_touch_ratio` failure was mis-attributed.**
   Phase 2A's own comment blamed morphological cleanup; direct
   debugging found the real, primary cause is upstream in `_segment`'s
   candidate generation and scoring (`docs/BORDER_TOUCH_RATIO_DECISION.md`).
2. **Zero downstream consumers** of `border_touch_ratio` -- confirmed by
   direct inspection of both registries and a regression test.
3. **Fixing it could change page-frame classifications broadly** --
   because the real bug lives in shared `_segment` logic used by every
   feature and by `page_frame.py`'s own separate edge-touch computation,
   not something isolated to the one named feature.
4. **`coverage_full`/page-frame-border conflict confirmed and
   quantified**: a margin thin enough for >=90% area coverage is
   structurally thinner than the border band the page-frame heuristic
   needs.
5. **Coverage was always image-relative, never page-relative** --
   confirmed by direct code read: `bounding_box_coverage` divides by the
   uploaded image's own `h*w`, with no confirmed/detected page polygon
   anywhere before this phase.
6. **Page-reference options compared** (`docs/PAGE_REFERENCE_MODEL.md`):
   automatic-only, user-confirmed-full-frame, user-defined-corners --
   all 3 implemented as real, tested modes.
7. **Recommendation followed**: implement the explicit page-reference
   model (Section 3) rather than continuing to assume image==page.
8. **Psychologist review still needed** before the 4 Phase 2A rules (or
   the redefined `coverage_full`) reach parent-facing output --
   `artifacts/expert_review/phase2a1_*.csv` (Section 7) exist precisely
   for this and remain unreviewed (`expert_review_status: pending_review`
   for all 10 executable rules).

## Gaps Phase 2A.1 closed

1. **No explicit page-reference model existed** -- `page_reference.py`
   added: 6 modes, a real polygon or `None`, a
   `page_relative_features_assessable` boolean every downstream gate now
   reads, and a real (not stubbed) `user_confirmed_full_frame`/
   `user_defined_page_corners` API for the future interactive app.
2. **`coverage_full`'s definition conflicted with page-frame
   detectability** -- redefined to a margin-based "approaches all 4
   page margins" test (`docs/PAGE_COVERAGE_DEFINITION_DECISION.md`),
   selected by definitional match/stability/explainability, not by
   trigger frequency. Real effect: baseline smoke image now triggers
   (previously did not); trigger rate among assessable cases went from
   25.81% to 58.06% on the 200-image sample.
3. **`border_touch_ratio`'s root cause was mis-documented** -- corrected,
   and the feature retired from downstream use (`confidence=0.0`,
   `missing=True`) rather than patched in a way that would not have
   fixed the actual bug.
4. **No canonical/resolution-normalized feature set existed** --
   `canonical_input.py` added for the 4 features Phase 2A found
   resize-sensitive, strictly separate from the original-image feature
   set, never touching brightness/contrast/colour (which remain
   evidence, per instruction).
5. **Only 4 rules had any expert-review coverage, and only generically**
   -- 3 new, more targeted forms (rule-level, threshold-level,
   page-rule-level) generated, all blank for a human reviewer.
6. **No structured per-rule record of page-reference requirement,
   feature version, robustness limitations, or review status existed**
   -- added to every registry-v2 rule.
7. **A real, previously-undetected wiring gap**: `apply_page_frame_gating`/
   `evaluate_v2_rules`/`page_frame_judge` all gated on the raw automatic
   `page_frame` status even after `page_reference` was introduced, so a
   user override had no actual effect on gating despite being correctly
   resolved. Found by the very integration test Section 9 required
   ("user-confirmed full-frame behavior is traceable") and fixed.

## Gaps that remain (honestly, not silently)

1. **None of the 4 Phase 2A rules, nor the redefined `coverage_full`,
   have been reviewed by an actual psychologist.** The forms exist; no
   review has happened.
2. **The real `_segment` candidate-selection/scoring bug that caused
   the `border_touch_ratio` failure is unfixed** -- it may still affect
   `page_frame.py`'s own edge-touch evidence and any feature computed
   from a border-touching mask, for real (not just synthetic) images.
   Deliberately not attempted this phase (dataset-wide blast radius, out
   of Section 5's one-feature scope).
3. **`canonical_features` has zero rule consumers** -- built and
   validated, but no rule currently reads it. Re-pointing a rule at a
   canonical value is a future, deliberate decision, not automatic.
4. **`user_confirmed_full_frame`/`user_defined_page_corners` have no UI**
   in the current Streamlit prototype -- the schema/API is real and
   tested, but nothing in `doar_prototype_app.py` lets a user actually
   invoke it yet (explicitly permitted to be schema-only per the task's
   "a minimal UI control may be added if it can be implemented without
   destabilizing the current app" -- judged not necessary to add this
   phase, since the batch pipeline and API are the higher-priority
   deliverable and the app was already exercised extensively without
   destabilizing it).
5. **0.15 (the margin threshold) and the 3 stroke-proxy thresholds
   remain explicitly invented/exploratory**, not expert-reviewed.
6. **No new detector, no LLM/Gemini/ChatGPT integration, no
   Angry/Fear/Happy/Sad retraining** -- explicitly out of scope, none
   attempted.
