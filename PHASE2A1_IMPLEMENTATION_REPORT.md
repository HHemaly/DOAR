# DOAR-TRACE Phase 2A.1 Implementation Report

**Status: real, executed, tested.** Branch
`feature/doar-trace-measurement-hardening`, checkpoint tag
`checkpoint/pre-doar-trace-phase2a1`, starting commit `bc4189e` (Phase
2A complete, 639 tests passing) → commit `e309d21`. Never merged to
`main`. Every number below comes from an actual run, not a projection.
Phase 2B was **not** started, per explicit instruction.

## 1. Branch and final commit

`feature/doar-trace-measurement-hardening` @ `e309d21` (this report's
own commit follows immediately after).

## 2. Baseline and final test results

- **Baseline** (before any Section 2+ edit): 639 passed, 0 failed;
  `ruff check` clean; `compileall` clean; Streamlit AppTest 5/5 passed;
  one full-page real smoke (`train/Angry/a11.jpg`:
  `page_frame_status=full_page_detected`, `bounding_box_coverage=0.8067`)
  and one cropped real smoke (`valid/Angry/a51_jpg...`:
  `page_frame_status=cropped_or_content_only`) both recorded.
- **Final**: **727 passed, 0 failed, 0 skipped** (`pytest tests/ -q`).
  `ruff check src main.py tests doar_prototype_app.py`: all checks
  passed. `python -m compileall -q src main.py tests doar_prototype_app.py`:
  clean. Streamlit AppTest: 5/5 passed (including a real-checkpoint
  run). 88 net new tests across 6 new test files plus targeted additions
  to 5 existing ones.

## 3. Page-reference design selected

6 modes exactly as specified (`page_reference.py`,
`docs/PAGE_REFERENCE_MODEL.md`): `auto_detected_page` (reuses Phase
2A's classical-CV heuristic, polygon = whole image),
`user_confirmed_full_frame` (explicit human assertion, polygon = whole
image, overrides any automatic reading), `user_defined_page_corners`
(4 validated, normalized pixel corners — can be genuinely smaller than
the whole image), and 3 non-assessable modes
(`cropped_or_content_only`/`uncertain`/`failed`, polygon always `None`,
never manufactured). Every page-relative gate now reads
`page_reference.page_relative_features_assessable`. A dedicated new
feature, `segmentation.page_relative_bounding_box_coverage`, re-expresses
coverage relative to the confirmed page instead of the raw image;
verified end-to-end to produce a genuinely different number for a real
smaller user-supplied polygon (0.5476 image-relative → 0.7291
page-relative on the same synthetic case).

## 4. Border-touch-ratio decision

**Retired** (`docs/BORDER_TOUCH_RATIO_DECISION.md`), not patched. Traced
the exact mechanism and found Phase 2A's own explanation ("morphological
cleanup erodes 1-2px") was incomplete: the true, primary cause is
upstream in `_segment`'s candidate generation
(`candidate_adaptive`'s BoxBlur-based test structurally fails near
thick, border-touching content) and candidate scoring (`(1 -
border_ratio)` rewards that exact failure) — confirmed by a regression
test that inspects the pre-cleanup candidate masks directly, not just
the final aggregate number. Retired rather than fixed because the real
bug lives in shared segmentation logic (dataset-wide blast radius, out
of a one-feature section's scope) and the feature itself has zero
downstream consumers (verified against both registries).
`confidence=0.0`/`missing=True` forced unconditionally; historical value
preserved for comparison.

## 5. Coverage-full decision

Redefined from image-relative area (`bounding_box_coverage >= 0.90`) to
a page-reference-gated, margin-based "approaches all 4 page margins
(<=0.15)" test (`docs/PAGE_COVERAGE_DEFINITION_DECISION.md`). 5
candidate definitions compared and scored on definitional match,
measurement stability, explainability, page-reference fit, and
expert-review suitability — explicitly never on trigger frequency.
Implemented as a post-processing override (`rule_engine_v2.redefine_coverage_full`)
that never edits `rules.py`/`rules_registry.json`.

## 6. Before/after page-frame audit

**Unchanged, byte-for-byte, across every Phase 2A.1 code change**: 120
images, 5 `full_page_detected` / 10 `likely_full_page` / 8 `uncertain`
/ 97 `cropped_or_content_only`, 12.5% assessable. Expected: no Phase
2A.1 change touched `page_frame.py`'s own classification logic.

## 7. Before/after robustness results

**Unchanged, byte-for-byte**: 880 measurements, 351 pass / 508
expected_change / 21 fail. Phase 2A.1 did not modify the invariance
harness or its transforms; the new `canonical_input.py` pipeline is a
separate, additive counterpart (4 features, resolution-only), not a
change to the original-image measurements this audit checks.

## 8. Rules changed, disabled, or retained

**Changed**: `PSY_AR_SIZE_FULL_015` (`coverage_full`) — new trigger
definition (Section 5 above); real trigger rate among assessable cases
went from 25.81% (8/31) to 58.06% (18/31) on the 200-image sample.
**Disabled**: none newly disabled. **Retained unchanged**: the other 9
executable rules' trigger logic (5 historical + 4 Phase 2A) and all 31
disabled rules. **New fields on every rule** (Section 8,
`docs/BORDER_TOUCH_RATIO_DECISION.md`/registry): `page_reference_requirement`,
`feature_version`, `known_robustness_limitations`, `expert_review_status`
— descriptive only, none change `allowed_output_level`.

## 9. Expert-review artifacts generated

3 new CSVs (`artifacts/expert_review/phase2a1_{rule,threshold,page_rule}_review_form.csv`),
all blank in every judgment column, built from real registry/rule-engine
data: 4 rows (the newly-activated rules), 9 rows (every executable rule
with a non-directly-sourced threshold, including the redefined
`coverage_full`), and 7 rows (every page-relative rule) respectively.
Regenerated after the `coverage_full` redefinition so their real
trigger-rate column is current, not stale.

## 10. Real-image smoke results

**32/32 real, non-test, class-balanced images** (8/class, seed=202)
processed end-to-end through `analyze_image` with zero errors, each
verified to carry `page_frame`, `page_reference`, `canonical_features`,
and `rule_evaluations` in its output — exceeding the required ≥30.
Combined with the 24-image Phase 2A smoke set from the prior report,
56 distinct real images have now been exercised through this exact
pipeline across the two phases.

## 11. Remaining risks

1. **No psychologist review has occurred** for any of the 4 Phase 2A
   rules or the redefined `coverage_full` — the forms exist, unfilled.
2. **The real `_segment` candidate-selection bug is unfixed** and may
   affect `page_frame.py`'s own edge-touch evidence and any
   border-touching real image, not just the synthetic case that
   surfaced it — the page-frame audit shows no observed discrepancy on
   this dataset, but the exposure is real and undiagnosed at scale.
3. **`canonical_features` has zero rule consumers** — built, tested,
   and separated correctly, but nothing reads it yet; re-pointing a
   rule at it is a deliberate future decision.
4. **No UI exists for `user_confirmed_full_frame`/`user_defined_page_corners`**
   in the Streamlit prototype — the schema/API is real and tested end
   to end, but a user cannot yet invoke it through the app.
5. **0.15 (margin threshold) and the 3 stroke-proxy thresholds remain
   explicitly invented/exploratory**, unreviewed by any domain expert.

## 12. Exact Phase 2B recommendation

**Do not begin automatically — this report is the stopping point, per
instruction.** If/when Phase 2B is authorized, in priority order: (a)
obtain actual psychologist review using the 6 generated CSV instruments
(3 from Phase 2A, 3 from this phase) before any of these rules reach
real parent-facing output; (b) investigate and, if warranted, fix the
`_segment` candidate-selection/scoring bug this phase traced but
deliberately did not touch, with its own dedicated full-audit rerun;
(c) decide whether to build a minimal UI for the page-reference user
declarations now that the underlying API is proven. None of these
require an object detector, an LLM, or any model retraining, and none
were attempted here.
