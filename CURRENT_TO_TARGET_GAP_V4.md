# Current-to-Target Gap (v4) — DOAR-TRACE Phase 2A

**Status: post-implementation record.** Supersedes `CURRENT_TO_TARGET_GAP_V3.md`
(Phase 1.5) for the areas Phase 2A touched — that document's own gap
list is not re-litigated here except where Phase 2A changed it.

## What Phase 2A set out to answer (Section 2 audit, done before implementation)

Before any code was written, the codebase was audited against the 9
required questions covering: whether any existing feature already
approximates page-frame visibility (no — confirmed nothing in
`analysis.py` ever checked it), which registry-v2 rules have a
sufficiently precise feature match to activate safely (narrowed during
implementation to the 4 documented in
`docs/STATIC_PROXY_RULE_POLICY.md`), where thresholds currently live and
their provenance (`rules_registry.json`'s 6 tier-1 rules, all
`directly_sourced`/`source_centre_with_invented_band`/
`invented_operational_standin`, none empirically derived until this
phase), and what "not_assessable" as a status would require touching
(`rules.py`'s output shape, `structured_report.py`'s Level A/B/C
filtering, and the judge layer). This audit's findings shaped every
later section and are reflected throughout the docs below rather than
repeated verbatim here.

## Gaps Phase 2A closed

1. **No page-frame assessability check existed anywhere.** Added
   `page_frame.py` (5 statuses, honest `uncertain`/`failed` fallbacks,
   classical-CV only) and wired it into every `analyze_image` run.
2. **Page-relative rules were never gated by page visibility** — a
   `coverage_full` or `placement_top` match on a tightly-cropped photo
   with no visible page was indistinguishable from a match on a real
   full-page scan. **Fixed**: `rule_engine_v2.apply_page_frame_gating()`
   post-processes the historical engine's output (never edits
   `rules.py` itself); `EN_COMPILED_PLACEMENT_CENTER_029` is gated
   internally. Verified by `page_frame_judge` (new 9th judge) on every
   case.
3. **No feature had ever been validated against a known-correct value.**
   22 synthetic ground-truth checks now exist
   (`docs/FEATURE_MEASUREMENT_VALIDATION.md`); 1 real pipeline
   limitation found and documented (`segmentation.border_touch_ratio`
   erosion at image edges).
4. **No feature had ever been tested for robustness to real-world image
   variation** (resize/rotation/compression/lighting/margins/crop/
   perspective/shadow/grayscale). 880 real measurements now exist
   (`docs/FEATURE_ROBUSTNESS_RESULTS.md`); `foreground_coverage`'s
   resize sensitivity and brightness/contrast sensitivity in
   segmentation thresholding are now documented, not hidden.
5. **Threshold provenance existed only informally in code comments** (no
   consistent vocabulary, no sensitivity data). Now: a 5-category
   provenance vocabulary applied to all 10 executable rules, plus a real
   300-image sensitivity sweep quantifying, for the first time, that
   `coverage_small` (the one directly-sourced threshold) triggers on
   ~0% of real images while `coverage_full` (invented) triggers on 86%
   (`docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`).
6. **`stroke.intensity_proxy`/`stroke.fragmentation` were computed for
   every case but wired to no evaluator** — Phase 1.5's own gap list
   (item 6) named this as "the single fastest remaining lever." Now
   wired: 3 new rules, real-data-derived quartile thresholds, structural
   mutual exclusivity between heavy/light pressure.
7. **`EN_COMPILED_PLACEMENT_CENTER_029` had the same problem** (a real
   feature match, unwired) — now activated, page-frame-gated.
8. **Only 8 judges existed**, none checking page-frame-gating
   correctness. **`page_frame_judge`** (9th) added — operational, not a
   stub, independently re-verifies the gating invariant from saved case
   output.
9. **No trigger-frequency data existed for any rule on real data.**
   `rule_trigger_distribution.csv` now covers all 41 registry-v2 rules
   (10 executable with real counts, 31 disabled with an explicit "why
   not" note) on a real 200-image sample.
10. **A latent Arabic-report crash** (`reports.py::_rule_rows` assumed
    `original_arabic` was never `None`) was exposed and fixed the moment
    `EN_COMPILED_*` rules first appeared in real `rule_evaluations` —
    they have no Arabic source text by construction (English-only
    compiled PDF).
11. **A latent evidence-grounding gap**: the new rules cited evidence
    IDs (`ev_feature_stroke_intensity_proxy`,
    `ev_feature_stroke_fragmentation`) that were never actually recorded
    in `analysis["evidence"]`, which `test_chat.py`'s grounding check
    caught the moment they were wired in. Fixed by adding 2 real
    `Evidence` records reusing the existing `FeatureValue` provenance.
12. **A real, pre-existing `registry_v2_build.py` bug**: `validation_status`
    checked `rule_id in _EXECUTABLE_STATIC_DIRECT_FEATURE_IDS` (a dict
    keyed by `observable`, not `rule_id`) — always `False`, so no rule's
    `validation_status` was ever correctly `IMPLEMENTED_UNVALIDATED`
    even when genuinely executable. Fixed.

## Gaps that remain (honestly, not silently)

1. **Only 12.5% (dataset audit) to ~16% (trigger-distribution sample)
   of this dataset is page-frame-assessable.** This is the single
   largest constraint on how often page-relative rules (10 of the 10
   executable rules touch page-relative or line-appearance features;
   7 of those 10 are page-gated) can say anything at all on this
   dataset. Not a bug — an honest, now-quantified property of the data.
2. **31 of 41 registry-v2 rules remain `disabled`** — unchanged in kind
   from Phase 1.5's gap #1, just now with 4 fewer disabled rules and an
   explicit per-rule "why not" audit
   (`docs/STATIC_PROXY_RULE_POLICY.md`) instead of a blanket statement.
3. **`segmentation.border_touch_ratio` has a real, documented
   measurement bug** (morphological cleanup erodes the image's own
   border row) — found, not fixed, out of this phase's validation-only
   scope.
4. **`foreground_coverage` is meaningfully sensitive to resizing**, and
   segmentation thresholds are sensitive to brightness/contrast on real
   photographic content — both now measured and documented, neither
   fixed.
5. **No new detector, no LLM/Gemini/ChatGPT integration, no
   Angry/Fear/Happy/Sad retraining** — all explicitly out of scope, none
   attempted.
6. **The 4 newly-activated rules' `confidence_ceiling` (0.10) is an
   explicitly invented, conservative value**, not clinician-assigned —
   stated plainly in `registry_v2_build.py` and
   `docs/STATIC_PROXY_RULE_POLICY.md`, not hidden behind a plausible-
   looking number.
7. **`coverage_full` (≥0.90) and genuine full-page detectability are
   close to mutually exclusive** under the current page-frame border
   band — a structural interaction between two independently-designed
   heuristics, discovered while migrating tests, not yet resolved or
   even necessarily in need of resolution (see
   `docs/PAGE_FRAME_ASSESSABILITY.md`'s closing section).
