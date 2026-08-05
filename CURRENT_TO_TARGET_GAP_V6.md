# Current-to-Target Gap (v6) — DOAR-TRACE Phase 2A.2

**Status: post-implementation record.** Supersedes `CURRENT_TO_TARGET_GAP_V5.md`
(Phase 2A.1) for the areas Phase 2A.2 touched -- UI/UX only, no analysis,
rule, feature, threshold, or aggregation logic changed.

## What Phase 2A.2's audit found (Section 2, presented to the user before any edit)

1. **12 Parent-view sections**, several showing internal identifiers.
2. **Useful-to-a-parent content existed but was buried** among technical
   detail (item 12's raw tables, rule_id-labeled expanders).
3. **Technical-only content correctly belonged in Technical View** but
   duplicated in Parent View too (item 12).
4. **A real contradiction found**: the page-frame warning (item 1) told
   parents page interpretation was suppressed, but item 4's
   `plain_language_observations()` call unconditionally still reported
   "covers X% of the page" a few lines later.
5. **Internal IDs visible to parents**: `rule_id` (items 3, 6, 10),
   `observable` (item 3 titles), `reference_ids` (item 10), raw
   `evidence_ids` (chat panel).
6. **No capability's absence was actively disguised as having run** --
   but nothing summarized unavailable capabilities up front, only in
   passing captions.
7. **Restructuring plan followed exactly**: UI-only, `doar_prototype_app.py`
   + `parent_view.py` + a `registry_v2_build.py` wording fix + 2 new
   `page_reference.py` API modes -- zero changes to `analysis.py`,
   `rules.py`, `rule_engine_v2.py`'s thresholds, or `structured_report.py`'s
   aggregation logic.

## Gaps Phase 2A.2 closed

1. **Parent View reduced from 12 mixed sections to exactly 5**, each
   built from real per-case data via new `parent_view.py` functions.
2. **The page-assessability contradiction is fixed**: `plain_language_observations()`
   and the new `build_overall_result_summary()` both take the resolved
   `page_reference` and never emit coverage/placement wording when it
   is not assessable.
3. **No internal ID, raw table, or technical enum reaches Parent View
   by default** -- verified by 5 dedicated regression tests exercising
   the real app, not just unit-level string checks.
4. **A minimal, real page-reference user control exists**: 4-choice
   sidebar radio, wired through `analyze_image_with_timing`'s new
   `user_page_declaration` parameter to the actual pipeline -- verified
   end to end (declaration reaches real rule gating, is persisted, is
   traceable in Technical View) for all 4 choices.
5. **A compact, accurate capability-status summary exists** in both
   views, at different levels of detail.
6. **Every executable rule's suggested question is now natural** --
   the mechanical "ask about the light line pressure appearance"
   pattern is gone for all 10 executable rules.
7. **Technical View reorganized into 11 named subsections**, with
   nothing deleted -- every value the old Technical View showed is
   still shown, plus new page-reference/canonical-feature/capability/
   declaration-traceability detail.
8. **13 `use_container_width=True` calls replaced with `width="stretch"`**
   -- zero deprecation warnings remain.

## Gaps that remain (honestly, not silently)

1. **No corner-editor UI** for `user_defined_page_corners` -- the API
   is real and tested; the prototype's UI only exposes the 4-choice
   declaration. Explicitly deferred, per instruction.
2. **The friendly-name/family mappings are a fixed, small vocabulary**
   (2 source documents, 4 evidence families) -- adding a new source PDF
   or rule family in a future phase requires extending
   `_SOURCE_FRIENDLY_NAMES`/`_FAMILY_FRIENDLY_NAMES` in `parent_view.py`,
   or the honest raw-string fallback will surface instead of a
   translated name.
3. **`capability_status()` is static across cases** -- it does not
   currently vary based on, e.g., whether the emotion model actually
   ran for a specific case (only the always-true fact that the
   capability exists "when a checkpoint is loaded").
4. **No new detector, no LLM/Gemini/ChatGPT integration, no model
   training changes, no rule-threshold changes, no aggregation-logic
   changes** -- explicitly out of scope, none attempted, verified by
   the fact that every real trigger-status check in this phase's tests
   (`PSY_AR_SIZE_FULL_015`, `EN_COMPILED_LINE_HEAVY_PRESSURE_030`)
   reproduces exactly the same status Phase 2A.1 would have produced
   for the same input.
5. **The 4 Phase 2A rules and the redefined `coverage_full` still have
   not been reviewed by an actual psychologist** -- unchanged from
   Phase 2A.1's gap list; this phase did not touch expert-review status.
