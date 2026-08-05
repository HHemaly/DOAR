# DOAR-TRACE Phase 2A.2 Implementation Report

**Status: real, executed, tested.** Branch `feature/doar-parent-view-clarity`,
checkpoint tag `checkpoint/pre-doar-trace-phase2a2`, starting commit
`52ff276` (Phase 2A.1 complete, 727 tests passing) → commit `7a2eec9`
(this report's commit follows immediately after). Never merged to
`main`. UI/UX only -- no analysis, rule, feature, threshold, or
aggregation-logic change; every trigger-status check in this phase's
tests reproduces exactly the Phase 2A.1 status for the same input.
Phase 2B was **not** started, per explicit instruction.

## 1. Branch and final commit

`feature/doar-parent-view-clarity` @ `7a2eec9`.

## 2. Before/after Parent-View structure

**Before**: 12 mixed sections (drawing-level summary, combined
patterns, individual suggestions showing raw `rule_id`/`observable` in
expander titles, what-was-measured, expressive-model result, "why each
suggestion was made" showing raw `rule_id`, contradictions, missing
evidence, mechanically-generated questions, sources showing raw
`reference_ids`, disclaimer, and an expandable all-41-rules/all-60-
features table pair) plus a chat panel showing raw `evidence_ids`.

**After**: exactly 5 (Overall result / What was observed / Possible
meaning / Questions and next steps / Limitations), per
`docs/PARENT_VIEW_INFORMATION_POLICY.md`. Overall result is generated
entirely from real per-case data by `parent_view.build_overall_result_summary()`
and matches the task's own worked-example shape (verified by a
dedicated test). Individual-observation expanders are titled
"Observation N: `<friendly family>`" (e.g. "line-appearance"), never a
raw rule ID. The all-41-rule/all-60-feature tables and raw
`evidence_id`/`reference_ids` display are gone from Parent View
entirely -- confirmed absent by dedicated regression tests against the
real running app, not just by code inspection.

## 3. Page-reference UI implementation

A single, clearly optional sidebar radio ("Does this image show the
complete sheet of paper?", 4 choices) wired through
`analyze_image_with_timing`'s new `user_page_declaration` parameter
into the real `analyze_image` pipeline -- no UI-only stub. 2 new
`page_reference.py` API modes (`user_declared_cropped`/
`user_declared_uncertain`) extend Phase 2A.1's existing
`user_confirmed_full_frame` so "No"/"I am not sure" can override an
automatic reading the same way "Yes" already could. The choice is
persisted and traceable purely from the saved `page_reference` fields
(no separate declaration file); Technical View's "Parent declaration
used" metric shows it via `describe_declaration_choice()`. No
corner-editor UI was built (explicitly deferred) -- Technical View
states this fact plainly.

## 4. Capability-status implementation

`parent_view.capability_status()` (3 tiers: WORKING/LIMITED/NOT
AVAILABLE, exact lists in Section 7 of the task) is shown in full in
Technical View's "Missing capabilities" section; Parent View shows only
`capability_status_summary_text()`, 1-2 plain sentences. Both are real,
static, and verified accurate against what the pipeline actually does
(e.g. "10 executable heuristic rules" matches the real registry count,
checked by test).

## 5. Information retained in Technical View

**Nothing was deleted.** Every value the pre-Phase-2A.2 Technical View
showed is still shown, reorganized under the 11 named subsections
Section 9 specifies (Input and page reference / Image quality and
segmentation / Objective features / Canonical features / Expressive-
content model / Rule evaluations / Combined-pattern calculation /
Evidence and provenance / Judges and verification / Missing
capabilities / Sources and registry), plus new content this phase adds
(page reference detail, canonical features, the full capability list,
declaration traceability, the required image-frame-vs-page-usage
label on the objective-features table). Verified directly: the raw
`rule_evaluations` dataframe in Technical View still carries real
`rule_id`s, checked against `analysis.json` in a dedicated test.

## 6. Tests/lint/compile/AppTest results

- **Baseline** (before any Section 2+ edit): 727 passed; ruff clean;
  compileall clean; AppTest 5/5; 3 real cases recorded (full-page,
  cropped, one triggering `EN_COMPILED_LINE_HEAVY_PRESSURE_030`).
- **Final**: **775 passed, 0 failed, 0 skipped**. `ruff check src
  main.py tests doar_prototype_app.py`: all checks passed.
  `python -m compileall -q src main.py tests doar_prototype_app.py`:
  clean. Streamlit AppTest: 5/5 passed. 48 net new tests across
  `test_phase2a2_parent_view.py` (33), `test_parent_view.py` (+16),
  `test_page_reference.py` (+16, including the new declaration-choice
  API), plus targeted edits to `test_prototype_app_smoke.py` (1
  migrated with a stated reason: the individual-suggestion expander
  check now looks for the friendly family label instead of the raw
  `rule_id`, since Section 4 requires hiding it).
- Zero `use_container_width` occurrences remain in `doar_prototype_app.py`
  (confirmed by grep and by a dedicated test); the app runs with no
  deprecation warning in a live AppTest run.

## 7. Paths for three example cases

No generated example JSON is committed (`outputs/` is gitignored, per
existing convention). Reproducible directly:

- Full-page: `train/Angry/a11.jpg` (page_reference → `auto_detected_page`,
  `PSY_AR_SIZE_FULL_015` → `weak_support`).
- Cropped: `valid/Angry/a51_jpg.rf.11492699ac81cdd89da4b5874324bcef.jpg`
  (page_reference → `cropped_or_content_only`,
  `PSY_AR_SIZE_FULL_015` → `not_assessable`).
- Line-proxy: `train/Angry/2-2_jpg.rf.aef2f6a8cf72176022ee3fe0bd2717ed.jpg`
  (`EN_COMPILED_LINE_HEAVY_PRESSURE_030` → `weak_support`, renders with
  "The lines appear relatively dark or thick..." in the live app --
  verified by `tests/test_phase2a2_parent_view.py::LineProxyWordingRemainsNonDiagnosticInAppTests`).

Run any of the 3 through `main.py analyze-image` or the app's own
upload flow to see both views live.

## 8. Remaining limitations

1. No corner-editor UI for `user_defined_page_corners` (API-only, by
   design this phase).
2. Friendly-name mappings (source documents, evidence families) are a
   fixed small vocabulary -- a new source PDF or rule family added in a
   future phase needs a corresponding mapping entry, or the honest raw
   fallback will surface instead of a translated name.
3. `capability_status()` is static across cases, not case-varying.
4. The 4 Phase 2A rules and the redefined `coverage_full` still have
   not been reviewed by an actual psychologist -- unchanged from Phase
   2A.1's gap list.
5. No new detector, no LLM integration, no model-training change, no
   rule-threshold change, no aggregation-logic change -- all explicitly
   out of scope, none attempted.

## 9. Exact recommended Phase 2B0 scope

**Do not begin automatically — this report is the stopping point, per
instruction.** If/when a next phase is authorized, the smallest,
highest-leverage scope ("Phase 2B0") would be: (a) actually collect
psychologist review using the 6 generated CSV instruments across Phase
2A and 2A.1 before any of these rules reach real parent-facing output
in a non-prototype setting; (b) decide whether a minimal
`user_defined_page_corners` UI (a simple 4-point click-to-mark control)
is worth building now that its API is proven end-to-end in this phase;
(c) revisit the `_segment` candidate-selection bug Phase 2A.1 traced
but deliberately did not fix. None of these require an object detector,
an LLM, or any model retraining, and none were attempted here.
