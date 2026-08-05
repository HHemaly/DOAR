# DOAR-TRACE Phase 2A Implementation Report

**Status: real, executed, tested.** Branch
`feature/doar-trace-measurement-validation`, starting commit `4ee5a64`
(Phase 1.5 complete, 610 tests passing) → this report's commit. Never
merged to `main`. Every number below comes from an actual run, not a
projection. Phase 2B was **not** started, per explicit instruction.

## 1. What this phase was

"Measurement validation, page-frame assessability, threshold audit, and
static-rule activation" — Phase 1.5 built the Level A/B/C reporting
architecture and a 41-rule registry, but never checked whether the
underlying measurements were correct, whether page-relative rules made
sense on this dataset's images, or whether any threshold's real-world
trigger behaviour had ever been observed. Phase 2A closes exactly those
3 gaps, plus activates the 4 rules Phase 1.5 explicitly flagged as "the
fastest remaining lever" (`CURRENT_TO_TARGET_GAP_V3.md` gap #6).

## 2. Files changed

**New**: `src/doar/page_frame.py`,
`src/doar/phase2a_{page_frame_audit,feature_ground_truth,
feature_invariance,threshold_sensitivity,rule_trigger_distribution}.py`,
`src/doar/rule_engine_v2.py`,
`tests/{test_phase2a_rule_engine_v2,test_phase2a_experiment_scripts}.py`,
`artifacts/phase2a/*` (6 CSVs + 2 JSON summaries),
`docs/{PAGE_FRAME_ASSESSABILITY,FEATURE_MEASUREMENT_VALIDATION,
FEATURE_ROBUSTNESS_RESULTS,THRESHOLD_PROVENANCE_AND_SENSITIVITY,
STATIC_PROXY_RULE_POLICY}.md`, `CURRENT_TO_TARGET_GAP_V4.md`, this
report. **Additively edited**: `src/doar/{analysis,schemas,
registry_v2_build,structured_report,judge_schemas,reports,case_output}.py`
(`case_output.py` required no code change — it already passes the full
analysis dict through, so the new judge/fields flow automatically),
`doar_prototype_app.py`, `DOAR_TRACE_MASTER_SPEC.md`,
`docs/RULE_PDF_COVERAGE_AUDIT_V2.md`,
`resources/psychology_sources/rules_registry_v2.json`,
`tests/{test_source_catalog_and_registry_v2,test_judge_schemas,
test_claims_pipeline_auto_wiring,test_structured_report,
test_prototype_app_smoke}.py`. **Never touched**: `rules.py`,
`rules_registry.json` (the historical production engine/registry) —
Phase 2A is entirely additive/parallel, per the explicit constraint.

## 3. Page-frame audit results

120-image, class-balanced, non-test sample: **only 12.5% assessable**
(5 `full_page_detected`, 10 `likely_full_page`, 8 `uncertain`, 97
`cropped_or_content_only`). Reported prominently before continuing
implementation, per the task's stop-condition instruction — proceeded
because Section 3's own `not_assessable` design exists specifically to
handle this. Cross-validated by an independent 200-image sample
(`rule_trigger_distribution.csv`): 31–32/200 (~15.5–16%) assessable.
Full detail: `docs/PAGE_FRAME_ASSESSABILITY.md`.

## 4. Feature ground-truth errors

22 synthetic checks: **20 pass, 1 not_applicable
(`colour.light_ratio` does not exist), 1 documented real fail**
(`segmentation.border_touch_ratio` — `_segment`'s morphological cleanup
erodes content touching the image's own border row; root-caused, not
fixed, out of this module's validation-only scope). Full detail:
`docs/FEATURE_MEASUREMENT_VALIDATION.md`.

## 5. Feature robustness results

880 real measurements (4 images × 20 transforms × 11 features):
**351 pass, 508 expected_change, 21 real documented fails** — mostly
`segmentation.foreground_coverage`'s sensitivity to resizing (13/21) and
brightness/contrast sensitivity in segmentation thresholding (7/21).
The 2 new stroke-proxy features had only 1 fail each out of 80
measurements. Full detail: `docs/FEATURE_ROBUSTNESS_RESULTS.md`.

## 6. Threshold sensitivity results

300 real images: the one directly-sourced threshold (`coverage_small`
≤0.20) triggers on **0%** across nearly its entire swept range; the one
clearly-invented threshold (`coverage_full` ≥0.90) triggers on **86%**
and stays above 77% across its swept range. Full sweep table and the 4
new rules' quartile-derived thresholds (from an 80-image sample,
seed=11): `docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`.

## 7. Newly executable rules

**4**: `EN_COMPILED_PLACEMENT_CENTER_029` (page-frame-gated,
93.55% trigger rate among its 31 assessable cases),
`EN_COMPILED_LINE_HEAVY_PRESSURE_030` (18.09%/199),
`EN_COMPILED_LINE_LIGHT_PRESSURE_031` (32.66%/199),
`EN_COMPILED_LINE_SHAKY_BROKEN_032` (20.10%/199) — real counts from a
200-image sample, `artifacts/phase2a/rule_trigger_distribution.csv`.
Wired via a new, additive, parallel evaluator (`rule_engine_v2.py`);
`rules.py`/`rules_registry.json` untouched. Heavy/light pressure are
mutually exclusive **by construction** (opposite quartiles of the same
distribution), verified across a dense 1001-value sweep. **10 of 41**
registry-v2 rules are now executable (up from 6).

## 8. Rules kept disabled and why

**31 of 41** remain `disabled`. 2 were specifically re-audited and
deliberately not activated despite plausible-sounding names —
`EN_COMPILED_EXCESSIVE_DETAIL_040` (no feature distinguishes "fixation"
from general edge/component density) and
`EN_COMPILED_NEGLECT_BACKGROUND_041` (requires a subject/figure
detector to distinguish "no background" from "the subject is the whole
drawing," which does not exist). The other 29 need an object/face/
shape/symbol detector, describe the drawing *process* rather than the
finished image, require multiple drawings over time, or have no
physical-scale reference available. Full per-rule audit:
`docs/STATIC_PROXY_RULE_POLICY.md`.

## 9. Real trigger distributions

`artifacts/phase2a/rule_trigger_distribution.csv` — all 41 rules, 200
real non-test images (50/class, seed=7), reproducible (byte-identical
across 2 independent runs). Notable: `PSY_AR_SIZE_SMALL_016` (the one
directly-sourced threshold) triggers on **0/31** assessable cases in
this independent sample too, consistent with §6's 300-image finding.
None of these counts are interpreted as psychological prevalence —
stated explicitly in the CSV-producing script's own docstring and in
every doc that cites it.

## 10. Parent and Technical view examples

Both views render against any real case directory under
`outputs/prototype_cases/` produced by `main.py analyze-image` or the
app's own upload flow — e.g. any of the 24 cases exercised in the §11
smoke run below. Representative code paths:
`doar_prototype_app.py` lines 167–323 (Parent view, including the new
page-frame-unverifiable warning and proxy-wording captions) and
319–517 (Technical view, including the new page-frame panel, threshold/
dependency-group columns, and measurement-validation panel). No
generated example JSON is committed to the repo (`outputs/` is
gitignored per existing convention) — run any real case through the
app to see both views live.

## 11. Exact test/lint/compile/smoke results

- Phase 1.5 baseline (start of this task): 610 passed, 0 failed.
- **Final: 639 passed, 0 failed, 0 skipped** (`pytest tests/ -q`).
- `ruff check src main.py tests doar_prototype_app.py`: all checks passed.
- `python -m compileall -q src main.py tests doar_prototype_app.py`: clean.
- **24 real, non-test, class-balanced images** (6/class, seed=99) run
  through the full `analyze_image` pipeline end-to-end (not a unit-test
  mock): **24/24 succeeded, 0 errors**.
- Streamlit `AppTest` suite: **5/5 passed**, including a real-checkpoint
  run.
- All 5 Phase 2A experiment scripts run **twice** each on their full
  sample: **byte-identical output both times**, every script (verified
  by direct diff, not just an assertion).

## 12. Top 5 unresolved risks

1. **Only ~12.5–16% of this dataset is page-frame-assessable** — the
   single largest constraint on how often 7 of the 10 executable rules
   can say anything at all here. Not fixable without either a real page
   detector or a different dataset; explicitly out of this phase's
   scope.
2. **`segmentation.border_touch_ratio` has a real, unfixed measurement
   bug** (border-row erosion in `_segment`'s morphological cleanup) —
   found and documented, not patched, since this module's scope is
   validation only.
3. **`coverage_full` (≥0.90) and genuine full-page detectability are
   close to mutually exclusive** under the current page-frame border
   band — discovered while migrating tests; not yet resolved, and it is
   not obvious which side (the coverage threshold or the border-band
   width) should move.
4. **`foreground_coverage`'s resize sensitivity** and segmentation's
   brightness/contrast sensitivity on real photographic content are
   real and documented but unaddressed — could shift any
   coverage-based rule's real trigger rate depending on upload
   pipeline/compression choices not currently controlled for.
5. **The 4 new rules' confidence ceilings (0.10) and the 3 stroke
   thresholds (real-data quartiles) are honestly labeled as invented/
   exploratory, not expert-reviewed** — the generated review CSVs from
   Phase 1.5 (`artifacts/expert_review/*.csv`) do not yet cover these 4
   rules; extending them is a natural next step but was not done here.

## Recommendation for Phase 2B

**Do not begin automatically — this report is the stopping point, per
instruction.** If/when Phase 2B is authorized, the highest-leverage next
steps in priority order appear to be: (a) extend the expert-review CSV
instruments to cover the 4 newly-activated rules and their thresholds;
(b) investigate whether the page-frame border-band width or the
`coverage_full` threshold should be reconciled, given finding §12.3; (c)
decide whether `segmentation.border_touch_ratio`'s erosion bug is worth
fixing given its actual usage surface. None of these require an object
detector, an LLM, or any model retraining.
