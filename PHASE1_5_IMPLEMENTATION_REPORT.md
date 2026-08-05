# DOAR-TRACE Phase 1.5 Implementation Report

**Status: real, executed, tested.** Branch `feature/doar-trace-foundation`,
starting commit `2e89ff5` → `2030d18` (compiled PDF added per your
instruction), checkpoint tag `checkpoint/pre-doar-trace-phase1-5`. Never
merged to main. Every section below was actually run, not projected.

## 1. Phase 1 audit (done first, before any Phase 1.5 code)

Full findings recorded in `CURRENT_TO_TARGET_GAP_V3.md` Section 1
(answers to the 8 required audit questions). Headline finding: Phase 1's
`structured_report.py` grouped rules by a 1:1 `target_construct`, so a
**single** triggered rule (e.g. `coverage_full` alone) was already a
"group of one" and rendered as a candidate drawing-level theme
(`self_esteem`) in the Parent view — confirmed by direct reproduction
before any fix was applied. This is the central bug Phase 1.5 fixes.

## 2. What was reused (not reimplemented)

`rule_schema.py`/`RULE_PROVENANCE` (left unchanged — still the correct,
tested provenance layer for the unrelated `evidence_rule_engine.py`
path), `judges.py::run_judges` (wrapped, not reimplemented, by 4 of the 8
v2 judges), `claim_verifier.py`/`judge_schemas.py` (Phase 1's real logic,
now wired automatically instead of ad hoc), `case_output.py::write_versioned`,
`emotion.py`'s calibrated model output, `parent_view.py`'s existing
plain-language helper functions.

## 3. Files changed

**New**: `src/doar/{source_catalog_build,construct_registry_build,claims_pipeline,expert_review_forms}.py`,
`resources/psychology_sources/{source_rule_catalog,construct_registry}.json`,
`docs/{CONSTRUCT_MAPPING_RATIONALE,AGGREGATION_POLICY,RULE_PDF_COVERAGE_AUDIT_V2}.md`,
`CURRENT_TO_TARGET_GAP_V3.md`, `artifacts/expert_review/*.csv` (3 files),
`tests/{test_source_catalog_and_registry_v2,test_claims_pipeline_auto_wiring,test_expert_review_forms}.py`.
**Fully rewritten**: `src/doar/registry_v2_build.py`,
`src/doar/structured_report.py`, `resources/psychology_sources/rules_registry_v2.json`,
`tests/test_structured_report.py`. **Additively edited**:
`src/doar/{case_output,judge_schemas}.py`, `doar_prototype_app.py`,
`tests/{test_judge_schemas,test_prototype_app_smoke}.py`,
`DOAR_TRACE_MASTER_SPEC.md`. **Deleted**: `tests/test_registry_v2.py`
(tested the fully-superseded Phase 1 internal API; its coverage is
re-provided, expanded, in `test_source_catalog_and_registry_v2.py`).

## 4. Source-row and rule counts

| Metric | Count |
|---|---|
| Source rows (both PDFs) | **67** (19 Arabic + 48 English) |
| Candidate-rule source rows | 62 (19 + 43) |
| Explicit relationships (near_duplicate/expands/related_but_distinct) | 31 |
| Total registry-v2 rules | **41** (19 original production IDs preserved + 22 new) |
| Directly executable (`individual_heuristic_only`) | 6 (unchanged from Phase 1 — no new detector built) |
| `disabled` (no detector/evaluator wired) | 35 |
| Observability classes used | all 6 that apply: `static_direct` (7), `static_detector` (24), `static_proxy` (3), `process_required` (3), `longitudinal_required` (3), `not_operational` (1) — `prompt_required`/`age_required` unused, no rule in either source needs them |

Full detail: `docs/RULE_PDF_COVERAGE_AUDIT_V2.md`.

## 5. Construct counts and mappings

**12 constructs** (`construct_registry.json`). **27 of 41 rules** map to
exactly one construct; **14 deliberately left unmapped** (personality-
trait or bidirectional source wording — `docs/CONSTRUCT_MAPPING_RATIONALE.md`
records every decision). No rule was force-mapped to manufacture
convergence.

## 6. Which combined patterns can now occur on real cases

**Exactly one, mechanically**: `fear_or_insecurity_pattern`, from
`PSY_AR_SIZE_SMALL_016` (triggered) + a calibrated Fear prediction
>=0.50 confidence from the expressive-content model — verified correct
by both unit tests (`tests/test_structured_report.py`) and a direct
`build_structured_analysis()` call with real rule-evaluation data
(`docs/AGGREGATION_POLICY.md`).

**On the 9 real/CLI cases run for this report (§8), zero reached Level
C** — not a bug: every one of these real dataset drawings had
`bounding_box_coverage` well above the 0.20 `coverage_small` threshold
(a 40-image sample of the `Fear` class found a minimum coverage of
**0.625** — the dataset's drawings are almost always high-coverage), so
`coverage_small` essentially never triggers on this dataset's real
images. This is reported honestly rather than engineering a cherry-picked
"success" case: the mechanism is correct and tested; its real-world
trigger rate on this specific dataset is low.

## 7. Exact test/lint/compile results

- Phase 1 baseline (start of this task): 567 passed, 0 failed, 0 skipped.
- **Final: 610 passed, 0 failed, 0 skipped.**
- `ruff check src main.py tests doar_prototype_app.py`: all checks passed.
- `python -m compileall -q src main.py tests doar_prototype_app.py`: clean.

## 8. Results across real smoke drawings (>=8 required)

9 real, non-test-split dataset images run through the actual
`main.py analyze-image` CLI with the real `efficientnet_b0_seed_42`
checkpoint (case_1-8), plus one small-synthetic-drawing demonstration
(case_9) attempting to reach the Fear/small-coverage combination on a
live checkpoint:

| Case | True class (folder) | Model prediction | Individual suggestions | Combined (Level C) | Claims accepted | `aggregation_judge` |
|---|---|---|---|---|---|---|
| 1 | Happy | Happy (0.87) | SIZE_FULL, PLACE_RIGHT | none | 2/2 | pass |
| 2 | Happy | Happy (0.94) | SIZE_FULL | none | 1/1 | pass |
| 3 | Fear | Fear (0.95) | SIZE_FULL | none | 1/1 | pass |
| 4 | Sad | Sad (0.93) | SIZE_FULL | none | 1/1 | pass |
| 5 | Fear | Fear (0.91) | SIZE_FULL | none | 1/1 | pass |
| 6 | Fear | Fear (0.92) | SIZE_FULL | none | 1/1 | pass |
| 7 | Fear | Fear (0.53) | SIZE_FULL, PLACE_RIGHT | none | 2/2 | pass |
| 8 | Angry | Angry (0.77) | SIZE_FULL | none | 1/1 | pass |
| 9 (synthetic tiny dot) | n/a | Sad (0.84) | SIZE_SMALL | none (model=Sad maps to a different construct than SIZE_SMALL's fear_or_insecurity_pattern) | 1/1 | pass |

**Every claim generated across all 9 cases was accepted** (0 rejected) —
expected, since claims are mechanically built from each case's own real
evidence. **`aggregation_judge` passed on all 9** — no policy violation
was ever produced. The locked final test split was not accessed (all 9
paths confirmed `split=="train"` in `outputs/phase5/manifest.csv` before
use, or fully synthetic).

## 9. Paths to parent and technical examples

`outputs/phase1_5_smoke/case_1/` through `case_9_synthetic_convergence_demo/`
— each has `structured_analysis.json`, `judges_v2.json`,
`generated_claims.json`, `verification_report.json`, and the full
existing case artifact set (`analysis.json`, `reports/*.html`, etc.).
Parent/Technical view rendering verified via the real Streamlit
`AppTest` harness (`tests/test_prototype_app_smoke.py`, 4/4 passing,
including a real-checkpoint case).

## 10. Unresolved decisions (flagged, not silently decided)

1. Whether the 27 rule→construct mappings are appropriate is an open
   question for a human reviewer — `artifacts/expert_review/construct_review_form.csv`
   exists precisely for this, unreviewed as of this report.
2. Whether to invest the small effort needed to wire
   `EN_COMPILED_LINE_HEAVY_PRESSURE_030`/`_LIGHT_PRESSURE_031`/
   `_SHAKY_BROKEN_032` to their already-computed proxy features
   (`stroke.intensity_proxy`, `stroke.fragmentation`) — this would add 3
   more executable rules with zero new detector work, the single fastest
   lever identified in this phase, but was not done here (a genuine new
   evaluator, however small, was judged out of this phase's "no new
   detector wiring" scope and left for your decision).
3. Whether `EN_COMPILED_PLACEMENT_CENTER_029` should also be wired
   (same category of decision as #2, using the already-computed
   `composition.centroid_normalized`).

## 11. Top five risks

1. **Construct mappings are unreviewed domain judgment calls** (see #1
   above) — could be wrong and currently drive real Level-B wording.
2. **The one real Level-C convergence path almost never fires on this
   dataset** (§6) — the feature might look "built but useless" until
   either more executable rules exist or a differently-composed dataset
   is used.
3. **`generated_claims.json`/`verification_report.json` are written for
   every case now**, adding real (small) per-case latency and storage —
   not measured precisely in this report.
4. **Two evidence schemas + a rule-schema/registry-v2-schema split**
   (Phase 1's `evidence_schema.py` vs. `trace_evidence.py`; `rule_schema.py`
   vs. `registry_v2_build.py`) is a growing surface area — fine today,
   a real maintenance cost if it grows further without consolidation.
5. **`aggregation_judge`'s policy re-implements `structured_report.py`'s
   thresholds independently** (by design, so it's a genuine second
   check) — but this means the two must be kept in sync by hand if the
   policy ever changes; no shared single source of truth enforces this
   today beyond the tests.

## 12. Exact recommendation for the next phase

Wire the 3 already-computed `static_proxy`/`static_direct` features
(`stroke.intensity_proxy` → heavy/light pressure, `stroke.fragmentation`
→ shaky/broken lines, `composition.centroid_normalized` → placement
center) into a small new evaluator — zero new detector training, real
increase in executable rule count from 6 to ~10, and a real chance of
unlocking a second Level-C construct (`tension_or_anger_pattern`, via
heavy-pressure + a confident Angry model prediction) on real data. This
is additive, bounded, and does not require object detection, an LLM, or
model retraining.

**Not started automatically, per your instruction.**
