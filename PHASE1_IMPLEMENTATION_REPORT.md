# DOAR-TRACE Phase 1 Implementation Report

**Status: real, executed, tested.** Branch `feature/doar-trace-foundation`,
based on `247562d` (tagged `checkpoint/pre-doar-trace-foundation`), never
merged to main. Every section below was actually run, not projected.

## What was reused (not reimplemented)

`schemas.py::Evidence/Analysis`, `case_output.py::write_versioned/
finalize_case`, `features.py::objective_feature_row/FeatureValue`,
`rules.py::evaluate_rules`, `concerns.py`'s convergence-strength policy
(referenced, not imported, for `structured_report.py`'s aggregator),
`judges.py::run_judges` and its `DIAGNOSTIC_PATTERNS`/`_ARABIC_DIAGNOSTIC`
(imported directly into `claim_verifier.py`, not duplicated),
`rule_schema.py`/`RULE_PROVENANCE` (this session's earlier work, reused
as the source/page/threshold-provenance layer for registry-v2),
`doar_prototype_app.py`/`parent_view.py`/`profile.py`/`chat.py` (extended,
not rewritten).

## What was added

| File | Purpose |
|---|---|
| `src/doar/trace_evidence.py` | 4A: `EvidenceRecordV2` (PASS/WARN/FAIL/ABSTAIN) + `EvidenceSetV2` |
| `src/doar/trace_evidence_adapters.py` | 4A: converts existing `FeatureValue`/`Evidence` into v2 records |
| `src/doar/objective_features_report.py` | 4B: builds `objective_features.json`'s per-feature envelope |
| `src/doar/registry_v2_build.py` | 4C: builds `rules_registry_v2.json` |
| `resources/psychology_sources/rules_registry_v2.json` | 4C: the draft registry itself (19 rules) |
| `docs/RULE_PDF_COVERAGE_AUDIT.md` | 4C: coverage audit, missing-PDF disclosure |
| `src/doar/structured_report.py` | 4D: `structured_analysis.json` planner + theme aggregator + contradiction detector |
| `src/doar/claim_verifier.py` | 4F: `Claim`/`verify_claims` and 7 deterministic checks |
| `src/doar/judge_schemas.py` | 4G: `JudgeVerdict` + all 8 judges (4 real wrappers, 4 honest stubs) |
| `DOAR_TRACE_MASTER_SPEC.md` | Policy: wording rules, rule ontology taxonomy, evidence-v2 contract |
| `CURRENT_TO_TARGET_GAP_V2.md` | Pre-implementation verification against code |
| Edits to `schemas.py`, `analysis.py`, `case_output.py`, `doar_prototype_app.py`, `tests/test_prototype_app_smoke.py` | Additive wiring: `objective_features` field, unconditional feature computation, `objective_features.json`/`structured_analysis.json` persistence, new Parent/Technical view sections |

9 new test files (`test_trace_evidence_v2.py`, `test_objective_features_persistence.py`,
`test_registry_v2.py`, `test_structured_report.py`, `test_claim_verifier.py`,
`test_judge_schemas.py`, plus extensions to `test_prototype_app_smoke.py`),
**96 new tests**, all passing.

## What remains unavailable (honestly, not silently)

- Object detection: still nothing (out of scope, per the task).
- 13 of 19 rules: still `missing_detector`/`disabled` — no new detector was built.
- `detection_judge`, `relation_judge`, `aggregation_judge`, `language_judge`: schema only, `status="not_implemented"` always, mechanically enforced (constructing a `not_implemented` verdict with no reason raises).
- `combined_hypothesis_only`: implemented and unit-tested, but **cannot occur on any real image today** — every one of the 6 currently-executable rules maps to a unique `target_construct` (no two rules share a construct), so the aggregator never has 2 rules to combine in practice. This is stated plainly in `docs/RULE_PDF_COVERAGE_AUDIT.md` and proven with synthetic data in `tests/test_structured_report.py`.
- No LLM anywhere (not attempted, out of scope for this phase).
- `child_drawing_rules_compiled.pdf` still does not exist in this repository; `rules_registry_v2.json` was built entirely from `التحليل النفسي للصور.pdf` instead.

## Assumptions made (all documented at the point they were made)

1. `trace_evidence.py`'s PASS/WARN/FAIL/ABSTAIN vocabulary is a judge-style verdict on evidence trust, deliberately distinct from `evidence_schema.py`'s existing extractor-availability vocabulary — the two coexist rather than one replacing the other.
2. `unit` is `None` for every objective feature in `objective_features.json` — `features.py` tracks no physical units today; inventing them would be fabrication.
3. `judge_status` is `"not_evaluated"` for every feature — no per-feature judge exists yet, honestly stated rather than defaulted to a fake "pass".
4. Registry-v2's `registry_v2_status` is `already_in_production_registry` for all 19 rows, never the task's default `candidate_unreviewed` — because none of them are actually new (see the PDF-coverage finding).
5. `psychologist_review_status` is `supplied_by_named_source_not_independently_reviewed` for all 19 rules — no second psychologist's review exists anywhere in this repository to cite.

## Rules imported from the PDF (all 19)

See `docs/RULE_PDF_COVERAGE_AUDIT.md` for the full page/section table. 19
of 19 PDF rows covered; 0 missing; 0 fabricated.

## Rules not operational, and why

- 12 rules (`static_detector`): no detector exists (eyes×3, animals×4, geometry, stars, circles, transport, hearts).
- 1 rule (`process_required`, flowers/clouds/sun): its precondition ("while distracted") is not observable from a static finished image, even with a perfect detector.
- 6 rules (`static_direct`) **are** operational — the only 6 that were already executable before this task, now additionally carrying the full registry-v2 taxonomy.

## Limitations of drawing-based low-mood/distress wording

The allowed cautious wording (`DOAR_TRACE_MASTER_SPEC.md` §1) is only
ever reachable at `combined_hypothesis_only` level, which — per above —
cannot occur with the current 6-rule executable set on real data. In
practice, today's system can only reach `question_generating` (a single,
low-confidence, unvalidated observation), never the stronger multi-feature
"possible pattern" wording the task's own example shows. This is a real,
current ceiling, not a design flaw: reaching it honestly requires either
more executable rules sharing a construct, or a real detector unlocking
tier-2 rules — both explicitly out of scope for this increment.

## No claim of clinical validation

Every rule in `rules_registry_v2.json` carries `scientific_support` values
already present in the production registry (`not_found_for_specific_claim`,
etc.) and a `psychologist_review_status` stating it was never
independently reviewed. Nothing in this increment changes or strengthens
any of those values.

## Exact commands used

```bash
git status && git branch --show-current && git rev-parse HEAD && git log --oneline -5
git tag checkpoint/pre-doar-trace-foundation
git branch feature/doar-trace-foundation && git checkout feature/doar-trace-foundation
python -m unittest discover -s tests -p "test_*.py"
python -m ruff check src main.py tests
python -m compileall -q src main.py tests doar_prototype_app.py
python main.py analyze-image --image "<real dataset image>" \
  --output outputs/doar_trace_smoke/case001 \
  --emotion-checkpoint outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt
```

## Exact test results

- Baseline (before this task's edits): 471 passed, 0 failed, 0 skipped.
- After 4A: 493 (471 + 15 + 7).
- After 4B: 493 (7 counted above were 4B's own).
- After 4C: 513 (+20).
- After 4D: 526 (+13).
- After 4E: 527 (+1).
- After 4F: 550 (+23).
- After 4G: 567 (+17).
- **Final: 567 passed, 0 failed, 0 skipped.** Ruff: clean. `compileall`: clean.
- Real-image smoke test: `analyze-image` against a real, non-test-split dataset drawing with a real trained checkpoint (`efficientnet_b0_seed_42`) — produced correct real output (`top_class=Happy, confidence=0.93`, matching the folder label), 59/59 objective features persisted, 1 candidate theme (`self_esteem`), all 4 real judges `pass`, all 4 interface-only judges `not_implemented`, one hand-built claim verified `passed=True` against it. The locked final test split was not accessed.

## Next recommended phase

Per the task's own "do not begin Phase 2 automatically" instruction, this
is a recommendation, not a start: the highest-value next step is either
(a) annotating a small real sample to pilot one `static_detector` (most
likely the eye/face family, since `PHASE3_DETECTOR_EVALUATION_PLAN.md`
already scopes it) so at least one tier-2 construct becomes real, or (b)
wiring `claim_verifier.py`/`judge_schemas.py` into `case_output.py` so
every case gets a persisted `judges_v2.json`/verification report
automatically rather than only being callable ad hoc as demonstrated
above. Both are additive and do not require an LLM or new detector
training beyond (a)'s own annotation/evaluation cycle.
