# Current-to-Target Gap (v2) — DOAR-TRACE Foundation

**Status: pre-implementation verification, written before any Phase 1 code
in this increment.** Every claim below was checked against the executable
code on branch `feature/doar-trace-foundation` (base commit `247562d`,
checkpoint tag `checkpoint/pre-doar-trace-foundation`), not assumed from
prior documentation.

## Baseline (recorded before any edit)

- Python 3.11.9, PyTorch 2.7.1+cu118 (CUDA available, Quadro P3200), scikit-learn 1.9.0.
- `python -m unittest discover -s tests -p "test_*.py"`: **471 passed, 0 failed, 0 skipped**.
- `ruff check src main.py tests`: all checks passed.
- `python -m compileall -q src main.py tests`: clean.

## Verified facts (Section 1 of the task)

| Question | Answer | Evidence |
|---|---|---|
| Rules in registry | 19 | `resources/psychology_sources/rules_registry.json` |
| Executable | 6 | tier-1 composition/placement rules, `rules.py::_TIER1_DISPATCH` |
| Blocked by missing detector | 13 | tier-2 rules, `rules.py::_status_for` returns `missing_detector` unconditionally |
| Objective feature count | 59 | live run of `features.py::objective_feature_row` this session |
| Features marked unavailable | 2 | `shape.enclosed_shape_count`, `shape.repetition_score` (honest NaN, not fabricated) |
| Persisted in normal non-fusion inference? | **No** | `emotion.py:85-95`: `objective_feature_row` is called only when `payload["checkpoint_type"] == "doar_fusion_bundle_v1"` |
| Object detection | Nothing genuine | `detectors/` is schema-only (`schema.py::DetectorResult`), imported by no user-facing path; `case_output.py:61` writes a hardcoded unavailable stub for every case |
| Report/chat functionality | Real, deterministic, no LLM | `reports.py` (bilingual HTML), `qa.py` (grounded deterministic Q&A), `chat.py`+`parent_view.py`+`profile.py`+`doar_prototype_app.py` (prior session, working dual-view prototype) |
| Judges | 6 real (`judges.py::run_judges`): quality, segmentation, feature, emotion, rule, safety — not the 8 this task specifies (detection/relation/model/aggregation/language are entirely new) |

## Missing input, resolved by substitution (documented, not silently worked around)

`child_drawing_rules_compiled.pdf` is named throughout this task's
instructions but **does not exist anywhere in this repository** — a
full-tree search (`find . -iname "*.pdf"`) finds exactly one PDF:
`التحليل النفسي للصور.pdf` (repo root). That PDF was already fully
extracted and traced to all 19 registry rules in this same session
(`RULE_FEATURE_COVERAGE.md`, `CURRENT_SYSTEM_AUDIT.md`) — a clean 1:1
mapping, zero PDF bullets left unmapped, zero registry rules without a
source sentence. **This increment uses that PDF as the sole rule-ontology
source.** Practical consequence for 4C: registry-v2's requirement to
capture "every distinct feature/rule row from the PDF" is, on the actual
available source, identical to "the existing 19 rules" — there is no
20th row to discover. This is stated plainly in
`docs/RULE_PDF_COVERAGE_AUDIT.md` rather than silently assumed.

## Relationship to this session's earlier Phase-1 evidence/rule work

Two tasks ago in this same session, an evidence-to-rule vertical slice
was already built: `src/doar/evidence_schema.py` (`EvidenceItem` with
`status ∈ {available, experimental, unavailable, failed,
insufficient_evidence, requires_manual_review}`), `rule_schema.py`
(`RuleV2` with source page/section/`threshold_source`),
`evidence_rule_engine.py` (outcome vocabulary `triggered/not_triggered/
insufficient_evidence/not_applicable/extractor_unavailable/
requires_manual_review/conflicting_evidence`), `evidence_adapter.py`,
`english_report.py`, `evidence_pipeline.py`. 471 tests currently depend
on these exact field names and vocabularies (49 of them written for this
exact code) and real GPU experiments consume the sibling `features.csv`
extraction path unchanged — **none of it is modified or renamed in this
increment.**

This task's 4A explicitly specifies a **different** evidence vocabulary
(`evidence_type`, `producer`/`producer_version`, `status ∈ {PASS, WARN,
FAIL, ABSTAIN}`, `provenance`, `source_evidence_ids`) — a judge-style
pass/fail/abstain verdict on a piece of evidence, not an
extractor-availability state. These are genuinely different axes (one
says "was this ever measured," the other says "do we trust the
measurement"), not a redundant rename, so **this increment adds a new,
separate module (`trace_evidence.py`) rather than renaming or replacing
`evidence_schema.py`.** Both coexist; `docs/RULE_PDF_COVERAGE_AUDIT.md`
and `PHASE1_IMPLEMENTATION_REPORT.md` cross-reference the relationship
explicitly so a future reader is not left guessing why two evidence
schemas exist.

## Gaps this increment closes (and nothing beyond)

1. No evidence-v2 schema with the PASS/WARN/FAIL/ABSTAIN + producer/provenance/source_evidence_ids vocabulary → **4A**.
2. Objective features never persisted outside the fusion-checkpoint branch → **4B**.
3. No registry-v2 draft with observability class, allowed-output level, evidence family, target construct, alternative explanations, psychologist-review status → **4C**.
4. No `structured_analysis.json` planner or dependency-aware theme aggregator → **4D**.
5. Parent/technical views exist but lack expandable full-feature/full-rule sections and the new structured-analysis fields → **4E**.
6. No deterministic claim-verifier → **4F**.
7. No judge-interface schemas for detection/relation/model/aggregation/language judges (5 of 8 required judge types don't exist even as an interface) → **4G**.

## Explicitly out of scope for this increment (per the task's own instruction)

New object detector, model training, Gemini/external LLM integration,
rewriting the training pipeline, replacing the production
`rules_registry.json`, merging to main.
