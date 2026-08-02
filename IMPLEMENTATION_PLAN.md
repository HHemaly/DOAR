# DOAR — Implementation Plan

Phased, incremental, each phase scoped to be reviewable on its own. Phases marked **APPROVAL GATE** must not start until you confirm the recommendation in `PROPOSED_ARCHITECTURE.md`/`DECISION_LOG.md`. Phases marked **SAFE — PROCEED** are behavior-preserving (bug fixes, docs, tests, additive schema fields) and will continue without re-confirming each one, per the working rules ("continue autonomously through safe audits, documentation corrections, tests, bug fixes, refactoring that preserves behaviour").

## Phase 0 — Mandatory initial audit — **DONE (this session)**
`CURRENT_STATE_AUDIT.md`, `RULE_COVERAGE_MATRIX.csv`, `RULE_SOURCE_REGISTER.csv`, `SCIENTIFIC_LIMITATIONS.md`, `PROPOSED_ARCHITECTURE.md`, this file, `DECISION_LOG.md`.

## Phase 1 — Safe local fixes (no material decisions) — **SAFE — PROCEED**
1. Fix `resolve_leakage()` so `conflicting_labels` images are added to `leaked_image_ids` (currently computed but not quarantined — `CURRENT_STATE_AUDIT.md` §2, item 1). Add a regression test asserting a synthetic conflicting-label pair is removed from `clean_dataset`.
2. Fix `main.py`'s `evaluate --deep-comparison` path (`CURRENT_STATE_AUDIT.md` §2, item 2) — either dispatch on checkpoint suffix like `probability_export.py` already does, or clearly deprecate the direct `evaluate` command in favor of `export-probabilities`/`evaluate-predictions` and update `docs/RUN_GUIDE_WINDOWS.md`/`LOCAL_WORKFLOW_WINDOWS.md` to match reality. Add a test that currently would fail (crash) and passes after the fix.
3. Mark `docs/PSYCHOLOGIST_SOURCE_AUDIT.md` as superseded by this audit (append a note at the top; do not delete — the PDF is in fact readable, see `CURRENT_STATE_AUDIT.md` §5.1) and commit the freshly-ingested draft registry output to `resources/psychology_sources/` (currently only exists ad hoc under `outputs/full_run/psych_ingest`).
4. Add `original_source_label` as an additive, optional field written by `analyze-image`/`predict-image` whenever the image path is inside a known ImageFolder split — non-destructive, does not touch inference (verified no leakage exists there today), just records provenance for later audit comparison. Add the `CONSISTENT/POSSIBLE_CONFLICT/UNCERTAIN/REVIEWED_CONFIRMED/REVIEWED_CORRECTED/ADJUDICATION_REQUIRED` status as a computed-after-the-fact field, never fed back into inference.
5. Version the per-case JSON outputs: replace `case_output.py::_write()`'s unconditional overwrite with an additive scheme (`analysis.v{n}.json` + a small `index.json`), preserving all prior versions — closes the immutability gap in `CURRENT_STATE_AUDIT.md` §6. `clinician_review.json`'s existing "don't clobber" pattern generalizes to this.
6. Expand test coverage for the gaps this audit found: a test that directly asserts 13/19 rules return `missing_detector` (documents the current state so future detector work is measured against a known baseline, not silently "fixed" by moving goalposts); a test that asserts `derive_concerns()` cannot converge even with `enabled=True` given rules-engine-shaped input (documents the source-type bug until Phase 2 fixes it).

None of the above changes the principal model, the database, any public interface, or the research protocol — all are additive/corrective within the existing design.

## Phase 2 — Rule tiering, free-drawing gating, aggregation fix — **SAFE — PROCEED** (extends already-tested logic; no new dependency, no model/DB/API change)
1. Add a `tier` field to every entry in `rules_registry.json` (values from `RULE_COVERAGE_MATRIX.csv`, already assigned this session) without changing any existing rule's wording, citations, or confidence ceiling.
2. Rewrite `rules.py::_status_for()` as tier-aware: Tier 1 evaluates as today; Tier 2 must check a real "positively detected" signal (which, until Phase 3 ships a detector, always resolves to `DETECTOR_UNAVAILABLE`/`missing_detector` — behavior-identical to today, just now for a documented reason rather than an implicit default branch); Tier 3 always returns `NOT_ASSESSABLE_CONTEXT_UNKNOWN` and is excluded from ordinary free-drawing inference unless prompt metadata is later supplied and approved.
3. Fix the `source_type` bug in `concerns.py`/`rules.py` (`CURRENT_STATE_AUDIT.md` §5.3) so evidence is tagged by its *actual* origin (objective measurement vs. clinician-symbolic hypothesis vs. model prediction), not a blanket string.
4. Implement the four-level aggregation vocabulary (`INSUFFICIENT`/`WEAK_HYPOTHESIS`/`POSSIBLE_FOR_EXPLORATION`/`PROFESSIONAL_REVIEW_SUGGESTED`) per `PROPOSED_ARCHITECTURE.md` §5, evidence-family-based per the spec's correlated-evidence rule (page-geometry rules must not double-count as independent confirmations of each other).
5. Full regression pass: every existing test must still pass; add new tests for tier gating and the fixed aggregation.

## Phase 3 — Detector layer — **APPROVED (2026-08-02), scope: all three detector types**
Approved order: face/person, then shape/symbol, then animal-species — all three attempted, per user decision in `DECISION_LOG.md` (overriding the narrower face+shape-only recommendation). Accepted risk: animal-species may not clear a defensible validation bar given the line-drawing-vs-photograph style gap; if so, `tiger_or_wolf`/`fox`/`squirrel`/`lion` revert to `DETECTOR_UNAVAILABLE` rather than ship unvalidated. No labeled validation data exists yet for any of the three — first concrete task is scoping an annotation effort (small held-out sample of the local dataset, hand-labeled for face/person presence, shape/symbol presence, and animal species) before any detector is judged "adequate."

## Phase 4 — Persistence upgrade (SQLite index layer) — **APPROVED (2026-08-02)**
Per `PROPOSED_ARCHITECTURE.md` §6 and `DECISION_LOG.md`. Additive only (new `doar.db` alongside existing files, existing JSON/CSV artifacts remain the source of truth for large content, SQLite indexes and tracks provenance/status/review-queue state), with a migration script that can be re-run idempotently against the existing `outputs/` tree, and a rollback path (the file-based system keeps working if the DB is deleted).

## Phase 5 — Emotion-model experiment expansion — **SAFE — PROCEED once scoped**
Per the working spec §11: before adding any new architecture, estimate relevance/compute cost/expected thesis value for each candidate (objective baseline — already done; existing CNN/MobileNetV3 — already done this session; EfficientNet, ConvNeXt-Tiny, frozen DINOv2 + shallow classifier, SigLIP2 — not yet attempted). Given this session's reduced 2-model/1-seed sweep was a time-budget compromise, not a final answer, the next concrete step is the full documented sweep (4+ architectures × seeds `[42,123,2026]`) already supported by `compare-deep-models` — this is re-running existing, already-verified code at full scope, not new development, so it proceeds without a separate approval gate. Report every metric the spec requires (macro-F1, balanced accuracy, per-class P/R/F1, confusion matrices, calibration, uncertainty, CIs, runtime, parameter count, model size, hardware/software/config/seed/split-ID/checkpoint-checksum) — `thesis.py`/`experiments.py` already capture most of this; verify coverage and extend where missing.

## Phase 6 — Label-quality audit — **SAFE — PROCEED once scoped**
SHA-256/perceptual-hash duplicate and cross-label-conflict detection already exist and are proven (`leakage.py`, exercised this session). New work: out-of-fold predictions, multi-seed/multi-model disagreement ranking, optional Confident Learning integration, and the blind professional-review queue UI in Streamlit. No model/DB/API decision implied — can proceed once Phase 1's `original_source_label`/audit-status fields exist to hang results on.

## Phase 7 — Optional external-AI audit layer — **APPROVAL GATE on model choice only**
The deterministic-validator architecture in `PROPOSED_ARCHITECTURE.md` §7 can be built and tested with a stub/mock model (or no model, always returning `CANNOT_DETERMINE`) without needing to pick a real LLM. Picking an actual external model (which one, and confirming the system remains fully functional with it disabled, per spec §13) is the gated part.

## Phase 8 — Testing, deliverables, documentation close-out — **SAFE — PROCEED, ongoing**
Expand `tests/` per the working spec §19 checklist as each phase above lands (tests are added alongside their feature, not batched at the end). Keep `MODEL_CARD.md`, `DATASET_CARD.md`, `VALIDATION_PROTOCOL.md`, `CHANGELOG.md`, `RULE_CATALOG.md` current as each phase completes — not written speculatively ahead of the work they describe.

---

## Sequencing note

Phases 1 and 2 do not depend on any approval and are the highest-value, lowest-risk next work — they fix two real bugs, close the biggest documentation-vs-reality gap (rule tiering matching what's actually wired), and fix a structural bug in the concern-aggregation engine, all while keeping every existing test green. Recommend starting there immediately after you've reviewed this audit, in parallel with your decision on Phases 3/4's approval gates.
