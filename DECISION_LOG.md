# DOAR — Decision Log

Append-only. Never edit or delete a prior entry — if a decision is superseded, add a new entry that references the old one.

---

## 2026-08-02 — Session start: mandatory initial audit performed

**Context**: user requested a full-scope continuation of DOAR toward a spontaneous-free-drawing, evidence-traceable decision-support system, with a mandatory initial audit before any material changes.

**Actions taken (no material/approval-requiring decisions among these):**
- Read/verified every core module in `src/doar/` against actual behavior (not just prior docs).
- Re-ran the full test suite (168/168 passing).
- Ran `main.py ingest-psychology-pdf` against `التحليل النفسي للصور.pdf` — succeeded (prior audits, dated one day earlier, claimed this was blocked; it is not). Output: 43 extracted draft statements, confirms the shipped 19-rule registry is thematically faithful to the source.
- Traced every rule in `rules_registry.json` against `rules.py::_status_for()` — found 13/19 rules structurally cannot fire (no detector exists) and the concern-aggregation engine cannot converge even if its `CONCERNS_ENABLED` flag were set to `True`, due to a `source_type` tagging bug.
- Found two functional bugs by executing the pipeline (not by reading alone): `resolve_leakage()` doesn't quarantine `conflicting_labels` images it detects; `evaluate --deep-comparison` crashes on deep/fusion checkpoints.
- Confirmed: no database, no REST API, no `.git` repository anywhere under the project root.
- Wrote `CURRENT_STATE_AUDIT.md`, `RULE_COVERAGE_MATRIX.csv`, `RULE_SOURCE_REGISTER.csv`, `SCIENTIFIC_LIMITATIONS.md`, `PROPOSED_ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, this file.

**No code was modified in `src/doar/` or `main.py` this session as part of the audit** — this was inspection and documentation only, per Section 3 of the working instructions ("Start by inspecting... Create or update [documents]"). Phase 1 fixes are planned but not yet applied — see `IMPLEMENTATION_PLAN.md`.

---

## 2026-08-02 — DECISION NEEDED: version control

**Finding**: `DOAR-main` (and its parent `DOAR/`) is not a git repository. No `.git` directory exists. There is no commit history to inspect, contrary to the working instructions' assumption ("The current local repository, its Git history... are the source of truth").

**Options considered:**
- (a) Leave as-is, no version control. Rejected — the working instructions explicitly require committing scoped changes ("Commit the completed scoped change if this repository is under Git"), and without git there is no way to review, revert, or diff future changes safely.
- (b) `git init` + one baseline commit capturing current state, then commit each scoped phase from `IMPLEMENTATION_PLAN.md` going forward.

**Decision (user-approved 2026-08-02): (b).** Proceeding with `git init` + `.gitignore` + baseline commit this session.

---

## 2026-08-02 — DECISION NEEDED: persistence / database

**Finding**: current persistence is 100% flat files (JSON/CSV), no SQL engine anywhere. `case_output.py::_write()` overwrites per-case AI-output files unconditionally on re-analysis (no version history); `review.py::append_reviews()` is genuinely append-only for structured psychologist reviews.

**Recommendation**: introduce SQLite as an additive index/query layer (not a replacement for file storage of large artifacts — images, checkpoints, embeddings stay as files). Full reasoning and trade-offs in `PROPOSED_ARCHITECTURE.md` §6.

**Decision (user-approved 2026-08-02): SQLite as an index layer.** Scheduled as Phase 4 in `IMPLEMENTATION_PLAN.md` — not started this session (Phase 1 comes first per sequencing).

---

## 2026-08-02 — DECISION NEEDED: detector-layer scope

**Finding**: 13 of 19 active rules (and every Tier-2-style content-conditional rule the spec anticipates) require an object/symbol/face/animal detector that does not exist anywhere in the codebase. Building these is real, non-trivial computer-vision scope with no existing validation data for any of the three candidate detector types (face/person, shape/symbol, animal-species).

**Recommendation**: prioritize face/person detection first (most likely to have a usable off-the-shelf starting point, though validation on actual line-drawing input is unproven), then shape/symbol (buildable from existing classical-CV connected-component infrastructure), and treat animal-species classification as possibly infeasible at an "adequate documented performance" bar given the style gap between children's line drawings and any realistic training distribution — recommend explicitly deciding to leave those rules permanently `DETECTOR_UNAVAILABLE`/`CATALOGUED_RESEARCH_ONLY` rather than shipping a low-confidence classifier presented as functional.

**Decision (user-approved 2026-08-02): attempt all three, including animal-species**, overriding the narrower recommendation. Accepted risk, recorded here: the animal-species classifier may not reach an "adequate documented performance" bar given the style gap between children's line drawings and realistic training distributions (almost all available animal-classification training data is photographic). If validation on real DOAR-style drawings fails to clear a documented, defensible bar, the corresponding 4 rules (`tiger_or_wolf`, `fox`, `squirrel`, `lion`) must revert to `DETECTOR_UNAVAILABLE` rather than ship a low-confidence classifier presented as functional — this reversion is not a failure of the plan, it is the plan working as designed. No labeled validation data for any of the three detector types exists yet; annotation effort is in scope for Phase 3 and will need its own time/cost estimate once started.

**Status**: approved. Scheduled as Phase 3 in `IMPLEMENTATION_PLAN.md`, after Phase 1/2.

---

## 2026-08-02 — DECISION NEEDED: external AI / API layer

**Finding**: no REST API and no external LLM integration exist today. Both are optional per the spec (system must work fully without either).

**Recommendation**: do not build either speculatively. Build the API only if a concrete consumer (e.g. a separate frontend) is identified. Build the external-AI audit layer's deterministic-validation scaffolding first (testable without a real model), and treat "which LLM, if any" as a separate, later approval-gated choice.

**Status**: no action planned unless requested.

---

## 2026-08-02 — Documentation correction

**Finding**: `docs/PSYCHOLOGIST_SOURCE_AUDIT.md` (dated 2026-07-19) incorrectly states the source PDF could not be read. It can be, and was, this session.

**Decision**: leave the file in place unmodified as a historical record (append-only principle applies to project documentation, not just data), but note the correction in `CURRENT_STATE_AUDIT.md` §11 and here, and mark it superseded in Phase 1 of `IMPLEMENTATION_PLAN.md` rather than deleting or silently rewriting it.

---

## 2026-08-02 — All four gating decisions resolved; Phase 1 begins

User approved, in one batch: (1) `git init` + baseline commit, (2) SQLite as a persistence index layer (Phase 4), (3) attempt all three detector types including animal-species (Phase 3, risk accepted above), (4) originally asked to wait for all three before starting Phase 1 — since all three were resolved in the same exchange, Phase 1 (safe, local, behavior-preserving fixes) starts now.

---

## 2026-08-02 — Phase 1 approved; Phase 2 scope presented and approved

User approved Phase 1 as delivered (commit `a9bbfe8`) and authorized Phase 2 (rule tiering + concern-engine fix) only, explicitly excluding Phase 3 detectors, the SQLite migration, any UI replacement, and the full model sweep. Requested, before any code: (1) confirmation the 4 manually-removed conflicting-label files were deleted only from a generated `outputs/full_run/leakage/clean_dataset` copy, never the original dataset; (2) current git branch/commits/push status; (3) the exact proposed rule-tier schema and concern-aggregation logic; (4) all 19 rules preserved with original Arabic wording, unsupported/detector-dependent rules classified (never silently enabled); (5) the concern-engine source-type fix, kept disabled/experimental; (6) new tests for tiering, missing detectors, unknown context, correlated evidence, contradictory evidence, prohibited diagnostic output; (7) full verification (tests/ruff/compileall/real case) after changes; (8) a separate commit with an exact report of what changed/is operational/remains unvalidated.

**Item 1 verified directly, not just asserted**: all 4 originally-named files confirmed present at their original paths in `C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing` with original June 24 timestamps; original dataset total file count confirmed unchanged at 3,688; the removed files exist only under the `outputs/full_run/leakage/clean_dataset/` copy that `materialize_clean_dataset()` builds via `shutil.copy2()` into a new destination, never touching the source. **Item 2**: branch `master`, commits `894d733` (baseline) → `a9bbfe8` (Phase 1), no remotes configured, nothing pushed anywhere.

The rule-tier schema and concern-aggregation design (item 3) were presented in full before any code was written, per the transcript.

---

## 2026-08-02 — Correction: the "source_type bug" description was inaccurate

While implementing the Phase 2 concern-engine fix, closer reading of `rules.py::evaluate_rules()` and `concerns.py::_source_type()` together showed that the defect described in `CURRENT_STATE_AUDIT.md` §5.3 and in this session's own prior message — "every rule evaluation is mistagged `psychologist_supplied_hypothesis`, so the diversity check can never see two source types" — is **not accurate**. That tagging is intentional and correct: every one of the 19 rules genuinely is a clinician-supplied hypothesis regardless of which measurable feature backs it, and two such rules firing together should not count as independent corroboration of anything.

The actual defect, confirmed by reading the exact line: `evaluate_rules(composition, colour, evidence): del colour, evidence` — the function discarded its `evidence` argument, which is where the emotion model's independent prediction (`ev_emotion_prediction`, source type `model_prediction`) lives. That is the only genuinely independent source anywhere in the system, and it never reached `derive_concerns()`.

This correction was presented to the user before implementation (not silently substituted), with a revised, more narrowly-scoped fix: stop discarding `evidence` in `evaluate_rules()`; pass the emotion-model evidence through to `derive_concerns()` as a real second source; leave `source_type` tagging in `rules.py` unchanged. User approved the corrected fix. `CURRENT_STATE_AUDIT.md` §5.3 and `RULE_COVERAGE_MATRIX.csv`'s concern-engine row are left as historical record of the original (inaccurate) finding, with a note pointing here, per the append-only/RETIRED_NOT_DELETED principle — not rewritten to hide the correction.

---

## 2026-08-02 — Phase 3 planning and validation prep (no detector adopted)

User approved Phase 2 (commit `c0ebb17`) and authorized Phase 3 *planning and
validation preparation only* — explicitly not adoption/enabling of any
detector. Produced: `DETECTOR_DEPENDENCY_MATRIX.csv` (13 Tier-2 rules ×
required object/detector capability/task type/free-drawing applicability/
candidate models & datasets/licence & compute/required annotations/validation
metrics/whether the rule stays scientifically unvalidated even if detection
works — always yes, confirmed per rule against `RULE_SOURCE_REGISTER.csv`);
`PHASE3_DETECTOR_EVALUATION_PLAN.md` (candidate comparison across
person/face, shapes/symbols/objects, and animals, each including photo-domain,
sketch-domain, and drawing-specific options); `docs/PHASE3_ANNOTATION_SCHEMA.md`;
and `src/doar/detectors/` (a `DetectorResult` contract + IoU/component-ID
matching metrics harness — scaffolding only, not imported by any user-facing
code path, no model or download involved).

**Key research findings, verified via web search rather than asserted from
memory**: COCO's 10 animal classes (bird/cat/dog/horse/sheep/cow/elephant/
bear/zebra/giraffe) do not include any of tiger/wolf/fox/squirrel/lion — the
animal category has no matching off-the-shelf detector vocabulary at all.
COCO's vehicle classes (car/truck/bus/train/airplane/boat/bicycle/motorcycle)
do match the `vehicles` rule directly. Quick, Draw! (Google, CC-BY 4.0, 345
categories) is treated per the user's explicit instruction as a
classification-of-isolated-sketch candidate only, not a detection source.
Two datasets not in the original architecture proposal were found and
evaluated: **ESRA** (Roboflow Universe, ~3,002 images of "objects inside
kids' drawings," licence/quality unverified) and **ChildlikeSHAPES** (Meta,
arXiv:2504.08022, Apr 2025, 16,075 pixel-annotated figure drawings — but
authenticity as real child-drawn data vs. artist-created "childlike style"
art is unconfirmed and must be checked before relying on it). A third,
**SceneDAPR** (ACM Web Conference 2024, scene-level free-hand drawings
including children), was considered and **rejected** for this project: it is
fundamentally Draw-A-Person-in-the-Rain data — an instructed, prompt-specific
test — and training on it risks learning instructed-prompt compositional
conventions that DOAR's own free-drawing constraint
(`SCIENTIFIC_LIMITATIONS.md` Section 4) exists specifically to avoid.

**Recommended first experiment**: circles, via classical CV (contour
circularity over DOAR's existing connected-component extraction) — zero new
dependency, zero download, tests whether the existing segmentation pipeline
even produces clean enough crops for any shape-classification approach to be
viable, before investing in any external dataset integration.

No detector was downloaded, adopted, or enabled. No rule's `tier`,
`activation_status`, or runtime `status` changed — verified against a real
`analyze-image` run (13 `DETECTOR_UNAVAILABLE`, 6 `IMPLEMENTED_UNVALIDATED`,
`concerns.json` still `[]`, identical to the Phase 2 verification run).
216/216 tests passing (12 new). Awaiting review before Phase 3 implementation
begins.

---

## 2026-08-02 — Literature review round 1 + Phase 3 validation-plan corrections

User approved Phase 3 planning (commit `eee8e35`) and requested, before any
detector pilot: a literature-based review of primary sources for additional
visual-feature candidates relevant to spontaneous free drawing, a comparison
against the existing 19 rules, and methodological corrections to the Phase 3
validation plan. No detector implementation was authorized or attempted.

**Literature review**: 6 candidates researched and recorded in
`LITERATURE_CANDIDATE_REGISTER.csv`, 2 read directly in full text via PMC
(`LIT_HFD_DEPRESSION_ADULT_004`, `LIT_HFD_DEPRESSION_ADULT_004`'s exact
statistics quoted from source; `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`,
likewise). **No genuinely supported new psychological rule candidate was
found** — every interpretive candidate was either directly self-contradictory
in the literature, structurally unmeasurable from DOAR's static-image input,
far beyond current detector scope, or population/protocol-mismatched (adults
and/or instructed Draw-A-Person tasks, not children's spontaneous drawing).
This is reported as a real finding, not a search failure — see
`IMPROVEMENT_REGISTER.md` category 1. **One genuinely supported, non-
psychological objective feature was found**: scribble-vs-representational
developmental stage, p<.001, buildable from existing DOAR feature
infrastructure with no new dependency — proposed as a Tier-1 descriptive
feature / possible emotion-model covariate, never as an interpretation.

**Comparison against the 19 existing rules** (`IMPROVEMENT_REGISTER.md`):
confirmed the eye rules' and animal rules' existing weaknesses are part of a
broader, repeated pattern (the overwhelming majority of rigorous drawing-
interpretation literature is instructed-protocol research); recommended
demoting face/eye detector work below its round-1 priority given how weak and
internally inconsistent even the best available instructed-protocol evidence
for that rule family turned out to be on closer reading; no rule was deleted,
disabled, or had its registry content changed — only the Phase 3 priority
ordering (a planning artifact) was revised.

**Validation-plan corrections applied to `PHASE3_DETECTOR_EVALUATION_PLAN.md`**
(marked CORRECTED in place, original text preserved, not deleted): 30–50
image samples are now explicitly feasibility-only, never a validation result;
annotators must be blind to folder labels and detector/model output; the
round-1 threshold table's "detector macro-F1 must exceed human kappa" is
retracted as statistically invalid (different metrics, not comparable
numerically) and replaced with a prevalence/CI/cost-informed per-class process;
final sample size is deferred to a staged design (feasibility → prevalence
estimate → sample-size calculation → full annotation) instead of a single
up-front number.

**ESRA/ChildlikeSHAPES/SceneDAPR re-verified via primary sources, not search
summaries**: Roboflow (ESRA) returned HTTP 403 to direct fetch — genuinely
unverified, not just unchecked. ChildlikeSHAPES' arXiv abstract does not
disclose drawing authenticity or licence — genuinely unconfirmed. SceneDAPR's
GitHub page was fetched successfully and fully verified: CC BY-NC 4.0,
access-by-request, 148 categories (6 core DAPR), 1,399 sketches (462 from
ages 8–18, not matched to DOAR's likely population either), confirmed
instructed DAPR protocol. Round 1's "experiment with ESRA" recommendation is
downgraded to "defer pending manual verification"; SceneDAPR's rejection
stands, now on verified rather than summarized grounds.

**Incident during this work, self-corrected**: a one-off script intended to
append 6 new rows to `RULE_SOURCE_REGISTER.csv` opened the file in write mode
before its read had fully completed, and crashed on a pre-existing malformed
row partway through — truncating the file to just its header and destroying
all 22 existing rows. Recovered immediately via `git checkout HEAD --
RULE_SOURCE_REGISTER.csv` (Phase 1's decision to initialize git paid for
itself here) with zero data loss, then re-done with an append-only script
that never reads or rewrites existing content. Recorded here per the
transparency standard applied throughout this log, not omitted because it
was self-corrected.

Nothing was activated, enabled, downloaded, or made user-facing. Awaiting
review before any detector implementation begins.

---

## 2026-08-02 — Phase 2 complete

Implemented exactly the corrected design: `tier` + `activation_status` fields added to all 19 registry rules (content unchanged, verified by diff — only cosmetic array reformatting plus the two new fields); `rules.py::_status_for()` refactored to explicit tier-aware dispatch (behavior-identical for every existing rule, verified by end-to-end test against the real registry); `evaluate_rules()` no longer discards `evidence`; `concerns.py` implements the four-level aggregation vocabulary; `CONCERNS_ENABLED` left `False`. 13 new/updated tests added (2 existing tests updated to reflect the new, spec-required `WEAK_HYPOTHESIS` behavior for correlated single-source rules, with the change documented in-line). Full suite: 204/204 passing. `ruff check`: clean. `compileall`: clean. Real end-to-end `analyze-image` run confirmed `rules.json` carries correct tier/activation_status fields and `concerns.json` stays `[]` in production. Committed separately from Phase 1. Per the user's explicit instruction: this is **not** described as scientifically validated — it is a correctness/completeness improvement to inert scaffolding. No rule produces different user-facing output than before Phase 2.
