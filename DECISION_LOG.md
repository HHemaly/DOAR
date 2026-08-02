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
