# Session Handoff

Written 2026-08-02 at the end of a long working session, for a fresh Claude
Code session (or human) to pick up with full context. Read this before
`CURRENT_STATE_AUDIT.md` or any other doc — it tells you what's current and
in what order everything happened.

---

## 1. Git state

**Branch**: `master` (only branch). **No remotes configured** — nothing has
been or can be pushed anywhere from this machine.

**Commit history** (oldest → newest):

| Commit | Summary |
|---|---|
| `894d733` | Baseline commit — first-ever commit of the pre-existing codebase (project had no git history before this session) |
| `a9bbfe8` | Phase 1 — 2 real bug fixes (leakage quarantine gap, `evaluate` crash), `original_source_label` provenance, versioned case outputs |
| `c0ebb17` | Phase 2 — rule tiers (`tier`/`activation_status` fields), corrected concern-engine evidence-passthrough fix (concerns stay disabled) |
| `eee8e35` | Phase 3 planning — detector dependency matrix, candidate comparison, annotation schema, evaluation scaffolding. No detector implemented. |
| `20966e8` | Literature review round 1 — 6 candidates, no new operational rule found; Phase 3 validation-plan corrections |
| `5840d65` | Corrections — CSV structural integrity fix (real data-loss incident caught and recovered via git), developmental-stage overreach corrected |
| `88f0863` | Literature review round 2 — broader 10-query search, 6-category comparison, `EXPERIMENT_PRIORITIZATION.md` (ranks dataset/label audit > model completion > objective-feature pilot > detector work) |
| `cff8342` | Phase 3A — preliminary end-to-end training + full-pipeline functional validation on the full uncleaned dataset |
| *(uncommitted)* | Phase 3A verification/documentation corrections — see §4 below, not yet committed as of this doc being written |

**Working tree status as of writing this doc**: `PHASE3A_RESULTS.md` modified
(corrections + 2 new verification sections, not yet committed) and this file
is new. Both will be committed together immediately after this doc is
written — see the end of this session's work.

---

## 2. What each phase actually changed (code, not just docs)

- **Phase 1** (`src/doar/leakage.py`, `src/doar/models.py`,
  `src/doar/label_provenance.py` [new], `src/doar/case_output.py`):
  `resolve_leakage()` now quarantines label-conflicting images (previously
  computed but not removed); `evaluate` raises a clear error instead of
  crashing on non-sklearn checkpoints; every case now records
  `original_source_label` + a `CONSISTENT/POSSIBLE_CONFLICT/UNCERTAIN` audit
  status; case JSON outputs are now versioned on change instead of silently
  overwritten (`case/versions/*.json` + `history.jsonl`).
- **Phase 2** (`src/doar/rules.py`, `src/doar/concerns.py`,
  `resources/psychology_sources/rules_registry.json`): every rule now carries
  `tier` (`tier_1_prompt_independent`/`tier_2_content_conditional`/
  `tier_3_prompt_or_age_dependent`) and `activation_status`. **No rule's
  runtime behavior changed** — this made the existing implicit fallthrough
  explicit. `evaluate_rules()` no longer discards the emotion model's
  evidence; `concerns.py` implements a 4-level aggregation vocabulary
  (`INSUFFICIENT`/`WEAK_HYPOTHESIS`/`POSSIBLE_FOR_EXPLORATION`/
  `PROFESSIONAL_REVIEW_SUGGESTED`). **`CONCERNS_ENABLED` stays `False` in
  production** — this is a correctness fix to disabled scaffolding, not new
  user-facing capability.
- **Phase 3 planning**: no `src/doar` code changed. Added
  `src/doar/detectors/` (schema + metrics scaffolding only, not imported by
  any user-facing path, no model/download).
- **Phase 3A**: no `src/doar` code changed (no bugs found — see §5).

Everything else across all phases was documentation, CSV registers, and
tests — see §7 for the full file inventory.

---

## 3. Current architecture and operational status

**Pipeline**: dataset → leakage gate → objective feature extraction
(`features.py`) → deep CNN emotion model (`deep/`) and/or classical
feature-based model (`models.py`) → optional fusion (`fusion/`) → tier-aware
psychologist rule evaluation (`rules.py`) → disabled concern-convergence
engine (`concerns.py`) → judges/safety scan (`judges.py`) → bilingual (EN/AR)
report generation (`reports.py`) → deterministic evidence-grounded Q&A
(`qa.py`) → versioned case persistence (`case_output.py`).

**Interfaces**: CLI (`main.py`, 32 commands) + Streamlit app
(`streamlit_app.py`, includes an upload-and-analyze flow added early this
session). No REST API exists or is planned without a concrete need.

**Persistence**: 100% flat JSON/CSV files. No database. Case outputs are
now versioned (Phase 1) but the review-master CSV was already the one
genuinely append-only artifact from before this session.

**What actually runs correctly, proven by execution this session** (not
just code-reading): the full CLI pipeline end-to-end, on real data, twice —
once on a leakage-cleaned subset (~2,171 images, an earlier session
checkpoint predating this handoff's scope) and once on the full uncleaned
dataset (3,688 images, Phase 3A, this session). Both produced real trained
checkpoints, real metrics, and real bilingual case reports.

---

## 4. Exact Phase 3A commands and results

Full detail, corrected wording, and 2 additional verification checks
(version-history round-trip, report structural inspection) are in
`PHASE3A_RESULTS.md` — read that file directly rather than this summary for
anything you plan to cite. Headline:

- Dataset: 3,688 images, **0 unreadable**, no exclusions (deliberately —
  Phase 3A tests wiring, not leakage-safe accuracy).
- Split: existing `train`/`valid`/`test` folder structure (already fixed,
  deterministic).
- Model: `mobilenet_v3_small`, seed 42, 10 epochs, batch 4 × grad-accum 4,
  224px, GPU, 520.6s training time.
- Validation-split metrics (n=310, **preliminary, not leakage-safe**):
  accuracy 0.729, macro-F1 0.704, balanced accuracy 0.713, ROC-AUC 0.905 —
  full per-class/confusion-matrix detail in `PHASE3A_RESULTS.md` §5.
- This run's macro-F1 is higher than an earlier leakage-cleaned run's
  (0.656) — **corrected framing**: consistent with possible leakage
  inflation *and* other uncontrolled experimental differences (different
  epoch count, different invocation path), not attributable to leakage
  alone; a controlled comparison has not been run.
- 4 real images (one per class) run through the full `analyze-image`
  pipeline — explicitly a **pipeline smoke test**, not a representative
  performance sample. 3/4 correct, zero errors/exceptions anywhere in the
  run log.
- Version-history round-trip independently verified on a real Phase 3A case
  (§7a of `PHASE3A_RESULTS.md`): unchanged re-analysis correctly creates no
  version; a genuine content change correctly archives exactly the files
  that changed, with the prior content recoverable and verified intact.
- Reports structurally inspected (§7b): zero encoding problems, zero
  unresolved placeholders, all required sections present, no
  cross-language contradiction found. **Arabic linguistic quality/fluency
  has NOT been validated by a qualified Arabic speaker — this remains an
  open item.**
- No bugs found this phase; no source code changed.

---

## 5. Files created or modified this phase (Phase 3A + its corrections)

Committed in `cff8342`: `PHASE3A_RESULTS.md` (created), `DECISION_LOG.md`
(entry added).

Modified after `cff8342`, to be committed next: `PHASE3A_RESULTS.md`
(corrected §6/§7 wording, added §7a/§7b/§9a), `SESSION_HANDOFF.md` (this
file, new).

Not committed (git-ignored, `outputs/` rule): everything under
`outputs/phase3a/` — see `PHASE3A_RESULTS.md` §9a for the exact paths,
sizes, and SHA-256 hashes of every artifact that matters (checkpoint,
manifest, metrics, loss curve, probability export).

---

## 6. Operational, disabled, and unvalidated components

**Operational** (proven by execution this session): leakage detection +
quarantine + override-with-audit-log; objective feature extraction; classical
and deep model training; fusion training + calibration; locked-test
evaluation guard; tier-aware rule dispatch (6/19 rules can produce
`weak_support`, at low confidence ceilings 0.05–0.25); bilingual report
generation; deterministic Q&A; versioned case persistence; Streamlit
upload-and-analyze.

**Disabled by design, correctness-verified but not user-facing**: the
concern-convergence engine (`CONCERNS_ENABLED = False` — every case's
`concerns.json` is `[]`).

**Unvalidated / not yet built**: 13/19 rules (`DETECTOR_UNAVAILABLE` — no
face/eye, animal, or shape/symbol detector exists); SQLite persistence layer
(approved, Phase 4, not started); the `LIT_EMOTION_FACE_ENCODING_007`
literature candidate (most promising lead found, full-text verification
blocked by a cookie wall, not acted on further); the objective
scribble/fragmentation feature (`LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006` —
explicitly corrected this session from "operational candidate" to
"hypothesis to pilot," since the source study used CNN features on a
stimulus-completion task, not DOAR's classical features on free drawings);
Arabic report linguistic quality (structure verified, fluency not — see §4).

---

## 7. Known limitations (compounding, not exhaustive — see `SCIENTIFIC_LIMITATIONS.md`)

- Dataset: 41% of the raw dataset was found to be cross-split duplicates or
  label-conflicting in an earlier leakage-detection pass; this has **not**
  been resolved into versioned exclusion manifests yet (that work was
  explicitly paused this session in favor of Phase 3A — see §8).
- Rules: 68% of the active rule registry can never fire (no detector); the
  ones that can fire are backed by a two-page, non-clinical Arabic source
  document, correctly reflected in very low confidence ceilings.
- Objective features are sensitive to paper size/drawing medium
  (`LIT_MATERIAL_MEDIUM_CONFOUND_010`), which the pipeline does not record —
  a genuinely new limitation found this session, not previously documented.
- No formal database-backed literature search was performed (WebSearch is a
  general web backend, not PubMed/PsycINFO/Scopus) — stated as a real
  limitation in `LITERATURE_SEARCH_LOG.md`, not hidden.
- Phase 3A's metrics are explicitly preliminary and not leakage-safe (§4).

---

## 8. Pending decisions and the paused dataset audit

**The dataset/label-quality audit was started, then explicitly paused** by
the user in favor of Phase 3A (fast functional verification), with an
explicit instruction that leakage will be addressed **before final thesis
evaluation** but is not the immediate objective. It is **queued, not
abandoned**. When resumed, the original scope was:

1. Exact, mutually-exclusive counts explaining the 41% exclusion (original
   readable / corrupt / exact-dup groups / near-dup groups / same-label
   dups / cross-label conflicts / other / final retained).
2. Confirmation that no original image/label was ever deleted/moved/
   renamed/overwritten/auto-relabeled (generated copies may change,
   originals must not) — this was already spot-verified informally earlier
   this session (file timestamps + counts checked directly against the
   source directory) but not yet done as a formal, versioned audit deliverable.
3. Versioned manifests for duplicate groups, exclusions, conflicts, and
   retained images (paths, hashes, labels, group IDs, reasons, provenance).
4. Verification that duplicate/near-duplicate groups never cross
   train/valid/test; class balance before/after; ranked (not changed)
   suspicious labels.
5. Tests for manifest integrity, mutually-exclusive counts, reproducibility,
   group leakage, source-dataset immutability.
6. A proposed (not run) next baseline experiment: fixed grouped split,
   candidate models, seeds, metrics + confidence intervals, calibration,
   compute/time estimate, model-selection criteria.

**Detector work (Phase 3B)** remains approved-but-not-started, now
explicitly reprioritized: `EXPERIMENT_PRIORITIZATION.md` ranks it **last**
of 4 options, behind dataset/label auditing, completing the emotion-model
baseline, and the (now-corrected, still-unvalidated) objective
scribble/fragmentation pilot. This session's final task (§9 of this
document, next section below) re-ranks 3 of those 4 options once more,
explicitly excluding dataset work per this session's final instruction.

**Literature review** is at "round 2 of an explicitly not-yet-exhaustive
process" — `LIT_EMOTION_FACE_ENCODING_007` is flagged as the best lead
requiring round-3 full-text verification before any further action.

---

## 9. Recommended next-phase ranking (A vs B vs C — see full reasoning in the final chat message of this session)

Comparing, as instructed: **A.** controlled emotion-model baseline
comparisons; **B.** objective visual-feature experiments; **C.** detector
implementation for unavailable rules. Dataset cleaning is excluded from this
specific ranking per the final instruction of this session (not because it
lacks priority — see §8 — but because it was scoped out of this particular
comparison).

Full ranking, criteria, and reasoning are in the chat response accompanying
this handoff (not duplicated here to avoid drift between two copies) — read
that response, or ask the next session to re-derive it from
`EXPERIMENT_PRIORITIZATION.md` (which ranked a superset including dataset
work) plus this document's §8/§9 framing.

---

## 10. Exact recommended prompt for the next session

```
Continue the DOAR project. Read SESSION_HANDOFF.md first, in full, before
anything else -- it has the current git history, architecture, operational
status, and pending decisions. Do not repeat any audit or literature review
already completed (see DECISION_LOG.md for the full history).

Resume the dataset/label-quality audit that was paused for Phase 3A
(SESSION_HANDOFF.md Section 8 has the original scope: mutually-exclusive
exclusion counts, source-image immutability confirmation, versioned
duplicate/exclusion/conflict manifests, cross-split leakage verification,
class balance before/after, integrity tests, and a proposed -- not run --
next baseline experiment). Do not clean the dataset or exclude anything
without separate approval; the audit's job is to measure and document, not
to modify. Stop for approval before running any training.
```

(This assumes the dataset audit is what gets resumed next, consistent with
this session's own instruction that leakage must be addressed before final
thesis evaluation. If a different phase is chosen instead after reviewing
the A/B/C ranking, adjust the second paragraph accordingly before using this
prompt.)
