# Session Handoff

Updated 2026-08-02, end of the Phase 4 working session, for a fresh Claude
Code session (or human) to pick up with full context. Read this before
`CURRENT_STATE_AUDIT.md` or any other doc — it tells you what's current and
in what order everything happened.

**Canonical phase terminology (preserve exactly, do not rename):**
- **Phase 1: Pipeline and Data-Safety Corrections**
- **Phase 2: Rule Tiers and Concern-Engine Correction**
- **Phase 3A: Preliminary End-to-End Training and Functional Validation**
- **Phase 3A Closure Verification** — commit `cf6512f`
- **Phase 4: Preliminary Controlled Emotion-Model Comparison** — this session
- Detector implementation for currently-unavailable rules was **proposed
  only, never implemented**. It is a **future roadmap item**, not a
  numbered phase (specifically, it is **not** "Phase 3B" — that label was
  used informally in an earlier draft of this document and is retired here).

---

## 1. Git state

**Branch**: `master` (only branch). **No remotes configured** — nothing has
been or can be pushed anywhere from this machine.

**Commit history** (oldest → newest):

| Commit | Phase | Summary |
|---|---|---|
| `894d733` | — | Baseline commit — first-ever commit of the pre-existing codebase (project had no git history before the first session) |
| `a9bbfe8` | Phase 1 | Pipeline and Data-Safety Corrections — 2 real bug fixes (leakage quarantine gap, `evaluate` crash), `original_source_label` provenance, versioned case outputs |
| `c0ebb17` | Phase 2 | Rule Tiers and Concern-Engine Correction — `tier`/`activation_status` fields, corrected concern-engine evidence-passthrough fix (concerns stay disabled) |
| `eee8e35` | (planning) | Detector dependency matrix, candidate comparison, annotation schema, evaluation scaffolding. No detector implemented — future roadmap item, see above. |
| `20966e8` | (planning) | Literature review round 1 — 6 candidates, no new operational rule found |
| `5840d65` | (corrections) | CSV structural integrity fix (real data-loss incident caught and recovered via git), developmental-stage overreach corrected |
| `88f0863` | (planning) | Literature review round 2 — broader 10-query search, 6-category comparison, `EXPERIMENT_PRIORITIZATION.md` |
| `cff8342` | Phase 3A | Preliminary end-to-end training + full-pipeline functional validation on the full uncleaned dataset |
| `cf6512f` | Phase 3A Closure Verification | Version-history round-trip re-verified, report structural inspection, wording corrections, artifact hash inventory, `SESSION_HANDOFF.md` v1 |
| *(this session, to be committed next)* | Phase 4 | Preliminary controlled one-seed emotion-model comparison — see §4 |

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
  evidence; `concerns.py` implements a 4-level aggregation vocabulary.
  **`CONCERNS_ENABLED` stays `False` in production.**
- **Detector planning** (not a numbered phase): no `src/doar` code changed
  beyond adding `src/doar/detectors/` (schema + metrics scaffolding only,
  not imported by any user-facing path, no model/download).
- **Phase 3A**: no `src/doar` code changed (no bugs found).
- **Phase 3A Closure Verification**: no `src/doar` code changed (doc/
  verification corrections only).
- **Phase 4** (`src/doar/deep/trainers.py`, `tests/test_trainer_regression.py`):
  `train_image_model()` now records `parameter_count` and
  `trainable_parameter_count` in its result (previously missing, needed for
  the model-comparison report). 3 new regression tests. No other code
  changed — the multi-model comparison itself reused
  `main.py compare-deep-models` → `deep/compare.py::run_deep_comparison()`
  unmodified.

Everything else across all phases was documentation, CSV registers, and
tests.

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
(`streamlit_app.py`, includes an upload-and-analyze flow). No REST API
exists or is planned without a concrete need.

**Persistence**: 100% flat JSON/CSV files. No database. Case outputs are
versioned (Phase 1); the review-master CSV is genuinely append-only.

**What actually runs correctly, proven by execution** (not just
code-reading): the full CLI pipeline end-to-end, on real data, multiple
times — a leakage-cleaned subset (~2,171 images, an earlier session), the
full uncleaned dataset for a single model (Phase 3A), and now the full
uncleaned dataset across **4 architectures** (Phase 4). All produced real
trained checkpoints, real metrics, real bilingual case reports.

---

## 4. Exact Phase 3A and Phase 4 commands and results

Full detail lives in `PHASE3A_RESULTS.md` and `PHASE4_RESULTS.md` — read
those directly for anything you plan to cite. Headline:

**Phase 3A**: `mobilenet_v3_small`, seed 42, 10 epochs, on the full uncleaned
dataset (3,688 images, 0 unreadable). Validation macro-F1 0.704, accuracy
0.729 (n=310, preliminary). 4-image pipeline smoke test: 3/4 correct, zero
errors. Version-history round-trip and report structural inspection both
verified correct (`PHASE3A_RESULTS.md` §7a/§7b). Arabic linguistic fluency
NOT validated by a qualified speaker — still open.

**Phase 4**: one-seed controlled screening of `small_cnn`,
`mobilenet_v3_small`, `resnet18`, `efficientnet_b0` — identical protocol
(same dataset/manifest as Phase 3A, same 224px/batch-4/grad-accum-4/10-epoch
budget, same AdamW/ReduceLROnPlateau policy, same evaluation code) via the
existing `compare-deep-models` command, not a new framework. Results
(validation split, n=310, **preliminary, single-seed, not leakage-safe**):

| Model | Params | Time | Macro-F1 | Accuracy | ECE |
|---|---|---|---|---|---|
| `small_cnn` | 93,764 | 329.6s | 0.435 | 0.497 | 0.091 |
| `mobilenet_v3_small` | 1,521,956 | 508.3s | 0.704 | 0.729 | 0.045 |
| `resnet18` | 11,178,564 | 453.8s | 0.715 | 0.735 | 0.108 |
| `efficientnet_b0` | 4,012,672 | 790.7s | **0.736** | **0.761** | 0.056 |

`mobilenet_v3_small`'s result here matches Phase 3A's independent run to 4
decimal places (0.7043 both times) — a genuine internal-consistency check,
not fabricated agreement. **No architecture is claimed superior from this
single-seed result** — see `PHASE4_RESULTS.md` §7. Total screening time
2,082.4s (~34.7 min). No bugs found beyond the parameter-count gap (fixed,
tested, see §2 above).

---

## 5. Files created or modified this session

**Phase 3A Closure Verification** (committed `cf6512f`): `PHASE3A_RESULTS.md`
(corrected + 2 new sections), `SESSION_HANDOFF.md` (v1, now superseded by
this v2).

**Phase 4** (to be committed next): `src/doar/deep/trainers.py` (parameter
count fields), `tests/test_trainer_regression.py` (3 new tests),
`PHASE4_RESULTS.md` (new), `SESSION_HANDOFF.md` (this file, v2).

**Not committed** (git-ignored, `outputs/` rule): everything under
`outputs/phase3a/` and `outputs/phase4/` — see each phase's `*_RESULTS.md`
§6/§9a for exact paths, sizes, and SHA-256 hashes of every artifact that
matters. `outputs/phase4/ARTIFACT_MANIFEST.tsv` has the full 25-row Phase 4
manifest.

---

## 6. Operational, disabled, and unvalidated components

**Operational** (proven by execution): leakage detection + quarantine +
override-with-audit-log; objective feature extraction; classical and deep
model training across 4 architectures now; fusion training + calibration;
locked-test evaluation guard; tier-aware rule dispatch (6/19 rules can
produce `weak_support`, confidence ceilings 0.05–0.25); bilingual report
generation; deterministic Q&A; versioned case persistence; Streamlit
upload-and-analyze.

**Disabled by design, correctness-verified but not user-facing**: the
concern-convergence engine (`CONCERNS_ENABLED = False` — every case's
`concerns.json` is `[]`).

**Unvalidated / not yet built**: 13/19 rules (`DETECTOR_UNAVAILABLE` — no
face/eye, animal, or shape/symbol detector exists — **future roadmap item,
not started**); SQLite persistence layer (approved, not started); the
`LIT_EMOTION_FACE_ENCODING_007` literature candidate (most promising lead
found, full-text verification blocked by a cookie wall); the objective
scribble/fragmentation feature (`LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006` —
corrected to "hypothesis to pilot," not operational); Arabic report
linguistic quality (structure verified, fluency not); **which of the 4
Phase 4 architectures is genuinely best — single-seed result only, no
multi-seed comparison run yet (proposed, not executed — see
`PHASE4_RESULTS.md` §9)**.

---

## 7. Known limitations (compounding, not exhaustive — see `SCIENTIFIC_LIMITATIONS.md`)

- Dataset: 41% of the raw dataset was found to be cross-split duplicates or
  label-conflicting in an earlier leakage-detection pass; **not yet**
  resolved into versioned exclusion manifests (paused in favor of Phase 3A/
  Phase 4, queued not abandoned — see §8).
- Rules: 68% of the active rule registry can never fire (no detector); the
  ones that can fire are backed by a two-page, non-clinical Arabic source
  document, correctly reflected in very low confidence ceilings.
- Objective features are sensitive to paper size/drawing medium
  (`LIT_MATERIAL_MEDIUM_CONFOUND_010`), which the pipeline does not record.
- No formal database-backed literature search was performed (WebSearch, not
  PubMed/PsycINFO/Scopus) — stated as a real limitation, not hidden.
- Phase 3A's and Phase 4's metrics are explicitly preliminary and not
  leakage-safe.
- Phase 4 is single-seed — no stability/variance information exists yet for
  any of the 4 architectures.

---

## 8. Pending decisions and the paused dataset audit

**The dataset/label-quality audit was started, then explicitly paused**
twice now (once for Phase 3A, once implicitly for Phase 4) in favor of
functional/comparison work, with an explicit instruction that leakage will
be addressed **before final thesis evaluation**. It is **queued, not
abandoned**. Original scope (unchanged since Phase 3A Closure Verification):

1. Exact, mutually-exclusive counts explaining the 41% exclusion.
2. Confirmation that no original image/label was ever deleted/moved/
   renamed/overwritten/auto-relabeled.
3. Versioned manifests for duplicate groups, exclusions, conflicts, and
   retained images (paths, hashes, labels, group IDs, reasons, provenance).
4. Verification that duplicate/near-duplicate groups never cross
   train/valid/test; class balance before/after; ranked (not changed)
   suspicious labels.
5. Tests for manifest integrity, mutually-exclusive counts, reproducibility,
   group leakage, source-dataset immutability.
6. A proposed (not run) next baseline experiment.

**Detector implementation** (future roadmap item, not a numbered phase,
not started): planning complete (`DETECTOR_DEPENDENCY_MATRIX.csv`,
`PHASE3_DETECTOR_EVALUATION_PLAN.md`), ranked last of the non-dataset
options in `EXPERIMENT_PRIORITIZATION.md` — every one of the 13 unavailable
rules stays scientifically unvalidated even if a detector works, so this
remains lower priority than dataset auditing or model comparison work.

**Literature review** is at "round 2 of an explicitly not-yet-exhaustive
process" — `LIT_EMOTION_FACE_ENCODING_007` is flagged as the best lead
requiring round-3 full-text verification before any further action.

**Phase 4's own proposal** (multi-seed comparison of the same 4
architectures, seeds 42/123/2026, ~105 min estimated GPU time, ~1.18GB
storage) is **not started** — awaiting approval per `PHASE4_RESULTS.md` §9.

---

## 9. Exact recommended prompt for the next session

Two reasonable next steps exist; pick based on current priority. Use
whichever prompt matches:

**If continuing model work (multi-seed comparison):**
```
Continue the DOAR project. Read SESSION_HANDOFF.md and PHASE4_RESULTS.md in
full before acting. Treat the repository at its current HEAD as the source
of truth. Do not repeat Phase 4's one-seed screening.

Proceed with the multi-seed comparison proposed in PHASE4_RESULTS.md
Section 9: the same 4 architectures (small_cnn, mobilenet_v3_small,
resnet18, efficientnet_b0), seeds 42/123/2026, identical protocol, on the
same uncleaned dataset (still explicitly preliminary, not leakage-safe --
label every result accordingly). Include calibration this time. Do not
clean the dataset, implement detectors, or activate psychological concerns.
Stop for review after training completes, before any test-split evaluation.
```

**If resuming the paused dataset/label-quality audit instead:**
```
Continue the DOAR project. Read SESSION_HANDOFF.md in full before acting --
Section 8 has the paused dataset-audit's exact scope (mutually-exclusive
exclusion counts, source-image immutability confirmation, versioned
duplicate/exclusion/conflict manifests, cross-split leakage verification,
class balance before/after, integrity tests, and a proposed -- not run --
next baseline experiment). Do not clean the dataset or exclude anything
without separate approval; the audit's job is to measure and document, not
to modify. Do not repeat Phase 3A or Phase 4's completed work. Stop for
approval before running any training.
```
