# Session Handoff

Updated 2026-08-02, end of the Phase 5 working session, for a fresh Claude
Code session (or human) to pick up with full context. Read this before
`CURRENT_STATE_AUDIT.md` or any other doc — it tells you what's current and
in what order everything happened.

**Canonical phase terminology (preserve exactly, do not rename):**
- **Phase 1: Pipeline and Data-Safety Corrections**
- **Phase 2: Rule Tiers and Concern-Engine Correction**
- **Phase 3A: Preliminary End-to-End Training and Functional Validation**
- **Phase 3A Closure Verification** — commit `cf6512f`
- **Phase 4: Preliminary Controlled Emotion-Model Comparison**
- **Phase 5: Preliminary Multi-Seed Confirmation** — this session
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
| `73b082f` | Phase 4 | Preliminary controlled one-seed emotion-model comparison — see §4 |
| *(this session, to be committed next)* | Phase 5 | Preliminary multi-seed (42/123/2026) confirmation of the 3 competitive pretrained architectures + post-hoc calibration — see §4 |

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
- **Phase 5**: **no `src/doar` code changed.** Trained 2 additional seeds
  (123, 2026) for the 3 competitive pretrained architectures via the
  existing `compare-deep-models` command, unmodified; ran the existing
  `main.py calibrate` (temperature scaling) on all 9 model×seed checkpoints
  for the first time in this project (Phase 3A/4 never calibrated); no bugs
  found in `src/doar`. One operator (CLI-usage, not code) mistake was made
  and self-corrected: `evaluate-predictions --output` takes a directory, not
  a file path — passing a `.json` path created a nested directory named
  `metrics.json`; caught immediately, fixed by moving the files up one
  level, no data was lost (see `PHASE5_RESULTS.md` §3 note).

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
full uncleaned dataset for a single model (Phase 3A), the full uncleaned
dataset across **4 architectures** (Phase 4), and now **3 seeds × 3
architectures** (9 model×seed checkpoints total) plus, for the first time in
this project, post-hoc temperature-scaling calibration on all 9 (Phase 5).
All produced real trained checkpoints, real metrics, real bilingual case
reports.

---

## 4. Exact Phase 3A, Phase 4, and Phase 5 commands and results

Full detail lives in `PHASE3A_RESULTS.md`, `PHASE4_RESULTS.md`, and
`PHASE5_RESULTS.md` — read those directly for anything you plan to cite.
Headline:

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

**Phase 5**: multi-seed confirmation (seeds 42 reused from Phase 4 + newly
trained 123, 2026) of the 3 competitive pretrained architectures only
(`small_cnn` not retrained — kept as a Phase 4 screening reference).
Identical protocol to Phase 4/seed-42, verified compatible before training.
Validation split only, n=310, **still preliminary, still not leakage-safe**.

| Model | Mean macro-F1 (n=3) | Std | Min | Max | Mean accuracy | Mean ECE (precal → postcal) |
|---|---|---|---|---|---|---|
| `mobilenet_v3_small` | 0.6951 | 0.0169 | 0.6714 | 0.7097 | 0.7215 | 0.0629 → 0.0417 |
| `resnet18` | 0.7179 | 0.0067 | 0.7120 | 0.7272 | 0.7430 | 0.1058 → 0.0588 |
| `efficientnet_b0` | 0.7291 | 0.0076 | 0.7186 | 0.7364 | 0.7527 | 0.0564 → 0.0543 |

`mobilenet_v3_small` ranks last on every one of the 3 seeds with no overlap
against the other two's ranges — the one unambiguous finding. `efficientnet_b0`
and `resnet18` swap 1st/2nd depending on seed; their mean gap (0.0112) is
smaller than either model's own seed-to-seed std, so **no statistical
superiority is claimed between them at n=3** — see `PHASE5_RESULTS.md` §5.
Temperature-scaling calibration was applied to all 9 checkpoints
(validation-only, never test): it improved NLL/Brier in all 9 cases but
worsened ECE in 3 of 9, showing calibration does not uniformly improve every
probability-quality metric. Macro-F1/accuracy are identical before and after
calibration for every run, as expected (temperature scaling doesn't change
arg-max predictions). Preliminary recommendation for a **future** end-to-end
pipeline test: `efficientnet_b0` (best mean macro-F1, best calibration, and
a much smaller checkpoint than `resnet18`) — but this is **not** a decision
to replace the production/default model, and no test-split evaluation
occurred. Full 9-run detail, per-class variation, and artifact hashes in
`PHASE5_RESULTS.md`.

---

## 5. Files created or modified this session

**Phase 3A Closure Verification** (committed `cf6512f`): `PHASE3A_RESULTS.md`
(corrected + 2 new sections), `SESSION_HANDOFF.md` (v1, now superseded by
this v2).

**Phase 4** (committed `73b082f`): `src/doar/deep/trainers.py` (parameter
count fields), `tests/test_trainer_regression.py` (3 new tests),
`PHASE4_RESULTS.md` (new), `SESSION_HANDOFF.md` (v2).

**Phase 5** (to be committed next): `PHASE5_RESULTS.md` (new),
`SESSION_HANDOFF.md` (this file, v3). **No `src/doar` or `tests/` files
changed** — no code changes were needed this phase (see §2 above).

**Not committed** (git-ignored, `outputs/` rule): everything under
`outputs/phase3a/`, `outputs/phase4/`, and `outputs/phase5/` — see each
phase's `*_RESULTS.md` §6/§9a for exact paths, sizes, and SHA-256 hashes of
every artifact that matters. `outputs/phase4/ARTIFACT_MANIFEST.tsv` (25 rows)
and `outputs/phase5/ARTIFACT_MANIFEST.tsv` (120 rows, ≈1.22 GB) cover their
respective phases in full.

---

## 6. Operational, disabled, and unvalidated components

**Operational** (proven by execution): leakage detection + quarantine +
override-with-audit-log; objective feature extraction; classical and deep
model training across 4 architectures, now with multi-seed (3-seed)
confirmation for the 3 competitive ones; deep-checkpoint temperature-scaling
calibration (validation-only, now exercised for the first time on 9
checkpoints); fusion training + calibration; locked-test evaluation guard;
tier-aware rule dispatch (6/19 rules can produce `weak_support`, confidence
ceilings 0.05–0.25); bilingual report generation; deterministic Q&A;
versioned case persistence; Streamlit upload-and-analyze.

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
linguistic quality (structure verified, fluency not); **which of the 3
competitive Phase 5 architectures is genuinely best — `efficientnet_b0` and
`resnet18` are statistically indistinguishable at n=3 seeds (see
`PHASE5_RESULTS.md` §5); a preliminary single-model recommendation exists
for a future pipeline test, but no test-split evaluation, production-model
replacement, or full pipeline run has occurred**.

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
- Phase 3A's, Phase 4's, and Phase 5's metrics are explicitly preliminary and
  not leakage-safe (all use the same uncleaned dataset and fixed splits).
- Phase 5 used only 3 seeds — enough to see that `mobilenet_v3_small` is
  consistently weakest, not enough to resolve `efficientnet_b0` vs.
  `resnet18` (their mean gap is smaller than either model's own seed
  variability). No 95% confidence interval is claimed anywhere for n=3.
- `small_cnn` (Phase 4's weakest architecture) was never given multi-seed
  confirmation — it was deliberately excluded from Phase 5's compute per
  explicit instruction, and remains a single-seed screening data point only.

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

**Phase 4's multi-seed proposal is now complete (Phase 5)** — see §4 above
and `PHASE5_RESULTS.md`. The next open decision is what to do with Phase 5's
preliminary recommendation (`efficientnet_b0`): the user's own closing
instruction for Phase 5 was to stop before test-split evaluation, production
replacement, full-pipeline execution, dataset cleaning, detector work, or
rule activation — so **an explicit go-ahead is needed before any of those**.

---

## 9. Exact recommended prompt for the next session

Three reasonable next steps exist; pick based on current priority. Use
whichever prompt matches:

**If proceeding to a final end-to-end pipeline test of the Phase 5 winner:**
```
Continue the DOAR project. Read SESSION_HANDOFF.md and PHASE5_RESULTS.md in
full before acting. Treat the repository at its current HEAD as the source
of truth.

Phase 5 preliminarily recommended efficientnet_b0 (checkpoint at
outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt or any of
the seed 123/2026 checkpoints under outputs/phase5/deep/runs/) for a final
end-to-end pipeline test, but explicitly stopped short of running it. Run
that model through the full DOAR pipeline (analyze-image or equivalent) on
a small number of held-out example images -- NOT the locked test split
unless the user explicitly authorizes unlocking it via --unlock-test
--confirm-final-evaluation, which is a separate, deliberate decision. Do not
treat this as replacing the production/default model without a separate,
explicit instruction to do so. Do not clean the dataset, implement
detectors, or activate psychological concerns.
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
to modify. Do not repeat Phase 3A, Phase 4, or Phase 5's completed work.
Stop for approval before running any training.
```

**If the user wants to formally decide between efficientnet_b0 and resnet18
first (rather than accepting Phase 5's preliminary lean):**
```
Continue the DOAR project. Read SESSION_HANDOFF.md and PHASE5_RESULTS.md in
full before acting. Phase 5 (n=3 seeds) found efficientnet_b0 and resnet18
statistically indistinguishable on mean validation macro-F1 -- their gap is
smaller than either model's own seed-to-seed standard deviation. Do not
manufacture false statistical confidence with more seeds under the same
non-leakage-safe protocol; instead ask the user whether they want (a) more
seeds under the current preliminary protocol, (b) to accept the practical
tie and pick the simpler/cheaper model, or (c) to defer the decision until
after the dataset audit produces a leakage-safe evaluation. Do not decide
this unilaterally.
```
