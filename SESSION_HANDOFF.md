# Session Handoff

Updated 2026-08-03, end of a Phase 7B continuation session, for a fresh
Claude Code session (or human) to pick up with full context. Read this
before `CURRENT_STATE_AUDIT.md` or any other doc — it tells you what's
current and in what order everything happened.

**Canonical phase terminology (preserve exactly, do not rename):**
- **Phase 1: Pipeline and Data-Safety Corrections**
- **Phase 2: Rule Tiers and Concern-Engine Correction**
- **Phase 3A: Preliminary End-to-End Training and Functional Validation**
- **Phase 3A Closure Verification** — commit `cf6512f`
- **Phase 4: Preliminary Controlled Emotion-Model Comparison**
- **Phase 5: Preliminary Multi-Seed Confirmation**
- **Phase 6: Selected-Model End-to-End Integration Validation**
- **Phase 7: Extended Experiment Programme and Model-Family Feasibility
  Review** — proposal only, no experiment in it has been run. See
  `PHASE7_RESULTS.md` and `PHASE7_EXPERIMENT_MATRIX.csv`.
- **Phase 7A: Leakage-Safe Dataset Readiness and Partition Design** —
  dataset analysis and split construction only, no model trained. See
  `PHASE7A_DATASET_READINESS.md` and `PHASE7_EXPERIMENT_MATRIX_REVISION.md`.
  **Its threshold-5 partition was explicitly provisional.**
- **Phase 7B: Duplicate-Policy Validation and Final Partition Refinement**
  — dataset analysis and split construction only, no model trained, tuned,
  calibrated, or evaluated. First part of this phase validated the
  near-duplicate detector via a blinded manual pair audit and adopted dHash
  (threshold 6) over aHash. **A same-phase continuation then found that
  choice was based on an insufficiently targeted sample and is likely too
  permissive** — see `PHASE7B_DUPLICATE_POLICY.md`, especially §§14–17.
  **No threshold or clustering policy is finalized. No partition is
  locked.** A human-review package now exists
  (`outputs/phase7b/human_review/`) and is the required next step before
  any partition can be locked.
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
| `1542505` | Phase 5 | Preliminary multi-seed (42/123/2026) confirmation of the 3 competitive pretrained architectures + post-hoc calibration — see §4 |
| `d0c5ca2` | Phase 5 correction | Fixed overreaching superiority/calibration claims, recorded Ruff honestly (documentation-only, no retraining, no artifacts touched) |
| `84623bd` | Phase 6 | Selected-model (`efficientnet_b0`) end-to-end integration validation against the Phase 3A `mobilenet_v3_small` reference — see §4 |
| `4aeb7db` | Phase 6 correction | Disclosed that 4 test-folder images were reused for integration smoke testing across Phase 3A and Phase 6 — test split is not completely untouched (documentation-only, no rerun) |
| `526c146` | Phase 7 | Extended Experiment Programme and Model-Family Feasibility Review — proposal-only, no training run; see §4 |
| `8f6d135` | Phase 7A | Leakage-Safe Dataset Readiness and Partition Design — dataset analysis and split construction only, no model trained; see §4 |
| `ef86d40` | Phase 7B | Duplicate-Policy Validation and Final Partition Refinement — blinded pair audit, dHash adopted over aHash, new (now-provisional) partition; see §4 |
| `5d4c56f` | Phase 7B (continuation) | Correction: dHash threshold 6 found insufficiently evidenced; complete-linkage comparison; human-review package produced; no partition locked; see §4 |
| `fac795e` | Phase 7B (second continuation) | Interactive blind human-review application (225 pairs, 4 nav sections), export logic, provisional conservative-policy comparison; no partition locked; see §4 |
| *(this session, to be committed next)* | Phase 7B (documentation touch-up) | Self-reference fix in this commit table only (commit hash `fac795e` was not yet known when the second continuation's own files were committed) |

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
- **Phase 5 correction** (commit `d0c5ca2`): documentation-only. Corrected
  overreaching wording in `PHASE5_RESULTS.md`/`SESSION_HANDOFF.md` (the
  `efficientnet_b0`/`resnet18` comparison was reframed from "comparable"/
  implicitly-tied to "inconclusive regarding superiority, no formal test
  run"), corrected the calibration conclusion to report NLL/Brier/ECE
  separately instead of implying one model was uniformly best, and recorded
  Ruff's actual result (792 findings, exit 1) honestly instead of omitting
  it. No retraining, no artifact changes, no `src/doar` changes.
- **Phase 6**: **no `src/doar` code changed** — no bug was found to fix.
  Ran `main.py analyze-image` (unmodified) 10 times: the same 4 Phase 3A
  smoke-test images through the selected Phase 5 `efficientnet_b0`
  checkpoint and the Phase 3A `mobilenet_v3_small` checkpoint, plus 2 extra
  runs re-exercising the version-history path from Phase 3A §7a with the
  new checkpoint. All 10 runs exited 0 with no errors/warnings found in the
  logs.
- **Phase 6 correction** (commit `4aeb7db`): documentation-only. Disclosed
  that Phase 3A and Phase 6 both reused the same 4 test-folder images for
  individual `analyze-image` integration smoke testing, and that their
  folder labels and predictions were inspected — the test split cannot be
  described as completely untouched. Reworded the "3 of 4 correct" result
  in both `PHASE6_RESULTS.md` and `SESSION_HANDOFF.md` to be explicit it is
  not a performance estimate. No rerun, no artifact changes, no `src/doar`
  changes.
- **Phase 7**: **no `src/doar` code changed — proposal only, nothing
  trained.** Produced `PHASE7_RESULTS.md` (feasibility report + research
  programme + governance + roadmap) and `PHASE7_EXPERIMENT_MATRIX.csv` (22
  proposed experiments across 5 families, full per-experiment
  specification). Grounded entirely in real inspection performed this
  phase: `pip list`, `torch.cuda` queries, `du`/`df`, exact model
  parameter counts (`torchvision`/`timm` construction, no download), the
  real leakage-report counts, and the existing codebase's already-built
  infrastructure — no training, no test-split access, no dataset or code
  changes.
- **Phase 7A** (`src/doar/partition.py` [new], `tests/test_partition.py`
  [new, 20 tests], `main.py` [new `build-partition` command]): no model
  trained or evaluated. Added a new duplicate-group/partition-design module
  reusing `dataset.py`'s existing hashing (SHA-256 exact, aHash near-dup)
  but computing the full duplicate-group structure across the whole
  dataset (not just cross-split pairs, unlike `leakage.assess_leakage`),
  detecting near-duplicate label conflicts for the first time (previously
  only exact-duplicate conflicts were checked), and constructing a new,
  deterministic, group-disjoint, class-stratified train/valid/test
  partition — written to `outputs/phase7a/`, never touching the original
  dataset or any prior-phase artifact. One real bug found and fixed:
  `compute_effective_counts()` iterated a raw Python `set` whose order
  depends on per-process hash randomization, making `effective_counts.json`
  byte-non-reproducible across runs despite correct numbers — fixed by
  sorting before every iteration, caught by running `main.py
  build-partition` twice and diffing hashes, now regression-tested. Ruff
  transiently gained 2 findings (unused imports in the new module) and was
  fixed back to the established 792 baseline before finalizing.
- **Phase 7B** (`src/doar/dataset.py` [additive: `_difference_hash`,
  `DHASH_THRESHOLD`], `src/doar/partition.py` [extended: `hash_field`
  parameter, `compute_dhash_column`, `identify_exposed_groups`,
  `exposed_group_ids` support, `verify_no_exposed_group_in_valid_or_test`],
  `tests/test_partition_phase7b.py` [new, 9 tests], `main.py`
  [`build-partition` gains `--hash-field`/`--exposed-image-ids`]): no model
  trained, tuned, calibrated, or evaluated. Ran a blinded manual pair audit
  (87 pairs, metadata held separate from judgment) that found aHash
  precision drops from 100% (Hamming 0-1) to 0% (Hamming 5, the existing
  default's own boundary), directly inspected the Phase 7A 137-image
  chained component (its direct edges were 3/3 real duplicates, its
  non-adjacent members were 0/3 related — chaining, not detector failure,
  confirmed mechanistically), and found dHash (difference hash, no more
  complex than the existing aHash, no new dependency) separates true from
  false near-duplicates far more cleanly on this dataset. Adopted dHash,
  threshold 6, keeping single-linkage clustering (the audit showed the
  threshold/method was the problem, not the linkage rule). Regenerated the
  partition: more usable data recovered (2,564 vs. Phase 7A's 2,305 train
  images) with fewer unnecessary exclusions (338 vs. 674 conflict images)
  and a far cleaner duplicate structure (largest component 17 vs. 137).
  Added previously-exposed-image group exclusion (the 4 Phase 6 smoke-test
  images' duplicate groups are force-assigned to train, never valid/test).
  Phase 7A's own artifacts and the original `_average_hash`/
  `NEAR_DUP_THRESHOLD` reproduction path are unmodified and re-verified
  unchanged.
- **Phase 7B continuation** (`src/doar/partition.py` [additive:
  `refine_with_complete_linkage`], `tests/test_partition_phase7b.py`
  [+5 tests]): no model trained. **Corrected the prior part's own
  threshold-6 conclusion.** A second, purpose-built blinded audit (42
  pairs, sampled directly at dHash distances 2–8, unlike the first audit's
  aHash-conditioned sample) found precision falls to 50% at distance 4 and
  to 0% at distance 5 — materially earlier than the first audit implied.
  Directly inspected the 17-image dHash-6 cross-label component
  (contact-sheet visual review): it is not one coherent cluster but a
  chain of ≥4 visually distinct sub-groups plus several clearly-unrelated
  singletons swept in transitively. Implemented and tested complete-linkage
  as a constrained alternative to single-linkage: it correctly isolates the
  clearest outlier and tightens the core cluster, but does not fully fix
  two remaining pairs where the underlying hash distance itself (not the
  clustering rule) is misleadingly small. Produced a full human-review
  package (`outputs/phase7b/human_review/pair_review.csv` [129 rows],
  `group_review.csv` [50 rows]) with every AI judgment explicitly labeled
  preliminary, not ground truth. **`outputs/phase7b/final_partition/`
  (threshold 6) was NOT regenerated or re-locked** — it remains on disk,
  byte-verified unchanged, for reference only. Also verified the exact
  dataset root (`C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`,
  confirmed two independent ways) and that all 3,688 manifest paths
  resolve (0 missing).
- **Phase 7B, second continuation** (`src/doar/human_review.py` [new],
  `phase7b_review_app.py` [new, Streamlit], `tests/test_human_review.py`
  [new, 29 tests], `main.py` [help-text correction only, no behavior
  change]): no model trained, no partition regenerated. Built a local,
  blind, point-and-click human-review interface covering 225 pairs across
  the same 4 categories the user specified (87 ambiguous, 71
  threshold-boundary — including an exhaustive add-on of every remaining
  dataset-wide dHash-distance-4 edge, 47 for the 17-image component's
  complete internal edge set, 20 sampled policy-change bridging edges).
  Blinding is enforced in code (`blind_item_for_display` is the one
  function allowed to decide what the reviewer's screen shows); decisions
  save immediately and incrementally; navigation supports back/forward,
  jump-to-item, and resume-where-you-left-off. Added export logic for 5
  files (`human_pair_reviews.csv`, `human_group_reviews.csv`,
  `reviewer_agreement_report.json`, `threshold_precision_summary.json` with
  Wilson 95% CIs, `unresolved_items.csv`) — none of which select or lock a
  policy. Produced a provisional (AI-preliminary-evidence-only, since no
  real human review has happened yet) comparison of conservative policies
  (exact+dHash≤2, ≤3, and a selectively-reviewed distance-4 tier) in
  `PHASE7B_DUPLICATE_POLICY.md` §25 — explicitly a comparison, not a
  selection. Rebuilt `outputs/phase7b/ARTIFACT_MANIFEST.tsv` (255 files,
  ≈15.1 MB). `outputs/phase7b/final_partition/` remains untouched.

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

**Interfaces**: CLI (`main.py`, 33 commands, including Phase 7A's new
`build-partition`) + Streamlit app
(`streamlit_app.py`, includes an upload-and-analyze flow). No REST API
exists or is planned without a concrete need.

**Persistence**: 100% flat JSON/CSV files. No database. Case outputs are
versioned (Phase 1); the review-master CSV is genuinely append-only.

**What actually runs correctly, proven by execution** (not just
code-reading): the full CLI pipeline end-to-end, on real data, multiple
times — a leakage-cleaned subset (~2,171 images, an earlier session), the
full uncleaned dataset for a single model (Phase 3A), the full uncleaned
dataset across **4 architectures** (Phase 4), **3 seeds × 3 architectures**
(9 model×seed checkpoints total) plus post-hoc temperature-scaling
calibration on all 9 (Phase 5), and now a **full single-image pipeline
integration test of the Phase 5-selected `efficientnet_b0` checkpoint**
against the Phase 3A `mobilenet_v3_small` checkpoint, side by side, on the
same 4 smoke-test drawings (Phase 6). All produced real trained checkpoints,
real metrics, real bilingual case reports.

---

## 4. Exact Phase 3A through Phase 7B commands and results

Full detail lives in `PHASE3A_RESULTS.md`, `PHASE4_RESULTS.md`,
`PHASE5_RESULTS.md`, `PHASE6_RESULTS.md`, `PHASE7_RESULTS.md` +
`PHASE7_EXPERIMENT_MATRIX.csv`, `PHASE7A_DATASET_READINESS.md` +
`PHASE7_EXPERIMENT_MATRIX_REVISION.md`, and `PHASE7B_DUPLICATE_POLICY.md` —
read those directly for anything you plan to cite. Headline:

**Phase 3A**: `mobilenet_v3_small`, seed 42, 10 epochs, on the full uncleaned
dataset (3,688 images, 0 unreadable). Validation macro-F1 0.704, accuracy
0.729 (n=310, preliminary). 4-image pipeline smoke test (one test-split
image per class): 3 of 4 predictions matched the folder label, zero errors —
**this is an integration-check observation, not a performance estimate**,
and it is the origin of the 4 test-split images that Phase 6 later disclosed
as non-blind (see the Phase 6 entry below and `PHASE6_RESULTS.md` §2).
Version-history round-trip and report structural inspection both verified
correct (`PHASE3A_RESULTS.md` §7a/§7b). Arabic linguistic fluency NOT
validated by a qualified speaker — still open.

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
and `resnet18` swap 1st/2nd depending on seed (their mean gap is ~0.0112
against population std ~0.0076/~0.0067); their observed ranges overlap and
their ranking flips on seed 2026. **This comparison is inconclusive
regarding superiority — no formal superiority or equivalence test was run,
and neither "comparable" nor "statistically indistinguishable" is claimed**
— see `PHASE5_RESULTS.md` §5. Temperature-scaling calibration was applied
to all 9 checkpoints (validation-only, never test): it improved NLL/Brier in
all 9 cases but worsened ECE in 3 of 9. Calibration behavior differed by
metric and phase: `efficientnet_b0` had the best mean pre-calibration
ECE/NLL/Brier, but `mobilenet_v3_small` had the lowest mean *post*-
calibration ECE (0.0417 vs. `efficientnet_b0`'s 0.0543), while
`efficientnet_b0` kept the best post-calibration NLL/Brier — no model was
uniformly best on every calibration metric. Macro-F1/accuracy are identical
before and after calibration for every run, as expected (temperature scaling
doesn't change arg-max predictions). Preliminary recommendation for a
**future** end-to-end pipeline test: `efficientnet_b0` (best mean macro-F1,
best pre-calibration ECE/NLL/Brier, best post-calibration NLL/Brier, and a
much smaller checkpoint than `resnet18`) — but this is **not** a decision to
replace the production/default model, and no test-split evaluation occurred.
Full 9-run detail, per-class variation, and artifact hashes in
`PHASE5_RESULTS.md`.

**Verification:** full test suite passed (226 tests). `compileall` passed
(exit 0). `ruff check .` **failed its pass/fail gate with 792 findings** —
reported honestly here as a failing Ruff run; no `src/doar` or `tests/` code
was changed in Phase 5 (see §2 above), so all 792 findings are pre-existing
and Phase 5 introduced no identified new one, but the codebase as a whole
does not currently pass Ruff.

**Phase 6**: end-to-end integration validation of the Phase 5-selected
`efficientnet_b0` checkpoint (seed 42, the highest validation macro-F1 of
the three Phase 5 seeds, 0.7364 — the Phase 5 calibrated reference copy at
`outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt`), run
through `main.py analyze-image` on the same 4 Phase 3A smoke-test drawings,
side by side against the Phase 3A `mobilenet_v3_small` checkpoint
(`outputs/phase3a/model/best.pt`) as reference. **All 8 runs (+2 more for
the versioning check) exited 0, zero errors/warnings/exceptions found.**

**Test-split disclosure (correction, added 2026-08-03):** no *aggregate*
test-set performance evaluation or model selection was performed — the
guarded `--split test` path was never invoked. But these 4 individual
test-folder images (the same 4 Phase 3A first used) **were** reused for
integration smoke testing, and their folder labels and both checkpoints'
predictions were inspected and reported. **The existing test split
therefore cannot be described as completely untouched.** These 4 cases are
the full, disclosed extent of test-split exposure across Phase 3A and
Phase 6 — no other test-split image or aggregate statistic has been
computed or inspected in any phase to date. A newly locked, leakage-safe
test set, disjoint from these 4 images, will be required for final thesis
evaluation. Full disclosure in `PHASE6_RESULTS.md` §2.

Both checkpoints agreed on the predicted class for all 4 drawings, and each
checkpoint's prediction matched the folder label on 3 of the 4 drawings
(same Fear→Angry miss both times — consistent with every architecture
tested so far). **This is an integration-test observation on 4 disclosed,
previously-inspected images, not a performance estimate.** Objective features
(quality/segmentation/composition/colour) were confirmed byte-identical
between the two checkpoints' runs for every drawing, as they must be — no
coupling bug found. Emotion-derived evidence (`ev_emotion_prediction`)
correctly differed per checkpoint and matched each run's own `analysis.json`
exactly; `rules.json` was identical between checkpoints for every drawing,
correctly explained by the fact that none of the 6 currently-evaluable
rules (all size/placement-based) consume emotion evidence. All 40 generated
report files (4 drawings × 2 checkpoints × 5 report types) were scanned for
placeholders/mojibake — zero issues. The version-history round-trip
(re-analyze unchanged → no `versions/`; remove checkpoint → exactly the 4
changed files archived, original prediction preserved; restore checkpoint →
exact reproduction) was repeated with the new checkpoint and confirmed
correct, replicating Phase 3A §7a's result under this different
architecture/calibration configuration. **No bug found, no code changed, no
default/production model changed, no aggregate test-split evaluation
performed.** Full detail in `PHASE6_RESULTS.md`.

**Verification:** full test suite passed (226 tests, unchanged from Phase 5
— no code changed so no new tests were needed). `compileall` passed (exit
0). `ruff check .` **792 findings, exit 1** — identical count to Phase 5,
confirming no new finding was introduced (`git status` showed zero tracked
files modified during the investigation itself).

**Phase 7: proposal only, nothing trained, nothing tuned, test split not
accessed.** A feasibility-grounded research programme (`PHASE7_RESULTS.md`
+ `PHASE7_EXPERIMENT_MATRIX.csv`, 22 proposed experiments across 5
families: deep-classifier screening, pretrained SSL embeddings,
interpretable feature baselines, fusion, and fine-tuning ablations). Key
feasibility findings from real (non-training) inspection this phase: 4 of
the 8 registered architectures (`small_cnn`, `mobilenet_v3_small`,
`resnet18`, `efficientnet_b0`) have ever been trained — `resnet50`,
`mobilenet_v3_large`, `convnext_tiny`, and `vit_b_16` are fully wired but
unused; `timm` (DINOv2, DenseNet, compact ViTs) and `open_clip_torch` are
already installed and mostly already wired into `embeddings.py` but never
exercised; the classical-feature baseline (`compare-models`, 6 model
families) and feature-ablation tool (`run-ablation`) already exist and have
never been run under the current manifest; early/late fusion machinery
(`fusion/trainer.py`, `fusion/late.py`) already exists and is unused. Real
dataset/leakage counts sourced directly from
`outputs/phase5/deep/leakage_gate/leakage_report.json`: 324 exact +
1,442 near cross-split duplicate groups + 48 label conflicts = 1,517/3,688
(41.1%) flagged images, 2,171 clean; **no subject/child-level grouping
exists in this dataset at all**, so any future "leakage-safe" partition can
only be image-level safe, never subject-level safe. Hardware: 6.44 GB VRAM
(Quadro P3200, confirmed), 47 GB free disk (current `outputs/phase3a`–
`phase6` footprint ≈1.65 GB). No architecture/family in the matrix requires
a second training framework — every new component identified (DenseNet121
registry entry, a `timm`-backed compact-ViT loader, HOG features, focal
loss, a "no augmentation" profile) is a small, targeted extension of
existing modules, detailed in `PHASE7_RESULTS.md` §1.8. Estimated total
compute for the full proposed programme: ≈5–9 hours GPU-inclusive
wall-clock, ≈4–6 GB new storage (§6) — not currently a blocking constraint.
**Stops after the proposal; no experiment has been approved to run.**

**Phase 7A: dataset analysis and split construction only, no model
trained.** Independently reproduced the existing leakage counts exactly
(324 exact + 1,442 near cross-split duplicate groups, 48 exact-only label
conflicts, 1,517/3,688 flagged — all 4 matched the recorded
`leakage_report.json` precisely). Went further than the existing
`leakage.assess_leakage()` (which only checks cross-split violations of the
*current* split) by computing the full duplicate-group structure across
the *entire* dataset via union-find over exact SHA-256 + aHash near-dup
(threshold 5, transitively closed): **2,106 groups from 3,688 images**, and
discovered that this reveals **124 label-conflict groups / 674 images
(18.3%)** — far more than the existing exact-only check's 48, because
near-duplicate pairs with mismatched labels were never checked before. A
threshold-sensitivity scan (0–8) showed the existing default of 5 already
produces a 137-image chaining artifact (single-linkage transitive closure
absorbing many non-duplicate, low-visual-entropy drawings); 13 of the 15
largest groups span multiple classes and are therefore already
(correctly) excluded as conflicts regardless. Built a new, deterministic,
group-disjoint, class-stratified partition (seed 42, `main.py
build-partition`, byte-reproducible across separate process runs — a real
non-determinism bug from Python's per-process set-hash randomization was
found and fixed along the way): **train 2,305 images / 1,273 independent
groups, valid 254 (all singleton groups), test 455 (all singleton groups,
newly locked), 674 excluded pending conflict review.** All required
verification checks pass (no group crosses partitions, no image in >1
partition, no unresolved conflict enters a supervised split, counts
consistent). Explicitly **not** claimed as "subject-independent" — no
child/subject identifier exists in this dataset, so this is
"image-group-disjoint and duplicate-controlled" only (see
`PHASE7A_DATASET_READINESS.md` §6). Produced a proposed (not applied)
revision to the Phase 7 experiment matrix: ConvNeXt-Tiny downgraded from
Recommended to probe-first Optional given the true effective training size
is 1,273 independent groups, not 2,821 raw images; DINOv2/OpenCLIP
frozen-embedding experiments relatively strengthened for the same reason.
**Stops after dataset-readiness verification; no Phase 7 Stage 0 experiment
has started.**

**Phase 7B: dataset analysis and split construction only, no model
trained, tuned, calibrated, or evaluated.** Treated Phase 7A's aHash-5
partition as explicitly provisional and validated it with a blinded manual
pair audit: 87 anonymized side-by-side pairs (exact duplicates, near-dup
pairs at every aHash Hamming distance 0-5, hard negatives at distance 6,
small groups, and direct/non-adjacent samples from the 3 largest chained
groups), judged with all split/label/category metadata held in a separate
file never opened during judgment. **Result: aHash precision was 100%
(Hamming 0-1, 95% Wilson CI [64.6%, 100%]) but fell to 0% at Hamming 5 —
the existing default's own boundary (95% CI [0%, 27.8%])** — corroborated
by a population-wide cross-label-edge ratio that climbs from 9.6% to 25.8%
across the same range. Directly inspected the 137-image component from
Phase 7A: its 3 sampled direct graph edges were all real duplicates; its 3
sampled non-adjacent members were all unrelated — confirmed the mechanism
is single-linkage chaining through a handful of recurring, low-visual-
entropy templates (a "sad face in rain" sketch, a generic circle-face
template, a branching "twig-person" sketch), not a wholesale detector
failure. Compared aHash against dHash (difference hash — equally simple,
no new dependency) on the same audit pairs: **every true-duplicate-like
pair had dHash distance <=5; every false-positive pair had dHash distance
>=16**, a clean separation aHash never showed. Adopted **dHash, threshold
6** (validated at full-dataset scale: largest component only 17 images,
vs. aHash-5's 137), keeping single-linkage clustering unchanged (the
audit showed the threshold/method was the problem, not the linkage rule).
Added previously-exposed-image handling: the 4 Phase 6 smoke-test images'
duplicate groups are force-assigned to train, verified to never enter
valid/test. Regenerated the partition: **train 2,564 images / 1,414
independent groups, valid 282 (all singleton), test 504 (all singleton,
newly locked, not inspected or predicted on), 338 excluded pending
conflict review (down from Phase 7A's 674)** — more usable data recovered
with a measurably cleaner duplicate structure, not just a different
threshold pick. Proposed (design only, not implemented) inverse-group-size
sample weighting as the default correction for train's remaining
within-group duplication, plus one limited future ablation (weighted vs.
unweighted, 3 seeds, one architecture). Phase 7A's own artifacts and the
original aHash reproduction path are unmodified and re-verified unchanged.
**Stops after the duplicate policy and regenerated partition are
verified; no Phase 7 Stage 0 experiment has started.**

**Phase 7B continuation — CORRECTION, same phase: dHash threshold 6 is
NOT finalized, no partition is locked.** The comparison above ("every
true-dup pair had dHash distance ≤5, every false-positive ≥16") was
computed on a sample confounded by construction: those 45 pairs were
originally selected because *aHash* flagged them as near-dup, with dHash
distance recorded only incidentally. A second, purpose-built audit sampled
directly from dHash's own near-dup edges at each exact distance 2–8 (42
new pairs, same blinding procedure): **precision was 100% at distance 2–3,
but fell to 50% at distance 4 and 0% at distance 5** — materially earlier
than the first comparison implied. Direct visual inspection of the
17-image dHash-6 cross-label component (contact sheet) found it is a chain
of at least 4 visually distinct sub-groups plus several clearly-unrelated
singletons (a noisy scribble, a notebook portrait, an unrelated photo), not
one coherent duplicate cluster. Implemented and tested **complete-linkage**
as a constrained alternative to single-linkage on this exact component: it
correctly isolates the clearest outlier and tightens the tightest
sub-cluster, but 2 of 6 resulting sub-clusters still pair visually
unrelated images because their *measured* hash distance (not the
clustering rule) is misleadingly small — complete-linkage fixes chaining
but not individual hash mismeasurement. **Produced a full human-review
package** (`outputs/phase7b/human_review/pair_review.csv` [129 rows],
`group_review.csv` [50 rows], plus supporting contact sheets) with every
AI judgment from both audits explicitly labeled a preliminary annotation,
not ground truth. `outputs/phase7b/final_partition/` (built at threshold 6
in the prior part of this session) was **not** regenerated or re-locked —
it remains on disk for reference only. Also verified: the exact dataset
root is `C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`
(confirmed two independent ways from the manifest), and all 3,688
manifest paths resolve (0 missing). **No threshold or clustering policy
is approved. Do not use `outputs/phase7b/final_partition/` as if it were
final until the human-review package has been reviewed.**

**Phase 7B second continuation — built the interactive review app and a
provisional conservative-policy comparison, still no policy locked.** The
static CSV human-review package above was hard to actually use, so
`phase7b_review_app.py` (Streamlit, backed by `src/doar/human_review.py`)
now presents all 225 pairs (87 ambiguous + 71 threshold-boundary,
including an exhaustive add-on covering every one of the 43 dataset-wide
dHash-distance-4 edges + 47 for the 17-image component's complete internal
edge set + 20 sampled policy-change bridges) blind, one at a time, with
incremental save, resume, and a 5-file export
(`human_pair_reviews.csv`/`human_group_reviews.csv`/
`reviewer_agreement_report.json`/`threshold_precision_summary.json` with
Wilson CIs/`unresolved_items.csv`). A **provisional** (AI-preliminary
evidence only — no real human review has happened yet) comparison of
exact+dHash≤2, ≤3, and a selectively-reviewed distance-4 tier is in
`PHASE7B_DUPLICATE_POLICY.md` §25: distances 2-3 show 100% precision (n=6,
wide CI) with near-identical structural cost (largest component stays 9);
distance 4 is where precision visibly splits (50%, n=6) without yet
showing threshold-6's structural blow-up (component 17, cross-label ratio
jump) — which is exactly why it's a candidate for edge-by-edge review
rather than blanket inclusion. **This is a comparison of options, not a
recommendation adopted by the project; no threshold, clustering method, or
partition has been selected or locked.** Launch:
`.\.venv\Scripts\Activate.ps1` then `python -m streamlit run
phase7b_review_app.py`, open http://localhost:8501 (see RUN_GUIDE_WINDOWS.md
§11).

---

## 5. Files created or modified this session

**Phase 3A Closure Verification** (committed `cf6512f`): `PHASE3A_RESULTS.md`
(corrected + 2 new sections), `SESSION_HANDOFF.md` (v1, now superseded by
this v2).

**Phase 4** (committed `73b082f`): `src/doar/deep/trainers.py` (parameter
count fields), `tests/test_trainer_regression.py` (3 new tests),
`PHASE4_RESULTS.md` (new), `SESSION_HANDOFF.md` (v2).

**Phase 5** (committed `1542505`): `PHASE5_RESULTS.md` (new),
`SESSION_HANDOFF.md` (v3). **No `src/doar` or `tests/` files changed** — no
code changes were needed this phase (see §2 above).

**Phase 5 correction** (committed `d0c5ca2`): `PHASE5_RESULTS.md` (wording
corrections), `SESSION_HANDOFF.md` (v4, matching corrections). Documentation
only.

**Phase 6** (committed `84623bd`): `PHASE6_RESULTS.md` (new),
`SESSION_HANDOFF.md` (v5). **No `src/doar` or `tests/` files changed** — no
bug was found this phase (see §2 above).

**Phase 6 correction** (committed `4aeb7db`): `PHASE6_RESULTS.md`
(test-split disclosure added), `SESSION_HANDOFF.md` (v6, matching
disclosure). Documentation only, no rerun.

**Phase 7** (committed `526c146`): `PHASE7_RESULTS.md` (new),
`PHASE7_EXPERIMENT_MATRIX.csv` (new, 22 rows), `SESSION_HANDOFF.md` (v7).
**No `src/doar` or `tests/` files changed — proposal only, nothing
trained** (see §2 above). Unlike `outputs/`, both new Phase 7 files are
**committed** (like `EXPERIMENT_PRIORITIZATION.md`,
`DETECTOR_DEPENDENCY_MATRIX.csv` before them) since they are planning
documents, not generated experiment artifacts. Phase 7 produced no
`outputs/` directory at all (nothing was run).

**Phase 7A** (committed `8f6d135`): `src/doar/partition.py` (new module),
`tests/test_partition.py` (new, 20 tests), `main.py` (new `build-partition`
command), `PHASE7A_DATASET_READINESS.md` (new),
`PHASE7_EXPERIMENT_MATRIX_REVISION.md` (new, proposes changes only —
`PHASE7_EXPERIMENT_MATRIX.csv` itself is untouched), `SESSION_HANDOFF.md`
(v8). Real code was added this phase (unlike Phase 7's pure proposal), but
no model was trained or evaluated — this is dataset analysis and
split-construction tooling.

**Phase 7B** (committed `ef86d40`): `src/doar/dataset.py` (additive:
`_difference_hash`, `DHASH_THRESHOLD`), `src/doar/partition.py` (extended:
`hash_field`, `compute_dhash_column`, `identify_exposed_groups`,
`exposed_group_ids` support, `verify_no_exposed_group_in_valid_or_test`),
`tests/test_partition_phase7b.py` (new, 9 tests), `main.py`
(`build-partition` gains `--hash-field`/`--exposed-image-ids`),
`PHASE7B_DUPLICATE_POLICY.md` (new), `SESSION_HANDOFF.md` (v9). No model
trained, tuned, calibrated, or evaluated.

**Phase 7B continuation** (committed `5d4c56f`): `src/doar/partition.py`
(additive: `refine_with_complete_linkage`), `tests/test_partition_phase7b.py`
(+5 tests, `CompleteLinkageRefinementTests`), `PHASE7B_DUPLICATE_POLICY.md`
(extensive correction — new §§14-18, renumbered §§9-13 → §§19-23),
`SESSION_HANDOFF.md` (v10). No model trained. **This is a
correction of the prior part's own conclusion**, not new independent work —
see §4 above.

**Phase 7B second continuation** (committed `fac795e`): `src/doar/human_review.py`
(new, core review/export logic), `phase7b_review_app.py` (new, Streamlit
UI), `tests/test_human_review.py` (new, 29 tests), `main.py` (help-text
correction only), `PHASE7B_DUPLICATE_POLICY.md` (new §§24-25),
`RUN_GUIDE_WINDOWS.md` (new §11, compileall/ruff file lists extended),
`outputs/phase7b/human_review/README.md` (updated with launch
instructions), `SESSION_HANDOFF.md` (this file, v11, plus a follow-up
documentation-only commit fixing this commit table's self-reference). No
model trained, no partition regenerated or locked.

**Not committed** (git-ignored, `outputs/` rule): everything under
`outputs/phase3a/`, `outputs/phase4/`, `outputs/phase5/`, `outputs/phase6/`,
`outputs/phase7a/`, and `outputs/phase7b/` — see each phase's
`*_RESULTS.md`/`PHASE7A_DATASET_READINESS.md`/`PHASE7B_DUPLICATE_POLICY.md`
for exact paths, sizes, and SHA-256 hashes of every artifact that matters.
`outputs/phase4/ARTIFACT_MANIFEST.tsv` (25 rows),
`outputs/phase5/ARTIFACT_MANIFEST.tsv` (120 rows, ≈1.22 GB),
`outputs/phase6/ARTIFACT_MANIFEST.tsv` (202 rows, ≈8.3 MB),
`outputs/phase7a/ARTIFACT_HASHES.json` (12 files), and
`outputs/phase7b/ARTIFACT_MANIFEST.tsv` (255 files, ≈15.1 MB, rebuilt this
continuation) cover their respective phases in full. Phase 7B's human-review
package (`outputs/phase7b/human_review/`, including the interactive
review app's item registry and new composite images) and contact sheets
(`outputs/phase7b/contact_sheets/`) are included in that same manifest.

---

## 6. Operational, disabled, and unvalidated components

**Operational** (proven by execution): leakage detection + quarantine +
override-with-audit-log; objective feature extraction; classical and deep
model training across 4 architectures, now with multi-seed (3-seed)
confirmation for the 3 competitive ones; deep-checkpoint temperature-scaling
calibration (validation-only, now exercised for the first time on 9
checkpoints); the full single-image `analyze-image` pipeline confirmed
integration-correct with a calibrated `efficientnet_b0` checkpoint
end-to-end (Phase 6), including version-history behavior under a changed
checkpoint configuration; fusion training + calibration; locked-test
evaluation guard; tier-aware rule dispatch (6/19 rules can produce
`weak_support`, confidence ceilings 0.05–0.25, all 6 currently
composition/placement-based, none consume emotion-model evidence yet);
bilingual report generation; deterministic Q&A; versioned case persistence;
Streamlit upload-and-analyze.

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
competitive Phase 5 architectures is genuinely best — the `efficientnet_b0`
vs. `resnet18` macro-F1 comparison is inconclusive regarding superiority at
n=3 seeds (overlapping ranges, ranking flips on seed 2026, no formal
superiority test run — see `PHASE5_RESULTS.md` §5); `efficientnet_b0` has
since been integration-tested through the full pipeline on 4 disclosed
test-split images (Phase 6, no defects found), but this is not a
performance validation and does not resolve the architecture question; no
*aggregate* test-split evaluation or production-model replacement has
occurred**.

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
7. **(added 2026-08-03, VERIFIED SATISFIED by Phase 7A)** The eventual
   leakage-safe test set must exclude the 4 test-folder images disclosed in
   `PHASE6_RESULTS.md` §2 (`2-1_jpg.rf.f967ea55654e6fa4badbe2906373b7f7.jpg`,
   `Fear_1_10_jpg.rf.eb405d0904de942eabf3841d65ea4d01.jpg`,
   `Happy_1_19_jpg.rf.87ed22895e6a85ffbf4d1c43c2420ac1.jpg`,
   `3-4_jpg.rf.f9a20de96b74b4e2ace5fb04c364791d.jpg`) — their labels and two
   different checkpoints' predictions on them are no longer blind. Phase 7A
   traced all 4 through the new partition and confirmed none land in the
   new `test` split (1 landed in the 137-image excluded-conflict group, 2
   in train, 1 in valid) — see `PHASE7A_DATASET_READINESS.md` §5.4. This
   was a consequence of the partitioning algorithm, not specifically
   targeted, and was checked directly rather than assumed.

**Detector implementation** (future roadmap item, not a numbered phase,
not started): planning complete (`DETECTOR_DEPENDENCY_MATRIX.csv`,
`PHASE3_DETECTOR_EVALUATION_PLAN.md`), ranked last of the non-dataset
options in `EXPERIMENT_PRIORITIZATION.md` — every one of the 13 unavailable
rules stays scientifically unvalidated even if a detector works, so this
remains lower priority than dataset auditing or model comparison work.

**Literature review** is at "round 2 of an explicitly not-yet-exhaustive
process" — `LIT_EMOTION_FACE_ENCODING_007` is flagged as the best lead
requiring round-3 full-text verification before any further action.

**Phase 4's multi-seed proposal is complete (Phase 5)** and **Phase 5's
preliminary `efficientnet_b0` recommendation has now been integration-tested
end-to-end through the pipeline (Phase 6), with no defects found** — see §4
above, `PHASE5_RESULTS.md`, and `PHASE6_RESULTS.md`. What remains open:
Phase 6 confirms the pipeline **works** with `efficientnet_b0`; it does not
resolve the still-inconclusive `efficientnet_b0` vs. `resnet18` macro-F1
question from Phase 5 §5, and does not by itself justify standardizing on
`efficientnet_b0`. The user's own closing instruction for Phase 6 was to
stop before changing the default/production model, *aggregate* test-split
evaluation, dataset cleaning, detector work, or rule activation — so **an
explicit go-ahead is needed before any of those**.

**Test-split disclosure (open item, added 2026-08-03):** Phase 3A and Phase
6 both reused the same 4 test-folder images (one per class) for individual
`analyze-image` integration smoke tests, and their folder labels and
predictions were inspected both times — see `PHASE6_RESULTS.md` §2. No
aggregate test-split statistic exists, but these 4 specific images are no
longer blind and must be excluded from whatever leakage-safe final
evaluation set gets constructed. This is now a **required input** to the
paused dataset audit above (item 7), not just a data-quality question — the
audit must also produce a test set disjoint from these 4 disclosed images.

**Phase 7's proposal is complete and pending review (added 2026-08-03):**
`PHASE7_RESULTS.md` + `PHASE7_EXPERIMENT_MATRIX.csv` propose 22 experiments
across 5 families (deep-classifier screening, pretrained SSL embeddings,
interpretable feature baselines, fusion, fine-tuning ablations), all
grounded in real feasibility inspection of this repository, dataset,
hardware, and installed libraries — see §4 above for the headline. **No
experiment in it has been approved to run.** Phase 7 also names the dataset
audit above as a hard prerequisite for its own §7 (locked final-evaluation
protocol) — the audit and Phase 7's Stage 0/1 development-time screening
are independent and can proceed in either order or in parallel, but Phase
7's *final* evaluation cannot happen until the audit's leakage-safe
partition exists.

**Phase 7A's dataset-readiness work is complete (added 2026-08-03), and its
threshold-5 partition has since been explicitly superseded for development
use by Phase 7B (below)** — `PHASE7A_DATASET_READINESS.md` +
`PHASE7_EXPERIMENT_MATRIX_REVISION.md` remain valid as the first-pass
analysis and are not retracted; Phase 7A's own artifacts are preserved
unchanged. This substantially advanced the paused dataset audit above:
items 1, 3, 4 (partial), and 7 got real, verified answers; items 2, 5
(partial), and 6 remain open.

**Phase 7B's duplicate-policy work is NOT complete — a same-phase
continuation corrected its own initial conclusion (added 2026-08-03):**
`PHASE7B_DUPLICATE_POLICY.md` — see §4 above for the headline. The first
part's blinded manual pair audit found the original aHash near-duplicate
detector's precision collapses to 0% at its own default threshold (5), and
that switching to dHash appeared to resolve this. **A second, more
targeted audit then found the specific choice of dHash threshold 6 was
based on a confounded sample and is likely too permissive** (precision
already falls to 50% at distance 4, 0% at distance 5 in the corrected,
properly-sampled evidence) **, and that unrestricted single-linkage leaves
a genuinely heterogeneous 17-image component even at threshold 6.**
Complete-linkage was implemented and shown to help but not fully resolve
this. **`outputs/phase7b/final_partition/` (built at threshold 6) is NOT
approved and must not be adopted as the standard split.** A human-review
package now exists (`outputs/phase7b/human_review/pair_review.csv` [129
rows], `group_review.csv` [50 rows], plus contact sheets), and **a second
continuation built an interactive, blind, point-and-click review app**
(`phase7b_review_app.py` — launch with `python -m streamlit run
phase7b_review_app.py` after activating the venv, open
http://localhost:8501) covering 225 pairs across 4 categories with
incremental save and a 5-file export — this is now the recommended way to
actually do the review, not editing the CSVs by hand. A **provisional**
(AI-preliminary-evidence-only) comparison of conservative policies
(exact+dHash≤2, ≤3, selectively-reviewed distance-4) exists in
`PHASE7B_DUPLICATE_POLICY.md` §25 — a comparison, not a selection.
Reviewing the app's pairs is the concrete next step. **Five concrete open
decisions now exist:**
1. **(supersedes the old "adopt Phase 7B's split" decision)** Use the
   review app to decide a final near-dup threshold and clustering policy
   (single-linkage, complete-linkage, or something else) — only then
   should a partition be regenerated and locked via `main.py
   build-partition`. Do not adopt `outputs/phase7b/final_partition/`
   (threshold 6) as-is.
2. Whether to accept `PHASE7_EXPERIMENT_MATRIX_REVISION.md`'s proposed
   changes (ConvNeXt-Tiny downgraded to probe-first; DINOv2/OpenCLIP
   strengthened) — these were reaffirmed using the (now provisional)
   threshold-6 partition's numbers in `PHASE7B_DUPLICATE_POLICY.md` §20;
   revisit once a reviewed partition exists. `PHASE7_EXPERIMENT_MATRIX.csv`
   itself remains unmodified.
3. What (if anything) to do about the excluded-conflict images (338 under
   the provisional threshold-6 policy, subject to change once a reviewed
   policy is adopted) — every one is still flagged
   `exclude_pending_human_review`; no review has happened yet.
4. Whether to adopt `PHASE7B_DUPLICATE_POLICY.md` §7's recommended default
   (inverse-group-size sample weighting for `train`'s remaining
   within-group duplication) and its proposed single ablation — design
   only, not implemented, not yet approved.
5. Whether to do the review yourself using `phase7b_review_app.py`, or have
   a future session help organize/summarize the exported results for your
   review (the actual approve/correct decisions require a human, not
   another AI pass — the app itself never preselects or infers an answer).

None of Phase 7's Stage 0 experiments have started; this is unchanged from
before Phase 7A/7B.

---

## 9. Exact recommended prompt for the next session

Six reasonable next steps exist; pick based on current priority. Use
whichever prompt matches. **Note:** as of the Phase 7B continuation
(2026-08-03), no near-dup threshold or clustering policy is approved and
`outputs/phase7b/final_partition/` (built at dHash threshold 6) must NOT be
treated as usable — a same-phase re-audit found it likely too permissive.
The human-review package must be resolved and a partition regenerated
before any option below that depends on "the split" can actually run.

**If completing the human review of Phase 7B's duplicate-policy package
(the actual next step, recommended first):**
```
Continue the DOAR project. Read SESSION_HANDOFF.md Section 2's "Phase 7B
second continuation" entry and PHASE7B_DUPLICATE_POLICY.md Sections 14-17,
24-25 in full before acting. I have reviewed (or want help
organizing/summarizing, but not resolving) my decisions from
phase7b_review_app.py's export at
outputs/phase7b/human_review/exports/human_pair_reviews.csv,
human_group_reviews.csv, reviewer_agreement_report.json, and
threshold_precision_summary.json. (If I have not run the app yet: launch
it with `.\.venv\Scripts\Activate.ps1` then `python -m streamlit run
phase7b_review_app.py`, open http://localhost:8501, and tell me to do the
review before proceeding -- do not regenerate a partition on
AI-preliminary evidence alone.) My decision on the near-dup threshold is
[state: e.g. "use dHash threshold 3", "use dHash threshold 4 with
complete-linkage clustering", etc.] and on clustering method is
[single-linkage / complete-linkage / other -- state which]. Regenerate the
partition via `main.py build-partition` with these exact settings, verify
it with the existing verification functions in src/doar/partition.py, and
update PHASE7B_DUPLICATE_POLICY.md to record this as the final, approved
policy (not provisional). Do not change the threshold/method yourself --
use exactly what I specify above. Do not train or evaluate any model in
this session.
```

**If approving Phase 7's Stage 0 experiments (only once a reviewed,
approved partition exists per the option above -- do NOT use
outputs/phase7b/final_partition/ as-is):**
```
Continue the DOAR project. Read SESSION_HANDOFF.md, PHASE7_RESULTS.md,
PHASE7A_DATASET_READINESS.md, PHASE7B_DUPLICATE_POLICY.md, and
PHASE7_EXPERIMENT_MATRIX_REVISION.md in full before acting, plus
PHASE7_EXPERIMENT_MATRIX.csv for exact per-experiment specs. Treat the
repository at its current HEAD as the source of truth. Confirm that the
Phase 7B human-review package has actually been resolved and a partition
regenerated/approved (see PHASE7B_DUPLICATE_POLICY.md's final-policy
section) before proceeding -- if not, stop and ask instead of using
outputs/phase7b/final_partition/ as-is.

I approve [the revision in PHASE7_EXPERIMENT_MATRIX_REVISION.md, reaffirmed
by PHASE7B_DUPLICATE_POLICY.md Section 20 / specific parts of it -- state
which] and Stage 0 [specify which rows, e.g. "C1, C2, C3, C5, A1" or "all
of Stage 0 as revised"] per PHASE7_RESULTS.md Section 5.1's execution
order. Use the reviewed, approved Phase 7B partition (new_split column) for
all new training/evaluation -- this supersedes Phase 7A's own split, which
was explicitly provisional. Exclude any row with new_split ==
"excluded_conflict". Implement any required new component listed in
PHASE7_RESULTS.md Section 1.8 for those rows only (e.g. the DenseNet121
registry.py entry for A1) with a matching regression test, following this
project's existing read-verify-backup-write and safe-training conventions.
Do not run any Stage 1/2/3 row without separate approval. Do not access the
new_split == "test" images for anything. Do not clean or modify the
dataset. Do not implement a detector or activate psychological concerns.
Consider (but do not implement without separate confirmation)
PHASE7B_DUPLICATE_POLICY.md Section 7's inverse-group-size sample
weighting recommendation for train. Save results under a new
outputs/phase8/ (or an appropriately named) directory without touching
outputs/phase3a-7b, and produce an artifact manifest with SHA-256 hashes.
```

**If resolving the excluded-conflict images (Phase 7B's label-conflict review
-- note this list is provisional, computed under the not-yet-approved
threshold-6 policy, and will change once a partition is regenerated per the
option above):**
```
Continue the DOAR project. Read SESSION_HANDOFF.md and
PHASE7B_DUPLICATE_POLICY.md (including Sections 14-17 on the still-open
threshold/clustering decision) in full before acting, plus
outputs/phase7b/final_partition/partition_manifest.csv (filter
conflict_status == "unresolved_label_conflict", 338 images across 103
groups under the provisional threshold-6 policy -- treat this count as
illustrative, not final, since it will likely change once a reviewed
threshold/clustering policy is adopted). Do not resolve any conflict
automatically or by heuristic -- this requires actual human review of
specific images/groups. Help review a batch [specify size] if asked to
summarize/organize the conflicts, but do not change any label, do not move
any image, and do not modify outputs/phase7b/final_partition/ (it is
provisional and superseded once a new partition is built) without an
explicit, documented decision for each group reviewed. Preserve full audit
trail of any resolution.
```

**If resuming the remaining paused dataset/label-quality audit items:**
```
Continue the DOAR project. Read SESSION_HANDOFF.md, PHASE7A_DATASET_READINESS.md,
and PHASE7B_DUPLICATE_POLICY.md in full before acting. Phase 7A/7B
substantially advanced the paused dataset audit in Section 8 (items 1, 3,
4 partial, and 7 now have verified answers). What remains open: item 2
(confirmation that no original image/label was ever
deleted/moved/renamed/overwritten/auto-relabeled -- Phase 7A/7B confirmed
the dataset is unchanged THIS session via hash spot-checks, but the full
historical confirmation since the dataset's original ingestion is still
open), item 5 (integrity tests exist via tests/test_partition.py and
tests/test_partition_phase7b.py, but group-leakage/reproducibility tests
for the ORIGINAL contaminated split specifically were not added), and item
6 (a proposed next baseline experiment beyond what Phase 7/7A/7B already
proposed). Do not clean the dataset or exclude anything further without
separate approval. Do not repeat Phase 3A through Phase 7B's completed
work. Stop for approval before running any training.
```

**If the user wants to formally decide between efficientnet_b0 and resnet18
first (rather than accepting Phase 5's preliminary lean):**
```
Continue the DOAR project. Read SESSION_HANDOFF.md and PHASE5_RESULTS.md in
full before acting. Phase 5 (n=3 seeds) found the efficientnet_b0 vs.
resnet18 macro-F1 comparison inconclusive regarding superiority -- their
observed ranges overlap, their ranking flips on seed 2026, and no formal
superiority or equivalence test was run. Do not manufacture false
statistical confidence with more seeds under the same non-leakage-safe
protocol, and do not describe the two models as "comparable" or
"statistically indistinguishable" without actually running a test for that;
instead ask the user whether they want (a) more seeds under the current
preliminary protocol, (b) a formal statistical comparison run on the
existing 3-seed data, (c) to pick a model on secondary criteria alone
(calibration, runtime, size) without resolving the macro-F1 question, or
(d) to defer the decision until a reviewed, approved Phase 7B partition
(the threshold-6 split is not yet approved -- see Section 8 above) supports
a more defensible comparison. Do not decide this unilaterally.
```

**If proceeding toward a production-model decision for `efficientnet_b0`
without further experiments:**
```
Continue the DOAR project. Read SESSION_HANDOFF.md, PHASE5_RESULTS.md,
PHASE6_RESULTS.md, PHASE7_RESULTS.md, PHASE7A_DATASET_READINESS.md, and
PHASE7B_DUPLICATE_POLICY.md in full before acting. Treat the repository at
its current HEAD as the source of truth.

Phase 6 confirmed efficientnet_b0 (checkpoint at
outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt) runs
correctly through the full DOAR pipeline with no integration defects, on 4
now-disclosed, non-blind smoke-test drawings only (PHASE6_RESULTS.md
Section 2) -- this is not a performance validation. Phase 7 proposes a much
larger experiment programme (revised by Phase 7A/7B) that has not been
approved. Phase 7B produced a candidate development split (dHash threshold
6, single-linkage) that a same-phase continuation found likely too
permissive and not yet approved -- see SESSION_HANDOFF.md Section 8 and
PHASE7B_DUPLICATE_POLICY.md Sections 14-17 -- so there is currently no
validated split ready to use. Do not replace the production/default model
without first resolving (or explicitly deciding to set aside) the
still-inconclusive efficientnet_b0 vs. resnet18 macro-F1 comparison from
PHASE5_RESULTS.md Section 5, and without considering whether any part of
Phase 7/7A/7B's programme (including the still-open duplicate-policy
decision) should run first. Ask the user how they want to proceed before
taking any action that changes the default model, runs an aggregate
evaluation on the locked test split, cleans the dataset, implements a
detector, or activates psychological concerns.
```
