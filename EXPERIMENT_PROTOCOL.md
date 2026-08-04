# Shared Experimental Protocol — Stage 0 Model Comparison

**Status: protocol definition. §0's gate currently BLOCKS every full run
described here.** This document is the single shared contract every
experiment in `EXPERIMENT_MATRIX.md` must follow, so that differences
between experiments are attributable to the stated independent variable,
not to an uncontrolled protocol difference.

---

## 0. The dataset gate (read this first)

Checked live, this session, via `src/doar/dataset_gate.py::check_clean_split_gate()`
— **not** a documentation claim, an executable function you can re-run
yourself at any time:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from doar.dataset_gate import check_clean_split_gate; import json; print(json.dumps(check_clean_split_gate('.'), indent=2))"
```

**Current result: `gate_passed: false`.** 4 of 8 checks fail:

| Check | Status | Why |
|---|---|---|
| Human Phase 7B duplicate review recorded | **FAIL** | `outputs/phase7b/human_review/app_data/decisions.json` does not exist |
| Human pair/group decisions exported | **FAIL** | `outputs/phase7b/human_review/exports/` does not exist |
| Duplicate-detection policy approved | **FAIL** | `outputs/phase7b/APPROVED_POLICY.json` does not exist |
| Partition manifest frozen | **FAIL** | `outputs/phase7b/final_partition/FROZEN.json` does not exist; the partition there is explicitly provisional |
| No duplicate group crosses splits | pass | verified via `outputs/phase7b/final_partition/leakage_verification_report.json`'s own real `ok:true` |
| Exposed images excluded from valid/test | pass | same source |
| Class counts and paths verified | pass | same source |
| Final test set untouched | pass | no `final_test_unlock_log.jsonl` entries for this manifest |

**Exact remediation steps** (identical to what `check_clean_split_gate()`
itself returns as `report["remediation"]`):

1. `.\.venv\Scripts\Activate.ps1 ; python -m streamlit run phase7b_review_app.py`
2. Complete the human review of all 225 pairs across the app's 4 tabs.
3. Click **Export** in the app sidebar.
4. A human reviews the exported evidence and writes
   `outputs/phase7b/APPROVED_POLICY.json` with fields `approved_by,
   approved_at, near_dup_threshold, hash_field, clustering_method, notes`.
5. Re-run `python main.py build-partition` with the approved
   threshold/hash-field/clustering method.
6. A human (or an explicitly-instructed session) writes
   `outputs/phase7b/final_partition/FROZEN.json` with fields `frozen_by,
   frozen_at, approval_record, manifest_sha256`, only after re-confirming
   `leakage_verification_report.json` shows `ok: true` for every check.
7. Re-run the gate check above — every check must report `pass: true`.

**Until then**: every command in this protocol may only be run in
`--smoke` mode (synthetic or tiny non-test-split real-image data,
`main.py run-stage0 --smoke`), which is always permitted and never
produces a reportable result. `src/doar/stage0_runner.py::run_stage0`
enforces this in code — calling it with `smoke=False` while the gate
fails raises `CleanSplitGateFailed`, it does not merely warn.

---

## 1. What every experiment shares

- **Manifest**: once the gate passes, the single frozen
  `outputs/phase7b/final_partition/partition_manifest.csv` (or whichever
  path `FROZEN.json` references) — never `outputs/phase5/manifest.csv`
  (the old, contaminated split) for anything reported as a Stage 0 result.
- **Label mapping**: `src/doar/dataset.py::CLASSES = ("Angry", "Fear",
  "Happy", "Sad")`, fixed order, enforced by
  `deep/datasets.py::build_loaders`'s `class_to_idx` check and by every
  classical trainer's `CLASSES.index(...)` lookup — the same order
  everywhere, never re-derived per experiment.
- **Image-level inclusion policy**: `readable == "true"` rows only
  (matching `extract.py`/`extract_hog_features`'s existing filter); rows
  flagged `excluded_conflict` by the partition are excluded from every
  training/evaluation split, never silently included.
- **Development seeds**: `{42, 123, 2026}` — matches
  `experiments.py::DEFAULT_SEEDS` and every existing Phase 4/5 config.
  Stage 0's *first pass* uses **one seed (42) only**, per §6 below;
  3-seed runs are a deliberate escalation, not the default.
- **Model-selection metric**: mean validation macro-F1 across seeds,
  ties broken by lower std, then higher balanced accuracy, then lower
  ECE — same tie-break order already used by
  `experiments.py::run_feature_experiment`'s and
  `deep/compare.py::aggregate_and_select`'s leaderboards; this session's
  two new modules (`handcrafted_comparison.py`,
  `deep/embedding_classifier.py`) use the identical order.
- **No test-set access during development**: enforced twice —
  structurally, no development code path in this protocol ever loads
  `split=="test"` rows at all (not merely gated); and procedurally, even
  once the gate passes, `test_guard.py::require_test_access` still blocks
  any command that *could* touch the test split without
  `--unlock-test --confirm-final-evaluation --initiated-by`.
- **Comparable training budgets**: classical/embedding/HOG experiments
  (A, B, C) have zero hyperparameter search (fixed sklearn defaults,
  matching the existing project convention that a search itself would be
  a hidden validation-leakage path for this dataset size) — the "budget"
  that varies is only which classifier families are compared, held equal
  across A/B/C. Deep experiments (D, E) use the identical
  freeze-then-unfreeze schedule, optimizer defaults, and epoch/patience
  budget as the existing Phase 4/5 screening runs (D) or an explicitly
  reduced probe budget (E, see `EXPERIMENT_MATRIX.md`).

## 2. Required metrics (never accuracy alone)

Every experiment's leaderboard/result JSON already includes, via the
shared `evaluation.py::compute_metrics` (verified complete this session,
`MODEL_EXPERIMENT_AUDIT.md` §7):

macro-F1 (primary) · balanced accuracy · per-class precision/recall/F1 ·
confusion matrix (raw + normalized) · one-vs-rest AUROC (macro,
sklearn-optional) · multiclass log loss · expected calibration error (ECE,
10 bins) · Brier score · plus `accuracy` and `weighted_f1` as extras.

**Not yet recorded anywhere** (a real, confirmed gap, not fixed this
session — see §7): parameter count is available per-model from
`sum(p.numel() for p in model.parameters())` but is not currently written
to any checkpoint or result JSON; training time IS recorded per-run
(`training_result.json`'s `training_seconds`) but is not surfaced into
`deep_comparison.json`'s top-level summary; inference time is not
measured anywhere; peak GPU memory is not measured anywhere outside the
unrelated `gpu_smoke.py` command. **Any thesis table claiming these four
figures for the 6 new experiments must first add real instrumentation and
re-verify it — do not estimate or fabricate them.**

## 3. Seeds, repetition, and uncertainty

- **Stage 0 first pass**: 1 seed (42), restricted budget — a feasibility
  check, not a comparison with statistical weight (§6).
- **Full development comparison**: ≥3 fixed seeds (42, 123, 2026) for any
  experiment recommended for escalation after Stage 0.
- Every leaderboard already reports **mean and std across seeds**
  (`mean_macro_f1`/`std_macro_f1`, etc.) — this is not new, it is the
  existing convention in `experiments.py`, `ablation.py`, `deep/compare.py`,
  `fusion/trainer.py`, and this session's two new modules.
- **Do not treat a numerical difference smaller than the larger of the two
  compared configurations' `std_macro_f1` as meaningful.** No formal
  significance test (e.g. a paired test across seeds) is implemented
  anywhere in this codebase today — comparisons in `EXPERIMENT_MATRIX.md`
  and any future results table must describe differences in terms of
  "exceeds/does not exceed seed-to-seed std," not p-values that do not
  exist.

## 4. Real vs. smoke vs. preliminary — the three-way distinction

Enforced in code, not just prose, by `thesis.py::DATA_PROVENANCE_LABELS`
and `build_cross_family_model_comparison()`'s required `data_provenance`
argument (raises `ValueError` on anything else):

- **`preliminary_contaminated_split`** — any result trained on
  `outputs/phase5/manifest.csv` or an equivalent (Phase 3A/4/5's existing
  checkpoints, and any pre-gate re-run of the same manifest). Never used
  for a thesis conclusion; reported only as historical/screening context.
- **`clean_development_split`** — a genuine Stage 0 result trained on the
  gate-approved, frozen manifest, evaluated on `valid` only. **Does not
  exist yet** — the gate has never passed in this repository's history.
- **`locked_test`** — a `--unlock-test`-gated final evaluation. Also does
  not exist for the clean split (and must not, until Stage 0 development
  is complete and a final model is chosen on `valid` alone).
- **`smoke_test_not_a_result`** — this session's own synthetic/tiny-data
  pipeline verification runs (`MODEL_EXPERIMENT_AUDIT.md` §12). Never
  compared numerically against the other three categories in any table.

## 5. Configuration and reproducibility

Every run must be launchable from a config file (`configs/training/*.toml`
for deep models, `configs/experiments/*.json` for classical/embedding/
handcrafted families) and must record, per Phase 5 of this task: run ID,
timestamp, git commit (`stage0_runner.py::_git_commit`), full resolved
configuration + its SHA-256 (`config.py::save_resolved`, pre-existing),
dataset manifest SHA-256 + split-policy version
(`provenance.py::build_manifest_and_split_provenance`, new this session),
model name + pretrained-weights spec, seed, hyperparameters, training
history, best checkpoint, validation predictions, metrics, confusion
matrix, calibration info, and failure status if applicable
(`stage0_runner.py`'s `STAGE_COMPLETE.json` / failed-stage entries in
`stage0_run_manifest.json`). Runtime/hardware info is captured by
`config.py::environment_metadata` (python/torch/cuda/git versions,
GPU name) for deep-model runs; classical/embedding runs record `platform`/
`sklearn` version in their own summary JSON.

**Overwrite prevention**: `stage0_runner.py::run_stage0` writes a
`STAGE_COMPLETE.json` marker per stage and skips (does not overwrite) a
completed stage on re-run unless `--no-resume` is passed —
this is the resumability mechanism itself, not a separate feature.
`deep/compare.py` has its own, independent skip-if-`training_result.json`-exists
logic for multi-seed deep comparisons.

## 6. Stage 0 sequence (once the gate passes)

```
1. A — objective-feature classical baseline
2. B — HOG/colour/geometry handcrafted comparison
3. C — DINOv2 frozen embeddings
4. D — DenseNet121
5. E — ConvNeXt-Tiny feasibility probe
6. F — fusion (only after A and C/D have real, matching-provenance artifacts)
```

Run with **one seed, a restricted epoch/hyperparameter budget** initially
(`main.py run-stage0 --manifest <frozen_manifest> --output outputs/stage0`,
no `--smoke` flag once the gate passes — smoke mode always uses seed 42/1
epoch regardless). **After Stage 0**, inspect
`outputs/stage0/stage0_run_manifest.json` and each stage's leaderboard,
then decide — by hand, informed by §3's std-based judgment, not
automatically — which experiments justify a full 3-seed run. **Do not run
an expensive exhaustive multi-seed sweep of all 6 families automatically.**
The final test set is never evaluated during this process.

## 7. Explicitly outstanding (not implemented this session)

- Parameter count / inference time / peak GPU memory instrumentation
  (§2) — needed before Phase 8's "performance vs. computational cost"
  table can be populated honestly.
- A CNN-penultimate-embedding fusion config (Experiment F / D3) — depends
  on Experiment D's own winner being selected first; only the DINOv2
  fusion config exists today.
- A formal seed-to-seed significance test (§3) — currently, only
  descriptive mean/std is available anywhere in this codebase.
- Real DINOv2 weight download/caching in this environment (Experiment C)
  — requires live network access, not attempted this session.
