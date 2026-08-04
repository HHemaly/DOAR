# Model Experiment Infrastructure Audit

**Status: evidence-based audit only. No model was trained on real data as
part of writing this document.** Every claim is anchored to an exact file,
function, or command actually read or run this session — cross-checked
against the filesystem, never inferred from `PHASE7_EXPERIMENT_MATRIX.csv`,
`PHASE7_EXPERIMENT_MATRIX_REVISION.md`, or any other prior-session document
without independent verification. Where this audit's own new pipelines are
described, "smoke-tested" means executed end-to-end this session against
synthetic and/or tiny real-image (non-test-split) data — never a
scientific result.

---

## 0. Governing fact: the dataset gate is CLOSED

See `EXPERIMENT_PROTOCOL.md` §0 and `SESSION_HANDOFF.md` for the full
gate report. In one line: `outputs/phase7b/human_review/app_data/decisions.json`
does not exist (zero human review decisions recorded), no
`outputs/phase7b/APPROVED_POLICY.json` exists, and
`outputs/phase7b/final_partition/` remains explicitly provisional. **No
model in this audit — old or new — may be trained on real data toward a
reportable thesis result until the gate passes.** This audit and all new
code in this session were built and verified in preparation/smoke mode
only.

---

## 1. Which architectures are genuinely implemented

`src/doar/deep/__init__.py::MODEL_NAMES` (verified by reading the file,
then by actually building each model):

| Model | Registered before this session | Verified buildable | Notes |
|---|---|---|---|
| `small_cnn` | yes | yes | Plain `nn.Sequential`, no pretrained weights (scratch only) |
| `mobilenet_v3_small` / `_large` | yes | yes (pre-existing checkpoints load, §2) | |
| `resnet18` / `resnet50` | yes | yes | |
| `efficientnet_b0` | yes | yes | |
| `convnext_tiny` | yes | yes | **Already fully wired** — Experiment E needed no registry change, only a new config |
| `vit_b_16` | yes | yes | Not part of the 6 requested experiments |
| `densenet121` | **no — added this session** | **yes, verified this session** (`registry.py::build_model`, forward pass, freeze/unfreeze, penultimate-embedding extraction all directly exercised, see §7) | Experiment D |

`registry.py::freeze_backbone`/`unfreeze_all` already generalize to any
registered model via `getattr(model, "fc"/"classifier"/"heads", None)` —
DenseNet121's single `nn.Linear` classifier matches the existing
`"classifier"` lookup with no special-casing needed.

## 2. Which checkpoints exist and load correctly

Unchanged from the immediately-preceding session's audit
(`CURRENT_CAPABILITY_AUDIT.md` §2, re-confirmed here by directory listing,
not re-loaded a second time): real `.pt`/`.joblib` checkpoints exist under
`outputs/phase3a/`, `outputs/phase4/`, `outputs/phase5/`, `outputs/full_run/`,
`outputs/exploratory_full/`. **No checkpoint exists for `densenet121`,
`convnext_tiny`, or any of the 6 new experiment families** — this is
expected, since none of them has ever been trained; this audit adds the
*capability*, not a trained artifact.

## 3. Which models were genuinely trained

Same as the prior audit: mobilenet_v3_small (Phase 3A/4/5), resnet18
(Phase 4/5), efficientnet_b0 (Phase 4/5), plus classical/fusion models
under `outputs/full_run/`. **Zero of the 6 newly-requested experiment
families (objective-feature-only baseline as a dedicated family,
HOG/colour/geometry, DINOv2, DenseNet121, ConvNeXt-Tiny, image+feature
fusion) have ever been trained on real data** — this session implemented
and smoke-tested their pipelines but did not train any of them for real,
per the closed gate.

## 4. Dataset manifests used for each run

Every existing trained checkpoint (§3) used `outputs/phase5/manifest.csv`
(or an equivalent copy under `outputs/full_run/`) — the **original,
duplicate-contaminated** manifest, train=2,821/valid=310/test=557. No
checkpoint anywhere in this repository was ever trained on
`outputs/phase7a/` or `outputs/phase7b/`'s partition outputs (neither has
a `.pt`/`.joblib` file under it) — confirmed again this session via
`find outputs/phase7a outputs/phase7b -iname "*.pt" -o -iname "*.joblib"`
returning nothing.

## 5. Whether leakage affected each reported result

Yes, all of them, without exception — `outputs/phase5/deep/leakage_gate/leakage_report.json`
shows `"status": "FAIL_LEAKAGE_DETECTED"` (324 exact cross-split
duplicates, 1,442 near-duplicate cross-split pairs, 48 conflicting-label
groups), and the training override
(`leakage_override_audit.jsonl`) explicitly cites "Phase 5: preliminary
multi-seed confirmation ... explicitly not leakage-safe." Every reported
macro-F1 for mobilenet_v3_small/resnet18/efficientnet_b0 must be read as
**preliminary, contaminated-split** — labeled as such in every comparison
table this session's tooling produces (`thesis.py::DATA_PROVENANCE_LABELS`,
§9 below).

## 6. Preprocessing, augmentation, optimizer, scheduler, loss, stopping, class-imbalance, seeds

All traced directly from `src/doar/deep/trainers.py::train_image_model`
(read in full this session):

- **Preprocessing**: resolved per-model via `deep/preprocessing.py::resolve_preprocessing`,
  hashed and stored in the checkpoint (`preprocessing_spec`/`preprocessing_hash`); the
  executed train/eval transforms are proven consistent with the recorded
  spec by `tests/test_preprocessing_consistency.py`.
- **Augmentation**: `augmentation="conservative"` (config-selectable) via
  `deep/augmentations.py::build_transforms`.
- **Optimizer**: `adamw` (default), `adam`, or `sgd` (momentum=0.9 hardcoded);
  two param groups (head vs. backbone) with independent learning rates
  (`head_learning_rate=3e-4`, `backbone_learning_rate=1e-4` defaults).
- **Scheduler**: `reduce_on_plateau` (default, `ReduceLROnPlateau(mode="max", patience=2, factor=.3)`
  on validation macro-F1), `cosine` (`CosineAnnealingLR(T_max=10)`), or `none`.
- **Loss**: `CrossEntropyLoss(weight=class_weights, label_smoothing=.05)` —
  label smoothing is hardcoded, not configurable.
- **Class-imbalance handling**: inverse-frequency class weights only
  (`class_weighting=True` default); no oversampling/weighted sampler in
  the loss path (a `WeightedRandomSampler` option exists in
  `deep/datasets.py::build_loaders` but is a separate, opt-in mechanism
  from the loss-level class weights).
- **Early stopping**: `patience=7` default epochs of no validation
  macro-F1 improvement.
- **Freeze/unfreeze staging**: real, not a placeholder — `freeze_epochs=3`
  default freezes the backbone (`registry.py::freeze_backbone`) until that
  epoch, then calls `unfreeze_all` and **rebuilds** the optimizer/scheduler
  from scratch (preserving the requested settings) — exactly the staged
  strategy Experiment D's spec requires. Only a two-stage (frozen →
  fully unfrozen) schedule exists; there is no partial "unfreeze last
  block only" option in the current trainer.
- **AMP**: enabled automatically on CUDA (`torch.autocast`/`GradScaler`),
  inert (disabled) on CPU — no separate CPU-AMP code path, confirmed no
  special flags are needed for CPU-only execution.
- **Gradient accumulation**: `grad_accum_steps` (function default 1; CLI
  default 4 — noted here since it's an easy-to-miss discrepancy between
  calling the function directly, as `stage0_runner.py` does, vs. the CLI).
- **Seeds**: no fixed project-wide default inside `trainers.py` itself;
  every existing Phase 4/5 config and this session's new configs use
  `{42, 123, 2026}` for 3-seed runs, matching `experiments.py::DEFAULT_SEEDS`
  and `PHASE7_EXPERIMENT_MATRIX.csv`'s own seed column.

## 7. Saved metrics and predictions

Checkpoint contents (`trainers.py`, confirmed exhaustively): `model_state,
optimizer_state, scheduler_state, scaler_state, epoch, model_name, classes,
seed, image_size, preprocessing_version, preprocessing_hash,
preprocessing_spec, resolved_weights_id, model_version, calibration_status,
validation, history, best_valid_macro_f1, early_stopping_stale_epochs,
model_family, configuration_sha256`. `evaluation.py::compute_metrics`
computes **every metric requested by this task's spec** (macro-F1,
balanced accuracy, per-class precision/recall/F1, confusion matrix,
one-vs-rest AUROC, multiclass log loss, ECE, Brier score) — verified by
reading the function; nothing on that checklist is missing from the
existing shared evaluation code. Per-sample predictions are saved as
`predictions.csv` by every classical/embedding/fusion trainer (including
the two new ones this session, §9).

**Genuinely new this session** (`stage0_runner.py`/`provenance.py`, per
Phase 5 of the protocol, since neither existed before): dataset manifest
SHA-256 + explicit split-policy-version string
(`provenance.py::build_manifest_and_split_provenance`, additive, does not
change `config.py::save_run_metadata`'s existing schema) and a
`STAGE_COMPLETE.json`/`stage0_run_manifest.json` pair recording run ID,
timestamp, git commit, and per-stage status.

## 8. Reproducibility from configuration files

Real and CLI-verified: `main.py train-image-model --config configs/training/resnet18.toml`
loads via `config.py::load_config` (whitelist-checked TOML),
`config.py::resolve` (CLI-overrides-TOML precedence),
`config.py::assert_all_config_consumed` (fails loudly on an
accepted-but-unused field). `configs/training/resnet18.toml` and
`configs/training/quick_cpu_smoke.toml` were the only two full-training
example configs before this session; `configs/training/densenet121.toml`
and `configs/training/convnext_tiny_probe.toml` (Experiment E, explicitly
commented as a single-seed feasibility probe, not the full comparison —
see `EXPERIMENT_MATRIX.md` Experiment E) were added this session following
the identical schema, and both were confirmed loadable and internally
consistent (no unconsumed-field errors) as part of this session's own
config-plumbing tests (§10).

## 9. GPU and CPU execution

**CUDA is available on this machine right now**: `torch.cuda.is_available()`
→ `True`, device = **Quadro P3200** (confirmed live this session, matching
the pre-existing comment in `deep/compare.py`). CPU-only execution was
directly exercised this session (all smoke tests below ran with
`device="cpu"`, no special flags beyond passing `device="cpu"`) and is
also covered by pre-existing tests (`test_trainer_regression.py`,
`test_preprocessing_consistency.py`). AMP auto-disables on CPU (§6); no
GPU-only code path exists that would break on CPU.

**Peak GPU memory is not currently recorded anywhere in the training or
comparison pipeline** — the only call to `torch.cuda.max_memory_allocated()`
in the whole repository is in the separate, one-off `gpu_smoke.py` command.
This is a real, confirmed gap against this task's "Peak GPU memory where
measurable" metric requirement — not implemented in this session (adding
real GPU-memory instrumentation to `trainers.py` without a GPU run to
verify it against would risk an unverified claim; flagged here as
outstanding work, see `EXPERIMENT_PROTOCOL.md`'s "not yet implemented"
list).

## 10. Whether model selection ever used test results

**No, confirmed by direct grep, not by trusting variable names.**
`deep/trainers.py` and `deep/compare.py` contain zero references to the
string `"test"` as a split value — `test_used: False` is a hardcoded
literal in both `trainers.py` (`training_result.json`) and `compare.py`
(`deep_comparison.json`). The only place `split == "test"` is ever
evaluated is behind `test_guard.py::require_test_access` (or
`models.py::evaluate_model`'s own inline equivalent check), both of which
require `--unlock-test --confirm-final-evaluation --initiated-by <name>`
and write an append-only audit log. This session's new code
(`experiments.py`-derived `handcrafted_comparison.py`,
`deep/embedding_classifier.py`, `stage0_runner.py`) never references
`split=="test"` at all — confirmed both by design (only `train`/`valid`
splits are ever loaded) and by a dedicated regression test (§10 of
`EXPERIMENT_PROTOCOL.md`'s test list).

---

## 11. What already existed vs. what this session built

**Already fully implemented before this session** (verified by direct code
reading, not assumed from `PHASE7_EXPERIMENT_MATRIX.csv`'s existence):

- **Experiment A (objective-feature classical baseline)**: 100% complete.
  `extract-features` → `features.csv` (real 59-dim `objective_feature_row`
  output) → `train-feature-model`/`compare-models` →
  `experiments.py::run_feature_experiment`, already comparing all 6
  requested classifier families (logistic regression, linear SVM, RBF SVM,
  random forest, extra trees, HistGradientBoosting) across 3 seeds, with
  full metrics. `configs/experiments/objective_features.json` already
  lists exactly this classifier menu. **Zero new glue code was needed.**
- **Feature-family ablation** (`ablation.py`): drop-one-family ablation
  over `quality/segmentation/composition/colour/stroke/shape` already
  exists — this is a different (exclude-list) shape from Experiment B's
  requested (include-list, HOG/colour/geometry) ablation, so Experiment B
  still needed new code (§12), but the pattern and `compute_metrics`
  plumbing were directly reusable.
- **Experiment F (fusion)**: substantially complete.
  `fusion/trainer.py::train_primary_fusion` already fuses the real
  objective-features CSV with any cached `embeddings.npz` via 3 methods
  (`early_scaled_concat`, `pca_early_fusion`, `mlp_early_fusion`), with
  provenance verification, calibration, and a validation-only winner
  selection. `configs/training/primary_fusion.toml` already exists for a
  resnet18 embedding backbone. Late fusion of independently-calibrated
  probability exports is also already implemented
  (`fusion/late.py`/`fusion/oof.py`/`fusion/probability.py`). **New work
  needed: one config per additional embedding backbone** (§13) — no
  fusion *logic* changes.
- **DINOv2 as an embedding backbone**: `deep/embeddings.py::_extractor`
  already dispatches any `backbone.startswith("dinov2")` string to
  `torch.hub.load("facebookresearch/dinov2", backbone)`, with full
  preprocessing/provenance recording. What did **not** already exist: a
  classical-classifier trainer that compares multiple classifier families
  on top of a *cached embeddings-only* representation the way
  `experiments.py` does for objective features (`fusion/embedding_comparison.py`
  exists but fixes exactly one `LogisticRegression` pipeline per
  representation, and only in the context of representation-vs-fusion
  comparison, not a multi-classifier menu) — this was new work (§14).
- **ConvNeXt-Tiny**: fully wired in `registry.py` already (§1). Zero
  registry code needed; only a new, explicitly-labeled single-seed
  feasibility-probe config (§13), per
  `PHASE7_EXPERIMENT_MATRIX_REVISION.md`'s downgrade from "Recommended" to
  "Optional feasibility probe first."

**Genuinely new this session** (implemented, smoke-tested, not run for
real per the closed gate):

1. **`src/doar/dataset_gate.py`** — the code-enforced clean-split gate
   (§0). Verified against the real, current repository state: correctly
   reports `gate_passed: False` with 4 of 8 checks failing for the exact,
   real reasons (no decisions.json, no exports, no approval record, no
   freeze marker), and correctly reports the other 4 checks (group-crossing,
   exposed-image, count/path verification, test-set-untouched) as
   genuinely passing by reading the real, already-computed
   `leakage_verification_report.json` — not by re-deriving them.
2. **`densenet121` registered** in `deep/__init__.py`/`deep/registry.py`/
   `deep/embeddings.py`'s finetuned-embedding path (§1). Build, forward
   pass, freeze/unfreeze, and 1024-dim penultimate-embedding extraction
   all directly verified this session.
3. **`src/doar/hog_features.py`** — dependency-light, pure-numpy HOG
   (Histogram of Oriented Gradients) feature extraction. Neither `cv2`
   (this environment's opencv 5.0.0 build ships **without**
   `HOGDescriptor`, confirmed empirically) nor `scikit-image` (not
   installed) provides HOG here, so this is a from-scratch implementation
   of the standard Dalal-Triggs algorithm, fixed parameters (64×64
   grayscale, 8×8-pixel cells, 9 unsigned orientation bins, 2×2-cell
   blocks, L2 normalization → 1,764-dim descriptor), never searched, per
   `PHASE7_EXPERIMENT_MATRIX.csv` row C4's own stated leakage-avoidance
   rationale. Verified: flat images produce an all-zero descriptor, edge
   images produce a nonzero, fully deterministic descriptor, and a real
   drawing produces a correctly-shaped 1,764-value row.
4. **`src/doar/handcrafted_comparison.py`** — Experiment B's HOG/colour/
   geometry group comparison, reusing `experiments.py`'s exact classifier
   menu (`_model`, imported not duplicated). "Geometry" is defined as
   `composition.* + segmentation.* + shape.*` (spatial/layout signal,
   deliberately excluding `quality.*` and `stroke.*`) — documented in the
   module and in `EXPERIMENT_MATRIX.md`.
5. **`src/doar/deep/embedding_classifier.py`** — Experiment C's
   embeddings-only classical-classifier comparison (linear probe +
   small regularized MLP by default, full menu selectable), reusing
   `experiments.py::_model` for anything beyond the two RQ3-specific
   defaults.
6. **5 new config files** (`configs/training/densenet121.toml`,
   `configs/training/convnext_tiny_probe.toml`,
   `configs/training/primary_fusion_dinov2.toml`,
   `configs/experiments/handcrafted_groups.json`,
   `configs/experiments/embedding_classifier_dinov2.json`).
7. **`provenance.py::build_manifest_and_split_provenance`** — manifest
   checksum + split-policy-version helper (additive, §7).
8. **`src/doar/stage0_runner.py`** — the config-driven, resumable,
   gate-checked Stage 0 orchestrator (`StageSpec`/`run_stage0` core,
   fully unit-testable with injected fake stages;
   `build_real_stage0_plan` wires the 6 real families). **Two real bugs
   were found and fixed while smoke-testing this session's own new
   code**: (a) `train_image_model`'s actual keyword arguments are
   `model_name`/`patience`/`optimizer_name`/`scheduler_name`, not
   `model`/`early_stopping_patience`/`optimizer`/`scheduler` as first
   written; (b) `train_image_model` requires a physical torchvision
   `ImageFolder`-style dataset root (`root/train/<class>/*`,
   `root/valid/<class>/*`), not a flat manifest CSV directory — fixed by
   inferring the real dataset root from the manifest's own `path` column
   for full runs, and by building a dedicated tiny synthetic `ImageFolder`
   tree (matching `tests/test_trainer_regression.py`'s own fixture
   pattern) for smoke runs.
9. **4 new `main.py` CLI commands**: `extract-hog-features`,
   `train-handcrafted-groups`, `train-embedding-classifier`, `run-stage0`
   (with `--smoke`, `--only`, `--no-resume`, `--gate-check-only`) — all
   exercised via the actual command line this session, not only as direct
   Python calls.
10. **`thesis.py` extended**: 3 new auto-discovered figure types
    (objective-feature, handcrafted-group, embedding-classifier
    leaderboards) plus a new `build_cross_family_model_comparison()`
    function that pulls each family's best result into one table,
    **requiring an explicit `data_provenance` label** on every call
    (`preliminary_contaminated_split` / `clean_development_split` /
    `locked_test` / `smoke_test_not_a_result`) — it raises `ValueError`
    on anything else, so a caller cannot accidentally build an unlabeled
    or mislabeled comparison table.

## 12. Smoke-test results (synthetic + tiny real-image data, not scientific results)

All of the following ran this session, end to end, exit-code-clean:

- Experiment A + B: full pipeline (`extract-features` →
  `extract-hog-features` → `train-handcrafted-groups`) run against both a
  40-image fully-synthetic dataset and a real, 32-image stratified subset
  of `outputs/phase5/manifest.csv`'s `train`/`valid` rows (no test-split
  rows used) via the actual `main.py run-stage0 --smoke` CLI command.
- Experiment C: classifier logic verified against a synthetic,
  well-separated 4-class embeddings cache (real DINOv2 extraction was
  **not** attempted — no cached weights exist in this environment's
  `torch.hub` cache directory, and a live download was deliberately not
  triggered during an unattended session; see `EXPERIMENT_PROTOCOL.md` for
  the exact command to run once network access is available).
- Experiment D (DenseNet121): 1-epoch, CPU-only, `pretrained_weights="none"`
  training run against a tiny synthetic `ImageFolder` (4 classes × 4
  images), completing in ~5 seconds via the real `train_image_model`
  function, through the real orchestrator.
- Experiment E (ConvNeXt-Tiny probe): same as D, also completed.
- Experiment F (fusion): the orchestrator correctly **failed** this stage
  in smoke mode with `"Artifact provenance verification failed: missing
  provenance on one or both artifacts"` — this is `fusion/trainer.py`'s
  own `provenance.verify_artifacts()` safety check correctly refusing to
  fuse Experiment C's synthetic (network-avoidant, provenance-free)
  smoke embeddings against Experiment A's real (but synthetic-manifest)
  features. **This is the safety check working as designed, not a
  pipeline defect** — deliberately not bypassed or faked around; see
  `EXPERIMENT_PROTOCOL.md` for what a genuine Stage F smoke/full run
  requires (matching provenance from a real or consistently-synthetic
  manifest all the way through).
- The orchestrator's own core logic (gate enforcement, smoke-mode bypass,
  dependency-failure propagation, resume/skip-if-complete) was verified
  in isolation with injected fake stages, independent of any real
  pipeline.

No result above should be read as evidence of model quality — every
number produced is either from fully synthetic data or an intentionally
tiny (24-32 image), non-representative real-image subset used purely to
prove the code path executes.
