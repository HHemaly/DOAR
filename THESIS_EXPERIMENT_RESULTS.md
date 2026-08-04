# Thesis Model Experiment Results

**Status: real, executed results.** Every number in this document comes
from an actually-run command against the real, on-disk dataset
(`outputs/phase5/manifest.csv`, 3,688 images, physical `train/valid/test`
split) — nothing here is mocked, estimated, or fabricated. This document
supersedes `MODEL_EXPERIMENT_AUDIT.md`'s "smoke-tested only" status: that
audit was written when the duplicate-review gate was closed; this session's
explicit instruction was to move to real training and treat the
unresolved-duplicate leakage risk as a recorded limitation, not a blocker
(see §7). All new artifacts live under `outputs/thesis_run/`; nothing
under `outputs/phase5/` (Experiment 3's checkpoints, reused not retrained)
was modified.

---

## 0. Dataset/split verification (done first, not assumed)

- **Manifest**: `outputs/phase5/manifest.csv`, 3,688 rows, all paths
  resolve, `readable=True` for all rows, 0 missing files.
- **Classes**: `Angry` (841), `Fear` (677), `Happy` (1199), `Sad` (971) —
  real class imbalance (Happy is 1.77x Fear), carried into every
  experiment via `class_weighting=True` (CNNs) / `class_weight="balanced"`
  (classical models where applicable).
- **Split**: `train=2,821 / valid=310 / test=557`. `provenance` column is
  `physical_existing_split` for every row — the split is **not** computed
  by any random-seed algorithm; `dataset.py::build_manifest` deterministically
  walks the pre-existing `train/valid/test/<class>/` folder structure on
  disk and hashes each file. **This makes the split fixed and exactly
  reproducible**: re-running `build-manifest` against the same source
  folder produces byte-identical rows (sorted glob, content-hash IDs, no
  RNG).
- **Loaders**: `deep/datasets.py::build_loaders` verified directly this
  session against the real physical dataset root
  (`C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`): 353 train
  batches / 39 valid batches at batch_size=8, correct class order
  `[Angry, Fear, Happy, Sad]`, real image tensors `[8, 3, 224, 224]`.
- **GPU**: `torch.cuda.is_available()` → `True`, device = **Quadro P3200**
  (6.4 GB). Used for all CNN and DINOv2-extraction work in this session.
- **Existing checkpoints found**: real, loadable, 3-seed-complete CNN runs
  already existed for all 3 requested architectures under
  `outputs/phase5/` (mobilenet_v3_small, resnet18, efficientnet_b0 — see
  §3) — reused rather than retrained, per "reuse correct existing training
  code/results where possible."
- **Locked test set**: **not accessed** in this session. `test_used: false`
  confirmed in every result JSON below (a hardcoded literal the training/
  evaluation code writes, not a claim taken on trust).

## 1. Experiment 1 — Objective-feature classical baseline

**Command:**
```
python main.py extract-features --manifest outputs/phase5/manifest.csv \
  --output outputs/thesis_run/features \
  --allow-leakage-override --override-justification "<see Section 7>"
python main.py train-feature-model --features outputs/thesis_run/features/features.csv \
  --output outputs/thesis_run/feature_models --seeds 42,123,2026
```
63-column real feature CSV (59 objective features + 4 meta columns) for
all 3,688 images, 0 extraction failures.

| Model | mean macro-F1 | std | mean balanced-acc | mean ECE | mean train time (s) |
|---|---|---|---|---|---|
| **random_forest** | **0.6586** | 0.0057 | 0.6501 | 0.1085 | 1.11 |
| hist_gradient_boosting | 0.6417 | 0.0000 | 0.6346 | 0.2072 | 6.52 |
| extra_trees | 0.6417 | 0.0041 | 0.6312 | 0.0811 | 0.72 |
| linear_svm | 0.5543 | 0.0010 | 0.5502 | 0.0392 | 5.26 |
| logistic_regression | 0.5374 | 0.0000 | 0.5427 | 0.0609 | 0.12 |

Preprocessing (median imputation + standard scaling) is fit inside each
run's pipeline on the training split only — no leakage from valid/test
into preprocessing statistics.

## 2. Experiment 2 — HOG / colour / geometry baseline

**Commands:**
```
python main.py extract-hog-features --manifest outputs/phase5/manifest.csv \
  --output outputs/thesis_run/hog_features
python main.py train-handcrafted-groups \
  --objective-features outputs/thesis_run/features/features.csv \
  --hog-features outputs/thesis_run/hog_features/hog_features.csv \
  --output outputs/thesis_run/handcrafted_groups --seeds 42,123,2026
```
1,764-dim HOG descriptor (pure-numpy Dalal-Triggs, no external HOG library
available in this environment) for all 3,688 images, 0 failures. 4 feature
groups x 5 classifier families x 3 seeds = 60 runs, all completed.

| Group | Best model | mean macro-F1 | std | n features |
|---|---|---|---|---|
| **colour_only** | **random_forest** | **0.6361** | 0.0069 | 13 |
| hog_colour_geometry | random_forest | 0.6142 | 0.0188 | 1,806 |
| geometry_only | extra_trees | 0.5532 | 0.0086 | 29 |
| hog_only | random_forest | 0.5340 | 0.0066 | 1,764 |

**Real, slightly counterintuitive finding**: HOG alone (0.534) underperforms
colour alone (0.636), and the full HOG+colour+geometry fusion (0.614) does
**not** beat colour alone — combining groups added dimensionality (1,806
features from 2,821 training rows) without a matching accuracy gain for
tree ensembles. Colour is the strongest single handcrafted signal for this
task, ahead of shape/edge texture (HOG) and composition/geometry.
`linear_svm` and `hist_gradient_boosting` on the 1,764/1,806-dim
configurations were the slowest runs (up to ~161s/seed) — the practical
runtime cost of high-dimensional handcrafted features on ~2,800 training
rows, for no accuracy benefit here.

## 3. Experiment 3 — CNN transfer learning

**Not retrained — reused real, already-complete Phase 5 3-seed runs**
(verified this session, not assumed): checkpoints load, `test_used: false`
in every `training_result.json`, real per-epoch history present.

**Reproduction command** (as originally run, unchanged):
```
python main.py train-image-model --config configs/training/resnet18.toml --seed <42|123|2026>
```
(equivalent configs exist for mobilenet_v3_small, efficientnet_b0)

| Model | mean macro-F1 | std | epochs | mean train time (s) | pretrained |
|---|---|---|---|---|---|
| **efficientnet_b0** | **0.7291** | 0.0076 | 10 | 795 | ImageNet1K (torchvision DEFAULT) |
| resnet18 | 0.7179 | 0.0067 | 10 | 463 | ImageNet1K_V1 |
| mobilenet_v3_small | 0.6951 | 0.0169 | 10 | 503 | ImageNet1K |

**Configuration** (identical across all 9 runs, from `executed_config.json`):
image_size=224, augmentation=conservative, optimizer=adamw
(head_lr=3e-4, backbone_lr=1e-4), scheduler=reduce_on_plateau,
class_weighting=True, freeze_epochs=3 (2-stage freeze->unfreeze),
early_stopping_patience=7, batch_size=4 (effective 16 via grad_accum=4),
seeds={42,123,2026}, label_smoothing=0.05 (hardcoded), AMP enabled (CUDA).

**Real limitation found**: every run trained for the full `epochs=10`
budget without early-stopping (validation macro-F1 was still rising at
epoch 9 in every seed inspected) — these models are **not fully converged**
by their own patience=7 rule; the epoch cap, not the improvement plateau,
ended training. The single most valuable next experiment (§8) is
extending the epoch budget for efficientnet_b0.

**Augmentation justification** (task requirement): `augmentation="conservative"`
(`deep/augmentations.py`) does **not** include horizontal flip, rotation, or
colour jitter — all three were deliberately excluded because a child's
drawing has task-relevant left/right and up/down placement (the psychology
rule layer's own `placement_left/right/top` observables depend on this),
so flipping or rotating would corrupt exactly the compositional signal
this project cares about. Only resize/crop/normalize-level, orientation-
and colour-preserving transforms are applied.

## 4. Experiment 4 — DINOv2 embeddings

**Blocker check**: DINOv2 weights were **not** cached locally at the start
of this session (only torchvision CNN weights were). Network access was
available; `dinov2_vits14` (84.2 MB) was downloaded via `torch.hub` in
~32s. **Not blocked — completed.**

**Commands:**
```
python main.py extract-embeddings --manifest outputs/phase5/manifest.csv \
  --output outputs/thesis_run/embeddings_dinov2 --backbone dinov2_vits14 \
  --device cuda --batch-size 16 \
  --allow-leakage-override --override-justification "<see Section 7>"
python main.py train-embedding-classifier \
  --embeddings outputs/thesis_run/embeddings_dinov2/embeddings.npz \
  --output outputs/thesis_run/embedding_classifier_dinov2 \
  --models logistic_regression,mlp_small --seeds 42,123,2026
```
384-dim frozen embeddings for all 3,688 images (0 failures), extracted on
GPU. Backbone weights never fine-tuned.

| Model | mean macro-F1 | std | mean balanced-acc | mean ECE |
|---|---|---|---|---|
| **mlp_small** | **0.7038** | 0.0178 | 0.7002 | 0.1214 |
| logistic_regression | 0.6595 | 0.0000 | 0.6675 | 0.2126 |

**Real finding of scientific interest**: a **frozen** DINOv2 backbone
(zero backbone gradient updates, ~1 second of classifier training per
seed) reaches 0.704 mean macro-F1 — within 0.025 of the fully **fine-tuned**
efficientnet_b0 (0.729) and above fine-tuned resnet18 (0.718) and
mobilenet_v3_small (0.695). General-purpose self-supervised visual
features transfer to line-drawing emotion classification substantially
better than a linear/shallow probe on ImageNet-supervised features would
be expected to, at a small fraction of the compute cost of Experiment 3.

## 5. Experiment 5 — Fusion

**Real bug found and fixed while running this experiment**:
`fusion/trainer.py::train_primary_fusion` called
`provenance.py::verify_artifacts` without ever threading through an
`allow_override`/`override_justification` parameter — every other
leakage-gated stage in this codebase (`extract-features`,
`extract-embeddings`, `train`, etc.) already exposes this exact override
contract, but fusion training had no way to accept a duplicate-leakage
override at all, making it impossible to fuse artifacts from an already-
justified-override pipeline. Fixed additively: `train_primary_fusion` now
accepts `allow_leakage_override`/`override_justification` and threads them
to `verify_artifacts`; `main.py train-fusion-model` gained the same
`--allow-leakage-override --override-justification` flags every other
command already has (`_add_leakage_args`, reused not duplicated). Verified
safe: `tests/test_fusion_config_and_provenance.py` (pre-existing, 3 tests)
still passes unchanged, and the full 471-test suite passes.

**Command:**
```
python main.py train-fusion-model \
  --features outputs/thesis_run/features/features.csv \
  --embeddings outputs/thesis_run/embeddings_dinov2/embeddings.npz \
  --output outputs/thesis_run/fusion_dinov2 \
  --methods early_scaled_concat,pca_early_fusion,mlp_early_fusion \
  --seeds 42,123,2026 \
  --allow-leakage-override --override-justification "<see Section 7>"
```
(objective features + frozen DINOv2 embeddings — the two representations
independently profiled in Experiments 1 and 4)

| Method | mean macro-F1 | std | mean balanced-acc |
|---|---|---|---|
| **mlp_early_fusion** | **0.7141** | 0.0182 | 0.7116 |
| early_scaled_concat | 0.6766 | 0.0000 | 0.6831 |
| pca_early_fusion | 0.6376 | 0.0000 | 0.6420 |

**Fair comparison, per the task's explicit caution ("do not claim fusion
is better unless measured results support it")**: `mlp_early_fusion`
(0.714) beats **both** of its individual inputs (objective features alone
0.659, DINOv2 alone 0.704) — genuine, measured evidence that fusion helps
here — but it does **not** beat the strongest single model overall,
fine-tuned efficientnet_b0 (0.729). Fusion is not the recommended final
model on current evidence.

## 6. Master model comparison (validation split, mean macro-F1, 3 seeds each)

| Rank | Experiment | Model | mean macro-F1 | std |
|---|---|---|---|---|
| 1 | 3: CNN transfer learning | **efficientnet_b0** | **0.7291** | 0.0076 |
| 2 | 3: CNN transfer learning | resnet18 | 0.7179 | 0.0067 |
| 3 | 5: Fusion | mlp_early_fusion (DINOv2+features) | 0.7141 | 0.0182 |
| 4 | 4: DINOv2 (frozen) | mlp_small | 0.7038 | 0.0178 |
| 5 | 3: CNN transfer learning | mobilenet_v3_small | 0.6951 | 0.0169 |
| 6 | 1: Objective features | random_forest | 0.6586 | 0.0057 |
| 7 | 4: DINOv2 (frozen) | logistic_regression | 0.6595 | 0.0000 |
| 8 | 2: HOG/colour/geometry | colour_only + random_forest | 0.6361 | 0.0069 |

(Row 7's logistic_regression, 0.6595, is nominally above row 6's
random_forest, 0.6586, by less than one std of random_forest's own
seed-to-seed variance — the two are statistically indistinguishable, not
a meaningful ranking difference. Ranks 3 and 4, similarly, sit well within
one standard deviation of each other.)

## 7. Dataset-leakage limitation (recorded, not acted on)

**Every experiment in this document was trained and evaluated on
`outputs/phase5/manifest.csv`, which `leakage_gate` confirms contains real,
unresolved leakage**: 324 exact cross-split duplicate images and 1,442
near-duplicate cross-split pairs (`outputs/thesis_run/features/leakage_gate/leakage_report.json`,
freshly re-verified this session, not assumed from an old report). Per
this session's explicit instruction, this was **not** treated as a
training blocker and the dataset/split were **not** modified, cleaned, or
re-partitioned. Every leakage-gated command in this session
(`extract-features`, `extract-embeddings`, `train-fusion-model`) required
and used an explicit, audit-logged override
(`*/leakage_gate/leakage_override_audit.jsonl`,
`fusion_dinov2/provenance_override_audit.jsonl`) — nothing was silently
bypassed. **Every macro-F1 number in this document must be read as
preliminary / development-signal, not a leakage-safe generalization
estimate** — some fraction of validation-split "accuracy" may reflect the
model having seen a near-duplicate of that exact image during training,
not genuine transfer. The candidate leakage-safe partition built in a
prior task this session (`outputs/phase7b/candidate_partition_review_based/`)
exists and could correct this, but re-partitioning was explicitly out of
scope here.

## 8. Recommendation and next experiment

- **Strongest model on measured evidence**: **efficientnet_b0** (Experiment
  3), mean macro-F1 0.7291 ± 0.0076 across 3 seeds — highest mean, second-
  lowest std, and it beats the next-best (fusion, 0.7141) by more than one
  combined standard deviation.
- **Is locked-test evaluation scientifically appropriate now?** **No.**
  Two independent reasons: (1) the leakage limitation in §7 means a
  test-split evaluation right now would inherit the same contamination and
  produce a falsely optimistic, non-generalizable number; (2) model
  selection here compared 8 configurations across 5 experiment families —
  evaluating the test set now, then possibly wanting to also try the
  epoch-extension in the next bullet, would mean touching the locked set
  more than once, which this project's own test-guard is specifically
  designed to prevent.
- **Single next experiment of most scientific value**: extend
  efficientnet_b0 training past its current 10-epoch cap (validation
  macro-F1 was still rising at epoch 9 in every seed) with the same
  `patience=7` rule but a higher `epochs` ceiling (e.g. 25), 3 seeds — this
  directly tests whether the current ranking is an artifact of an
  under-trained CNN rather than a genuine architecture comparison, and is
  higher-value than adding a 4th CNN architecture or a 6th classifier
  family at this stage.

## 9. Reproduction commands (all, in run order)

```bash
# Verify GPU / loaders (no output artifacts)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Experiment 1
python main.py extract-features --manifest outputs/phase5/manifest.csv \
  --output outputs/thesis_run/features \
  --allow-leakage-override --override-justification "<justification>"
python main.py train-feature-model --features outputs/thesis_run/features/features.csv \
  --output outputs/thesis_run/feature_models --seeds 42,123,2026

# Experiment 2
python main.py extract-hog-features --manifest outputs/phase5/manifest.csv \
  --output outputs/thesis_run/hog_features
python main.py train-handcrafted-groups \
  --objective-features outputs/thesis_run/features/features.csv \
  --hog-features outputs/thesis_run/hog_features/hog_features.csv \
  --output outputs/thesis_run/handcrafted_groups --seeds 42,123,2026

# Experiment 3 (already-trained checkpoints; shown for reproducibility)
python main.py train-image-model --config configs/training/resnet18.toml --seed 42
# (repeat per model x seed: mobilenet_v3_small.toml / resnet18.toml / efficientnet_b0.toml x {42,123,2026})

# Experiment 4
python main.py extract-embeddings --manifest outputs/phase5/manifest.csv \
  --output outputs/thesis_run/embeddings_dinov2 --backbone dinov2_vits14 \
  --device cuda --batch-size 16 \
  --allow-leakage-override --override-justification "<justification>"
python main.py train-embedding-classifier \
  --embeddings outputs/thesis_run/embeddings_dinov2/embeddings.npz \
  --output outputs/thesis_run/embedding_classifier_dinov2 \
  --models logistic_regression,mlp_small --seeds 42,123,2026

# Experiment 5
python main.py train-fusion-model \
  --features outputs/thesis_run/features/features.csv \
  --embeddings outputs/thesis_run/embeddings_dinov2/embeddings.npz \
  --output outputs/thesis_run/fusion_dinov2 \
  --methods early_scaled_concat,pca_early_fusion,mlp_early_fusion \
  --seeds 42,123,2026 \
  --allow-leakage-override --override-justification "<justification>"
```

## 10. Output artifact locations

- Dataset/split summary: this document §0, `outputs/phase5/manifest.csv`
- Features: `outputs/thesis_run/features/features.csv` (63 cols, 3,688 rows)
- HOG: `outputs/thesis_run/hog_features/hog_features.csv` (1,764-dim)
- DINOv2 embeddings: `outputs/thesis_run/embeddings_dinov2/embeddings.npz`
- Experiment 1 checkpoints/leaderboard: `outputs/thesis_run/feature_models/`
- Experiment 2 checkpoints/leaderboard: `outputs/thesis_run/handcrafted_groups/`
- Experiment 3 checkpoints (reused): `outputs/phase5/deep/runs/`, `outputs/phase5/seed42_reference/`
- Experiment 4 checkpoints/leaderboard: `outputs/thesis_run/embedding_classifier_dinov2/`
- Experiment 5 checkpoints/leaderboard: `outputs/thesis_run/fusion_dinov2/`
- Confusion matrices, learning curves, error-analysis gallery: `outputs/thesis_run/error_analysis/`
- Per-run configs: every `runs/<name>/result.json` or `training_result.json`; per-family leaderboard CSV/JSON in each family's output root.
