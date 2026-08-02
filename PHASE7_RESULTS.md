# Phase 7 — Extended Experiment Programme and Model-Family Feasibility Review

**Status: PROPOSAL ONLY. Nothing in this document was executed.** No model
was trained or tuned, the test split was not accessed, the dataset was not
cleaned or modified, no detector was implemented, no psychological rule was
activated, and the default model was not changed. Every number attributed
to a *proposed* experiment in this document is an **estimate**, clearly
labeled as such, extrapolated from real Phase 3A–6 measurements or from
one-time, training-free inspection commands (parameter counts, `pip list`,
disk/GPU queries) run during this feasibility review — never a training
run. Every number attributed to Phase 3A–6 itself is the real, already-
reported result from `PHASE3A_RESULTS.md`–`PHASE6_RESULTS.md`, not
re-derived here.

This phase does not renumber or reopen Phase 1–6. It proposes a roadmap of
**future, unstarted** phases; nothing here should be read as a claim that
any listed experiment has occurred.

---

## 1. Repository and feasibility inspection

### 1.1 Dataset — exact structure, not estimated

From `outputs/phase5/manifest.csv` (byte-identical to Phase 3A/4's manifest,
per `PHASE5_RESULTS.md` §1) and `outputs/phase5/deep/leakage_gate/leakage_report.json`:

| Split | Angry | Fear | Happy | Sad | Total |
|---|---|---|---|---|---|
| train | 641 | 502 | 921 | 757 | **2,821** |
| valid | 77 | 45 | 110 | 78 | **310** |
| test | 123 | 130 | 168 | 136 | **557** |
| **Total** | | | | | **3,688** |

Fear is the smallest class in every split — consistent with the persistent
Fear→Angry confusion documented in every phase from 3A through 6.

**Leakage/duplication reality** (from the leakage gate's own report, real
counts, not the previously-cited round "41%" restated without a source):

| Category | Count |
|---|---|
| Exact cross-split duplicate groups | 324 |
| Near-duplicate cross-split groups (perceptual-hash threshold 5) | 1,442 |
| Conflicting-label groups | 48 |
| **Total flagged image IDs** | **1,517 / 3,688 (41.1%)** |
| Clean (non-flagged) rows | 2,171 |
| Subject/child-level grouping available | **No** (`subject_grouping_available: false`) |

The 41% figure already used in `SESSION_HANDOFF.md` is now traced to this
exact report rather than restated from memory. The absence of subject-level
grouping is a real, structural limitation: even a perfect duplicate-image
removal cannot guarantee the *same child* never appears in both train and a
future test set, because no child/subject identifier exists in this dataset
at all. Any leakage-safe partition this project builds (§7) can only be
**image-level** leakage-safe, not **subject-level** leakage-safe, and must
say so explicitly rather than imply a stronger guarantee than the data
supports.

### 1.2 Hardware — measured, not assumed

- GPU: Quadro P3200, confirmed via `torch.cuda.get_device_properties` — **6.44 GB total VRAM**, matching the batch-4/grad-accum-4 protocol already in use since Phase 3A.
- Disk: **47 GB free** on the working volume (`df -h`, measured this phase). Current cumulative footprint of `outputs/phase3a`–`outputs/phase6` is **≈1.65 GB** (45M + 393M + 1.2G + 8.5M, measured via `du -sh`). Storage is not currently tight, but is budgeted explicitly per experiment family in §5 rather than assumed unlimited, because several proposed architectures (ConvNeXt-Tiny, ViT-B/16) have checkpoint sizes an order of magnitude larger than anything trained so far (§1.6).

### 1.3 Installed libraries — checked, not assumed

`pip list` inside the project's own `.venv`, checked this phase:

| Package | Version | Relevance |
|---|---|---|
| `torch` | 2.7.1+cu118 | Already the training backend for Phase 3A–6 |
| `torchvision` | 0.22.1+cu118 | Already the backbone source for all 8 registered architectures |
| `timm` | 1.0.28 | **Already installed, currently unused anywhere in `src/doar`.** Provides DenseNet, ConvNeXt, and (critically) DINOv2 ViT variants — see §1.6 |
| `open_clip_torch` | 2.32.0 | Already installed and already wired into `embeddings.py` (`openclip:` backbone prefix) |
| `scikit-learn` | 1.9.0 | Already the backend for all classical models (`experiments.py`, `fusion/trainer.py`) — includes `LogisticRegression`, `SVC`, `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, `MLPClassifier`, `PCA`, all already imported somewhere in the codebase |
| `opencv-python-headless` | 5.0.0.93 | Already installed; provides `cv2.HOGDescriptor` as a no-new-dependency HOG path (§2, Family C) |
| `Pillow` | 11.3.0 | Already the image I/O backend throughout |

**Not installed:** `scikit-image` (an alternative, more common HOG implementation — small optional add), `xgboost`, `lightgbm` (not needed — `HistGradientBoostingClassifier` already covers the "tree-based models" requirement without a new dependency).

**Pretrained-weight licensing** — stated with appropriate hedging, not
fabricated: torchvision's ImageNet weights (used by all 8 registered
architectures, DenseNet121, and ConvNeXt-Tiny) are distributed under
permissive terms consistent with the BSD-style torchvision license and are
already in production use in this project since Phase 3A. DINOv2's model
weights, per Meta's public repository, are released under Apache 2.0.
OpenCLIP's LAION-trained weights are released under permissive open
licenses (varies by exact checkpoint). **These license terms were not
independently re-verified from source during this feasibility review** —
before any external distribution or publication of a model derived from
DINOv2/OpenCLIP/timm weights, the exact current license text for the
specific checkpoint used should be checked, not assumed from this summary.
For internal thesis research use, all of the above are consistent with how
this project already uses torchvision's own pretrained weights.

### 1.4 Existing training/evaluation infrastructure — already extensive

`main.py` exposes 31 commands today, including several **already built and
never used in Phase 3A–6**: `run-ablation`, `explain-gradcam`,
`explain-features`, `review-agreement`, `check-training-readiness`,
`validate-dataset`, `generate-thesis-outputs`, `compare-embeddings`,
`train-fusion-model`, `calibrate-fusion`, `train-late-fusion`,
`generate-oof-probabilities`. This is a mature, already-tested research
framework, not a thin CLI wrapper — **Phase 7 finds no justification for a
second training framework anywhere in this review.**

Concretely reusable without modification:

| Capability | Existing entry point |
|---|---|
| Multi-seed deep-CNN training/comparison | `main.py compare-deep-models` → `deep/compare.py` (used unmodified in Phase 4/5) |
| Single-model deep training | `main.py train-image-model` → `deep/trainers.py::train_image_model()` |
| Deep-checkpoint calibration | `main.py calibrate` → `deep/calibration.py` (used in Phase 5) |
| Classical-feature model training/comparison | `main.py compare-models` → `experiments.py::run_feature_experiment()` (6 model families already implemented: `logistic_regression`, `linear_svm`, `rbf_svm`, `random_forest`, `extra_trees`, `hist_gradient_boosting` — **never yet run under the current manifest**) |
| Feature-family ablation | `main.py run-ablation` → `ablation.py::run_feature_ablation()` (already implemented, never yet run under the current manifest) |
| Frozen-embedding extraction (DINOv2, OpenCLIP, torchvision backbones, or a fine-tuned checkpoint's penultimate layer) | `main.py extract-embeddings` → `deep/embeddings.py` (DINOv2 support already coded via `torch.hub`, **never yet exercised**) |
| Early fusion (features + embeddings) | `main.py train-fusion-model` → `fusion/trainer.py::train_primary_fusion()` |
| Late fusion (probability-level) | `main.py train-late-fusion` / `apply-late-fusion` → `fusion/late.py`, `fusion/probability.py` (3 methods already implemented: equal, validation-weighted, logistic-meta) |
| Out-of-fold stacking support | `main.py generate-oof-probabilities` → `fusion/oof.py` |
| Locked-test-split guard | `src/doar/test_guard.py::require_test_access()` (already gates `evaluate`, `evaluate-predictions`, `apply-late-fusion --split test`) |

### 1.5 Existing model checkpoints and results (reference, not re-derived)

| Source | Architectures with real results | Seeds |
|---|---|---|
| Phase 3A | `mobilenet_v3_small` | 42 |
| Phase 4 | `small_cnn`, `mobilenet_v3_small`, `resnet18`, `efficientnet_b0` | 42 |
| Phase 5 | `mobilenet_v3_small`, `resnet18`, `efficientnet_b0` | 42, 123, 2026 |
| Phase 6 | (integration test only, no new training) `efficientnet_b0` seed 42 vs. Phase 3A `mobilenet_v3_small` | n/a |

**8 architectures are registered in `src/doar/deep/registry.py`; only 4 have
ever been trained** (`small_cnn`, `mobilenet_v3_small`, `resnet18`,
`efficientnet_b0`). `resnet50`, `mobilenet_v3_large`, `convnext_tiny`, and
`vit_b_16` are fully wired (build, freeze/unfreeze, embedding extraction for
3 of the 4) but have **never been run once**, in any phase.

### 1.6 Real parameter counts and extrapolated checkpoint sizes

Measured this phase via `torchvision.models`/`timm` model construction
(random init, no download, no training):

| Architecture | Params | Extrapolated checkpoint size* |
|---|---|---|
| `mobilenet_v3_small` (Phase 3–6 real) | 1,521,956 | 18.5 MB (real) |
| `resnet18` (Phase 4–6 real) | 11,178,564 | 134.3 MB (real) |
| `efficientnet_b0` (Phase 4–6 real) | 4,012,672 | 48.6 MB (real) |
| **DenseNet121 (proposed, new)** | 6,957,956 | ≈84 MB |
| **ConvNeXt-Tiny (proposed, already registered)** | 27,823,204 | ≈337 MB |
| **ViT-B/16 (proposed, already registered)** | 85,801,732 | ≈1.04 GB |
| **DeiT-Tiny/16, compact ViT via `timm` (proposed, new)** | 5,525,188 | ≈67 MB |
| **ViT-Small/16 or DeiT-Small (compact ViT alternative)** | 21,667,204 | ≈262 MB |

*Extrapolated using **12.1 bytes/parameter**, the empirical ratio derived
from all 3 real Phase 5 checkpoints (`best.pt` size ÷ `parameter_count`,
consistently 12.01–12.15 across mobilenet/resnet18/efficientnet_b0 — this
reflects fp32 weights [4 bytes] + AdamW's 2 momentum buffers [8 bytes],
12 bytes/param exactly, confirming the checkpoint includes full optimizer
state). This is a real, derived-from-data extrapolation factor, not a
guess — but it is still an *estimate* for architectures never actually
checkpointed.

**`timm` already exposes 8 DINOv2 ViT variants** (`vit_small_patch14_dinov2`
through `vit_giant_patch14_dinov2`), confirmed by listing installed models
this phase — the smallest (`vit_small_patch14_dinov2`, embedding dim 384)
is the natural choice for a 6 GB GPU and a ~3,700-image dataset.

### 1.7 Existing objective DOAR features — already extensive

`src/doar/features.py::objective_feature_row()` already computes **~50
features** across 5 families used throughout Phase 1–6's rule-evidence
pipeline: `quality.*` (11 features: blur, contrast, brightness, entropy,
noise proxy, etc.), `segmentation.*` (9), `composition.*` (11, including
centroid, margins, quadrant balance, symmetry), `colour.*` (11), `stroke.*`
(5, including a **crude gradient-magnitude edge-density proxy**, not a true
oriented-gradient histogram), and `shape.*` (3, of which 2 are structurally
`UNAVAILABLE` — no shape detector exists, correctly surfaced as missing per
this project's D4 "never fake a missing feature" convention, not silently
zero-filled).

**Gap identified**: no true HOG (Histogram of Oriented Gradients) feature
exists. `stroke.edge_density` is a simple gradient-magnitude threshold, not
an orientation histogram — a real, addressable gap for Family C (§2.3).

### 1.8 What can reuse existing infrastructure vs. what needs new components

| Needs | Status |
|---|---|
| DenseNet121 training | **Zero new components** — one `registry.py` entry, same pattern as the 7 existing architectures |
| ConvNeXt-Tiny training | **Zero new components** — already fully wired, simply never run |
| ViT-B/16 (existing, frozen probe) | **Zero new components** — already fully wired, simply never run |
| Compact ViT (DeiT-Tiny/Small via `timm`) | **New, small**: `registry.py` currently only wraps `torchvision`; a `timm`-backed model-loading path is genuinely new code (§2.1, Family A) |
| DINOv2 frozen embeddings | **Near-zero** — extraction already coded (`embeddings.py`); a *pure* frozen-embedding classifier (no objective-feature fusion) may need a small extension since `train_primary_fusion` currently always expects both a features CSV and an embeddings NPZ — verify or add an embeddings-only mode |
| OpenCLIP frozen embeddings | **Zero new components** — already coded and already an installed dependency |
| Classical-feature baselines (logistic/SVM/tree) | **Zero new components** — `compare-models` already implements all 6 model families; simply never run under the current manifest |
| HOG features | **New, small** — a new extractor function in `features.py` (or a sibling module), using either `scikit-image` (new dependency) or already-installed OpenCV's `cv2.HOGDescriptor` (no new dependency, preferred) |
| Feature-family ablation | **Zero new components** — `run-ablation` already implemented, never run under the current manifest |
| Early fusion (embeddings + features) | **Zero new components**, with a possible small extension for dimensional-imbalance control (PCA is already imported in `fusion/trainer.py`'s dependency set — verify it's wired into the actual fusion path used, or wire it in) |
| Late fusion | **Zero new components** — 3 methods already implemented |
| Freeze-depth ablation | **Zero new components** — `freeze_epochs` already a `train_image_model()` parameter |
| Augmentation-strength ablation (moderate/strong vs. conservative) | **Zero new components** — all 3 profiles already implemented |
| Augmentation on/off ablation | **New, trivial** — `augmentations.py` has no `"none"` profile; adding one is a ~4-line change (empty ops list) |
| Focal loss | **New, small** — `trainers.py` currently hardcodes `CrossEntropyLoss`; a `FocalLoss` class plus a `loss_name` parameter is a well-scoped addition following the existing `optimizer_name`/`scheduler_name` parameterization pattern |

**No experiment in this proposal requires a second training framework, a
new CLI tool outside `main.py`, or a new persistence format.** Every new
component is a small, targeted extension of an existing module.

### 1.9 Architecture-specific protocol exceptions (already known, restated for completeness)

Per `PHASE4_RESULTS.md` §3 (already-established project language, not
reopened here): `small_cnn` has no pretrained backbone and therefore never
freezes (`registry.py`'s `freeze_backbone()` is a no-op for it by design).
This exception is unchanged and applies identically to any future
`small_cnn` work. All newly proposed architectures (DenseNet121,
ConvNeXt-Tiny, ViT-B/16, compact ViT) **do** have torchvision/timm-pretrained
backbones and follow the standard freeze/unfreeze path like
`mobilenet_v3_small`/`resnet18`/`efficientnet_b0` already do — no new
architecture-specific exception is anticipated beyond what §2's ablations
(E1) deliberately test.

### 1.10 Verification (this phase produced no code change)

`git status` at the end of this phase shows only `PHASE7_RESULTS.md` and
`PHASE7_EXPERIMENT_MATRIX.csv` as new files — no file under `src/doar/` or
`tests/` was touched. Run anyway, for confirmation: `pytest tests/` — 226
passed, 9 pre-existing warnings, 0 failures (identical to Phase 5/6).
`python -m compileall src main.py` — exit 0. `ruff check .` — 792 findings,
exit 1, identical count to Phase 5/6, confirming zero new findings from
this phase (consistent with zero source changes).

---

## 2. Research programme

Full per-experiment specification (all 20+ required fields) for all 22
proposed experiments is in **`PHASE7_EXPERIMENT_MATRIX.csv`** (deliverable
#2). This section gives the organizing research questions, hypotheses, and
Recommended/Optional/Not-justified rankings per the user's requested
structure; it does not repeat the full CSV content.

### 2.1 Family A — Deep image classifiers

| Candidate | Rank | Why |
|---|---|---|
| `mobilenet_v3_small`, `resnet18`, `efficientnet_b0` (existing) | **Already validated** — not re-proposed; continue as the Phase 5 baseline | See §1.5 |
| **DenseNet121** | **Recommended** | Zero new engineering (§1.8); a genuinely different inductive bias (dense connectivity) not represented by the existing 3; mid-size (7.0M params) |
| **ConvNeXt-Tiny** | **Recommended** | Zero new engineering — already registered, never run; represents the "modernized ConvNet" design family distinct from the existing 3 |
| **Compact ViT (DeiT-Tiny/16, via `timm`)** | **Optional** — single-seed frozen-backbone feasibility probe first (A3), multi-seed only if it clears an explicit advancement rule (A4) | Small new integration; dataset size (2,821 train images) is a real risk factor for transformers, so this is framed as a feasibility test, not an expected win — **do not assume it will improve on the CNNs** |
| **ViT-B/16 (existing, 86M params)** | **Optional**, capped at a single frozen-head probe, never multi-seed | Zero new engineering, but its scale (≈1.04 GB/checkpoint, §1.6) is disproportionate to this dataset; run once for the record, not pursued further regardless of outcome |

This directly follows the instruction to not assume newer/larger/
transformer-based models will help: three of five rows are explicitly
capped at a single non-competitive probe, and the compact-ViT campaign
(A4) is gated behind an advancement rule rather than run by default.

### 2.2 Family B — Pretrained self-supervised feature representations

**Research question:** does most of this task's linearly-separable signal
already exist in a general-purpose pretrained representation, without any
drawing-specific fine-tuning?

- **DINOv2 frozen embeddings + logistic regression (B1)** and **+ RBF SVM
  (B2)**: **Recommended** — both reuse already-coded extraction, near-zero
  marginal engineering cost beyond a possible embeddings-only classifier
  path, and directly answer a central scientific question of this
  programme.
- **+ shallow MLP (B3)**: **Optional** — completes the
  linear/kernel/neural progression cheaply, but flagged as the most
  overfitting-prone of the three given ~2,821 training rows vs. 384
  embedding dimensions.
- **OpenCLIP embeddings as a second SSL family (B4)**: **Optional**, offered
  as a bonus beyond the user's explicit request because it costs nothing —
  the dependency and extraction path already exist.

**Scientific value vs. adding another end-to-end CNN**: a frozen-embedding
classifier answers a *different* question than another fine-tuned CNN
(Family A) — it isolates how much signal is already present in a
general-purpose representation versus how much end-to-end domain adaptation
adds. If B1/B2 come within a small, resolvable margin of the best
fine-tuned CNN, that is direct evidence that fine-tuning's contribution is
modest for this task — a genuinely different, complementary finding to
"which CNN architecture is best," not a redundant screening run. If they
underperform substantially, that is equally informative: it argues
fine-tuning (or fusion, Family D) is necessary. Either outcome is reported
(governance rule 6, §4).

### 2.3 Family C — Interpretable feature baselines

All of logistic regression (C1), SVM (C2), and tree-based models (C3) are
**Recommended** — this entire family costs almost nothing (CPU-only, all 6
classical model types already implemented in `experiments.py`) and is the
project's **required interpretable baseline**, never yet run as a
controlled multi-seed comparison under the current manifest despite the
infrastructure existing since before Phase 3A. Adding real HOG features
(C4) is **Recommended** as a small, well-scoped, genuinely new feature
family that directly tests whether a textbook interpretable descriptor adds
value beyond DOAR's existing bespoke features. Feature-family ablation (C5,
reusing the already-built `run-ablation`) is **Recommended** as a
free, high-interpretability diagnostic.

All normalization/imputation/scaling in this family is fitted on the
training split only, inside `sklearn.Pipeline` objects, exactly as already
implemented in `experiments.py::_model()` — this is not a new leakage
control, it is confirmation that the existing implementation already
satisfies the "training data only" requirement.

### 2.4 Family D — Fusion experiments

All four fusion comparisons (D1: 3-way alone/alone/fused baseline; D2:
DINOv2+features early fusion; D3: fine-tuned-CNN-embeddings+features early
fusion; D4: late fusion of class probabilities+feature-classifier
probabilities) are **Recommended** — the entire early- and late-fusion
machinery already exists (`fusion/trainer.py`, `fusion/late.py`,
`fusion/probability.py`) and has never been exercised on this dataset.

**What is fused, at what stage, explicitly:**
- **D2/D3 (early fusion):** the model's *learned representation*
  (embedding vector, either generic-DINOv2 or task-fine-tuned-CNN) is
  concatenated with the ~50-dimensional objective-feature vector *before*
  a single classifier is fit on the combined vector.
- **D4 (late fusion):** two already-independently-trained models' *output
  class-probability distributions* are combined *after* both have made
  their own independent prediction — no joint representation is learned.

**Dimensional-imbalance control (explicit, not assumed handled):** DINOv2
embeddings (384-dim) or CNN penultimate embeddings (512–1,280-dim depending
on architecture) vastly outnumber the ~50 objective features. Left
unaddressed, a naive concatenation would let the embedding block dominate
any distance- or regularization-based classifier purely by dimension count,
not genuine signal. The matrix (D2/D3 rows) specifies PCA-based
dimensionality reduction of the embedding block (using `sklearn.PCA`,
already imported in `fusion/trainer.py`'s own dependency set) to a
comparable order of magnitude before fusion, with the reduced dimensionality
itself validation-selected from a small grid — not left to a fusion method
to silently absorb.

**Overfitting control:** all fusion classifiers are the same
`Pipeline(SimpleImputer, StandardScaler, <classifier>)` pattern already used
throughout `experiments.py` and `fusion/trainer.py`, fit on train only,
selected on validation only, with the same 3-seed evaluation discipline as
every other family.

### 2.5 Family E — Fine-tuning and training ablations

- **Freeze-depth sweep (E1: frozen / partial [current default] / full
  fine-tune)**: **Recommended** — tests a previously-unquestioned protocol
  default using an already-existing parameter (`freeze_epochs`).
- **Focal loss vs. class-weighted cross-entropy (E2)**: **Recommended** —
  directly targets the one persistent, already-documented weakness visible
  in every architecture across Phase 3A–6 (Fear→Angry confusion, Fear being
  the smallest class in every split). Requires a small, new `FocalLoss`
  implementation.
- **Augmentation on vs. off (E3)**: **Recommended** — a foundational
  comparison never actually run (only different augmentation *strengths*
  have ever been available, never "none"). Requires a trivial addition to
  `augmentations.py`.
- **Augmentation strength sweep beyond E3 (E4)**: **Optional, explicitly
  conditional** on E3 first showing augmentation helps at all — this is the
  concrete mechanism preventing "post-hoc expansion of the model list merely
  because results are disappointing" (governance rule 8, §4): if E3 shows
  no benefit, E4 does not run.

**Explicitly not assumed:** none of fine-tuning depth, focal loss,
augmentation, or fusion is assumed to help. Every experiment's hypothesis
(full detail in the CSV) states a falsifiable, two-sided expectation, and
every stopping rule requires retaining and reporting a negative result
rather than silently dropping it.

---

## 3. Complete experiment matrix

See **`PHASE7_EXPERIMENT_MATRIX.csv`** — 22 experiments (5 in Family A, 4 in
Family B, 5 in Family C, 4 in Family D, 4 in Family E), each with all 20+
required fields: experiment ID, family, research question, falsifiable
hypothesis, independent/primary/secondary dependent variables, baseline,
controlled variables, dataset split and leakage controls, preprocessing/
architecture exceptions, primary and secondary metrics, hyperparameter
search space, tuning budget, seed count, estimated compute time/memory/
storage, expected thesis value, advancement rule, stopping/exclusion
criterion, main confounders/limitations, required implementation changes,
and integration difficulty.

---

## 4. Experimental governance — how this proposal enforces the 10 required rules

1. **Validation-only selection**: every experiment's `primary_metric` is
   computed on the validation split; the CSV's `dataset_split_and_leakage_controls`
   column repeats this constraint on every row. No experiment in this
   matrix touches the test split.
2. **One final locked evaluation**: proposed in §7 below, explicitly
   deferred until the shortlist (§5) and analysis plan are frozen — this
   proposal does not run it, only specifies the protocol.
3. **Equal/normalized tuning budgets**: every family-A/deep-classifier
   screening experiment uses the *identical* zero-HP-search protocol Phase
   4/5 already used (no per-architecture tuning advantage); every
   classical-model experiment (Family C) reuses `experiments.py`'s existing
   fixed hyperparameters (no per-model tuning advantage); fusion and
   embedding-classifier experiments (Family B/D) use small, explicitly
   equal-sized grids across competing configurations (documented per-row in
   the CSV's `hyperparameter_search_space`/`tuning_budget` columns).
4. **Identical splits/evaluation code wherever valid**: every experiment
   reuses the existing `outputs/*/manifest.csv`-derived splits and the
   existing `export-probabilities`→`evaluate-predictions` evaluation code
   path (deep models) or the existing `compute_metrics`-based path
   (classical/fusion models) — no new evaluation code is proposed.
5. **Multi-seed for shortlisted systems**: every experiment defaults to 3
   seeds (42/123/2026, the project's own established default set); single-
   seed runs are explicitly reserved for feasibility *probes* (A3, A5) that
   must clear a stated advancement rule before any multi-seed campaign.
6. **Preserve negative results**: every experiment's `stopping_exclusion_criterion`
   column explicitly states what happens on a null/negative result — always
   "retain and report," never "delete and don't mention." This is stated as
   a hard requirement in the per-experiment specs, not left implicit.
7. **Configuration/seed/hardware/runtime/parameter-count/checkpoint/hash
   tracking**: every experiment reuses the existing `training_result.json` +
   `executed_config.json` + SHA-256 artifact-manifest pattern already used
   in Phase 3A–6 (§1.4) — no new tracking mechanism is proposed.
8. **No post-hoc model-list expansion on disappointing results**: enforced
   concretely by the conditional experiments (A4 conditional on A3, E4
   conditional on E3) — their advancement rules are the literal mechanism
   preventing scope creep, not just a stated principle.
9. **No statistical-superiority claims without appropriate evidence**: every
   experiment's hypothesis is written as a falsifiable, margin-based
   comparison (e.g. "exceeds X by more than the combined seed-to-seed std"),
   directly carrying forward the correction already made in
   `PHASE5_RESULTS.md`'s correction commit (`d0c5ca2`) — no experiment in
   this matrix is designed to declare a winner from an unresolvable gap.
10. **Separation of technical vs. psychological/clinical validity**: every
    experiment in this matrix measures classification performance
    (macro-F1, calibration, etc.) only. None proposes activating any
    psychological rule, concern, or clinical claim — that remains
    out of scope per this phase's explicit constraints (see Status, above,
    and §9).

### Handling the existing duplicate-contaminated splits

The current `train`/`valid`/`test` split (§1.1) is **usable for all of
Family A–E's *development* work** (architecture screening, ablations,
fusion method comparison) because:
- Every comparison in this matrix is *relative* (architecture A vs.
  architecture B, fusion vs. no fusion, loss X vs. loss Y) under the *same*
  contaminated split — the contamination affects all arms of any given
  comparison equally, so *relative* rankings from this data remain a
  reasonable basis for **shortlisting**, even though *absolute* metric
  values are inflated and not trustworthy as final performance claims.
- This is the same logic already used, correctly, throughout Phase 3A–6
  (each time labeled "preliminary, not leakage-safe").

It **cannot** support any final thesis performance claim, because
duplicate/near-duplicate images crossing train→test artificially inflate
apparent generalization (the model may have memorized a near-identical
training image). This is why governance rule 2 requires a **separate,
newly-constructed, leakage-safe test set** before any number from this
programme is reported as a final result (§7).

---

## 5. Dependency-aware execution order, shortlist, and rejected/deferred items

### 5.1 Execution order (dependency graph, not a calendar)

```
Stage 0 (no dependencies, can run in parallel):
  C1, C2, C3, C5   -- classical baselines + ablation (CPU-only, already-built)
  A1, A2           -- DenseNet121, ConvNeXt-Tiny screening (new/zero-cost registry additions)
  B1, B2           -- DINOv2 + logistic/SVM (embedding extraction is the only shared prerequisite)
  A5               -- ViT-B/16 single frozen probe (already-built, zero cost, run once)

Stage 1 (depends on Stage 0 artifacts):
  C4               -- HOG features (needs new feature code; independent of C1-C3/C5's results but reuses the same extract-features pass)
  B3, B4           -- MLP on DINOv2 embeddings; OpenCLIP embeddings (reuse B1's embeddings.npz / add a second one)
  A3               -- compact ViT frozen probe (independent, but sequenced after A1/A2 so Family A's shortlist winner is known before Family D needs it)
  E1, E2, E3       -- fine-tuning ablations on the Family A shortlist winner (needs A1/A2/A5 -- and the existing Phase4/5 efficientnet_b0 result -- to know which architecture to ablate)

Stage 2 (depends on Stage 0/1 shortlist decisions):
  A4               -- compact ViT multi-seed (ONLY if A3's advancement rule fires)
  E4               -- augmentation-strength sweep (ONLY if E3's advancement rule fires)
  D2, D3           -- early fusion (needs B1's DINOv2 embeddings AND the Family A shortlist winner's penultimate embeddings)
  D4               -- late fusion (needs the Family A shortlist winner's probabilities AND C1-C3's best classical model's probabilities)

Stage 3 (needs all prior stages' shortlist candidates):
  D1               -- the 3-way alone/alone/fused summary comparison
  Shortlist freeze -- see §5.2
```

### 5.2 Concise shortlist of recommended experiments (in priority order)

1. **C1–C3, C5** (classical baselines + ablation) — highest thesis value
   per unit of compute (CPU-only, minutes total), and a required
   interpretability baseline this project does not yet have under
   controlled conditions.
2. **A1, A2** (DenseNet121, ConvNeXt-Tiny screening) — highest thesis value
   per unit of engineering effort (zero-to-trivial new code).
3. **B1, B2** (DINOv2 + linear/kernel classifiers) — directly answers
   whether fine-tuning is necessary at all, a central open question.
4. **E1, E2, E3** (freeze-depth, focal loss, augmentation on/off) — targets
   real, already-observed weaknesses (Fear confusion) and untested protocol
   defaults, on the existing shortlist winner only (no combinatorial
   explosion across all architectures).
5. **D1–D4** (fusion) — the most direct test of whether combining sources
   helps, gated behind Stage 0/1 producing the components to fuse.

### 5.3 Explicitly rejected or deferred, with reasons

| Item | Status | Reason |
|---|---|---|
| Full multi-seed campaign for `vit_b_16` (86M params, full fine-tune) | **Rejected outright** | Storage cost alone (≈1.04 GB × 3 seeds ≈ 3.1 GB for one architecture) is disproportionate to a dataset this small; A5 caps it at one frozen-head probe, never revisited regardless of outcome |
| A second/third compact-ViT variant beyond DeiT-Tiny | **Deferred**, conditional on A3/A4 | Adding transformer variants before the first one clears its advancement rule would be exactly the "assume transformers help" pattern the user explicitly prohibited |
| `xgboost`/`lightgbm` gradient boosting | **Rejected** | `HistGradientBoostingClassifier` (already installed via scikit-learn, already implemented in `experiments.py`) already covers the "tree-based models" requirement without adding a new dependency; no stated research question needs a different boosting library |
| A brand-new training/evaluation framework | **Rejected outright** | §1.4/§1.8 found no experiment in this programme that the existing `main.py` + `src/doar/deep`/`fusion`/`experiments.py` infrastructure cannot run with at most a small, targeted extension |
| Deeper/wider MLP variants beyond B3's single shallow probe | **Deferred**, conditional on B3 showing a resolvable improvement over B1/B2 | Same "no speculative complexity expansion" principle as E4/A4 |
| Gabor filters / Local Binary Patterns / other hand-crafted texture families beyond HOG (C4) | **Deferred**, conditional on C4 showing HOG itself adds value | Adding a second speculative hand-crafted feature family before the first is validated repeats the same scope-creep risk |
| Immediate leakage-safe dataset reconstruction | **Deferred to its own future phase** (§9) — not part of this experiment programme | The dataset audit (paused since Phase 3A, `SESSION_HANDOFF.md` §8) is a separate, larger undertaking than any single experiment family here; §7 proposes its protocol but does not execute it |

---

## 6. Estimated total and per-stage compute/storage costs

All figures are **estimates**; real, extrapolated, or CPU-negligible per
the CSV's own per-row sourcing (§1.6's 12.1 bytes/param factor for storage;
Phase 4/5's own measured per-seed training times for same-order-of-magnitude
architectures for compute). None of these numbers were produced by running
anything this phase.

| Stage | Rows included | Est. GPU/CPU time | Est. new storage |
|---|---|---|---|
| Stage 0 | C1,C2,C3,C5,A1,A2,B1,B2,A5 | ≈2–2.5 hours (dominated by A1/A2/A5's GPU training; C/B rows are CPU-minutes each) | ≈2.6 GB (mostly A2's ConvNeXt-Tiny 3-seed checkpoints ≈2.0 GB + A1's DenseNet121 ≈0.5 GB + A5's single ViT-B/16 checkpoint ≈1.0 GB — **already exceeds the other 6 rows combined**) |
| Stage 1 | C4,B3,B4,A3,E1,E2,E3 | ≈3–4 hours (dominated by E1/E2/E3's 6–9 training runs each on the shortlist winner) | ≈1–1.5 GB (E1/E2/E3's new checkpoint configs; C4/B3/B4 are CPU-negligible) |
| Stage 2 (conditional) | A4,E4,D2,D3,D4 | 0 to ≈2 hours, entirely conditional on Stage 0/1 advancement rules firing | 0 to ≈1.5 GB, same conditionality |
| Stage 3 | D1 | Negligible (reuses prior outputs, no new training) | Negligible |
| **Total (worst case, all conditional experiments fire)** | | **≈8–9 hours GPU-inclusive wall-clock** | **≈5.5–6 GB** |
| **Total (expected case, some conditional experiments do not fire)** | | **≈5–6 hours** | **≈4 GB** |

Against **47 GB free disk** (§1.2), even the worst case uses roughly
12% of current headroom — not a blocking constraint, but large enough
relative to Phase 5's own 1.2 GB footprint that a **checkpoint-retention
policy** should be decided before Stage 0 begins (e.g., keep `best.pt` for
every seed but delete `last.pt` for non-shortlisted architectures once
`training_result.json` is safely recorded) rather than assumed unlimited.
This is a recommendation for the next session to decide, not a decision
made here.

---

## 7. Proposed locked final-evaluation protocol (proposal only, not executed)

This section specifies **how** a genuinely leakage-safe final evaluation
should eventually be built — it does not build it. Building it is a
future phase (§9), not part of this experiment programme.

1. **Duplicate-group resolution using existing infrastructure**:
   `src/doar/leakage.py::assess_leakage()` and `resolve_leakage()` already
   compute exact (`exact_cross_split_leakage`) and near-duplicate
   (`near_cross_split_leakage`, perceptual-hash threshold 5) groups, and
   already materialize a clean/quarantine split (`materialize_clean_dataset()`).
   The paused dataset audit's own scope (`SESSION_HANDOFF.md` §8, items
   1–6) already calls for exactly this — Phase 7 does not duplicate that
   scope, it depends on it.
2. **Group-level (not image-level) partitioning**: each of the 324 exact +
   1,442 near-duplicate groups must be assigned to exactly **one** of
   train/valid/test as a whole group, never split across two — this is
   already how `assess_leakage()` detects violations (`exact_cross_split_leakage`/
   `near_cross_split_leakage` entries **are** cross-split group violations);
   the fix is to re-partition by group membership, not by individual image,
   before any final split is drawn.
3. **No subject-level guarantee available** (§1.1) — this must be stated as
   an explicit, permanent limitation of any leakage-safe set built from
   this dataset, not silently omitted. If subject/child identifiers become
   available in a future data-collection round, subject-level grouping
   should be added retroactively; until then, "leakage-safe" in this
   project means **image-level** leakage-safe only.
4. **New, disjoint from all prior exposure**: the eventual locked test set
   must exclude not only cross-split duplicate groups but also the 4
   specific images already disclosed as non-blind in `PHASE6_RESULTS.md`
   §2 (`2-1_jpg.rf...`, `Fear_1_10_jpg.rf...`, `Happy_1_19_jpg.rf...`,
   `3-4_jpg.rf...`) — already required in `SESSION_HANDOFF.md` §8 item 7.
5. **Frozen shortlist before unlocking**: per governance rule 2 (§4), the
   final shortlist (architecture, fusion method, calibration choice) must
   be frozen — based entirely on Families A–E's validation-split results
   under the *current*, contaminated split — **before** the newly-built
   locked test set is unlocked even once, using the existing
   `--unlock-test --confirm-final-evaluation` guard
   (`src/doar/test_guard.py`) exactly as already implemented.
6. **One evaluation, one report**: the locked set is evaluated exactly
   once per shortlisted candidate, with the result reported regardless of
   outcome — no re-unlocking to "try again" after seeing a disappointing
   number, consistent with governance rules 8/9.
7. **Class balance verification**: report class balance before/after the
   new partition (already part of the paused audit's scope, item 4) to
   confirm the group-level re-partitioning did not introduce a new,
   unintended imbalance beyond what §1.1 already documents.

---

## 8. Risks, confounders, and limitations

- **No subject-level leakage control exists or is achievable with the
  current metadata** (§1.1, §7 item 3) — a hard, structural limitation, not a
  process gap that more effort resolves.
- **Small dataset relative to several proposed architectures**:
  2,821 training images is modest for ConvNeXt-Tiny (27.8M params) and
  especially for any ViT variant — overfitting risk is real and is why
  Family A's larger/transformer candidates are capped at probes or
  explicitly conditional escalation, not run at full multi-seed scale by
  default.
- **Domain gap for pretrained SSL embeddings**: DINOv2/OpenCLIP were
  pretrained on natural photographs, not children's drawings — a lower
  Family B result does not mean the embeddings are "bad," it means the
  domain gap is real; this must be stated in any Family B write-up, not
  treated as a simple win/loss.
- **Internet/hub access not yet verified**: both DINOv2 (`torch.hub`) and
  the compact-ViT `timm` path require a one-time download from an external
  host on first use. This has **not been tested from this environment**
  during Phase 7 — a real, unverified feasibility dependency that should be
  confirmed before Stage 0's B1/B2/A3 rows are scheduled, not assumed to
  work.
- **Correlated feature families**: many DOAR objective features are derived
  from the same underlying segmentation mask (§1.7) — tree-based feature
  importances (C3) can be misleading under this correlation and must be
  reported with that caveat, not as clean, independent importances.
- **Equal-protocol screening is not equal to per-architecture-optimal
  tuning**: Family A's zero-HP-search convention (inherited from Phase 4/5,
  governance rule 3) means an architecture that is more sensitive to its
  learning rate/schedule than the others could be under-represented — this
  is a deliberate trade-off for comparability, not an oversight, but should
  be stated plainly in any final write-up.
- **This entire programme, until §7 is executed, remains on the same
  duplicate-contaminated split as Phase 3A–6** — every macro-F1 number this
  programme will produce is a *development-time, relative* signal for
  shortlisting, not a final performance claim, exactly as already
  established and now further reinforced by this section.

---

## 9. Roadmap — distinguishing future phases (proposed, not started, not renumbering anything completed)

None of the phases below has begun. Listing them here is planning only.

- **Future phase: Extended model-family screening** — Family A (§2.1):
  DenseNet121, ConvNeXt-Tiny, compact-ViT probe.
- **Future phase: Classical and objective-feature baselines** — Families B
  and C (§2.2–2.3): DINOv2/OpenCLIP frozen embeddings, classical models on
  existing objective features, HOG, feature-family ablation.
- **Future phase: Fine-tuning ablations** — Family E (§2.5): freeze-depth,
  focal loss, augmentation on/off (and conditionally, strength).
- **Future phase: Fusion experiments** — Family D (§2.4): early and late
  fusion comparisons.
- **Future phase: Shortlist confirmation** — cross-family comparison of
  every family's best candidate under identical multi-seed evaluation,
  producing the single frozen shortlist referenced in §7 item 5.
- **Future phase: Leakage-safe final evaluation** — building the newly
  locked test set per §7, then the single, one-time unlocked evaluation of
  the frozen shortlist.
- **Future phase: Psychological or professional review** — remains entirely
  separate from and subsequent to all of the above; no experiment in this
  programme touches rule activation, concern-engine enablement, or any
  clinical claim (§4, governance rule 10).

Each future phase above will require its own explicit approval before any
training begins, per this phase's own closing instruction (§10).

---

## 10. Explicitly out of scope for this phase (per user instruction)

The following were **not** done and should not be inferred from this
document: training any model; tuning any hyperparameter; accessing the
test split in any way; cleaning or modifying the dataset; implementing any
detector; activating any psychological rule or concern; changing the
default/production model; starting any of the future phases listed in §9.
**This phase stops here.** Nothing in this experiment programme should be
started until the proposal (this document and `PHASE7_EXPERIMENT_MATRIX.csv`)
has been reviewed and explicitly approved.
