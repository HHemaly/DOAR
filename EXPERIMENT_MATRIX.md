# Experiment Matrix — Stage 0 Model Comparison Programme

**Status: planning document. No experiment below has been executed on
real data as a reportable result — the dataset gate is closed (see
`EXPERIMENT_PROTOCOL.md` §0). Every research question, hypothesis, and
variable below is either directly adapted from `PHASE7_EXPERIMENT_MATRIX.csv`
(cross-referenced row IDs given for traceability, not re-invented) or new
to this session, clearly marked.** Adding more models does not
automatically strengthen the thesis — each experiment tests one specific,
falsifiable research question against the *same* controlled protocol
(`EXPERIMENT_PROTOCOL.md`), and a null/negative result is exactly as
reportable as a positive one.

---

## Experiment A — Objective-feature classical baseline (RQ1)

*Cross-references `PHASE7_EXPERIMENT_MATRIX.csv` rows C1 (linear), C2
(RBF SVM), C3 (tree-based).*

- **Research question**: How well do interpretable, already-implemented
  objective drawing features (quality/segmentation/composition/colour/
  stroke/shape, 59 columns from `features.py::objective_feature_row`)
  predict emotion class without any deep image representation?
- **Hypothesis**: These features (designed for rule-evidence, not
  classification) underperform the fine-tuned CNN baselines by a wide
  margin, but are not necessarily useless — a macro-F1 meaningfully above
  chance (0.25 for 4 balanced classes) would still be a genuine, reportable
  finding about how much emotion-relevant signal is captured by simple,
  interpretable geometry/colour/quality statistics alone.
- **Independent variable**: Classifier family (logistic regression, linear
  SVM, RBF SVM, random forest, extra trees, HistGradientBoosting — all six
  already implemented in `experiments.py::_model`).
- **Controlled variables**: same manifest/split, same 59-feature input,
  same 3 seeds, `SimpleImputer(median)` for the 2 permanently-missing
  `shape.*` columns (never fabricated, per `features.py`'s own design).
- **Primary metric**: macro-F1 (validation). Full metric set per
  `EXPERIMENT_PROTOCOL.md` §3.
- **Infrastructure status**: **100% pre-existing**, zero new code required
  (`MODEL_EXPERIMENT_AUDIT.md` §11). Config: `configs/experiments/objective_features.json`.
- **Exact commands** (once the gate passes): see `MODEL_RUN_GUIDE_WINDOWS.md`.

## Experiment B — HOG + colour + geometry handcrafted-feature comparison (RQ2)

*Cross-references C4 (HOG) and C5 (family ablation); the specific
hog/colour/geometry-only 4-way isolation is new to this session (C5's
existing ablation is a drop-one-family, not an isolate-one-group, design).*

- **Research question**: Does adding real HOG (edge-orientation texture)
  features improve on the existing objective features, and which single
  handcrafted-feature group (HOG, colour, or geometry) carries the most
  standalone signal?
- **Hypothesis**: HOG adds complementary signal not captured by the
  existing composition/colour features (the pre-existing `stroke.edge_density`
  is only a crude gradient-magnitude proxy, not orientation-aware);
  falsified if `hog_colour_geometry` does not exceed `colour_only` and
  `geometry_only` by a margin exceeding seed variability.
- **Independent variable**: Feature group — `hog_only`, `colour_only`
  (`colour.*`), `geometry_only` (`composition.* + segmentation.* + shape.*`),
  `hog_colour_geometry` (union of the three). HOG parameters are **fixed a
  priori, never searched** (64×64/8×8-cell/9-bin/2×2-block, 1,764-dim),
  matching C4's own stated rationale: a HOG-parameter search would be a
  hidden validation-leakage path.
- **Controlled variables**: same classifier menu as Experiment A (reused,
  not reimplemented), same seeds, same splits.
- **Infrastructure status**: **new this session** —
  `src/doar/hog_features.py` (extraction) +
  `src/doar/handcrafted_comparison.py` (group comparison). Smoke-tested
  end-to-end on both synthetic and real (32-image, train/valid-only)
  data. Config: `configs/experiments/handcrafted_groups.json`.

## Experiment C — DINOv2 frozen-embedding classifier (RQ3)

*Cross-references B1 (linear probe), B2 (RBF SVM), B3 (MLP).*

- **Research question**: How effective are frozen, self-supervised DINOv2
  embeddings — no backbone fine-tuning — on this relatively small (2,821
  train image) drawing dataset?
- **Hypothesis**: A linear classifier on frozen DINOv2 embeddings reaches
  within 0.05 macro-F1 of the best fine-tuned CNN baseline
  (efficientnet_b0, 0.729 mean valid macro-F1, contaminated-split,
  Phase 5), suggesting most of the task's linearly-separable signal
  requires no domain-specific fine-tuning. A nonlinear classifier (MLP) is
  *not* hypothesized to meaningfully outperform the linear probe, given
  the small training set relative to embedding dimensionality — a null
  result here is expected and informative, not a failure.
- **Independent variable**: Classifier on top of the SAME frozen
  embeddings — `logistic_regression` (linear probe, default) vs.
  `mlp_small` (a small, `early_stopping`-regularized MLP, default) vs.
  optionally the full shared classifier menu.
- **Controlled variables**: backbone version pinned explicitly
  (`dinov2_vits14`, `facebookresearch/dinov2` via `torch.hub`) and
  recorded in every artifact — never silently upgraded. Embedding
  extraction is label-free by construction (`deep/embeddings.py::extract_embeddings`
  never reads the `class` column). The backbone is **never fine-tuned** in
  this experiment (that would be a different, unrequested experiment).
- **Infrastructure status**: embedding extraction was **already fully
  implemented** (`deep/embeddings.py`); the classifier-comparison layer is
  **new this session** (`src/doar/deep/embedding_classifier.py`),
  smoke-tested against a synthetic embeddings cache. **Real DINOv2
  extraction has not been attempted in this environment** — no cached
  `facebookresearch/dinov2` weights exist under this machine's
  `torch.hub` cache, so first use requires live network access (see
  `MODEL_RUN_GUIDE_WINDOWS.md` for the exact command and what to check
  first).

## Experiment D — DenseNet121 transfer learning (RQ4)

*Cross-references A1.*

- **Research question**: Does a densely-connected CNN (DenseNet121, 7.0M
  params) reach macro-F1 competitive with the already-screened
  architectures (mobilenet_v3_small 0.695, resnet18 0.718,
  efficientnet_b0 0.729 mean valid macro-F1, all contaminated-split,
  Phase 4/5)?
- **Hypothesis**: DenseNet121's mean validation macro-F1 (3 seeds) falls
  within the range already spanned by the 3 existing architectures — dense
  connectivity is not expected to be a decisive advantage or disadvantage
  for this task; a result outside that range in either direction is the
  actually interesting/reportable outcome.
- **Independent variable**: Architecture (densenet121 vs. the 3 existing).
- **Controlled variables**: identical protocol to Phase 4/5's own
  screening — same freeze-then-unfreeze schedule (`freeze_epochs=3`), same
  optimizer/scheduler defaults, same augmentation, zero hyperparameter
  search (a screening run, not a tuning run, exactly like the 3 existing
  architectures were run).
- **Infrastructure status**: **new this session** — DenseNet121 added to
  `deep/registry.py`/`deep/__init__.py::MODEL_NAMES` (build, forward pass,
  freeze/unfreeze, and penultimate-embedding extraction all directly
  verified). `train_image_model` itself required zero changes — it is
  already generic over any registered model name. Config:
  `configs/training/densenet121.toml`. 1-epoch CPU smoke run completed
  this session in ~3 seconds against a tiny synthetic `ImageFolder`.

## Experiment E — ConvNeXt-Tiny feasibility probe (RQ5)

*Cross-references A2, as revised by `PHASE7_EXPERIMENT_MATRIX_REVISION.md`
(downgraded from "Recommended" to "Optional feasibility probe first").*

- **Research question**: Does ConvNeXt-Tiny (27.8M params — the largest
  architecture screened so far, ~7× resnet18's param count) provide
  sufficient validation benefit over efficientnet_b0 to justify its
  computational cost, or does it primarily add overfitting risk against a
  ~1,273-independent-group effective training set?
- **Hypothesis**: ConvNeXt-Tiny's mean validation macro-F1 is **not**
  assumed superior to efficientnet_b0 merely for being newer/larger —
  falsified only if it exceeds efficientnet_b0's own screened result by a
  margin exceeding seed variability. The project's own governance rule
  (established in `PHASE7_EXPERIMENT_MATRIX_REVISION.md`) explicitly
  rejects "bigger automatically helps" as a default assumption.
  **Advancement rule**: run the single-seed, epoch-restricted probe
  (`configs/training/convnext_tiny_probe.toml`) first; only build and run
  a full 3-seed config if the probe's validation macro-F1 meaningfully
  exceeds efficientnet_b0's screened result. If it does not, **stop here
  and record that decision** — do not escalate to 3 seeds by default.
- **Independent variable**: Architecture (convnext_tiny vs. the 3+1
  existing).
- **Infrastructure status**: **already fully wired** in `registry.py`
  before this session (zero registry changes needed) — only the
  explicitly-labeled, restricted-budget probe config is new
  (`configs/training/convnext_tiny_probe.toml`: 1 seed, 15-epoch cap
  instead of 50, `early_stopping_patience=5`). Smoke-tested this session
  (1-epoch CPU run, ~3 seconds, synthetic data).

## Experiment F — Image + objective-feature fusion (RQ6)

*Cross-references D1 (framing), D2 (DINOv2 early fusion), D3 (finetuned-CNN
early fusion), D4 (late fusion).*

- **Research question**: Does fusing an image-based representation (a
  fine-tuned CNN's own penultimate embedding, or frozen DINOv2 embeddings)
  with the objective drawing features improve macro-F1 over either
  information source alone?
- **Hypothesis**: image-alone (the best available fine-tuned CNN) beats
  objective-features-alone (Experiment A's winner); fusion of the two
  beats both alone by a margin exceeding seed variability. Each clause is
  independently falsifiable and should be reported as such, not collapsed
  into a single pass/fail.
- **Independent variable**: Information source/fusion stage — image alone
  vs. objective-features alone vs. early fusion (`early_scaled_concat`,
  `pca_early_fusion`, `mlp_early_fusion`) vs. late fusion (equal /
  validation-weighted / logistic-meta, over independently-calibrated
  probability exports).
- **Controlled variables**: exactly the same train/valid samples across
  every arm (`fusion/trainer.py::_load` hard-fails on any
  split/label/sample-ID mismatch between the features CSV and the
  embeddings cache — a real, enforced consistency check, not a
  convention). Missing numeric objective features are imputed
  deterministically (median, training-fit only) — never dropped silently.
- **Infrastructure status**: **substantially pre-existing**
  (`fusion/trainer.py`, `fusion/late.py`, `fusion/oof.py`,
  `fusion/calibrate.py`) — only a new config per additional embedding
  backbone was needed this session
  (`configs/training/primary_fusion_dinov2.toml`; a CNN-penultimate-embedding
  fusion config is not yet created — see `EXPERIMENT_PROTOCOL.md`'s
  outstanding-work list, it depends on Experiment D's own winner being
  selected first). **Smoke-test finding, not a defect**: in the
  orchestrator's smoke mode, this stage correctly *fails* with an
  artifact-provenance error, because Experiment C's smoke-mode synthetic
  embeddings deliberately carry no real provenance record — this is
  `fusion/trainer.py`'s own safety check (`provenance.verify_artifacts`)
  refusing to fuse mismatched artifacts, exactly as designed. A genuine
  full or smoke run of Experiment F requires Experiments A and C (or D) to
  have produced artifacts from a *consistent* manifest first.

---

## Summary table

| Experiment | RQ | Family cross-ref | New code this session? | Smoke-tested? | Full run status |
|---|---|---|---|---|---|
| A: objective-feature classical | RQ1 | C1-C3 | No | Yes (real + synthetic images) | **Blocked by gate** |
| B: HOG/colour/geometry | RQ2 | C4, C5 (new isolation logic) | Yes | Yes (real + synthetic images) | **Blocked by gate** |
| C: DINOv2 frozen embeddings | RQ3 | B1-B3 | Yes (classifier layer) | Partial (classifier logic only; real extraction needs network) | **Blocked by gate** |
| D: DenseNet121 | RQ4 | A1 | Yes (registry) | Yes (CPU, synthetic) | **Blocked by gate** |
| E: ConvNeXt-Tiny probe | RQ5 | A2 | No (config only) | Yes (CPU, synthetic) | **Blocked by gate** |
| F: fusion | RQ6 | D1-D4 | Yes (1 new config) | Correctly refuses on mismatched smoke artifacts | **Blocked by gate** |

**No experiment above may proceed to a full, reportable run until
`EXPERIMENT_PROTOCOL.md` §0's gate passes.**
