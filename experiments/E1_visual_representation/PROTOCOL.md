# E1-A Protocol — Controlled Architecture Screening

**Status: DEVELOPMENT / VALIDATION SCREENING. NOT a locked test result.**
Locked Test (512 images) is never evaluated in this phase.

**IMPORTANT — two methodologies exist under this one protocol document,
clearly distinguished:**

- **§§1–7 below (original)** describe **E1-A1 — Transfer-Head Development
  Screening**, the CPU-only preliminary run. Preserved as real, secondary/
  supporting evidence — see `E1A_CPU_PRELIMINARY_STATUS.md`. Not the
  primary E1-A result.
- **§8 (new, added for GPU migration)** describes **E1-A2 — Standardized
  Frozen Linear-Probe Benchmark**, the PRIMARY E1-A methodology, portable
  to Colab GPU. `E1A_T2_standardized_linear_probe.csv` is the primary
  comparison table; E1-B finalists are ranked from E1-A2, not E1-A1.

## 0. Frozen inputs (never modified)

- T0 manifest: `outputs/t0_automated/final_partition/partition_manifest.csv`,
  SHA-256 `4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d`,
  verified by `scripts/build_e1a_manifest.py` before every run (refuses to
  proceed on a mismatch).
- Train = 2,599 / Valid = 284 / Test (locked, unused) = 512 / Excluded
  conflict = 293. All Train+Valid image paths verified to resolve on this
  machine (2,883/2,883, 0 missing) before any run.
- Dataset root resolved this session:
  `C:\Users\ZZ01G7865\Downloads\DOAR\Combined_Drawing-20260807T152850Z-1-001\Combined_Drawing\`.

## 1. Hardware / environment (readiness check)

- **GPU: none. CPU-only** (`torch.cuda.is_available() == False`).
- PyTorch 2.13.0+cpu, torchvision 0.28.0+cpu, open_clip 2.32.0,
  transformers 4.57.6, timm 1.0.28, scikit-learn 1.9.0.
- 8 CPU cores. ~19 GB free disk at session start.
- This is a real, load-bearing constraint on E1-A's design (see §3).

## 2. Existing infrastructure reused

- `src/doar/deep/registry.py::build_model/freeze_backbone/resolve_weights`
  — unmodified, used directly for all 6 CNN/ViT candidates.
- `src/doar/deep/preprocessing.py::resolve_preprocessing/build_train_transform/
  build_eval_transform` — unmodified; each torchvision backbone's OWN
  official pretrained preprocessing (resize/crop/mean/std/interpolation) is
  derived from its own `weights.transforms()`, not forced to one common
  spec (per instruction: never handicap a model to fake identical
  preprocessing when its official pretrained representation requires
  something else).
- `src/doar/deep/augmentations.py::augmentation_ops("conservative")` —
  unmodified; the SAME conservative profile (small affine ±7°/±3%
  translate/±5% scale + mild color jitter) applied identically across every
  CNN/ViT candidate's training split. Validation always deterministic
  (resize/crop/normalize only, no augmentation).
- **NOT reused, and why**: `deep/datasets.py::build_loaders` (uses
  `torchvision.datasets.ImageFolder`, which infers split membership from
  directory layout — incompatible with the T0 partition's manifest-driven
  split assignment, since the original on-disk `train/valid/test` folders
  do NOT match the new T0 split). `deep/trainers.py::train_image_model`
  (a real, load-bearing shared trainer, but its per-epoch history schema
  only records `epoch/train_loss/valid_macro_f1/learning_rate` — missing
  train_accuracy, val_loss, val_accuracy, val_balanced_accuracy, and
  per-epoch runtime, all explicitly required by E1A_T4). Modifying either
  shared module risked destabilizing other experiments that depend on
  their exact current behavior; instead, a small, E1-A-scoped module
  (`scripts/e1a_common.py`) reuses every pure model/transform building
  block above and implements only the manifest-based `Dataset` and the
  richer epoch loop.

## 3. Frozen-backbone execution strategy (CPU-only adaptation)

Every candidate uses a genuinely frozen backbone (no gradient, never
unfrozen) — but HOW that frozen computation is reused differs by
candidate family, disclosed explicitly:

- **CNN/ViT (MobileNetV3-Small, ResNet18, EfficientNet-B0, DenseNet121,
  ConvNeXt-Tiny, ViT-B/16)**: full image forward pass through the frozen
  backbone EVERY epoch (not cached), because training augmentation
  (§7 below) changes the image each epoch — a cached single-pass embedding
  would silently discard the augmentation policy this task explicitly
  requires. Only the replaced final classifier layer receives gradients.
- **Foundation representations (CLIP ViT-B/32, DINOv2 ViT-S/14, SigLIP2
  ViT-B/16)**: embeddings extracted ONCE per image (no augmentation, frozen
  encoder, single forward pass, cached to `.npz`), then a single
  `nn.Linear` head trained epoch-by-epoch on the cached vectors. This
  matches the task's own explicit language for this family ("compute/cache
  embeddings where practical... train the same simple classifier/head
  policy where possible") and is the only way frozen-backbone training on
  these larger encoders is tractable on CPU-only hardware within a
  reasonable wall-clock budget.

This is a real, disclosed methodological difference between the two
families, not an inconsistency — CNN/ViT candidates see augmented images
every epoch; foundation-model candidates train their head on fixed,
un-augmented cached representations. Both remain genuinely frozen-backbone
per the task's core requirement.

## 4. Candidate registry

| Key | Model | Family | Weights |
|---|---|---|---|
| `mobilenet_v3_small` | MobileNetV3-Small | CNN | torchvision IMAGENET1K |
| `resnet18` | ResNet18 | CNN | torchvision IMAGENET1K |
| `efficientnet_b0` | EfficientNet-B0 | CNN | torchvision IMAGENET1K |
| `densenet121` | DenseNet121 | CNN | torchvision IMAGENET1K |
| `convnext_tiny` | ConvNeXt-Tiny | CNN | torchvision IMAGENET1K |
| `vit_b_16` | ViT-B/16 | Transformer | torchvision IMAGENET1K |
| `clip_vit_b32` | CLIP ViT-B/32 | Foundation | open_clip, `openai` |
| `dinov2_vits14` | DINOv2 ViT-S/14 | Foundation | `facebookresearch/dinov2` hub |
| `siglip2_vit_b16` | SigLIP2 ViT-B/16 | Foundation | open_clip, `timm/ViT-B-16-SigLIP2` (`webli`) |

DINOv2 variant note: the smallest DINOv2 release (ViT-S/14, 21M params) was
selected over larger DINOv2 variants specifically for CPU tractability —
disclosed, not hidden; a larger DINOv2 backbone remains a candidate for a
future GPU-available session if warranted.

## 5. Common training recipe (fairness)

| Setting | Value |
|---|---|
| Seed | 42 |
| Max epochs | 30 |
| Early stopping | patience=5 on Validation Macro-F1, best checkpoint restored |
| Optimizer | AdamW |
| Head learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss, label_smoothing=0.05, class-weighted (inverse frequency) |
| Batch size | 16 (image models), 64 (cached-embedding models) |
| Scheduler | none (fixed LR — screening only, no LR-search in E1-A) |
| Augmentation | `conservative` profile (image models only) |
| Image size | model's own native pretrained size (224 for every candidate here) |

Identical across every candidate — no per-model hyperparameter tuning in
E1-A.

## 6. What E1-A does NOT do

No backbone fine-tuning (frozen throughout — that is E1-B). No
hyperparameter search. No Test-set evaluation. No psychological
interpretation of errors. No claim of clinical/diagnostic validity.

## 7. Throughput / cost stop-rule (Section 17)

Calibration run (200 train + 100 valid images, 2 epochs, steady-state
2nd-epoch timing projected to the full 2,883-image epoch): see
`raw/throughput_calibration.json` and the end-of-run report for the exact
measured numbers and the go/no-go decision made for each candidate before
committing to a full 30-epoch budget.

---

## 8. E1-A2 — Standardized Frozen Linear-Probe Benchmark (PRIMARY methodology, GPU-portable)

### 8.1 Why this methodology replaces E1-A1 as primary

E1-A1 used two different training strategies for different model families
(full-image augmented training for CNN/ViT vs. cached-embedding linear
probes for foundation models) and under-reported foundation models'
resource cost (only their ~2–3k-parameter head, not their full frozen
encoder). E1-A2 fixes both: **every** candidate — CNN, ViT, and foundation
alike — goes through the exact same pipeline:

```
image -> model-native deterministic pretrained preprocessing (NO augmentation)
      -> frozen pretrained encoder/backbone
      -> ONE embedding per image, extracted once, cached
      -> the SAME linear 4-class classifier
      -> trained on ALL 2,599 Train embeddings, selected on ALL 284 Validation embeddings
```

**No training augmentation anywhere in E1-A2** — every embedding is
extracted once, deterministically, from the encoder's own official
pretrained preprocessing. This is a genuine like-for-like REPRESENTATION
comparison, not a representation-plus-augmentation-policy comparison.

### 8.2 Embedding definition (never the classification logits)

The representation immediately BEFORE the original classifier/head:
torchvision CNN/ViT backbones have their final classifier layer replaced
with `nn.Identity` (same technique `src/doar/deep/embeddings.py::
_finetuned_extractor` already uses for fine-tuned DOAR checkpoints, reused
here for fresh ImageNet-pretrained backbones); CLIP/SigLIP2 use
`encode_image`; DINOv2 uses its own forward pass's `x_norm_clstoken` —
none of these three have a classification head in their base pretrained
form, so their native output already IS the pre-classification
representation.

### 8.3 Portability (dataset-root / output-root / device)

`scripts/e1a2_common.py::load_split_rows(dataset_root)` resolves every
image as `dataset_root / relative_path` from the frozen T0 manifest —
**never** the T0 manifest's own `path` column (which is machine-specific).
`scripts/prepare_e1_dataset.py --dataset-root <path> --verify-only` checks
the manifest SHA-256, split counts (2599/284/512), and that every Train/
Validation file resolves, before any embedding extraction is attempted.
`--device auto|cpu|cuda` (`resolve_device`) resolves to CUDA when
available, fails clearly if `cuda` is explicitly requested but unavailable,
and reports GPU name/CUDA version/VRAM/PyTorch version.
`scripts/run_e1a_all.py --dataset-root <path> --output-root <path>` never
assumes outputs live inside the repository — large artifacts (embeddings,
checkpoints) go to `--output-root`; only small tables/figures (Section
8.6) are written into the repo's own `tables/`/`figures/` directories.

### 8.4 Embedding cache invalidation

Cache key = `sha256({model_key, manifest_sha256, split, weights_id,
preprocessing_hash})`. A cached `.npz` is reused ONLY if its companion
`_meta.json`'s recorded cache key exactly matches AND its recorded
image-ID set exactly matches the requested rows — any difference in
model, T0 manifest content, split, pretrained weights, or preprocessing
triggers full re-extraction, never a silent stale reuse.

### 8.5 Resource accounting (corrected)

Every candidate's result JSON records, separately and explicitly:
`encoder_full_parameter_count` (the WHOLE frozen backbone, e.g. 86M for
CLIP ViT-B/32 — never just the ~2k-parameter head),
`trainable_head_parameter_count`, `embedding_dimension`,
`pretrained_weights_id`, `one_time_embedding_extraction_seconds_total`,
`encoder_inference_latency_seconds_per_image`,
`classifier_only_latency_seconds_per_image`,
`end_to_end_latency_seconds_per_image` (sum of the two latencies — never
compares a cached-head-only latency against a full CNN's end-to-end
latency as if equivalent), `peak_gpu_vram_mb` (GPU runs only),
`embedding_cache_bytes`, `checkpoint_size_bytes`.

### 8.6 Outputs

Large artifacts (`.npz` embeddings, `.pt` checkpoints) go to
`--output-root` only (gitignored even if that happens to be inside the
repo). Small artifacts committed to the repo: `tables/E1A_T2–T6*.csv`,
`figures/E1A_F1–F6.{png,pdf,csv}`, `RUN_MANIFEST.json`'s own small JSON
(if produced under a repo-relative output-root during a smoke test —
production Colab runs write it to Drive). Validation predictions
(`best_valid_predictions.csv`: `image_id, true_class, predicted_class,
prob_Angry, prob_Fear, prob_Happy, prob_Sad`) are written per-model under
`--output-root/checkpoints/<model>/` — large-artifact territory, not
committed, but exactly the schema needed later for calibration, paired
comparison, statistical tests, error analysis, and E4 fusion.

### 8.7 Resume behavior

`--resume`: reuses a valid embedding cache whenever its recorded identity
matches exactly (8.4); skips a model entirely if `RUN_MANIFEST.json`
already records `status: "trained"` under the exact same configuration
signature (`max_epochs`, `patience`, `seed`, `manifest_sha256`); resumes
an interrupted linear-probe run from its last saved epoch
(`run_linear_probe`'s own `last.pt`) when the embedding dimension and
model key match. A failed OPTIONAL candidate is recorded in
`RUN_MANIFEST.json` with its error and traceback; the run continues with
the remaining candidates rather than aborting.

### 8.8 Candidates and expected omissions

Same 9 candidates as E1-A1 (§1). SigLIP2 ViT-B/16 was found, in E1-A1, to
take ~19.5s/image for embedding extraction alone on CPU (~15.6h projected
for the full dataset) — GPU execution is expected to resolve this
(SigLIP2 is a standard-sized ViT-B/16 encoder; the CPU slowdown was
disproportionate to its parameter count, suggesting an inefficient CPU
code path rather than a fundamentally expensive model) and it remains a
full candidate under E1-A2 on Colab. If it is still disproportionately
slow on GPU, `run_e1a_all.py` will record it as `failed` in
`RUN_MANIFEST.json` with the real measured error and continue with the
remaining 8 candidates, per Section 17's "continue other valid candidates"
instruction.
