# Phase 4 — Preliminary Controlled Emotion-Model Comparison (One-Seed Screening)

**Status: PRELIMINARY, NOT LEAKAGE-SAFE.** Same dataset as Phase 3A (all
3,688 currently-readable images, no cleaning, no duplicate removal, no
relabeling). Do not cite any number here as final thesis evidence — the
known cross-split duplication documented in `CURRENT_STATE_AUDIT.md` §2
applies equally to every model in this comparison. **This is a single-seed
screening.** No architecture is claimed scientifically superior — see §7.

Phase 3A checkpoints/results were not touched (independently re-verified by
SHA-256 — see §6). All Phase 4 outputs live in a separate directory,
`outputs/phase4/`.

---

## 1. Models inspected and screened

`src/doar/deep/registry.py` supports 8 architectures: `small_cnn`,
`mobilenet_v3_small`, `mobilenet_v3_large`, `resnet18`, `resnet50`,
`efficientnet_b0`, `convnext_tiny`, `vit_b_16`.

**Selected for screening**: `small_cnn`, `mobilenet_v3_small`, `resnet18`,
`efficientnet_b0` — this is exactly `src/doar/deep/compare.py`'s existing
`DEFAULT_MODELS` list, confirming it was already the codebase's own
considered "practical default" set, not a new choice invented for this
phase. `mobilenet_v3_large`, `resnet50`, `convnext_tiny`, and `vit_b_16`
were excluded as impractical for one-seed screening on this hardware (6 GB
Quadro P3200) — each is substantially larger (`resnet50` ~25M params,
`convnext_tiny` ~28M, `vit_b_16` ~86M) and would multiply per-model runtime
without being likely candidates for a "small, practical" deployment target.

**Infrastructure reused, not reimplemented**: `main.py compare-deep-models`
→ `src/doar/deep/compare.py::run_deep_comparison()` → `src/doar/deep/trainers.py::train_image_model()`
already implements exactly the controlled, identical-protocol, multi-model
comparison this phase needed, including automatic comparison-bar-chart and
per-model training-curve generation. No parallel training framework was
built. The one real gap found: **parameter count was not recorded anywhere**
in the training output, despite being a required comparison field — fixed
as a small, tested addition to `train_image_model()` (see §8), not worked
around with an external script.

---

## 2. Exact controlled protocol

| Axis | Value | Identical across all 4 models? |
|---|---|---|
| Dataset | `C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`, all 3,688 readable images, no exclusion | Yes — same manifest file, byte-identical content confirmed against Phase 3A's manifest via diff |
| Split | Existing `train`/`valid`/`test` folders | Yes |
| Preprocessing | `resolve_preprocessing()` per model — ImageNet-standard normalization for all 3 pretrained torchvision models | Yes for the 3 pretrained models; **No for `small_cnn`** — see §3 |
| Augmentation | `"conservative"` profile | Yes |
| Image size | 224×224 | Yes |
| Batch size / grad accumulation | 4 physical / 4 accumulation steps (effective batch 16) | Yes — 6 GB-safe default, unchanged from Phase 3A |
| Epochs (training budget) | 10 | Yes — matches Phase 3A exactly |
| Optimizer | AdamW | Yes |
| Learning rates | head 3×10⁻⁴, backbone 1×10⁻⁴ (differential) | Yes as configured; **effectively No for `small_cnn`** — see §3 |
| Scheduler | ReduceLROnPlateau (patience 2, factor 0.3, mode max) | Yes |
| Early stopping | patience 7 | Yes (did not trigger for any model within 10 epochs) |
| Class weighting | Enabled | Yes |
| Checkpoint selection | Highest per-epoch validation macro-F1 → `best.pt` | Yes — identical rule, built into the trainer once |
| Evaluation code | `export-probabilities` → `evaluate-predictions` on the **validation** split | Yes — identical for all 4, same code path as Phase 3A |
| Seed | 42 | Yes (single seed — this is a screening pass, not the multi-seed comparison) |
| Calibration | Not run this pass | Yes (uniformly skipped, to keep the screening fast — see §9 for the multi-seed proposal, which does include it) |

## 3. Architecture-specific differences that cannot reasonably be identical

- **`small_cnn` has no pretrained weights at all** (a custom 3-conv-layer
  network with no published checkpoint) — it necessarily trains from
  scratch, while the other 3 use ImageNet `DEFAULT` pretrained weights. This
  is inherent to what `small_cnn` is, not a protocol inconsistency.
- **`small_cnn` skips the freeze/unfreeze backbone schedule entirely**
  (`registry.py`: `if freeze_epochs and model_name != "small_cnn":
  freeze_backbone(model)`) — freezing a backbone that was never pretrained
  has no meaning, so it trains all parameters from epoch 0, while the other
  3 freeze their backbone for the first 3 epochs then unfreeze.
- **Consequently, `small_cnn` effectively trains entirely at the "backbone"
  learning rate (1×10⁻⁴), not the "head" rate (3×10⁻⁴)**: the differential-LR
  parameter-group logic (`_build_param_groups`) assigns parameters to the
  head group by matching attribute names (`fc`/`classifier`/`heads`);
  `small_cnn` is a plain `nn.Sequential` with no such named submodule, so
  every one of its parameters lands in the backbone group. This is a real,
  previously-undocumented consequence of applying a naming heuristic
  designed for torchvision models to a custom `nn.Sequential` — noted here,
  not treated as a bug to fix mid-screening (fixing it would itself be a
  protocol change requiring separate consideration).
- **Preprocessing normalization differs for `small_cnn`**: the 3 pretrained
  models use their weights-derived ImageNet normalization; `small_cnn`'s
  `resolve_preprocessing()` output reflects "scratch" (no pretrained
  statistics to derive from).
- **Parameter counts differ by two orders of magnitude** (93,764 for
  `small_cnn` vs. up to 11.2M for `resnet18`) — this is the comparison's own
  subject matter, not something to normalize away.

---

## 4. Actual screening results (validation split, n=310, PRELIMINARY)

| Model | Params | Train time | Accuracy | Macro-F1 | Balanced acc. | Log loss | Brier | ECE | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| `small_cnn` | 93,764 | 329.6s | 0.497 | 0.435 | 0.443 | 1.194 | 0.645 | 0.091 | 0.722 | 0.457 |
| `mobilenet_v3_small` | 1,521,956 | 508.3s | 0.729 | 0.704 | 0.713 | 0.733 | 0.386 | 0.045 | 0.905 | 0.778 |
| `resnet18` | 11,178,564 | 453.8s | 0.735 | 0.715 | 0.716 | 0.839 | 0.397 | 0.108 | 0.906 | 0.782 |
| `efficientnet_b0` | 4,012,672 | 790.7s | **0.761** | **0.736** | **0.742** | 0.712 | 0.363 | 0.056 | **0.914** | **0.797** |

Total screening time: **2,082.4 seconds (~34.7 minutes)** for all 4 models,
1 seed, 10 epochs each — close to the pre-run estimate of ~35–45 minutes
(§9 of the original Phase 4 proposal).

**`mobilenet_v3_small`'s result here (macro-F1 0.7043, 508.3s) is
consistent with Phase 3A's independently-run result on the same protocol
(macro-F1 0.7043, 520.6s)** — the two invocation paths (`train-image-model`
directly vs. `compare-deep-models`) produced matching validation macro-F1 to
4 decimal places, differing only in wall-clock time by ~2% (expected
run-to-run variance, not a methodological difference). This is a genuine,
useful internal consistency check, not fabricated agreement.

### Per-class detail

<details>
<summary>small_cnn</summary>

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| Angry | 0.508 | 0.416 | 0.457 | 77 |
| Fear | 0.271 | 0.289 | 0.280 | 45 |
| Happy | 0.618 | 0.809 | 0.701 | 110 |
| Sad | 0.364 | 0.256 | 0.301 | 78 |

Confusion matrix: `[[32,13,17,15],[10,13,11,11],[7,5,89,9],[14,17,27,20]]`
</details>

<details>
<summary>mobilenet_v3_small</summary>

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| Angry | 0.734 | 0.610 | 0.667 | 77 |
| Fear | 0.542 | 0.711 | 0.615 | 45 |
| Happy | 0.856 | 0.864 | 0.860 | 110 |
| Sad | 0.684 | 0.667 | 0.675 | 78 |

Confusion matrix: `[[47,11,8,11],[4,32,2,7],[6,3,95,6],[7,13,6,52]]`
</details>

<details>
<summary>resnet18</summary>

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| Angry | 0.710 | 0.636 | 0.671 | 77 |
| Fear | 0.585 | 0.689 | 0.633 | 45 |
| Happy | 0.768 | 0.873 | 0.817 | 110 |
| Sad | 0.825 | 0.667 | 0.738 | 78 |

Confusion matrix: `[[49,6,18,4],[6,31,4,4],[6,5,96,3],[8,11,7,52]]`
</details>

<details>
<summary>efficientnet_b0</summary>

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| Angry | 0.724 | 0.714 | 0.719 | 77 |
| Fear | 0.582 | 0.711 | 0.640 | 45 |
| Happy | 0.846 | 0.900 | 0.872 | 110 |
| Sad | 0.806 | 0.641 | 0.714 | 78 |

Confusion matrix: `[[55,5,11,6],[7,32,2,4],[7,2,99,2],[7,16,5,50]]`
</details>

Plots (real data, no interpolation): `outputs/phase4/deep/figures/deep_comparison.png`
(mean±std bar chart, std=0 since n=1 seed), `curve_<model>_seed_42.png` ×4
(per-epoch validation macro-F1), `tradeoffs.png` (macro-F1 vs. training time
and vs. parameter count, both axes real measured values).

---

## 5. Comparing beyond accuracy — runtime, size, stability, complexity

- **`small_cnn`** is far behind on every accuracy metric and is not a
  competitive candidate on this screening — but it is also ~35% faster than
  `mobilenet_v3_small` and two orders of magnitude smaller (93.8K vs. 1.5M+
  params), which may still matter for a low-resource deployment target. Its
  ECE (0.091) is also the second-worst, suggesting poorly calibrated
  confidence alongside weak accuracy.
- **`mobilenet_v3_small`** is the fastest of the 3 competitive models
  (508.3s) and has the best calibration (ECE 0.045) despite not having the
  top accuracy — a real operational advantage if calibrated confidence
  matters more than peak macro-F1.
- **`resnet18`** trained faster than `mobilenet_v3_small` in this run
  (453.8s vs. 508.3s) despite having ~7× the parameters — plausibly because
  larger, more GPU-parallelism-friendly operations amortize better than
  MobileNet's depthwise-separable convolutions at this batch size, though
  this is a single-seed observation, not confirmed. Its ECE (0.108) is the
  worst of the 4, and its checkpoint is by far the largest on disk (134MB
  vs. 18–49MB for the others) due to AdamW's per-parameter optimizer state
  scaling with model size — a real operational-complexity cost for
  deployment/versioning, not just an accuracy question.
- **`efficientnet_b0`** has the best accuracy/macro-F1/balanced-accuracy/
  ROC-AUC/PR-AUC of the 4 in this screening, but also took the longest to
  train (790.7s, ~1.6× `mobilenet_v3_small`'s time) — the accuracy advantage
  comes with a real compute-cost tradeoff, not a free win.
- **No stability signal beyond a single seed exists yet** — every model's
  std-of-macro-F1 is trivially 0 (n=1). Nothing here indicates run-to-run
  stability; that is exactly what the multi-seed proposal (§9) is for.

---

## 6. Verification and preservation

- **Phase 3A untouched**: independently re-verified via SHA-256 against the
  hashes recorded in `PHASE3A_RESULTS.md` §9a — `manifest.csv`, `best.pt`,
  `last.pt`, `loss_curve.png`, and `eval/metrics.json` all match exactly.
  No Phase 3A file was read-modified or overwritten by this phase.
- **New versioned directory**: all Phase 4 output lives under
  `outputs/phase4/`, structurally separate from `outputs/phase3a/`.
- **Artifact manifest**: `outputs/phase4/ARTIFACT_MANIFEST.tsv` — 25 files,
  path + size + SHA-256 for every checkpoint, metrics file, probability
  export, comparison JSON, figure, and audit log. Key entries:

| Artifact | Size | SHA-256 (first 16 hex) |
|---|---|---|
| `outputs/phase4/deep/runs/small_cnn_seed_42/best.pt` | 1,136,273 B | `f07eeb9845326...` |
| `outputs/phase4/deep/runs/mobilenet_v3_small_seed_42/best.pt` | 18,491,543 B | `80b073e07a294...` |
| `outputs/phase4/deep/runs/resnet18_seed_42/best.pt` | 134,261,865 B | `17409c5ccdc42...` |
| `outputs/phase4/deep/runs/efficientnet_b0_seed_42/best.pt` | 48,586,090 B | `275ebdf82bcc8...` |
| `outputs/phase4/deep/deep_comparison.json` | 11,103 B | `0c1001743c83d...` |

Full 25-row manifest in the TSV file (not duplicated here). **Total Phase 4
storage: 411,149,934 bytes (~392.1 MB)** — `resnet18`'s two checkpoints
(best+last, ~134MB each) dominate this; per-checkpoint size scales with
AdamW's optimizer-state overhead (2 extra buffers per parameter), not just
raw parameter count, which is why `resnet18` (11.2M params) has a larger
checkpoint than `efficientnet_b0` (4.0M params) by a wider margin than the
parameter-count ratio alone would suggest.
- **All of `outputs/phase4/` is git-ignored** (same pre-existing
  `outputs/` rule as Phase 3A) — confirmed via `git check-ignore -v`.

---

## 7. What this screening does NOT establish

**No architecture is claimed scientifically superior from this result.**
This is one seed per model, on an uncleaned, non-leakage-safe dataset, with
no confidence intervals and no test-set evaluation. `efficientnet_b0`'s
lead (macro-F1 0.736 vs. `resnet18`'s 0.715 vs. `mobilenet_v3_small`'s
0.704) could plausibly be seed noise rather than a genuine architecture
effect — single-seed gaps of this size (0.02–0.03 macro-F1) are well within
the kind of run-to-run variance seen elsewhere in this project's own
multi-seed data. The correct purpose of this result is to **narrow the
candidate set and estimate cost** for the multi-seed comparison (§9), not
to select a final model.

---

## 8. Failures and fixes

**Failures**: none. All 4 training runs and all 4 evaluation runs completed
with exit code 0; `grep` across every log file found no
errors/exceptions/tracebacks beyond the expected, non-error
`FAIL_LEAKAGE_DETECTED` gate report (correctly overridden with an
audit-logged justification, exactly as designed).

**Fix made** (before training, not a mid-run patch): `train_image_model()`
in `src/doar/deep/trainers.py` did not record parameter count anywhere,
despite it being a required field for any model-comparison report. Added
`parameter_count` and `trainable_parameter_count` to the returned/saved
result dict — 3 new regression tests added in
`tests/test_trainer_regression.py::ParameterCountTests` (presence/
positivity, `small_cnn` vs. `resnet18` sanity-ordering, persistence to
`training_result.json`). Full suite re-run clean (225/225) after the change
and again after the full screening run.

---

## 9. Proposed multi-seed comparison (NOT run — proposal only)

- **Models**: same 4 (`small_cnn`, `mobilenet_v3_small`, `resnet18`,
  `efficientnet_b0`) — this screening found no reason to add or drop a
  candidate; `small_cnn` stays as the deliberately-weak baseline reference.
- **Seeds**: `42, 123, 2026` — the project's existing standard seed set,
  already used in earlier sessions' multi-seed runs, kept for consistency.
- **Protocol**: identical to §2 above, unchanged, for every seed.
- **Runtime estimate**: ~35 minutes × 3 seeds ≈ **~105 minutes (~1.75
  hours)** total GPU time, extrapolated linearly from this screening's
  measured per-model times (a reasonable first-order estimate; actual
  per-seed variance in epoch count via early stopping could shift this
  ±10–15%).
- **Storage estimate**: ~392 MB × 3 ≈ **~1.18 GB** (before any cleanup of
  `last.pt` duplicates, which could be dropped post-selection to roughly
  halve this).
- **Metrics + confidence intervals**: mean ± std macro-F1 across the 3
  seeds per model (already what `aggregate_and_select()` in
  `deep/compare.py` computes); given only 3 seeds, a std-based interval is
  more honest than a t-distribution CI — report mean±std, not a fabricated
  95% CI from n=3.
- **Calibration + uncertainty**: run `calibrate-fusion`-style temperature
  scaling (already available via `train_image_model`'s `--calibration`
  flag, skipped in this screening to save time) on each seed's best
  checkpoint; report before/after NLL/ECE/Brier per model, mean across
  seeds.
- **Model-selection criterion**: highest mean validation macro-F1 across
  the 3 seeds, tie-break on lower std (exactly `aggregate_and_select()`'s
  existing rule — reused, not reinvented). Test split stays locked until a
  separate, explicit, approved final-evaluation step.
- **Decision rule for this specific proposal**: only start the multi-seed
  run after either (a) explicit approval to proceed on the current uncleaned
  dataset with results clearly labeled non-final, or (b) the paused
  dataset/label-quality audit has produced a cleaned manifest to run on
  instead. Not run in this phase either way — awaiting direction.
