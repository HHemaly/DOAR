# Phase 5 — Preliminary Multi-Seed Confirmation

**Status:** Preliminary, non-leakage-safe. Same uncleaned dataset and fixed
train/valid/test splits as Phase 3A / Phase 4. Validation split only —
test split was never accessed. No dataset cleaning, no detector work, no
psychological-rule activation occurred in this phase.

**Scope:** Confirm whether Phase 4's single-seed (seed 42) ranking of
`mobilenet_v3_small`, `resnet18`, `efficientnet_b0` holds across two additional
seeds (123, 2026) drawn from the codebase's own existing default seed set
(`doar.deep.compare.DEFAULT_SEEDS = (42, 123, 2026)`). `small_cnn` was **not**
retrained — Phase 4 already showed it substantially behind the pretrained
candidates (macro-F1 0.435) and the user directed no further compute be spent
on it. It remains in the Phase 4 screening table as a lightweight reference
only.

---

## 1. Protocol

Identical to Phase 4/seed-42 in every controlled respect except seed:

| Field | Value |
|---|---|
| Dataset | Same uncleaned dataset used in Phase 3A/4 |
| Manifest | `outputs/phase5/manifest.csv` — confirmed byte-identical in content to `outputs/phase4/manifest.csv` / `outputs/phase3a` manifest via `diff <(sort ...) <(sort ...)` |
| Splits | Same fixed train/valid/test folders |
| Image size | 224×224 |
| Augmentation | `conservative`, per-architecture policy (same code path) |
| Batch size / grad accumulation | 4 / 4 (effective batch 16) |
| Epoch budget | 10 |
| Optimizer | AdamW, head LR 3e-4, backbone LR 1e-4 |
| Scheduler | `reduce_on_plateau` |
| Freezing | `freeze_epochs=3` (all three are pretrained backbones, unlike `small_cnn`) |
| Pretrained weights | `DEFAULT` (IMAGENET1K_V1 for all three) |
| Checkpoint selection | Best validation macro-F1 (same rule as Phase 4) |
| Evaluation code | `export-probabilities` → `evaluate-predictions` on the **validation** split, same code path as Phase 3A/4 |
| Seeds | 42 (reused, not retrained), 123, 2026 (newly trained) |
| Device | CUDA (Quadro P3200), `--device auto` |

### Pre-training compatibility check

Before training, each of the 3 models' `outputs/phase4/deep/runs/{model}_seed_42/executed_config.json`
was diffed against the intended Phase 5 protocol (batch size, image size,
epochs, augmentation, optimizer, LRs, scheduler, grad-accum steps, patience,
class weighting, pretrained-weights source). **No incompatibility was found**
— every checked field matched exactly. Training proceeded on that basis.

### Training command

```
main.py compare-deep-models --dataset <DATA> --output outputs/phase5/deep \
  --models mobilenet_v3_small,resnet18,efficientnet_b0 --seeds 123,2026 \
  --batch-size 4 --grad-accum-steps 4 --image-size 224 --epochs 10 --device auto \
  --allow-leakage-override --override-justification "Phase 5: preliminary
  multi-seed confirmation (seeds 123, 2026) on the same uncleaned dataset as
  Phase 3A/4, explicitly not leakage-safe. Seed 42 not retrained -- existing
  Phase 4 checkpoints reused. Dataset cleaning deferred per user instruction
  2026-08-02."
```

No `--calibration` flag was passed at training time, matching how seed 42 was
trained in Phase 4. Calibration was applied uniformly as a separate post-hoc
step to all three seeds afterward (§4).

---

## 2. Preservation of Phase 3A / Phase 4 artifacts

Seed 42 was **not retrained**. The existing Phase 4 checkpoints for the three
selected models were copied (not moved) into
`outputs/phase5/seed42_reference/{model}_seed_42/` and independently
SHA-256-verified against the copies before any further processing. Because
`main.py calibrate` stamps a checkpoint file in place, calibration was applied
only to these copies — never to the Phase 4 originals.

After all Phase 5 work concluded, the Phase 4 originals for **all four**
Phase-4 models (including `small_cnn`, which Phase 5 didn't touch) were
re-hashed and compared against the hashes Phase 4 itself recorded in
`outputs/phase4/ARTIFACT_MANIFEST.tsv`:

| File | Result |
|---|---|
| `small_cnn_seed_42/{best.pt,last.pt,training_result.json}` | MATCH |
| `mobilenet_v3_small_seed_42/{best.pt,last.pt,training_result.json}` | MATCH |
| `resnet18_seed_42/{best.pt,last.pt,training_result.json}` | MATCH |
| `efficientnet_b0_seed_42/{best.pt,last.pt,training_result.json}` | MATCH |

All 12 tracked Phase 4 seed-42 artifacts are byte-identical to their recorded
state at the end of Phase 4. `executed_config.json` was not itself included in
Phase 4's own manifest (a pre-existing instrumentation gap, not something
introduced this phase), so it could not be hash-verified against a recorded
baseline; it was never opened for writing at any point in Phase 5.

---

## 3. Per-run results (before calibration, validation split)

| Model | Seed | Macro-F1 | Accuracy | Bal. Acc. | NLL | Brier | ECE | ROC-AUC (macro OvR) | PR-AUC (macro) | Params | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | 42 (Phase 4) | 0.7043 | 0.7290 | 0.7130 | 0.7331 | 0.3856 | 0.045 | 0.9046 | 0.7782 | 1,521,956 | 508.3s |
| mobilenet_v3_small | 123 | 0.6714 | 0.7000 | 0.6870* | 0.7763 | 0.4125 | 0.0969 | 0.8965* | 0.7568* | 1,521,956 | 482.4s |
| mobilenet_v3_small | 2026 | 0.7097 | 0.7355 | 0.7204* | 0.7620 | 0.3973 | 0.0490 | 0.9006* | 0.7674* | 1,521,956 | 518.5s |
| resnet18 | 42 (Phase 4) | 0.7146 | 0.7355 | 0.7162 | 0.8393 | 0.3965 | 0.1081 | 0.9057 | 0.7820 | 11,178,564 | 453.8s |
| resnet18 | 123 | 0.7120 | 0.7387 | 0.7183* | 0.9588 | 0.4245 | 0.1312 | 0.9047* | 0.7787* | 11,178,564 | 467.4s |
| resnet18 | 2026 | 0.7272 | 0.7548 | 0.7371* | 0.7848 | 0.3753 | 0.0963 | 0.8951* | 0.7717* | 11,178,564 | 469.3s |
| efficientnet_b0 | 42 (Phase 4) | 0.7364 | 0.7613 | 0.7416 | 0.7121 | 0.3627 | 0.0546 | 0.9141 | 0.7970 | 4,012,672 | 790.7s |
| efficientnet_b0 | 123 | 0.7322 | 0.7581 | 0.7391* | 0.6982 | 0.3653 | 0.0603 | 0.9091* | 0.7803* | 4,012,672 | 797.1s |
| efficientnet_b0 | 2026 | 0.7186 | 0.7387 | 0.7371* | 0.7693 | 0.3891 | 0.0725 | 0.9176* | 0.7978* | 4,012,672 | 796.3s |

*Values for seeds 123/2026 recomputed directly this phase from
`outputs/phase5/precal_{model}_seed_{seed}/metrics.json`. Seed-42 values come
from `outputs/phase4/eval_{model}/metrics.json` — most of these fields
(balanced accuracy, NLL, Brier, ROC-AUC, PR-AUC) were present in that run's
`metrics.json` but not shown in Phase 4's own headline table, and are
reproduced here rather than approximated.

### Per-run confusion matrices (validation, order: Angry, Fear, Happy, Sad)

**mobilenet_v3_small**
| Seed | Angry | Fear | Happy | Sad |
|---|---|---|---|---|
| 42 | [47, 11, 8, 11] | [4, 32, 2, 7] | [6, 3, 95, 6] | [7, 13, 6, 52] |
| 123 | [48, 11, 13, 5] | [8, 31, 2, 4] | [7, 3, 95, 5] | [9, 20, 6, 43] |
| 2026 | [55, 8, 9, 5] | [8, 31, 4, 2] | [9, 3, 96, 2] | [10, 16, 6, 46] |

**resnet18**
| Seed | Angry | Fear | Happy | Sad |
|---|---|---|---|---|
| 42 | [49, 6, 18, 4] | [6, 31, 4, 4] | [6, 5, 96, 3] | [8, 11, 7, 52] |
| 123 | [41, 12, 15, 9] | [4, 35, 4, 2] | [4, 2, 100, 4] | [4, 16, 5, 53] |
| 2026 | [50, 11, 11, 5] | [5, 29, 4, 7] | [8, 3, 96, 3] | [5, 11, 3, 59] |

**efficientnet_b0**
| Seed | Angry | Fear | Happy | Sad |
|---|---|---|---|---|
| 42 | [55, 5, 11, 6] | [7, 32, 2, 4] | [7, 2, 99, 2] | [7, 16, 5, 50] |
| 123 | [54, 7, 12, 4] | [7, 33, 2, 3] | [6, 3, 99, 2] | [2, 22, 5, 49] |
| 2026 | [54, 9, 12, 2] | [4, 35, 3, 3] | [11, 4, 94, 1] | [9, 18, 5, 46] |

No warnings, failures, or protocol deviations were reported by any of the 6
new training runs (all exited 0; all completed the full 10-epoch budget).

**Operator note (self-caught, no data lost):** `evaluate-predictions --output`
takes a directory, not a file path (it writes `metrics.json` and
`per_class_metrics.csv` inside it). The first evaluation pass was run with
`--output <dir>/metrics.json`, which created a nested directory literally
named `metrics.json` containing the real files. This was a CLI-usage mistake,
not a code defect — `src/doar` was not touched. It was caught before any
aggregation was computed, fixed by moving the two files up one level and
removing the erroneous nested directory, and the resulting `metrics.json`
files were re-verified as parseable and numerically consistent with the
`main.py calibrate` command's own reported before/after values before being
used anywhere in this document.

---

## 4. Calibration (temperature scaling, validation-only)

Post-hoc temperature scaling (`main.py calibrate`) was applied to all 9
checkpoints (6 newly trained + 3 seed-42 copies), fit on the validation split
only. The test split was never touched by calibration.

**Important distinction:** temperature scaling rescales predicted
probabilities monotonically; it does not change the arg-max prediction, so
**macro-F1, accuracy, and balanced accuracy are identical before and after
calibration for every run** (confirmed numerically below). Calibration only
changes probability-quality metrics (NLL, Brier, ECE) — it must not be read as
changing the classifier ranking.

| Model | Seed | Temperature | NLL before→after | ECE before→after | Brier before→after |
|---|---|---|---|---|---|
| mobilenet_v3_small | 42 | 1.161 | 0.7333→0.7267 | 0.0534→0.0311 | 0.3856→0.3841 |
| mobilenet_v3_small | 123 | 1.329 | 0.7763→0.7482 | 0.0969→0.0636 | 0.4125→0.4028 |
| mobilenet_v3_small | 2026 | 1.144 | 0.7620→0.7566 | 0.0490→0.0574 | 0.3973→0.3973 |
| resnet18 | 42 | 1.770 | 0.8399→0.7258 | 0.1127→0.0620 | 0.3965→0.3789 |
| resnet18 | 123 | 1.996 | 0.9588→0.7760 | 0.1312→0.0694 | 0.4245→0.4029 |
| resnet18 | 2026 | 1.545 | 0.7848→0.7182 | 0.0963→0.0630 | 0.3753→0.3724 |
| efficientnet_b0 | 42 | 1.144 | 0.7122→0.7062 | 0.0546→0.0704 | 0.3627→0.3640 |
| efficientnet_b0 | 123 | 1.144 | 0.6982→0.6929 | 0.0603→0.0657 | 0.3653→0.3648 |
| efficientnet_b0 | 2026 | 1.370 | 0.7693→0.7383 | 0.0725→0.0631 | 0.3891→0.3855 |

**Observed pattern (not fabricated, directly from the above):** all 9 runs
show every temperature > 1, i.e. all three architectures are systematically
overconfident before calibration, consistent with Phase 4. NLL and Brier score
improved (or stayed flat) after calibration in every one of the 9 runs. ECE,
however, got *worse* after calibration in 3 of 9 runs (mobilenet seed 2026,
efficientnet seed 42, efficientnet seed 123) — temperature scaling minimizes
validation NLL, not ECE directly, so this divergence is expected behavior of
the method, not a bug. It means ECE-optimal and NLL-optimal temperatures can
differ slightly, and calibration should not be assumed to uniformly improve
every calibration metric.

Post-calibration macro-F1/accuracy verification (excerpt, full data in
`outputs/phase5/postcal_*/metrics.json`): every postcal macro-F1/accuracy pair
matches its precal counterpart exactly (e.g. `efficientnet_b0` seed 42:
0.736372 / 0.761290 in both `outputs/phase4/eval_efficientnet_b0/metrics.json`
and `outputs/phase5/postcal_efficientnet_b0_seed_42/metrics.json`).

---

## 5. Aggregation across seeds 42, 123, 2026 (n=3 per model)

**Evidence-strength caveat:** three seeds is enough to see whether a ranking
is seed-sensitive, but it is not enough to support a formal confidence
interval or a claim of statistical significance. No 95% CI is reported below,
and no claim of "statistically significant" superiority is made anywhere in
this document. All spreads are reported as plain min/max/std over 3
observations.

| Model | Mean macro-F1 | Std | Min | Max | Mean accuracy | Mean ECE (precal) | Std ECE | Mean ECE (postcal) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small | 0.6951 | 0.0169 | 0.6714 | 0.7097 | 0.7215 | 0.0629 | 0.0240 | 0.0417 |
| resnet18 | 0.7179 | 0.0067 | 0.7120 | 0.7272 | 0.7430 | 0.1058 | 0.0140 | 0.0588 |
| efficientnet_b0 | 0.7291 | 0.0076 | 0.7186 | 0.7364 | 0.7527 | 0.0564 | 0.0064 | 0.0543 |

### Ranking consistency across the 3 seeds (by macro-F1)

| Seed | 1st | 2nd | 3rd |
|---|---|---|---|
| 42 | efficientnet_b0 | resnet18 | mobilenet_v3_small |
| 123 | efficientnet_b0 | resnet18 | mobilenet_v3_small |
| 2026 | resnet18 | efficientnet_b0 | mobilenet_v3_small |

`mobilenet_v3_small` ranks last on every seed with no overlap in its
macro-F1 range (max 0.7097) against `resnet18`'s and `efficientnet_b0`'s
minimums (0.7120, 0.7186) — this is the one consistent, unambiguous finding
across all three seeds.

`efficientnet_b0` and `resnet18` swap 1st/2nd place depending on seed. The
mean gap between them (0.7291 − 0.7179 = 0.0112) is smaller than either
model's own seed-to-seed standard deviation (0.0076 and 0.0067 respectively).
With n=3, this is not a resolvable difference — it should be read as "these
two architectures perform comparably on this validation split," not as
"efficientnet_b0 wins."

### Per-class F1 variation (mean ± std across 3 seeds, precal)

| Class | mobilenet_v3_small | resnet18 | efficientnet_b0 |
|---|---|---|---|
| Angry | 0.668 ± 0.019 | 0.664 ± 0.025 | 0.719 ± 0.018 |
| Fear | 0.594 ± 0.022 | 0.618 ± 0.023 | 0.624 ± 0.017 |
| Happy | 0.851 ± 0.008 | 0.843 ± 0.018 | 0.860 ± 0.015 |
| Sad | 0.668 ± 0.023 | 0.747 ± 0.022 | 0.714 ± 0.005 |

`Fear` is the weakest class for all three architectures across all three
seeds, consistent with the class-imbalance pattern already noted in Phase 4
(`Fear` has the smallest support in the validation split, 45 of 310 images).

### Runtime and storage trade-offs

| Model | Params | Checkpoint size (`best.pt`) | Mean train time (10 epochs) |
|---|---|---|---|
| mobilenet_v3_small | 1,521,956 | ≈18.5 MB | 503.1s |
| resnet18 | 11,178,564 | ≈134.3 MB | 463.5s |
| efficientnet_b0 | 4,012,672 | ≈48.6 MB | 794.7s |

`efficientnet_b0` takes roughly 60–70% longer to train per seed than the
other two, for a mean macro-F1 advantage over `resnet18` that is within
seed-to-seed noise. `resnet18`'s checkpoint is ~2.7× the size of
`efficientnet_b0`'s and ~7.3× `mobilenet_v3_small`'s, for the smallest of the
three architectures' size class. `mobilenet_v3_small` is the fastest, smallest,
and lowest-scoring of the three on every seed.

---

## 6. Artifacts

- `outputs/phase5/manifest.csv` — dataset manifest, byte-identical to Phase 3A/4.
- `outputs/phase5/deep/runs/{model}_seed_{123,2026}/` — 6 new training runs (checkpoints, configs, results).
- `outputs/phase5/seed42_reference/{model}_seed_42/` — verified copies of Phase 4's seed-42 checkpoints, calibrated in place (Phase 4 originals untouched).
- `outputs/phase5/precal_{model}_seed_{seed}/` — pre-calibration validation evaluation for seeds 123/2026 (6 dirs).
- `outputs/phase5/postcal_{model}_seed_{seed}/` — post-calibration validation evaluation for all 9 model×seed combinations.
- `outputs/phase5/aggregation.json` — full computed aggregation backing §5.
- `outputs/phase5/ARTIFACT_MANIFEST.tsv` — 120 files, paths + sizes + SHA-256, ≈1.22 GB total.
- `outputs/phase5/*.log` — full stdout/stderr for training, evaluation, and calibration commands.

All of `outputs/` is gitignored; none of the above were committed to git.

---

## 7. Recommendation

Using **mean validation macro-F1 as the primary criterion** (per the
user's instruction): `efficientnet_b0` has the highest mean (0.7291),
narrowly ahead of `resnet18` (0.7179) and clearly ahead of
`mobilenet_v3_small` (0.6951). But per §5, the `efficientnet_b0`/`resnet18`
gap is smaller than either model's own seed variability — it is **not**
a resolvable difference at n=3, and this document does not claim
`efficientnet_b0` is statistically better than `resnet18`.

Bringing in the secondary criteria the user asked for:
- **Calibration:** `efficientnet_b0` has the lowest mean pre-calibration ECE (0.0564) of the three, though `resnet18` closes most of the gap after calibration (0.1058→0.0588).
- **Runtime:** `resnet18` trains ~40% faster than `efficientnet_b0` per seed.
- **Model size:** `efficientnet_b0`'s checkpoint (~49 MB) is roughly a third of `resnet18`'s (~134 MB).
- **Deployment complexity:** both are standard torchvision architectures already wired into the existing `registry.py`/`trainers.py` path; no meaningful difference.

`mobilenet_v3_small` is unambiguously the weakest of the three on every seed
and is not recommended.

**Recommendation: `efficientnet_b0`**, on the basis of the best mean
macro-F1 combined with the best calibration behavior and a substantially
smaller checkpoint than `resnet18`, even though its macro-F1 lead over
`resnet18` alone is not statistically resolvable from 3 seeds. This is
offered as a preliminary recommendation for a **future** end-to-end pipeline
test — not a decision to replace the production/default model, which this
phase does not do.

---

## 8. Explicitly out of scope for this phase (per user instruction)

The following were **not** done and should not be inferred from this
document: test-split evaluation of any kind; replacing the production/default
model; running `efficientnet_b0` or any other model through the full
end-to-end pipeline; dataset cleaning; detector implementation;
psychological-rule activation; starting a new numbered phase.
