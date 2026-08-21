# E1-A1 — Transfer-Head Development Screening (CPU, preliminary)

**Status: DEVELOPMENT SCREENING. Secondary/supporting evidence only.**
**Not promoted to a final E1-A result.** The primary E1-A methodology is
**E1-A2 — Standardized Frozen Linear-Probe Benchmark** (see `PROTOCOL.md`
and `tables/E1A_T2_standardized_linear_probe.csv`), run on GPU (Colab)
using the portable infrastructure added after this preliminary phase.

## Why this run exists, and why it was superseded

This was the first working E1-A implementation, built and run on the
local CPU-only development machine (no CUDA available) before GPU access
was arranged. It used two different training strategies depending on
model family — full-image forward passes with training augmentation every
epoch for the 6 CNN/ViT candidates (`e1a_common.py::run_epoch_loop` fed by
`ManifestImageDataset`), vs. cached-embedding linear-head training for the
3 foundation candidates (CLIP/DINOv2/SigLIP2). That split-methodology
design, and its resource accounting (which under-reported foundation
models' true encoder cost by citing only the ~2–3k-parameter linear head),
are both corrected in E1-A2's single standardized methodology. This run
is preserved as real, honest development evidence — not deleted, not
silently reused as if it were the primary result.

## Environment

CPU-only (no CUDA), PyTorch 2.13.0+cpu, torchvision 0.28.0+cpu, 8 CPU
cores, hardcoded to the local Windows dataset path and `device="cpu"` —
exactly the portability gaps E1-A2's infrastructure fixes.

## Results as of this writing

| Model | Status | Best epoch | Epochs completed | Val Macro-F1 | Sec/epoch |
|---|---|---|---|---|---|
| MobileNetV3-Small | ✅ complete | 11 | 17 (early-stopped) | 0.748 | 74.6 |
| CLIP ViT-B/32 | ✅ complete | 29 | 30 (not early-stopped) | 0.724 | 0.42 |
| ConvNeXt-Tiny | ✅ complete | 12 | 18 (early-stopped) | 0.713 | 985.8 |
| DINOv2 ViT-S/14 | ✅ complete | 9 | 15 (early-stopped) | 0.640 | 0.15 |
| ResNet18 | ✅ complete | 21 | 27 (early-stopped) | 0.638 | 219.8 |
| EfficientNet-B0 | ✅ complete | 8 | 14 (early-stopped) | 0.637 | 223.8 |
| DenseNet121 | ✅ complete | 12 | 18 (early-stopped) | 0.660 | 2174.6 |
| ViT-B/16 | ⏳ unfinished | — | — | — | (calibrated: ~32.8 min/epoch projected) |
| SigLIP2 ViT-B/16 | ❌ omitted | — | — | — | measured ~19.5s/image → ~15.6h projected for embedding extraction alone; disproportionate on CPU-only hardware, reported per the throughput/cost stop rule rather than silently skipped |

**No fabricated numbers**: DenseNet121 and ViT-B/16 were still running in
the background when GPU migration began; their CPU results are left blank
here, not estimated or backfilled. If they finish before this file is next
updated, their real numbers will be added — never guessed.

## Methodological limitations of this preliminary run

1. **Two different training strategies for different model families**
   (full-image augmented training vs. cached-embedding linear probe) —
   not a fair apples-to-apples representation comparison in the sense
   E1-A2 provides; CNN/ViT models here also received training-time
   augmentation the foundation models did not.
2. **Resource accounting was incomplete**: CLIP/DINOv2 results recorded
   only their ~2–3k-parameter trainable head as "n_trainable" without
   equally prominent reporting of the full frozen encoder's real parameter
   count (86M for CLIP ViT-B/32, 21M for DINOv2 ViT-S/14) alongside it —
   corrected in E1-A2's resource-accounting table (`E1A_T6_resource_
   comparison.csv`).
3. **Hardcoded Windows absolute paths** (`C:\Users\ZZ01G7865\...`) baked
   directly into the working manifest, non-portable to Colab or any other
   machine — corrected by E1-A2's `--dataset-root` + T0 `relative_path`
   resolution.
4. **CPU-only**: no GPU utilization possible; ConvNeXt-Tiny and ViT-B/16
   in particular were measured as disproportionately slow (up to ~16.4h
   worst-case projected for ViT-B/16 alone) — the direct motivation for
   migrating E1-A execution to Colab GPU.

## Why GPU (Colab) execution was chosen for the primary result

The CPU-only local machine has no CUDA device. Full-image forward passes
through larger backbones (ConvNeXt-Tiny, ViT-B/16) and any foundation
model with a real classification-relevant preprocessing pipeline are slow
enough on CPU alone (measured, not assumed — see the calibration table
above) that completing all 9 candidates under a fair, identical epoch
budget within a reasonable wall-clock time requires GPU acceleration.
E1-A2 was designed from the start to be portable (dataset-root/output-root/
device flags, resumable, cache-keyed by configuration hash) specifically
so it can run unmodified on Colab and be reproduced from any machine.

## Preserved artifacts (unmodified, not deleted)

`scripts/e1a_common.py`, `scripts/run_e1a_candidate.py`,
`scripts/calibrate_throughput.py`, `scripts/build_e1a_manifest.py`,
`raw/*_result.json`, `raw/*_epoch_history.json`, `raw/*_smoke.json`,
`raw/throughput_calibration.json`, `embeddings/clip_vit_b32_*.npz`,
`embeddings/dinov2_vits14_*.npz`, `logs/sequential_run.log` — all kept
exactly as produced by this preliminary phase.
