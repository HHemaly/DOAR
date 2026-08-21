# E1-A Thesis Sync Block

**Status: DEVELOPMENT / VALIDATION SCREENING.** NOT a locked Test result.
No model has been selected. No E1-B/E1-C work has started.

**One-line summary:** E1-A screens 9 pretrained visual representations
(5 CNNs, ViT-B/16, and 3 foundation encoders — CLIP/DINOv2/SigLIP2) for a
four-class children's-drawing emotion classification task under the
frozen, leakage-controlled T0 split (`checkpoint/t0-clean-split-v1`,
Train=2,599/Valid=284/Test=512-locked). A CPU-only preliminary pass
(E1-A1, secondary evidence) has produced real, partial results; the
primary, standardized methodology (E1-A2 — identical frozen-encoder →
one cached embedding per image → shared linear head, for every candidate,
with no augmentation) is implemented and portable to Colab GPU, but has
not yet been executed at full scale on real data — this phase's
deliverable is the GPU-portable infrastructure itself, not a finished
screening result.

**E1-A1 (CPU, preliminary, secondary) — real numbers so far, not
promoted to primary:** MobileNetV3-Small 0.748, CLIP ViT-B/32 0.724,
ConvNeXt-Tiny 0.713, DenseNet121 0.660, DINOv2 ViT-S/14 0.640, ResNet18
0.638, EfficientNet-B0 0.637 (Validation Macro-F1, best early-stopped
epoch). ViT-B/16 was still running when GPU migration began (~32.8
min/epoch on CPU, calibrated). SigLIP2 ViT-B/16 was omitted on CPU
(~19.5s/image measured, ~15.6h projected for embedding extraction alone)
— re-attempted under E1-A2 on GPU, not assumed infeasible there. Full
detail: `../E1A_CPU_PRELIMINARY_STATUS.md`.

**E1-A2 (GPU-portable, primary methodology) — infrastructure status:**
implemented and unit-tested (14 new tests, `tests/test_e1a2_common.py`) —
dataset-root/output-root/device portability, frozen-backbone penultimate-
embedding extraction with cache invalidation keyed by model+manifest-SHA+
split+weights+preprocessing, identical linear-probe training/early-
stopping/logging for every candidate, resumable via `RUN_MANIFEST.json`.
Validated end-to-end with a tiny (16 train / 8 valid image, 2-epoch)
smoke run on CPU — proves the full pipeline (verify → extract → train →
tables → figures) works, but those smoke numbers are deliberately not
reported here or anywhere as a result (too small to be meaningful, and
were removed from `tables/`/`figures/` after the smoke run). **No E1-A2
GPU run has been executed on the real 2,599/284 split as of this
document.**

**Limitation to disclose, always:** E1-A1's two different per-family
training strategies (augmented full-image training for CNN/ViT vs.
cached-embedding linear probes for foundation models) make its own
cross-family comparison less clean than E1-A2's single standardized
methodology — this is exactly why E1-A2 exists and is primary.

**Next step:** run `run_e1a_all.py` on Colab GPU against the real
2,599/284 split for all 9 candidates (`notebooks/DOAR_GPU_RUNNER.ipynb`),
producing the real `E1A_T2–T6` tables and `E1A_F1–F6` figures, before any
E1-A architecture ranking or E1-B shortlist is finalized.
