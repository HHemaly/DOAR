# E1 — Visual Representation and Fine-Tuning

**E1-A: Controlled Architecture Screening.** Which pretrained visual
representation provides the strongest useful representation of children's
drawings for the four-class Angry/Fear/Happy/Sad task, under the same
leakage-controlled dataset (T0, `checkpoint/t0-clean-split-v1`), training
protocol, and computational budget?

**Status: DEVELOPMENT / VALIDATION SCREENING.** The locked 512-image Test
set is never touched in this phase. Two methodologies exist here —
**E1-A1** (CPU-only preliminary, secondary evidence, see
`E1A_CPU_PRELIMINARY_STATUS.md`) and **E1-A2** (the PRIMARY, GPU-portable
standardized linear-probe benchmark — see `PROTOCOL.md` §8). See
`thesis/THESIS_SYNC_BLOCK.md` for the thesis-ready summary.

## Directory guide

- `PROTOCOL.md` — full methodology for both E1-A1 (§§1–7) and E1-A2 (§8, primary).
- `E1A_CPU_PRELIMINARY_STATUS.md` — honest status of the CPU dev run (E1-A1).
- `config/` — exact training configuration.
- `scripts/`:
  - **E1-A2 (primary, GPU-portable)**: `prepare_e1_dataset.py` (T0 verification via `--dataset-root`), `e1a2_common.py` (portable embedding-extraction + linear-probe library), `run_e1a_all.py` (one-command runner, resumable), `build_e1a2_tables.py`, `make_e1a2_figures.py`.
  - **E1-A1 (preserved, CPU dev)**: `build_e1a_manifest.py`, `e1a_common.py`, `run_e1a_candidate.py`, `calibrate_throughput.py` — unmodified, not the primary path.
- `raw/` — E1-A1 per-candidate result JSON/epoch history (preserved).
- `checkpoints/`, `embeddings/` — **gitignored** (large artifacts; E1-A2 writes these to `--output-root`, not the repo).
- `tables/` — `E1A_T1` (E1-A1, preserved) + `E1A_T2–T6` (E1-A2, primary, populated after a real GPU run).
- `figures/` — `E1A_F1–F6` (PNG+PDF+source CSV), populated after a real GPU run.
- `statistics/`, `error_analysis/` — derived comparison statistics / confusion matrices.
- `thesis/` — THESIS_SYNC_BLOCK.md.
- `requirements-e1.txt` — E1-specific Python dependencies (never force-downgrades Colab's CUDA PyTorch).

See also `notebooks/DOAR_GPU_RUNNER.ipynb` (repo root) — the Colab launcher; contains no E1 scientific logic of its own, only calls into the scripts here.

## Quick start

**Verify the frozen T0 split on any machine:**
```
python scripts/prepare_e1_dataset.py --dataset-root <path-to-Combined_Drawing> --verify-only
```

**Run E1-A2 (the primary methodology), locally or on Colab:**
```
python scripts/run_e1a_all.py \
  --dataset-root <path> --output-root <path> \
  --device auto --seed 42 --max-epochs 30 --patience 5 --resume
```

**Tiny smoke validation only (never the full experiment):**
```
python scripts/run_e1a_all.py --dataset-root <path> --output-root <path> \
  --device cpu --smoke --models mobilenet_v3_small,clip_vit_b32
```

**E1-A1 (preserved CPU dev harness, secondary):**
```
python scripts/build_e1a_manifest.py
python scripts/run_e1a_candidate.py --model resnet18
```

## Constraints honored throughout

No commit/push until approved. No Test-set evaluation. No E6/A1/A2/
scientific-rule modification. No Gemini/API calls. Seed=42 throughout. T0
manifest SHA-256 verified before every run; the frozen split is never
regenerated or edited.
