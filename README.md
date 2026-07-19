# DOAR v3

An evidence-traceable research pipeline for objective analysis of children's
drawings. Outputs are non-diagnostic and require professional review.

> **Full local workflow:** see [`docs/LOCAL_WORKFLOW_WINDOWS.md`](docs/LOCAL_WORKFLOW_WINDOWS.md)
> for the complete, verified PowerShell command sequence (dataset validation →
> features → deep training → embeddings → fusion → calibration → late fusion →
> ablation → explainability → thesis outputs → locked final test), including how
> to choose the correct PyTorch/CUDA wheel for a Quadro P3200 **after** checking
> `nvidia-smi` (no CUDA command is hardcoded as verified).

## Pipeline capabilities (CLI)

Dataset & leakage: `validate-dataset`, `check-training-readiness`,
`build-manifest` (blocks cross-split exact/near-duplicate/subject leakage).
Features & embeddings: `extract-features`, `extract-embeddings`
(`finetuned:<ckpt>` for penultimate-layer embeddings).
Models: `train-image-model`, `compare-deep-models`, `train-feature-model`,
`train-fusion-model`, `compare-embeddings`.
Calibration & fusion: `calibrate`, `calibrate-fusion`, `export-probabilities`,
`train-late-fusion`, `apply-late-fusion` (all sample_id-aligned, validation-only).
Evaluation & thesis: `evaluate` (locked test), `evaluate-predictions`,
`run-ablation`, `generate-thesis-outputs`.
Explainability: `explain-features` (tabular importance),
`explain-gradcam` (visual attention) — kept strictly separate.
Case tools: `analyze-image`, `predict-image`, `qa` (EN/AR),
`review-agreement`, `ingest-psychology-pdf`, `gpu-smoke`.

All model selection, calibration, weighting and stacking use the **validation**
split; the **test** split is locked behind `--unlock-test
--confirm-final-evaluation` with an audit log.

## Quick start (PowerShell)

```powershell
cd doar_v3
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ml,cv,dev]"
python main.py analyze-image --image "C:\path\drawing.png" --output "outputs\case_001"
python -m unittest discover -s tests -v
python -m streamlit run streamlit_app.py
```

The dataset split name is `valid`, never `val`:

```powershell
python main.py build-manifest `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --output "outputs\dataset\manifest.csv"

python main.py extract-features `
  --manifest "outputs\dataset\manifest.csv" `
  --output "outputs\features\v3_1"
```

Real training metrics are intentionally absent until the local dataset audit is
run and the locked-test policy is followed.
