# Local training workflow - Windows PowerShell

Commands use the environment's Python directly; activation is unnecessary.

## 1. Environment

```powershell
cd C:\path\to\DOAR\doar_v3
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[ml,cv,deep,embeddings,ui,dev]"
```

For NVIDIA CUDA, install the matching PyTorch wheel from the official PyTorch
selector before installing the remaining extras.

## 2. Dataset and hardware readiness

```powershell
.\.venv\Scripts\python.exe main.py validate-dataset `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --output "outputs\readiness\dataset"

.\.venv\Scripts\python.exe main.py check-training-readiness `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --output "outputs\readiness\hardware"
```

## 3. Manifest and objective features

```powershell
.\.venv\Scripts\python.exe main.py build-manifest `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --output "outputs\dataset\manifest.csv"

.\.venv\Scripts\python.exe main.py extract-features `
  --manifest "outputs\dataset\manifest.csv" `
  --output "outputs\features\v3_1"
```

## 4. CPU smoke and feature baselines

```powershell
.\.venv\Scripts\python.exe main.py train-image-model `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --model small_cnn --output "outputs\experiments\cpu_smoke" `
  --seed 42 --epochs 1 --batch-size 4 --image-size 128 --device cpu

.\.venv\Scripts\python.exe main.py compare-models `
  --features "outputs\features\v3_1\features.csv" `
  --output "outputs\experiments\objective_features" `
  --seeds "42,123,2026"
```

## 5. Deep model, embeddings, and primary fusion

```powershell
.\.venv\Scripts\python.exe main.py train-image-model `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --model resnet18 --output "outputs\experiments\resnet18_seed42" `
  --seed 42 --epochs 50 --batch-size 16 --device auto

# Equivalent configuration-driven run:
.\.venv\Scripts\python.exe main.py train-image-model `
  --config "configs\training\resnet18.toml" `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing"

.\.venv\Scripts\python.exe main.py extract-embeddings `
  --manifest "outputs\dataset\manifest.csv" `
  --backbone resnet18 --output "outputs\embeddings\resnet18" --device auto

.\.venv\Scripts\python.exe main.py train-fusion-model `
  --features "outputs\features\v3_1\features.csv" `
  --embeddings "outputs\embeddings\resnet18\embeddings.npz" `
  --output "outputs\experiments\primary_fusion_resnet18" `
  --seeds "42,123,2026"

# Configuration-driven fusion:
.\.venv\Scripts\python.exe main.py train-fusion-model `
  --config "configs\training\primary_fusion.toml"
```

Every configuration-driven run writes `resolved_config.json` and its SHA-256.
Explicit CLI values override TOML values. Unknown TOML fields are rejected.

Expected outputs include checkpoints, validation result JSON, prediction CSV,
embedding metadata, failures, and validation leaderboards. Accuracy is not
estimated in advance.

## 6. Case inference and reports

```powershell
.\.venv\Scripts\python.exe main.py analyze-image `
  --image "C:\path\to\drawing.png" `
  --emotion-checkpoint "outputs\experiments\resnet18_seed42\best.pt" `
  --output "outputs\cases\case_001"

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## 7. Final test

Do not run until preprocessing, features, model family, fusion, calibration,
and configuration are frozen. The final command requires both unlock flags and
an initiator code. It must be run exactly once for the frozen experiment.
