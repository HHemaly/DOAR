# DOAR — Local Workflow (Windows / PowerShell)

Every command below has been verified to parse and (where marked **CPU-verified**)
run on synthetic data. Commands marked **needs dataset** or **needs GPU** require
your real `Combined_Drawing` dataset and/or an NVIDIA GPU and have NOT been
executed in this environment — no results are fabricated.

Dataset layout expected (4 classes, `valid` — not `val`):
```
Combined_Drawing/
  train/{Angry,Fear,Happy,Sad}/*.jpg
  valid/{Angry,Fear,Happy,Sad}/*.jpg
  test/ {Angry,Fear,Happy,Sad}/*.jpg   # locked; never used for selection/calibration/stacking
```

---

## 1. Environment

```powershell
python --version                 # expect 3.11.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ml,cv,dev]"    # numpy, Pillow, scikit-learn, opencv, pytest, ruff
```

### PyTorch / CUDA — verify before installing (Quadro P3200)

> **Do not blindly copy a CUDA wheel command.** Your Quadro P3200 is a Pascal GPU
> (compute capability 6.1), which current PyTorch still supports, and Python 3.11
> is supported by torch ≥ 2.2. **But the correct wheel depends on your installed
> driver.** First check:
> ```powershell
> nvidia-smi        # note the "CUDA Version" shown top-right (driver's max CUDA)
> ```
> Then pick the matching wheel from the official selector
> <https://pytorch.org/get-started/locally/>. If your driver supports CUDA ≥ 12.1,
> the cu121 wheel is the usual choice:
> ```powershell
> # Only after confirming nvidia-smi shows CUDA >= 12.1:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> If `nvidia-smi` fails or shows an older CUDA, use the CPU wheel (`pip install torch
> torchvision`) or the cu118 wheel per the selector. Verify with:
> ```powershell
> python main.py gpu-smoke --output outputs\gpu     # reports cuda_used honestly
> ```
> `gpu-smoke` prints `cuda_used: true` only if a forward+backward+step+inference
> actually ran on CUDA. If it prints `false`, the GPU path is NOT verified.

Optional extras: `pip install -e ".[deep]"` (torch stack), `".[embeddings]"`
(open_clip), `".[ingest]"` (pypdf), `".[ui]"` (streamlit).

---

## 2. Dataset validation, manifest, leakage gate

```powershell
$DATA = "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing"
python main.py validate-dataset       --dataset $DATA --output outputs\validate      # needs dataset
python main.py check-training-readiness --dataset $DATA --output outputs\readiness    # needs dataset
python main.py build-manifest         --dataset $DATA --output outputs\manifest.csv  # needs dataset
```
`build-manifest` writes a leakage report (exact + near-duplicate cross-split,
conflicting labels, and subject-level leakage if a `subject_id` column exists).
Training/extraction commands enforce the gate and **block on leakage** unless you
pass `--allow-leakage-override --override-justification "<reason>"` (logged to an
audit file).

---

## 3. Objective features

```powershell
python main.py extract-features --manifest outputs\manifest.csv --output outputs\features   # needs dataset
```

---

## 4. Deep image models (GPU recommended; 6 GB-safe defaults)

```powershell
# single model from a TOML config (edit batch_size=4 for 6 GB):
python main.py train-image-model --config configs\training\resnet18.toml               # needs GPU

# multi-seed comparison of baseline + transfer models (batch 4, mixed precision):
python main.py compare-deep-models --dataset $DATA --output outputs\deep --batch-size 4  # needs GPU
```
Selection is by mean **validation** macro-F1; the test split is never used.

---

## 5. Embeddings (generic + fine-tuned) and representation comparison

```powershell
# generic pretrained backbone:
python main.py extract-embeddings --manifest outputs\manifest.csv --backbone resnet18 `
    --output outputs\emb_generic                                                        # needs GPU

# fine-tuned emotion checkpoint (penultimate layer):
python main.py extract-embeddings --manifest outputs\manifest.csv `
    --backbone "finetuned:outputs\deep\runs\resnet18_seed_42\best.pt" `
    --output outputs\emb_ft                                                             # needs GPU

# 5-way comparison: objective / generic / finetuned / obj+generic / obj+finetuned:
python main.py compare-embeddings --features outputs\features\features.csv `
    --generic outputs\emb_generic\embeddings.npz `
    --finetuned outputs\emb_ft\embeddings.npz --output outputs\emb_compare              # needs dataset
```

---

## 6. Primary fusion + calibration

```powershell
python main.py train-fusion-model --config configs\training\primary_fusion.toml         # needs dataset
python main.py calibrate-fusion --bundle outputs\...\fusion.joblib `
    --features outputs\features\features.csv `
    --embeddings outputs\emb_generic\embeddings.npz --output outputs\fusion_cal          # needs dataset
```
Calibration is fitted on **validation only**; raw probabilities are preserved and
a reliability diagram + before/after ECE/Brier/NLL are written.

---

## 7. Probability export, late fusion, stacking (aligned by sample_id)

```powershell
python main.py export-probabilities --model outputs\...\model.joblib `
    --features outputs\features\features.csv --output outputs\exports\m1.json `
    --splits train,valid                                                                # needs dataset
python main.py train-late-fusion --base outputs\exports\m1.json outputs\exports\m2.json `
    --method validation_weighted_late_fusion --output outputs\late                      # CPU-verified (given exports)
python main.py apply-late-fusion --model outputs\late\late_fusion_model.json `
    --base outputs\exports\m1.json outputs\exports\m2.json --split valid --output outputs\late_applied
```
Stacking (`--method logistic_probability_meta`) requires genuine OOF `fold_id` on
the train exports.

---

## 8. Ablations, explainability, thesis outputs

```powershell
python main.py run-ablation --features outputs\features\features.csv --output outputs\ablation  # needs dataset
python main.py explain-features --model outputs\...\model.joblib `
    --features outputs\features\features.csv --output outputs\explain_feat                # needs dataset
python main.py explain-gradcam --image path\to\drawing.jpg `
    --checkpoint outputs\deep\runs\resnet18_seed_42\best.pt --output outputs\gradcam       # needs GPU
python main.py generate-thesis-outputs --output outputs                                   # CPU-verified
```
`generate-thesis-outputs` builds figures ONLY from experiment outputs that exist;
each figure is paired with its source data. Nothing is generated from invented
results.

---

## 9. Final test (locked — run once, deliberately)

```powershell
python main.py evaluate --manifest outputs\manifest.csv `
    --checkpoint outputs\deep\runs\<winner>\best.pt --split test `
    --unlock-test --confirm-final-evaluation --initiated-by "Ahmed"                       # needs dataset
```
Requires BOTH confirmation flags and writes an audit-log entry. Only run after the
architecture, config, preprocessing and model-selection criteria are frozen.

---

## 10. Single-case analysis, reports, Q&A, psychologist review

```powershell
python main.py analyze-image --image path\to\drawing.jpg --output outputs\case            # CPU-verified
python main.py qa --analysis outputs\case\analysis.json --question "what colours?" --language en
python main.py review-agreement --master outputs\review\review_master.csv --output outputs\agreement
python main.py ingest-psychology-pdf --pdf resources\psychology_sources\notes.pdf `
    --output resources\psychology_sources\draft_rules.json                                # needs pypdf
```
`review-agreement` reports "unavailable" until **real** psychologist reviews are
collected (synthetic/incomplete reviews are excluded; nothing is fabricated).

---

## 11. Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
Torch-dependent tests skip cleanly when torch is not installed.
