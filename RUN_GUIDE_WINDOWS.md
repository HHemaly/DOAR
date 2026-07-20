# DOAR — Verified Windows Run Guide (P3200, 6 GB)

This is the exact, copy‑paste PowerShell workflow, verified against the actual
code and CLI (`python main.py --help` → 32 commands). It keeps the **test split
locked** until every model‑selection decision is frozen, and resolves dataset
leakage by **removing** leaked images (never by override).

- Repo root: `C:\Users\Ahmed\Documents\Hoda\DOAR_3\DOAR-main\DOAR-main`
- Dataset:   `C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`
  (expected `train\ valid\ test\`, each with `Angry Fear Happy Sad`)

Set these once per PowerShell session:

```powershell
cd "C:\Users\Ahmed\Documents\Hoda\DOAR_3\DOAR-main\DOAR-main"
$DATA = "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing"
$OUT  = "outputs"
```

---

## 1. Virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install libraries

Install **PyTorch first**, matched to your GPU, so the extras don't pull a wrong
build. Check your driver's CUDA version, then use the official selector
(https://pytorch.org/get-started/locally/). The P3200 (Pascal, CC 6.1) is
supported by current torch.

```powershell
nvidia-smi     # note "CUDA Version" (top-right). If >= 12.1, use the cu121 wheel:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# (If nvidia-smi fails or shows older CUDA, use the cu118 or CPU wheel per the selector.)

# Then the project + all optional extras (torch already satisfied, not replaced):
python -m pip install -e ".[ml,cv,dev,deep,embeddings,ui,ingest]"
```

`ml`=scikit-learn/pandas · `cv`=opencv · `deep`=torch/torchvision · `embeddings`=
open-clip · `ui`=streamlit · `ingest`=pypdf · `dev`=pytest/ruff/mypy.

## 3. Verify environment (Python, imports, tests, CUDA, GPU, dataset, output)

```powershell
python --version                                  # expect 3.11.x
python -c "import numpy, PIL, sklearn, torch, torchvision; print('imports OK, cuda=', torch.cuda.is_available())"
python -m compileall src main.py streamlit_app.py
python -m ruff check src tests main.py streamlit_app.py
python -m unittest discover -s tests -p "test_*.py"      # full suite

# Real GPU verification (forward+backward+step+inference; reports peak VRAM).
python main.py gpu-smoke --output "$OUT\gpu"
# Only if it prints  "cuda_used": true  is the GPU path verified.

# Dataset structure + readiness + writable output.
python main.py validate-dataset --dataset "$DATA" --output "$OUT\dataset_check"
python main.py check-training-readiness --dataset "$DATA" --output "$OUT\readiness"
```

If `gpu-smoke` shows `cuda_used: false`, training will run on CPU (very slow) —
stop and fix the torch/CUDA install before continuing.

## 4. What is already complete / which outputs are valid

- **Code is complete and tested** (166 unit tests pass; Ruff clean). All 32 CLI
  commands parse. Preprocessing is consistent across training/inference; the
  final‑test guard is enforced; leakage detection + provenance are wired.
- **No trained models, metrics, embeddings, reports or thesis figures exist
  yet** — those are produced by *your* runs below. Nothing is pre‑fabricated.
- Valid outputs are only the ones created by the commands you run; each writes
  its own folder under `outputs\` (see §9).

## 5. Blocking issues fixed for you

- `resolve-leakage` command added: builds a clean, leakage‑free ImageFolder
  dataset so deep training (which needs a directory) can proceed **without
  override**. (Committed on branch `claude/final-pretraining-fixes`.)
- `resnet18.toml` uses `batch_size = 16`, too large for 6 GB — the commands below
  override with `--batch-size 4` (+ gradient accumulation).

## 6. Resolve leakage FIRST (no override)

```powershell
python main.py resolve-leakage --dataset "$DATA" --output "$OUT\leakage"
```

Read `outputs\leakage\leakage_report.json`:

- **`leakage_status: PASS`** → the dataset is clean. Use the ORIGINAL dataset for
  every step:  `$DS = $DATA`
- **`leakage_status: FAIL_LEAKAGE_DETECTED`** → leaked images were quarantined and
  a clean ImageFolder was materialized. Use it for every step:
  `$DS = "$OUT\leakage\clean_dataset"`
  Inspect `quarantine.csv` to see what was removed. **Do not** pass
  `--allow-leakage-override` for thesis experiments — overridden results are not
  scientifically valid.

Pick the split source now (one line):

```powershell
$rep = Get-Content "$OUT\leakage\leakage_report.json" | ConvertFrom-Json
$DS = if ($rep.leakage_ok) { $DATA } else { "$OUT\leakage\clean_dataset" }
"Using dataset: $DS"
```

## 7. Full pipeline (correct order; test split stays locked)

### 7.1 Manifest + objective features (train/valid only)
```powershell
python main.py build-manifest    --dataset "$DS" --output "$OUT\manifest.csv"
python main.py extract-features  --manifest "$OUT\manifest.csv" --output "$OUT\features"
```

### 7.2 Objective baseline models (multi‑seed, validation‑selected)
```powershell
python main.py train-feature-model --features "$OUT\features\features.csv" `
  --output "$OUT\feature_models" --seeds 42,123,2026
```
Leaderboard: `outputs\feature_models\validation_leaderboard.json` (selection on
validation macro‑F1; `test_used=false`).

### 7.3 Deep‑model comparison across seeds (6 GB‑safe)
```powershell
python main.py compare-deep-models --dataset "$DS" --output "$OUT\deep" `
  --models "small_cnn,mobilenet_v3_small,resnet18,efficientnet_b0" `
  --seeds 42,123,2026 --batch-size 4 --grad-accum-steps 4 --image-size 224 `
  --epochs 50 --device auto
```
Winner (by mean validation macro‑F1): `outputs\deep\deep_comparison.json`.
Per‑run checkpoints: `outputs\deep\runs\<model>_seed_<seed>\best.pt`.
This is long (4 models × 3 seeds). **Resume a single interrupted model** with:
```powershell
python main.py train-image-model --config configs\training\resnet18.toml `
  --dataset "$DS" --output "$OUT\deep\runs\resnet18_seed_42" --batch-size 4 `
  --grad-accum-steps 4 --resume "$OUT\deep\runs\resnet18_seed_42\last.pt"
```

### 7.4 Embeddings (generic + fine‑tuned from the winning checkpoint)
```powershell
# generic pretrained backbone:
python main.py extract-embeddings --manifest "$OUT\manifest.csv" `
  --backbone resnet18 --device auto --batch-size 16 --output "$OUT\emb_generic"

# fine-tuned emotion checkpoint (use the winning run's best.pt):
python main.py extract-embeddings --manifest "$OUT\manifest.csv" `
  --backbone "finetuned:$OUT\deep\runs\resnet18_seed_42\best.pt" `
  --device auto --batch-size 16 --output "$OUT\emb_finetuned"
```

### 7.5 Primary fusion (validation‑selected) + calibration
```powershell
python main.py train-fusion-model `
  --features "$OUT\features\features.csv" `
  --embeddings "$OUT\emb_generic\embeddings.npz" `
  --output "$OUT\fusion" --seeds 42,123,2026
# calibrate the WINNING bundle on validation (raw preserved):
$win = (Get-Content "$OUT\fusion\fusion_leaderboard.json" | ConvertFrom-Json).winner.checkpoint
python main.py calibrate-fusion --bundle "$win" `
  --features "$OUT\features\features.csv" `
  --embeddings "$OUT\emb_generic\embeddings.npz" --output "$OUT\fusion_calibrated"
```
Reliability diagram + before/after ECE/Brier/NLL: `outputs\fusion_calibrated\`.

### 7.6 Ablations + representation comparison
```powershell
python main.py run-ablation --features "$OUT\features\features.csv" --output "$OUT\ablation" --seeds 42,123,2026
python main.py compare-embeddings --features "$OUT\features\features.csv" `
  --generic "$OUT\emb_generic\embeddings.npz" `
  --finetuned "$OUT\emb_finetuned\embeddings.npz" --output "$OUT\emb_compare"
```

### 7.7 Explainability (kept separate: tabular vs visual)
```powershell
$featmodel = (Get-ChildItem "$OUT\feature_models\runs" -Recurse -Filter model.joblib | Select-Object -First 1).FullName
python main.py explain-features --model "$featmodel" --features "$OUT\features\features.csv" --output "$OUT\explain_features"
python main.py explain-gradcam --image "$DS\valid\Happy\<pick-one>.png" `
  --checkpoint "$OUT\deep\runs\resnet18_seed_42\best.pt" --output "$OUT\gradcam"
```

### 7.8 LOCKED final test — run ONCE, only after freezing all choices
Freeze the architecture, config, preprocessing and the selected winner first.
Build a manifest that includes test, then evaluate the winner with all three
confirmation flags (writes an audit‑log entry):
```powershell
python main.py build-manifest --dataset "$DS" --output "$OUT\manifest_full.csv"
python main.py evaluate --manifest "$OUT\manifest_full.csv" `
  --checkpoint "$OUT\deep\runs\resnet18_seed_42\best.pt" --split test `
  --unlock-test --confirm-final-evaluation --initiated-by "Ahmed" `
  --output "$OUT\final_test"
```

### 7.9 Thesis figures + tables (only from real outputs)
```powershell
python main.py generate-thesis-outputs --output "$OUT"
```
Figures + matching source data: `outputs\thesis\figures\`, `outputs\thesis\data\`,
`outputs\thesis\tables\`, `outputs\thesis\thesis_manifest.json`.

### 7.10 Single‑image bilingual case report
```powershell
python main.py analyze-image --image "$DS\valid\Happy\<pick-one>.png" --output "$OUT\case_001"
```
Writes `analysis.json` + `reports\{professional_en,professional_ar,parent_en,parent_ar,bilingual}.html`.

## 8. 6 GB safety + resume

- Keep `--batch-size 4` and `--grad-accum-steps 4` (effective batch 16) at 224px.
- Mixed precision is automatic on CUDA.
- Resume a single deep run with `train-image-model --resume <...\last.pt>` (§7.3).
- Objective/fusion/ablation stages are fast and re‑runnable.

## 9. Output locations & purpose

| Path | Purpose |
|---|---|
| `outputs\gpu\gpu_smoke.json` | GPU verification (cuda_used, peak VRAM) |
| `outputs\dataset_check\`, `outputs\readiness\` | dataset structure + readiness |
| `outputs\leakage\leakage_report.json` | leakage status + quarantine list |
| `outputs\leakage\clean_dataset\` | leakage‑free ImageFolder (if leakage found) |
| `outputs\manifest.csv` (+`.summary.json`) | train/valid manifest + leakage summary |
| `outputs\features\features.csv` (+`extraction_metadata.json`) | objective features + provenance |
| `outputs\feature_models\validation_leaderboard.json` | baseline model leaderboard |
| `outputs\deep\deep_comparison.json`, `outputs\deep\runs\...\best.pt` | deep comparison + checkpoints |
| `outputs\emb_generic\`, `outputs\emb_finetuned\` | embeddings + `embedding_metadata.json` |
| `outputs\fusion\fusion_leaderboard.json` | fusion winner (validation) |
| `outputs\fusion_calibrated\` | calibrated bundle + reliability diagram |
| `outputs\ablation\ablation.{json,csv}`, `outputs\emb_compare\` | ablations + representation comparison |
| `outputs\explain_features\`, `outputs\gradcam\` | tabular importance / Grad‑CAM |
| `outputs\final_test\` + `final_test_unlock_log.jsonl` | locked test metrics + audit |
| `outputs\thesis\` | thesis figures, data, tables, manifest |
| `outputs\case_001\reports\*.html` | bilingual single‑image reports |

## 10. Launch the Streamlit psychologist‑review interface

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```
In the sidebar, set the **Case folder** to a case directory (e.g.
`outputs\case_001`). The structured review tab writes to a shared
`review_master.csv`; compute agreement (real reviews only) with:
```powershell
python main.py review-agreement --master "outputs\review_master.csv" --output "outputs\agreement"
```

---

### Known limitations (honest)
- Actual accuracy/confusion/kappa come only from your runs; none are fabricated.
- GPU path is verified only if `gpu-smoke` printed `cuda_used: true` on your P3200.
- Symbol/eye/animal detectors don't exist → those psychological rules stay
  `missing_detector`; concern profiles are disabled until a real taxonomy exists.
