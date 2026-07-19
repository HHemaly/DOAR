# Local execution

## Environment

Use Python 3.11 and create a new environment; do not reuse the legacy `.venv`.

```powershell
cd doar_v3
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ml,cv,ui,dev]"
```

## Audit and analyze

```powershell
python main.py build-manifest `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --output "outputs\dataset\manifest.csv"

python main.py analyze-image `
  --image "C:\path\to\drawing.png" `
  --output "outputs\cases\case_001"

python main.py extract-features `
  --manifest "outputs\dataset\manifest.csv" `
  --output "outputs\features\v3_1"
```

The feature cache contains `features.csv`, a versioned schema, extraction
metadata, per-case structured feature JSON, analysis artifacts, and
`failures.csv`. Failures are reported rather than silently skipped.

## Train and validate

The initial reproducible baseline is a feature-based logistic regression. Model
fitting uses `train`; model selection and reporting use `valid`.

```powershell
python main.py train `
  --manifest "outputs\dataset\manifest.csv" `
  --output "outputs\training\baseline" `
  --seed 42

python main.py evaluate `
  --manifest "outputs\dataset\manifest.csv" `
  --checkpoint "outputs\training\baseline\feature_logistic_regression.joblib" `
  --split valid `
  --output "outputs\evaluation\valid"
```

Only perform the final locked test once the experiment is frozen:

```powershell
python main.py evaluate `
  --manifest "outputs\dataset\manifest.csv" `
  --checkpoint "outputs\training\baseline\feature_logistic_regression.joblib" `
  --split test --unlock-test --confirm-final-evaluation `
  --initiated-by "researcher-code" `
  --output "outputs\evaluation\final_test"
```

This writes `final_test_unlock_log.jsonl` with the UTC timestamp, manifest,
checkpoint and configuration hashes, confirmation flags, and initiating code.

No real-data metrics are included in the repository; they must be produced
locally from the audited dataset.

## Psychologist interface

```powershell
python -m streamlit run streamlit_app.py
```

Enter a generated case directory in the sidebar. Reviews are appended to
`clinician_review.json`; the original AI analysis is preserved.
