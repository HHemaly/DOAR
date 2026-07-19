# DOAR v3

An evidence-traceable research pipeline for objective analysis of children's
drawings. Outputs are non-diagnostic and require professional review.

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
