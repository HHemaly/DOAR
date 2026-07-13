# DOAR — Drawing Observation & Analysis Report

A research system that analyzes children's drawings with computer vision,
measurable visual features, an optional supervised image classifier, structured
rule-based indicators, claim validation, and **safe, non-diagnostic**
interpretation.

> **This is not a diagnostic tool.** Children's drawings alone cannot establish
> any psychiatric diagnosis, trauma, abuse, or clinical state. Every
> parent-facing output uses cautious language and carries a mandatory
> disclaimer. Human psychologist validation is kept strictly separate from AI
> output.

---

## Scientific separation (the core design principle)

The system never mixes these five levels, and never merges their confidences:

| Level | What it is | Where in code |
|---|---|---|
| **L1 Objective observation** | measurable facts (size, colour %, contours, OCR text) | `pipeline._extract_features`, `src/data/inspect_dataset` |
| **L2 Dataset ground-truth label** | the folder/class label — never reinterpreted | `metadata.label_from_dataset` |
| **L3 ML prediction** | trained classifier: predicted class + probability + top-k + correctness | `src/models/` |
| **L4 Rule-based indicators** | cautious, non-diagnostic, literature-tagged rules | `pipeline._evaluate_v1_rules`, `src/psychological_rules_v2` |
| **L5 Psychologist validation** | human approve / reject / uncertain | claim `validator_status`, review reports |

The emotion component is a **feature-based heuristic**, explicitly labelled as
such — it is *not* a trained emotion model and its confidence is never presented
as a model result.

---

## Repository layout

```
DOAR/
  config/thresholds.json         centralised, justified thresholds
  src/
    data/       inspect_dataset.py, split.py            (dataset tooling)
    models/     dataset.py, classifier.py, train.py, evaluate.py
    utils/      reproducibility.py
    reports/    thesis_collate.py, output_manager.py
    psychological_rules_v2.py, emotion_heuristic.py, claim_builder.py
    numeric_validator.py, ocr_validator.py, psych_safety_validator.py,
    visual_claim_validator.py, safety_policy.py, final_response_judge.py,
    parent_ai_helper.py, arabic_translator.py
  ui/app.py                      Gradio local demo
  notebooks/DOAR_Colab.ipynb     Colab runner (calls modules only)
  scripts/build_colab_notebook.py
  tests/test_pipeline.py         critical-component + smoke tests
  pipeline.py                    interpretation pipeline (single/batch)
  analyze_dataset.py             batch interpretation convenience script
  main.py                        unified CLI
  requirements.txt
  AUDIT_REPORT.md                Phase-1 audit
  README.md
```

---

## Setup — VS Code on Windows (Python 3.11)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your dataset path once (either edit the constant in `main.py` /
`pipeline.py`, or use an environment variable):

```bat
set DOAR_DATASET=C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing
set DOAR_OUTPUT=outputs
```

> For an NVIDIA GPU, install the CUDA build of PyTorch instead of the default:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---

## Setup — Google Colab

Open `notebooks/DOAR_Colab.ipynb` in Colab and run the cells top to bottom.
It mounts Drive, clones/updates the repo, installs deps, and calls the same
`main.py` commands. Core logic lives in the modules, not the notebook.

---

## Commands (the full workflow)

```bat
:: 1. Inspect the dataset -> CSVs, statistics JSON, distribution figures
python main.py inspect --data "%DOAR_DATASET%" --out outputs

:: 2. Build a deterministic, leak-safe 70/15/15 split
python main.py split --out outputs

:: 3. Train a classifier (transfer=ResNet18 default; also baseline/mobilenet/efficientnet)
python main.py train --out outputs --model transfer --epochs 25

:: 4. Evaluate the best checkpoint on the untouched test split
python main.py evaluate --out outputs

:: 5. Interpretation pipeline on a single drawing
python main.py analyze-image --input "path\to\drawing.jpg" --question "What does it show?"

:: 6. Interpretation pipeline on a folder (5 per class; use --max 0 for all)
python main.py analyze-dataset --data "%DOAR_DATASET%" --max 5

:: 7. Collate thesis figures + tables into outputs/thesis/
python main.py thesis --out outputs

:: Launch the local UI (parents / children / psychologists)
python ui\app.py
```

---

## Outputs

```
outputs/
  dataset_analysis/   dataset_summary.csv, class_distribution.csv,
                      duplicates.csv, corrupted_files.csv,
                      dataset_statistics.json, figures/*.png|svg
  splits/             split.csv, split_meta.json  (leakage_ok flag)
  training/           best_model.pt, last_model.pt, training_history.csv,
                      training_config.json, training_log.txt, classes.json,
                      reproducibility.json, figures/training_curves.*
  evaluation/         metrics.json, classification_report.csv,
                      per_class_metrics.csv, predictions_test.csv,
                      figures/confusion_matrix.*, per_class_f1.*,
                      confidence_distribution.*, ...
  <timestamp>/        per-image interpretation: analysis_en.json,
                      analysis_ar.json, report_card.png
  thesis/             figures/, tables/, thesis_results_summary.md
```

Every per-image analysis contains both `analysis_en` and `analysis_ar`
(English + Arabic), gentle questions, safety note, disclaimer, and the final
10-point judge verdict.

---

## Testing

```bat
python -m pytest tests/ -v
:: or, without pytest:
python tests\test_pipeline.py
```

The suite covers dataset discovery, image loading, feature extraction, claim
building, numeric validation, safety-language blocking, graceful behaviour when
CLIP/torch are missing, the leak-safe split, and a full-pipeline smoke test on a
synthetic drawing (no dataset, torch, or network required).

---

## Reproducibility

Fixed seeds throughout; every training/eval run writes `reproducibility.json`
with Python version, library versions, device/GPU, seed, git commit, and
timestamp. The split is deterministic given `(summary.csv, seed)`.

---

## Safety policy

- Only cautious wording: *may suggest / could indicate / might reflect /
  requires contextual interpretation*.
- Hard blocks on diagnostic/clinical claims ("the child has depression",
  "this proves trauma", etc.).
- Sensitive claims require stricter validation thresholds.
- Mandatory disclaimer on every parent-facing output.
- A 10-point final judge sanitises or replaces unsafe answers before display.

---

## Honest limitations

- **A trained classifier must be produced by you**, on your data — this repo
  ships the training/evaluation *code*, not pretrained weights or metrics. No
  accuracy, confusion matrix, or psychologist agreement is fabricated.
- Drawing-based indicators are weak, context-dependent, and non-diagnostic.
- Object/symbol detection (SAM + CLIP) is optional; without it, object-level
  claims are skipped rather than guessed. CLIP cosine similarity is *not* a
  calibrated probability and is treated conservatively.
- OCR on child handwriting is error-prone; low-confidence text is not shown.
- Arabic translation preserves caution but is machine-generated; have a human
  review sensitive wording.

---

## Recommended next research steps

1. Run `inspect` on the full dataset and review class imbalance / duplicates.
2. Train baseline vs transfer; keep the model that generalises best on val.
3. Add Grad-CAM heatmaps for selected test cases (state clearly they show
   classifier attention, not psychological meaning).
4. Collect real psychologist reviews and compute inter-rater agreement
   (Cohen's / Fleiss' kappa) — only with genuine reviews.
5. Ablations: classifier-only vs classifier+features; full-image vs crop-based
   CLIP validation.

See `AUDIT_REPORT.md` for the full Phase-1 audit and architecture rationale.
