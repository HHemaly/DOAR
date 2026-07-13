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

## Primary environment — Google Colab (GPU)

**Colab is the recommended and primary environment**, especially for training
and evaluation (free GPU). Open **`notebooks/DOAR_Colab.ipynb`** in Colab, set a
T4 GPU (*Runtime → Change runtime type → T4 GPU*), and run the cells top to
bottom. The notebook:

1. mounts Google Drive
2. clones/updates the repo
3. installs dependencies
4. sets the dataset path (a `Combined_Drawing` folder in your Drive)
5. inspects the dataset → CSVs + figures
6. builds the leak-safe split
7. **trains & compares baseline + MobileNetV3 + ResNet18**, selects the winner
   on **validation** accuracy
8. evaluates the winner **once** on the untouched test set
9. generates per-image reports (technical / parent EN+AR / psychologist) + Grad-CAM
10. collates thesis outputs and copies everything back to Drive

Core logic lives in `src/` — the notebook only calls it, so Colab and VS Code
run identical code. Regenerate the notebook with
`python scripts/build_colab_notebook.py`.

### Colab commands (what the notebook runs)

```bash
python main.py inspect       --data "$DATASET" --out "$OUTPUT"
python main.py split         --out "$OUTPUT"
python main.py train-compare --out "$OUTPUT" --epochs 25   # baseline + mobilenet + resnet18
python main.py evaluate      --out "$OUTPUT"               # (winner already evaluated by train-compare)
python main.py reports       --data "$DATASET" --out "$OUTPUT" --max 6 --checkpoint "$CKPT"
python main.py thesis        --out "$OUTPUT"
```

**Model selection is honest:** all three models train on the same leak-safe
split; the winner is chosen by validation accuracy; only the winner touches the
test set, exactly once.

---

## Secondary environment — VS Code on Windows (Python 3.11)

Local Windows works where it already did (single-image analysis, batch
interpretation, the UI, and the report structure via `--synthetic`). Heavy
training is possible but slow without a GPU — prefer Colab for that.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

set DOAR_DATASET=C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing
set DOAR_OUTPUT=outputs
```

> NVIDIA GPU: install the CUDA PyTorch build:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

```bat
:: Interpretation pipeline on one drawing
python main.py analyze-image --input "path\to\drawing.jpg" --question "What does it show?"

:: Interpretation pipeline on a folder (5 per class; --max 0 for all)
python main.py analyze-dataset --data "%DOAR_DATASET%" --max 5

:: Demonstrate the report structure with no dataset or model
python main.py reports --synthetic --out outputs

:: Local UI (parents / children / psychologists)
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
  training/<model>/   per-model checkpoints + history + curves
  model_comparison/   comparison.csv/json, selected_model.json,
                      figures/model_comparison.*
  evaluation/         metrics.json, classification_report.csv,
                      per_class_metrics.csv, predictions_test.csv,
                      figures/confusion_matrix.*, per_class_f1.*,
                      confidence_distribution.*, ...   (WINNER only)
  examples/<case>/    original.*, annotated.png, crops/, gradcam.png,
                      analysis.json, technical_report.html,
                      parent_report_en.html, parent_report_ar.html,
                      psychologist_review.html
  psychologist_review/review_master.csv
  thesis/             figures/, tables/ (incl. psychologist_agreement.json),
                      thesis_results_summary.md
```

Every `examples/<case>/analysis.json` follows the documented **AnalysisRecord**
schema (`src/reports/schema.py`) and keeps the five scientific levels separate.
The `model_prediction` field is a labelled placeholder until a classifier is
trained, then auto-filled — no report code changes.

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
