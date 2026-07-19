# DOAR v2 — Local Setup & Run Guide (Windows / VS Code)

> **Important:** This system is for supportive observation only. It is NOT a diagnostic tool.
> Every output includes a mandatory disclaimer. Do not use it to label or diagnose children.

---

## 1. Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.11.x | https://python.org/downloads |
| Git | any | https://git-scm.com |
| VS Code | any | https://code.visualstudio.com |

Make sure Python is in your PATH:
```
python --version    # should print Python 3.11.x
```

---

## 2. Project folder structure

After cloning / unzipping:

```
DOAR-main/
├── config/
│   └── thresholds.json           ← threshold configuration
├── src/                          ← all Python modules
│   ├── arabic_translator.py      ← Arabic translation (new)
│   ├── claim_builder.py
│   ├── emotion_heuristic.py
│   ├── final_response_judge.py
│   ├── numeric_validator.py
│   ├── ocr_validator.py
│   ├── output_manager.py         ← output saving (new)
│   ├── parent_ai_helper.py
│   ├── psych_safety_validator.py
│   ├── psychological_rules_v2.py
│   ├── safety_policy.py
│   └── visual_claim_validator.py
├── ui/
│   └── app.py                    ← Gradio local UI (new)
├── outputs/                      ← auto-created on first run
├── pipeline.py                   ← main pipeline entry point (new)
├── analyze_dataset.py            ← batch dataset runner (new)
├── requirements.txt
├── setup.bat                     ← Windows one-click setup
└── DOOR_QA_30_04_26 (2).ipynb   ← original notebook (unchanged)
```

---

## 3. First-time setup

### Option A — One-click (Windows)
```bat
# Open a terminal in DOAR-main, then:
setup.bat
```

### Option B — Manual
```bat
# 1. Open VS Code terminal in DOAR-main
# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate

# 4. Install packages
pip install opencv-python Pillow numpy matplotlib
pip install easyocr
pip install sentence-transformers
pip install deep-translator
pip install gradio
pip install tqdm
```

> **Every time** you open a new terminal: `venv\Scripts\activate`

---

## 4. Configure paths

Open **`pipeline.py`** in VS Code and edit lines 25–26:

```python
DATASET_ROOT = r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
```

That's the only change needed to run locally. The `OUTPUT_DIR` default saves
results in an `outputs/` folder next to pipeline.py.

---

## 5. Run on a single image

```bat
venv\Scripts\activate
python pipeline.py --image "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\Happy\img001.jpg"
```

Optional flags:
```bat
python pipeline.py --image drawing.jpg --question "Is the child happy?"
python pipeline.py --image drawing.jpg --no-arabic   # skip Arabic translation
python pipeline.py --image drawing.jpg --no-ocr      # skip text detection
```

---

## 6. Run on the full dataset (batch mode)

```bat
venv\Scripts\activate
python analyze_dataset.py
```

Or with custom options:
```bat
python analyze_dataset.py --max 10           # 10 images per class
python analyze_dataset.py --max 0            # ALL images (takes a while)
python analyze_dataset.py --no-arabic        # faster — skip translation
python analyze_dataset.py --root "C:\path\to\Combined_Drawing"
```

---

## 7. Launch the UI demo

```bat
venv\Scripts\activate
python ui\app.py
```

Then open your browser at: **http://127.0.0.1:7860**

The UI lets you:
- Upload a drawing (drag & drop)
- Ask a parent question
- Switch output language to Arabic
- See the parent-facing answer, gentle questions, report card, and technical JSON
- View the safety judge status

To share with a supervisor or colleague (creates a temporary public link):
```bat
python ui\app.py --share
```

---

## 8. Outputs

Every analysis run creates a timestamped folder:

```
outputs/
└── 2025-06-24_14-30-00/
    ├── per_image/
    │   └── drawing_name/
    │       ├── analysis_en.json    ← full English analysis
    │       ├── analysis_ar.json    ← Arabic translation
    │       └── report_card.png     ← visual report card (psychologist view)
    ├── thesis_figures/
    │   ├── emotion_distribution.png
    │   ├── rule_frequency.png
    │   ├── validation_summary.png
    │   └── feature_overview.png
    └── summary_report.json         ← dataset-level summary
```

The `analysis_en.json` and `analysis_ar.json` both contain the keys:
- `analysis_en.parent_answer` — parent-facing text (English)
- `analysis_ar.parent_answer` — same text in Arabic
- `analysis_en.gentle_questions` — follow-up questions
- `final_judgment` — safety check result (PASS / REWRITE_REQUIRED / BLOCK)

---

## 9. Optional: PaddleOCR (better text detection)

EasyOCR is installed by default and works without a GPU.
For better accuracy on Arabic text or messy handwriting, install PaddleOCR:

```bat
pip install paddlepaddle
pip install paddleocr
```

The pipeline automatically prefers PaddleOCR when available.

---

## 10. Optional: CLIP visual validation

CLIP improves object detection accuracy but requires PyTorch (large download):

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

Then pass `--clip` to the pipeline (not yet wired to CLI; load in pipeline.py manually).

---

## 11. For Google Colab (future use)

If you later want to run in Colab, uncomment the Google Drive section at the
top of `pipeline.py` (lines 28–31) and comment out the local paths:

```python
# LOCAL (VS Code):
# DATASET_ROOT = r"C:\Users\Ahmed\Downloads\..."

# COLAB (uncomment these):
from google.colab import drive
drive.mount('/content/drive')
DATASET_ROOT = "/content/drive/MyDrive/Masters/Datasets/Combined_Drawing"
OUTPUT_DIR   = "/content/drive/MyDrive/Masters/DOAR_outputs"
```

---

## 12. Quick reference — all commands

```bat
# Setup (run once)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Activate (every new terminal)
venv\Scripts\activate

# Single image
python pipeline.py --image "path\to\drawing.jpg"

# Single image with question
python pipeline.py --image "path\to\drawing.jpg" --question "Why is the drawing so dark?"

# Batch dataset (5 per class)
python analyze_dataset.py

# Batch dataset (all images, no Arabic)
python analyze_dataset.py --max 0 --no-arabic

# Launch UI
python ui\app.py

# Launch UI with public share link
python ui\app.py --share
```

---

## Safety policy

Every output from this system:
1. Uses **only cautious wording** ("may suggest", "could indicate", "might reflect")
2. **Never** says the child "has depression", "is aggressive", "has trauma", or any clinical label
3. Always includes the mandatory disclaimer:
   > *"Drawing-based psychological indicators are not diagnostic on their own..."*
4. Passes a **10-point safety judge** before any text is shown
5. Flags or blocks outputs that contain diagnostic or alarming language
