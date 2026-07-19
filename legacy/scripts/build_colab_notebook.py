"""
build_colab_notebook.py — generate notebooks/DOAR_Colab.ipynb.

The notebook only *calls* the src/ modules — it contains no core implementation.
Regenerate with:  python scripts/build_colab_notebook.py
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "DOAR_Colab.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(lines)}


cells = [
    md("# DOAR — Google Colab Runner\n",
       "\n",
       "This notebook **only calls** the `src/` modules. All core logic lives in\n",
       "Python files in the repo, so Colab and VS Code run the exact same code.\n",
       "\n",
       "**Order:** mount Drive → clone/update repo → install → set paths →\n",
       "inspect → split → train → evaluate → analyze examples → thesis outputs → copy to Drive.\n",
       "\n",
       "> Non-diagnostic research tool. Every parent-facing output carries a disclaimer."),

    md("## 1. Mount Google Drive (optional but recommended)"),
    code("from google.colab import drive\n",
         "drive.mount('/content/drive')"),

    md("## 2. Clone or update the repository"),
    code("import os\n",
         "REPO_URL = 'https://github.com/HHemaly/DOAR.git'\n",
         "REPO_DIR = '/content/DOAR'\n",
         "if not os.path.exists(REPO_DIR):\n",
         "    !git clone $REPO_URL $REPO_DIR\n",
         "else:\n",
         "    !cd $REPO_DIR && git pull\n",
         "%cd $REPO_DIR"),

    md("## 3. Install dependencies\n",
       "Colab already has torch + torchvision; we add the light extras."),
    code("!pip install -q opencv-python Pillow matplotlib scikit-learn easyocr \\\n",
         "    sentence-transformers deep-translator gradio tqdm"),

    md("## 4. Set the dataset path\n",
       "Point this at your dataset in Drive (or a local Colab copy)."),
    code("import os\n",
         "# Example: a Combined_Drawing folder inside your Drive\n",
         "DATASET = '/content/drive/MyDrive/Masters/Datasets/Combined_Drawing'\n",
         "OUTPUT  = '/content/DOAR/outputs'\n",
         "os.environ['DOAR_DATASET'] = DATASET\n",
         "os.environ['DOAR_OUTPUT']  = OUTPUT\n",
         "assert os.path.isdir(DATASET), f'Dataset not found: {DATASET}'\n",
         "print('Dataset:', DATASET)"),

    md("## 5. Inspect the dataset\n",
       "Writes CSVs, `dataset_statistics.json`, and distribution figures."),
    code("!python main.py inspect --data \"$DATASET\" --out \"$OUTPUT\""),

    md("## 6. Build a leak-safe train/val/test split (70/15/15)"),
    code("!python main.py split --out \"$OUTPUT\""),
    code("import json\n",
         "meta = json.load(open(f'{OUTPUT}/splits/split_meta.json'))\n",
         "print('Split totals:', meta['split_totals'])\n",
         "print('Leakage OK  :', meta['leakage_ok'])"),

    md("## 7. Train & compare 3 models (baseline + 2 transfer) — GPU\n",
       "Trains **baseline + MobileNetV3 + ResNet18** on the SAME leak-safe split,\n",
       "selects the winner on **validation** accuracy, and evaluates the winner\n",
       "**once** on the untouched test set. This is the recommended Colab path.\n",
       "\n",
       "> Enable the GPU: *Runtime -> Change runtime type -> T4 GPU*."),
    code("!python main.py train-compare --out \"$OUTPUT\" --epochs 25 --batch-size 32"),
    code("import json\n",
         "sel = json.load(open(f'{OUTPUT}/model_comparison/selected_model.json'))\n",
         "print('Winner (selected on validation):', sel['model'],\n",
         "      'val_acc=', sel['best_val_acc'])\n",
         "from IPython.display import Image\n",
         "Image(f'{OUTPUT}/model_comparison/figures/model_comparison.png')"),

    md("## 8. Winner's test-set metrics\n",
       "`train-compare` already evaluated the winner once. Inspect the results\n",
       "(re-run `evaluate` only if you change the checkpoint)."),
    code("m = json.load(open(f'{OUTPUT}/evaluation/metrics.json'))\n",
         "for k in ('accuracy','balanced_accuracy','macro_f1','weighted_f1'):\n",
         "    print(f'{k:>16}: {m[k]}')"),
    code("import json\n",
         "m = json.load(open(f'{OUTPUT}/evaluation/metrics.json'))\n",
         "print('accuracy       :', m['accuracy'])\n",
         "print('balanced acc   :', m['balanced_accuracy'])\n",
         "print('macro F1       :', m['macro_f1'])\n",
         "print('weighted F1    :', m['weighted_f1'])"),
    code("from IPython.display import Image\n",
         "Image(f'{OUTPUT}/evaluation/figures/confusion_matrix.png')"),

    md("## 9. Generate per-image report folders (technical / parent EN+AR / psychologist)\n",
       "Builds `examples/<case>/` with annotated image, crops, Grad-CAM (via the\n",
       "winning checkpoint), all four HTML reports, and seeds the psychologist\n",
       "review master CSV."),
    code("CKPT = json.load(open(f'{OUTPUT}/model_comparison/selected_model.json'))['checkpoint']\n",
         "!python main.py reports --data \"$DATASET\" --out \"$OUTPUT\" --max 6 --checkpoint \"$CKPT\""),
    code("# Structure-only demo (no dataset/model needed):\n",
         "# !python main.py reports --synthetic --out \"$OUTPUT\""),

    md("## 10. Collate thesis figures, tables, examples, agreement"),
    code("!python main.py thesis --out \"$OUTPUT\""),

    md("## 11. Copy outputs back to Drive"),
    code("import shutil, os\n",
         "DEST = '/content/drive/MyDrive/Masters/DOAR_outputs'\n",
         "os.makedirs(DEST, exist_ok=True)\n",
         "shutil.copytree(OUTPUT, DEST, dirs_exist_ok=True)\n",
         "print('Copied outputs to', DEST)"),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {OUT} ({len(cells)} cells)")
