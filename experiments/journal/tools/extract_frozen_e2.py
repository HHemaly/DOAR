"""Generate src/doar_journal/_frozen_*.py from the archived notebooks, VERBATIM.

The foreground and feature code used in the manuscript exists only inside
notebook cells. This tool copies the exact function source text (via
ast.get_source_segment) into importable modules, records the source cell and
the notebook SHA-256 in each module header, and adds exactly one documented
derivative function (`extract_features_with_mask`) whose only change is that
the mask builder is passed in instead of hard-wired. Run it again to verify:
it fails if the notebooks changed.

  python experiments/journal/tools/extract_frozen_e2.py
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "colab/archive/DOAR_21_8_26_V1.ipynb"
PAPER = ROOT / "colab/paper/E2_A0_R1_V2_FROZEN_UNSEEN_HOLDOUT.ipynb"
ARCHIVE_SHA = "b7edd22e84ccef6554018ce6883dff20d2445459bd813083dd2d6686dd5f4ac4"
OUT = ROOT / "src/doar_journal"

HEADER = '''"""GENERATED FILE -- do not edit by hand.

Verbatim function bodies extracted by experiments/journal/tools/extract_frozen_e2.py
from {source} (sha256 {sha}), cell {cell} ({label}).
Thresholds, constants and logic are unchanged from the runs reported in the
archived notebook; only module imports were added so the functions can run
outside Colab.
"""
# flake8: noqa
# ruff: noqa
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

'''


def cell_source(nb_path: Path, idx: int) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][idx]["source"])
    return "\n".join(line for line in src.split("\n") if not line.lstrip().startswith(("!", "%")))


def functions(src: str, names: list[str]) -> dict[str, str]:
    tree = ast.parse(src)
    found = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"functions not found: {missing}")
    return {n: found[n] for n in names}


def write_module(name: str, header_kwargs: dict, funcs: dict[str, str], extra: str = "") -> Path:
    body = HEADER.format(**header_kwargs) + "\n\n".join(funcs.values()) + "\n" + extra
    compile(body, name, "exec")
    path = OUT / name
    path.write_text(body, encoding="utf-8")
    return path


INJECTABLE = '''

# ---------------------------------------------------------------------------
# DERIVATIVE (not in the archive): identical to extract_features except that
# the mask builder is a parameter. `mask_fn(image_rgb)` must return the same
# 4-tuple as build_foreground_mask: (mask, background_lab, threshold, noise).
# tests/test_journal_stage_c.py checks that passing build_foreground_mask
# reproduces extract_features exactly.
# ---------------------------------------------------------------------------
'''


def main() -> int:
    sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if sha != ARCHIVE_SHA:
        raise SystemExit(f"archive notebook changed: {sha}")
    paper_sha = hashlib.sha256(PAPER.read_bytes()).hexdigest()

    c20 = cell_source(ARCHIVE, 20)
    f20 = functions(c20, ["safe_divide", "entropy_from_histogram", "get_border_pixels",
                          "build_foreground_mask", "symmetry_score", "calculate_colorfulness",
                          "extract_features"])
    ef = f20["extract_features"]
    if ef.count("build_foreground_mask(") != 1:
        raise SystemExit("unexpected number of mask-builder calls in extract_features")
    injectable = (ef.replace("def extract_features(\n    row\n):", "def extract_features_with_mask(\n    row,\n    mask_fn,\n):", 1)
                    .replace("build_foreground_mask(", "mask_fn(", 1))
    if injectable == ef or "def extract_features_with_mask" not in injectable:
        raise SystemExit("could not derive extract_features_with_mask")
    write_module("_frozen_e2_features.py", {"source": ARCHIVE.name, "sha": sha, "cell": 20,
                                            "label": "E2-A0 objective feature extraction (51 numeric)"},
                 f20, INJECTABLE + injectable + "\n")

    c22 = cell_source(ARCHIVE, 22)
    f22 = functions(c22, ["get_border_pixels", "build_a0_mask", "robust_mad", "build_border_selector",
                          "build_r1_mask", "qc_status_from_ratio", "mask_iou"])
    write_module("_frozen_r1v1.py", {"source": ARCHIVE.name, "sha": sha, "cell": 22,
                                     "label": "E2-A0-R1 pilot (R1-V1)"}, f22)

    c3 = cell_source(PAPER, 3)
    f3 = functions(c3, ["robust_mad", "get_border_pixels", "cleanup_components", "build_a0_mask",
                        "build_border_selector", "border_samples_with_side_ids", "estimate_support_v2",
                        "suppress_smooth_border_floods", "build_v2_mask", "border_foreground_fraction",
                        "v2_qc_status", "measurement_policy"])
    write_module("_frozen_v2.py", {"source": PAPER.name, "sha": paper_sha, "cell": 3,
                                   "label": "frozen A0 / R1-V2 definitions (verbatim from archive cells 22-24)"}, f3)
    # E1-B adaptation code: cell 6 definitions, with `evaluate` taken from cell 7
    # (the bfloat16 -> float32 patch that all 27 reported E1-B runs used).
    c6 = cell_source(ARCHIVE, 6)
    tree6 = ast.parse(c6)
    keep = [n for n in tree6.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name not in ("make_example_panel", "evaluate")]
    defs6 = {n.name: ast.get_source_segment(c6, n) for n in keep}
    defs6["evaluate"] = functions(cell_source(ARCHIVE, 7), ["evaluate"])["evaluate"]
    header = HEADER.format(source=ARCHIVE.name, sha=sha, cell="6 (+ evaluate from cell 7)",
                           label="E1-B controlled representation adaptation")
    header += E1B_IMPORTS
    body = header + "\n\n".join(defs6.values()) + "\n" + E1B_CONFIGURE
    compile(body, "_frozen_e1b.py", "exec")
    (OUT / "_frozen_e1b.py").write_text(body, encoding="utf-8")
    print("generated _frozen_e2_features.py, _frozen_r1v1.py, _frozen_v2.py, _frozen_e1b.py")
    return 0


E1B_IMPORTS = '''import csv
import gc
import json
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageOps
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    import open_clip
except ImportError:  # CPU test environments without open_clip
    open_clip = None

'''

E1B_CONFIGURE = '''

# ---------------------------------------------------------------------------
# NOT in the archive: the notebook set these as Colab globals in cell 6.
# configure() sets the same names on this module so run_one() behaves exactly
# as it did there, but with the manifest, output root and seeds of a new run.
# Defaults are the archived E1-B values.
# ---------------------------------------------------------------------------
CLASSES = ["Angry", "Fear", "Happy", "Sad"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
MODELS = ["siglip2_vit_b16", "clip_vit_b32", "efficientnet_b0"]
MODES = ["frozen", "partial", "full"]
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
HEAD_LR = 0.0003
ENCODER_LR = 3e-05
WEIGHT_DECAY = 0.0001
LABEL_SMOOTHING = 0.05
PARTIAL_TARGET = 0.25
NUM_WORKERS = 2
QUAL_IDS = set()


def configure(*, train_rows, valid_rows, out_root, dataset_root, manifest_sha, source_commit,
              device="cuda", **overrides):
    g = globals()
    g["train_rows"] = list(train_rows)
    g["valid_rows"] = list(valid_rows)
    g["OUT_ROOT"] = Path(out_root)
    g["DATASET_ROOT"] = Path(dataset_root)
    g["EXPECTED_T0_SHA"] = manifest_sha  # the archive stored the split manifest hash under this name
    g["SOURCE_COMMIT"] = source_commit
    g["DEVICE"] = device
    if device == "cuda" and torch.cuda.is_available():
        g["AMP_DTYPE"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        g["AMP_DTYPE"] = torch.float32
    g["CLASS_WEIGHTS"] = compute_class_weights(g["train_rows"])
    for k, v in overrides.items():
        if k not in g:
            raise KeyError(f"unknown setting {k}")
        g[k] = v
'''


if __name__ == "__main__":
    raise SystemExit(main())
