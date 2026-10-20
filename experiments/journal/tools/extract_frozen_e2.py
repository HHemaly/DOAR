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
    print("generated _frozen_e2_features.py, _frozen_r1v1.py, _frozen_v2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
