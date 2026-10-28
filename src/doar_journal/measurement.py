"""Stage C: propagation of mask error into the archived E2 objective features.

The 51 numeric E2 features are computed by the archived `extract_features`
code. `features_with_mask` runs that exact code with a supplied mask (e.g. a
human reference mask or a baseline's mask) so that feature differences are
attributable to the mask alone.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import _frozen_e2_features as fe
from .stats import icc_2_1

NON_FEATURE_COLUMNS = {
    "image_id", "split", "true_class", "relative_path", "qc_flag",
    # mask-engineering metadata of the A0 builder, not drawing measurements
    "mask_lab_threshold", "estimated_background_noise",
}


def _fixed_mask_fn(mask: np.ndarray):
    m = np.asarray(mask).astype(np.uint8)

    def fn(image_rgb):
        if image_rgb.shape[:2] != m.shape:
            raise ValueError(f"mask {m.shape} does not match image {image_rgb.shape[:2]}")
        return m, np.full(3, np.nan), float("nan"), float("nan")
    return fn


def archived_features(path: str | Path, meta: dict | None = None) -> dict:
    """The archived pipeline end to end (A0 mask inside), for equivalence checks."""
    row = {"path": str(path), "image_id": "", "split": "", "true_class": "", "relative_path": "",
           **(meta or {})}
    result, _mask = fe.extract_features(row)
    return result


def features_with_mask(path: str | Path, mask: np.ndarray, meta: dict | None = None) -> dict:
    row = {"path": str(path), "image_id": "", "split": "", "true_class": "", "relative_path": "",
           **(meta or {})}
    result, _mask = fe.extract_features_with_mask(row, _fixed_mask_fn(mask))
    return result


def numeric_features(result: dict) -> dict[str, float]:
    out = {}
    for k, v in result.items():
        if k in NON_FEATURE_COLUMNS:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def mask_dependence(path: str | Path, mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, bool]:
    """Which features change when only the mask changes (True = mask-dependent)."""
    fa = numeric_features(features_with_mask(path, mask_a))
    fb = numeric_features(features_with_mask(path, mask_b))
    dep = {}
    for k in fa:
        a, b = fa[k], fb.get(k, float("nan"))
        same = (math.isnan(a) and math.isnan(b)) or (a == b)
        dep[k] = not same
    return dep


def feature_agreement(ref: list[dict[str, float]], est: list[dict[str, float]]) -> dict[str, dict]:
    """Per feature: n valid pairs, MAE, median absolute relative error, ICC(2,1)."""
    keys = sorted(set().union(*[r.keys() for r in ref]))
    out = {}
    for k in keys:
        pairs = np.array([(r.get(k, np.nan), e.get(k, np.nan)) for r, e in zip(ref, est)], float)
        ok = ~np.isnan(pairs).any(axis=1)
        p = pairs[ok]
        n_nan_est = int(np.isnan(pairs[:, 1]).sum() - np.isnan(pairs).all(axis=1).sum())
        if len(p) < 3:
            out[k] = {"n": int(len(p)), "mae": None, "median_rel_err": None, "icc_2_1": None,
                      "n_missing_only_in_estimate": n_nan_est}
            continue
        abs_err = np.abs(p[:, 0] - p[:, 1])
        denom = np.where(np.abs(p[:, 0]) > 1e-12, np.abs(p[:, 0]), np.nan)
        rel = abs_err / denom
        out[k] = {"n": int(len(p)), "mae": float(abs_err.mean()),
                  "median_rel_err": float(np.nanmedian(rel)) if np.isfinite(rel).any() else None,
                  "icc_2_1": icc_2_1(p), "n_missing_only_in_estimate": n_nan_est}
    return out


# Focused journal subset: one or two mask-dependent features per visual construct.
# All 49 features are still computed and reported in the supplement.
FOCUS_FEATURES = {
    "foreground_fraction": "coverage",
    "bbox_area_ratio": "extent",
    "centroid_x_ratio": "placement (horizontal)",
    "centroid_y_ratio": "placement (vertical)",
    "occupied_grid_fraction_4x4": "spatial spread",
    "thickness_proxy_mean_ratio": "apparent stroke thickness",
    "foreground_mean_darkness": "mark darkness",
    "foreground_mean_saturation": "colour saturation",
    "foreground_hue_entropy": "colour variety",
    "vertical_symmetry_iou": "symmetry",
    "connected_component_count": "fragmentation",
}

# Pre-registered reporting policies compared in C4 (PREREGISTRATION.md, C9).
SIMPLE_RULE_FRACTION = (0.005, 0.60)   # V2 foreground fraction outside this range -> withhold


def policy_reports(v2_policy: str, v2_fraction: float) -> dict[str, bool]:
    """Whether each policy would REPORT the measurements of an image."""
    lo, hi = SIMPLE_RULE_FRACTION
    return {"report_all": True,
            "v2_policy": v2_policy == "MASK_USABLE",
            "simple_fraction_rule": lo <= v2_fraction <= hi}
