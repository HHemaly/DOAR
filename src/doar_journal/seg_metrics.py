"""Segmentation agreement metrics for binary drawing-mark masks.

Conventions
* masks are boolean arrays of identical shape; True = drawing mark
* when BOTH masks are empty the image has no marks to find: IoU/F1 are
  defined as 1.0 and `both_empty` is set so the case can be excluded or
  reported separately
* boundary F1 (Csurka et al., BMVC 2013 style): a boundary pixel of one mask
  counts as matched if a boundary pixel of the other lies within `tol` px.
  Thin strokes make region IoU harsh, so both are reported.
"""

from __future__ import annotations

import cv2
import numpy as np


def confusion(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = np.asarray(pred, bool)
    ref = np.asarray(ref, bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    return {"tp": int(np.sum(pred & ref)), "fp": int(np.sum(pred & ~ref)),
            "fn": int(np.sum(~pred & ref)), "tn": int(np.sum(~pred & ~ref))}


def region_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    c = confusion(pred, ref)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    both_empty = (tp + fp + fn) == 0
    if both_empty:
        return {**c, "iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "both_empty": True,
                "pred_fraction": 0.0, "ref_fraction": 0.0}
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = pred.size
    return {**c, "iou": tp / (tp + fp + fn), "precision": prec, "recall": rec, "f1": f1,
            "both_empty": False, "pred_fraction": (tp + fp) / n, "ref_fraction": (tp + fn) / n}


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m > 0) & (eroded == 0)


def boundary_f1(pred: np.ndarray, ref: np.ndarray, tol: int = 2) -> float:
    bp, br = _boundary(pred), _boundary(ref)
    if not bp.any() and not br.any():
        return 1.0
    if not bp.any() or not br.any():
        return 0.0
    dist_to_ref = cv2.distanceTransform((~br).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((dist_to_ref[bp] <= tol).mean())
    rec = float((dist_to_pred[br] <= tol).mean())
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def all_metrics(pred: np.ndarray, ref: np.ndarray, tol: int = 2) -> dict:
    m = region_metrics(pred, ref)
    m["boundary_f1"] = boundary_f1(pred, ref, tol)
    m["boundary_tol_px"] = tol
    return m
