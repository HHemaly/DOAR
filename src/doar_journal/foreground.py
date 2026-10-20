"""Stage C: foreground (drawing-mark) extractors under one interface.

Every method maps an RGB uint8 image (H x W x 3, full resolution, no resize --
exactly as the archived E2 runs) to a boolean mask, True = drawing mark.

Frozen methods (verbatim code in the generated _frozen_* modules):
  a0     build_a0_mask           (E2-A0 baseline, Lab distance to a border model)
  r1v1   build_r1_mask           (R1-V1 development pilot)
  v2     build_v2_mask           (R1-V2, support-aware, frozen before the holdout)

Classical baselines (parameters fixed here BEFORE any reference mask exists,
taken from common defaults, never tuned on the evaluation data):
  otsu_gray        global Otsu on luminance; marks = darker class
  sauvola          Sauvola local threshold (window 25, k 0.2); marks = darker
  niblack          Niblack (window 25, k 0.2); marks = darker
  adaptive_gauss   OpenCV adaptive Gaussian mean (block 35, C 10); marks = darker
  otsu_lab_dist    Otsu on Lab distance from the image's median colour (colour-aware)

The classical baselines assume dark marks on a lighter support; that
assumption is part of what the regime-wise analysis is meant to expose, so it
is NOT patched per image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import _frozen_r1v1, _frozen_v2

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
ADAPTIVE_BLOCK = 35
ADAPTIVE_C = 10


@dataclass
class MaskResult:
    mask: np.ndarray                       # bool, H x W
    meta: dict = field(default_factory=dict)


def _gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def a0(image_rgb: np.ndarray) -> MaskResult:
    r = _frozen_v2.build_a0_mask(image_rgb)
    return MaskResult(np.asarray(r["mask"]) > 0, {"threshold": float(r["threshold"])})


def r1v1(image_rgb: np.ndarray) -> MaskResult:
    r = _frozen_r1v1.build_r1_mask(image_rgb)
    return MaskResult(np.asarray(r["mask"]) > 0, {})


def v2(image_rgb: np.ndarray) -> MaskResult:
    r = _frozen_v2.build_v2_mask(image_rgb)
    mask = np.asarray(r["mask"]) > 0
    ratio = float(mask.mean())
    border = _frozen_v2.border_foreground_fraction(np.asarray(r["mask"]))
    qc = _frozen_v2.v2_qc_status(foreground_ratio=ratio, border_fg_ratio=border,
                                 support_regime=r["support_regime"])
    policy = _frozen_v2.measurement_policy(qc_status=qc, support_confidence=r["support_confidence"])
    return MaskResult(mask, {"support_regime": r["support_regime"],
                             "support_confidence": r["support_confidence"],
                             "qc_status": qc, "measurement_policy": policy,
                             "foreground_ratio": ratio, "border_fg_ratio": border})


def otsu_gray(image_rgb: np.ndarray) -> MaskResult:
    g = _gray(image_rgb)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return MaskResult(g <= t, {"threshold": float(t)})


def sauvola(image_rgb: np.ndarray) -> MaskResult:
    from skimage.filters import threshold_sauvola
    g = _gray(image_rgb).astype(np.float64)
    return MaskResult(g <= threshold_sauvola(g, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K), {})


def niblack(image_rgb: np.ndarray) -> MaskResult:
    from skimage.filters import threshold_niblack
    g = _gray(image_rgb).astype(np.float64)
    return MaskResult(g <= threshold_niblack(g, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K), {})


def adaptive_gauss(image_rgb: np.ndarray) -> MaskResult:
    g = _gray(image_rgb)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                              ADAPTIVE_BLOCK, ADAPTIVE_C)
    return MaskResult(m > 0, {})


def otsu_lab_dist(image_rgb: np.ndarray) -> MaskResult:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.linalg.norm(lab - np.median(lab.reshape(-1, 3), axis=0), axis=2)
    d8 = np.clip(d / max(d.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return MaskResult(d8 > t, {"threshold": float(t)})


METHODS = {
    "a0": a0, "r1v1": r1v1, "v2": v2,
    "otsu_gray": otsu_gray, "sauvola": sauvola, "niblack": niblack,
    "adaptive_gauss": adaptive_gauss, "otsu_lab_dist": otsu_lab_dist,
}
FROZEN = ("a0", "r1v1", "v2")


def load_rgb(path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
