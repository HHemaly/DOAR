"""Does a confidence / assessability state predict measurement error?

Inputs are per-image pairs (confidence score, error). Higher score = the
method claims more confidence. Outputs:
* risk-coverage curve: keep the most-confident fraction c of images; risk =
  mean error among kept images
* AURC (area under the risk-coverage curve; lower is better) and the
  reference values for random ordering and the oracle ordering
* AUROC of the score for separating good from bad measurements (bad = error
  above a pre-registered threshold)
Ordinal states are mapped to scores with explicit, documented orders; ties
are broken by averaging over all tie orders (expected value), not by data order.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

POLICY_SCORE = {"MASK_USABLE": 2, "MASK_REVIEW_REQUIRED": 1, "MASK_UNRELIABLE": 0}
CONFIDENCE_SCORE = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
A0_QC_SCORE = {"OK": 2, "VERY_LOW_FOREGROUND": 1, "NO_FOREGROUND": 0}


def risk_coverage(scores, errors) -> dict:
    s = np.asarray(scores, float)
    e = np.asarray(errors, float)
    n = len(s)
    order = np.argsort(-s, kind="stable")
    # Expected curve under random tie-breaking: average errors within tie blocks.
    e_sorted = e[order].copy()
    s_sorted = s[order]
    start = 0
    for i in range(1, n + 1):
        if i == n or s_sorted[i] != s_sorted[start]:
            e_sorted[start:i] = e_sorted[start:i].mean()
            start = i
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(e_sorted) / np.arange(1, n + 1)
    oracle = np.cumsum(np.sort(e)) / np.arange(1, n + 1)
    return {"coverage": coverage.tolist(), "risk": risk.tolist(),
            "aurc": float(risk.mean()), "aurc_random": float(e.mean()), "aurc_oracle": float(oracle.mean()),
            "n": n}


def risk_at_coverage(curve: dict, c: float) -> float:
    cov = np.asarray(curve["coverage"])
    idx = int(np.searchsorted(cov, c - 1e-12))
    return float(np.asarray(curve["risk"])[min(idx, len(cov) - 1)])


def auroc_bad(scores, errors, bad_threshold: float) -> float | None:
    """AUROC of -score for detecting bad cases (error > threshold)."""
    e = np.asarray(errors, float)
    bad = e > bad_threshold
    if bad.all() or (~bad).all():
        return None
    return float(roc_auc_score(bad, -np.asarray(scores, float)))
