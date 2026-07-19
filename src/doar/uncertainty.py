from __future__ import annotations

import numpy as np


def summarize_ensemble(probability_sets: list[list[float]], calibrated: bool) -> dict:
    if not probability_sets:
        return {
            "status": "unavailable", "warnings": ["missing_model_probabilities"],
            "entropy": None, "margin": None, "disagreement": None,
        }
    values = np.asarray(probability_sets, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid ensemble probability matrix")
    values = values / values.sum(axis=1, keepdims=True)
    fused = values.mean(axis=0)
    order = np.argsort(fused)[::-1]
    entropy = float(-(fused * np.log(np.maximum(fused, 1e-12))).sum())
    disagreement = float(np.mean(np.std(values, axis=0)))
    warnings = []
    if not calibrated:
        warnings.append("uncalibrated_confidence")
    if fused[order[0]] < .5:
        warnings.append("low_confidence")
    if disagreement > .12:
        warnings.append("model_disagreement")
    return {
        "status": "available",
        "fused_probabilities": fused.tolist(),
        "entropy": entropy,
        "margin": float(fused[order[0]] - fused[order[1]]),
        "disagreement": disagreement,
        "uncertainty": "high" if warnings else "low",
        "warnings": warnings,
        "method": "mean_probability_with_disagreement",
    }
