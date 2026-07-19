"""
calibrate.py — validation-only calibration of the primary fusion classifier (Item 7).

Temperature scaling fitted on VALIDATION probabilities only (test never used).
Raw probabilities are preserved; a calibrated bundle stores the temperature and
applies it at inference and final-test evaluation. Reports ECE / Brier / NLL
before and after, and produces reliability-diagram data + figure.

The numerical core (fit_fusion_temperature / reliability_bins) is pure-numpy and
CPU-tested. calibrate_fusion_bundle loads the sklearn bundle and produces the
validation probabilities (needs the [ml] extra + aligned features/embeddings).
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..evaluation import expected_calibration_error


def _temperature_scale(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Recalibrate probability rows with a temperature via log-prob softmax."""
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    scaled = logits / max(temperature, 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(y: np.ndarray, proba: np.ndarray) -> float:
    return float(-np.log(np.maximum(proba[np.arange(len(y)), y], 1e-12)).mean())


def _brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def fit_fusion_temperature(valid_proba, valid_labels) -> dict:
    """Grid-search temperature minimizing VALIDATION NLL. Reports before/after."""
    proba = np.asarray(valid_proba, dtype=float)
    y = np.asarray(valid_labels, dtype=int)
    candidates = np.geomspace(0.25, 5.0, 200)
    losses = [_nll(y, _temperature_scale(proba, float(t))) for t in candidates]
    best_i = int(np.argmin(losses))
    T = float(candidates[best_i])
    after = _temperature_scale(proba, T)
    return {
        "method": "temperature_scaling",
        "fit_split": "valid",
        "test_used": False,
        "temperature": T,
        "raw_preserved": True,
        "validation_nll_before": _nll(y, proba),
        "validation_nll_after": _nll(y, after),
        "validation_ece_before": expected_calibration_error(y, proba),
        "validation_ece_after": expected_calibration_error(y, after),
        "validation_brier_before": _brier(y, proba),
        "validation_brier_after": _brier(y, after),
    }


def reliability_bins(y, proba, bins: int = 10) -> list[dict]:
    """Per-bin mean confidence vs accuracy for a reliability diagram (pure)."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    out = []
    for low in np.linspace(0, 1, bins, endpoint=False):
        high = low + 1 / bins
        sel = (conf >= low) & (conf < high)
        out.append({
            "bin_low": float(low), "bin_high": float(high),
            "count": int(sel.sum()),
            "mean_confidence": float(conf[sel].mean()) if sel.any() else None,
            "accuracy": float(correct[sel].mean()) if sel.any() else None,
        })
    return out


def _save_reliability_diagram(y, raw, calibrated, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    for label, proba, colour in (("raw", raw, "#e74c3c"), ("calibrated", calibrated, "#27ae60")):
        bins = reliability_bins(y, proba)
        xs = [b["mean_confidence"] for b in bins if b["mean_confidence"] is not None]
        ys = [b["accuracy"] for b in bins if b["accuracy"] is not None]
        ax.plot(xs, ys, marker="o", label=label, color=colour)
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram (validation)"); ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(p.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def apply_temperature(proba, temperature: float) -> np.ndarray:
    return _temperature_scale(np.asarray(proba, dtype=float), float(temperature))


def calibrate_fusion_bundle(bundle_path, features_csv, embeddings_npz, output) -> dict:
    """Load a fusion bundle, fit temperature on validation, save a calibrated
    bundle + calibration.json + reliability diagram. Requires sklearn."""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - needs [ml]
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc
    from ..dataset import CLASSES
    from .trainer import _load  # reuse the aligned loader

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_names, joined = _load(features_csv, embeddings_npz)
    valid = [item for item in joined if item[3] == "valid"]
    if not valid:
        raise ValueError("No aligned validation rows for calibration")
    x_valid = np.stack([np.concatenate((item[1], item[2])) for item in valid])
    y_valid = np.asarray([CLASSES.index(item[4]) for item in valid])

    raw_proba = model.predict_proba(x_valid)
    result = fit_fusion_temperature(raw_proba, y_valid)
    calibrated_proba = apply_temperature(raw_proba, result["temperature"])

    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _save_reliability_diagram(y_valid, raw_proba, calibrated_proba, out / "reliability_diagram")

    # Calibrated bundle preserves the raw model + records the temperature.
    bundle["calibration"] = {"method": "temperature_scaling", "status": "calibrated",
                             "temperature": result["temperature"], "fit_split": "valid"}
    calibrated_path = out / "fusion_calibrated.joblib"
    joblib.dump(bundle, calibrated_path)
    result["calibrated_bundle"] = str(calibrated_path)
    return result
