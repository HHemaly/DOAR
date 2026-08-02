"""
feature_importance.py — objective-feature (tabular) explainability (Item 15).

Global permutation importance, linear coefficients, and per-image local
contributions for objective-feature models. This is TABULAR importance and is
kept strictly separate from Grad-CAM (visual). It does NOT localize anything in
the image and must never be presented as image localization.

The permutation-importance core is pure-numpy with an injectable predict_proba,
so it is CPU-tested without sklearn.
"""

from __future__ import annotations
import numpy as np

DISCLAIMER = (
    "Objective-feature importance is a tabular attribution over measured drawing "
    "features. It does NOT localize regions in the image and is not a Grad-CAM "
    "heatmap. It is not causal or clinical evidence."
)


def _macro_f1(y_true, proba):
    from ..evaluation import compute_metrics
    y_true = np.asarray(y_true, int)
    return compute_metrics(y_true, proba.argmax(1), proba)["macro_f1"]


def permutation_importance(predict_proba, X, y, feature_names,
                           n_repeats: int = 5, seed: int = 0) -> dict:
    """Global permutation importance = baseline macro-F1 minus macro-F1 after
    permuting each feature column (averaged over n_repeats). predict_proba maps
    (n, d) -> (n, C). Pure-numpy; injectable predict for CPU tests."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    baseline = _macro_f1(y, predict_proba(X))
    importances = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(baseline - _macro_f1(y, predict_proba(Xp)))
        importances.append({
            "feature": name,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })
    importances.sort(key=lambda r: -r["importance_mean"])
    return {
        "method": "permutation_importance",
        "attribution_type": "tabular_objective_features",
        "baseline_macro_f1": baseline,
        "n_repeats": n_repeats,
        "importances": importances,
        "disclaimer": DISCLAIMER,
    }


def linear_coefficients(coef, feature_names, class_names) -> dict:
    """Per-class linear coefficients (valid only for linear models)."""
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 1:
        coef = coef[None, :]
    per_class = {}
    for i, cls in enumerate(class_names[:coef.shape[0]]):
        per_class[cls] = {feature_names[j]: float(coef[i, j])
                          for j in range(min(len(feature_names), coef.shape[1]))}
    return {"method": "linear_coefficients",
            "attribution_type": "tabular_objective_features",
            "per_class": per_class, "disclaimer": DISCLAIMER}


def local_contributions(coef, x, feature_mean, feature_names, class_names,
                        predicted_class_index: int) -> dict:
    """Per-image local contribution for a LINEAR model:
    contribution_j = coef[class, j] * (x_j - mean_j). Pure-numpy."""
    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 1:
        coef = coef[None, :]
    x = np.asarray(x, dtype=float)
    mean = np.asarray(feature_mean, dtype=float)
    row = coef[predicted_class_index]
    contribs = row * (x - mean)
    ranked = sorted(
        ({"feature": feature_names[j], "contribution": float(contribs[j])}
         for j in range(len(feature_names))),
        key=lambda r: -abs(r["contribution"]))
    return {
        "method": "linear_local_contributions",
        "attribution_type": "tabular_objective_features",
        "predicted_class": class_names[predicted_class_index],
        "contributions": ranked,
        "disclaimer": DISCLAIMER,
    }
