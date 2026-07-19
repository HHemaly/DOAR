"""
late.py — late fusion + stacking over common-format probability exports (Item 8).

Consumes exports written by evaluation.save_probability_export (one per base
model). Everything aligns by sample_id (never row order); mismatches fail loudly.
Selects/fits on VALIDATION only; the test split is never used here. Saves the
fitted fusion model (weights or logistic meta-model) so it can be reloaded and
applied to validation, test and single-sample inference. Uncertainty is computed
per sample.

Methods:
  equal_late_fusion               — mean of base probabilities (no fit)
  validation_weighted_late_fusion — 2-model weight grid on validation log-loss
  logistic_probability_meta       — stacking; REQUIRES genuine OOF train exports
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..evaluation import (load_probability_export, align_exports, compute_metrics,
                          CLASS_ORDER)
from .probability import (equal_late_fusion, validation_weighted_late_fusion,
                          probability_meta_features, validate_oof_folds)


def _per_sample_uncertainty(base_mats: list[np.ndarray], calibrated: bool) -> list[dict]:
    """Uncertainty for each sample from its per-model probability vectors."""
    from ..uncertainty import summarize_ensemble
    n = base_mats[0].shape[0]
    out = []
    for i in range(n):
        rows = [mat[i].tolist() for mat in base_mats]
        out.append(summarize_ensemble(rows, calibrated))
    return out


def _fuse_valid(exports, method, calibrated):
    """Fit/apply the method on the VALIDATION split. Returns
    (sample_ids, y_true, fused_proba, fitted_model_dict, base_mats)."""
    ids, y_true, base_mats, _ = align_exports(exports, "valid")

    if method == "equal_late_fusion":
        fused = equal_late_fusion(*base_mats)["probabilities"]
        model = {"method": method, "weights": [1 / len(base_mats)] * len(base_mats)}
    elif method == "validation_weighted_late_fusion":
        if len(base_mats) != 2:
            raise ValueError("validation_weighted_late_fusion expects exactly 2 base models")
        res = validation_weighted_late_fusion(base_mats[0], base_mats[1], y_true)
        fused = res["probabilities"]
        model = {"method": method, "weights": res["weights"], "fit_split": "valid"}
    elif method == "logistic_probability_meta":
        # Genuine OOF probabilities are required on the TRAIN split for stacking.
        tr_ids, tr_y, tr_mats, tr_folds = align_exports(exports, "train")
        if tr_folds is None:
            raise ValueError("logistic_probability_meta requires OOF fold_id on train exports")
        validate_oof_folds(tr_ids, tr_folds)
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - needs [ml]
            raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc
        train_meta = probability_meta_features(*tr_mats)
        valid_meta = probability_meta_features(*base_mats)
        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        clf.fit(train_meta, tr_y)
        fused = clf.predict_proba(valid_meta)
        model = {"method": method, "fit_split": "valid", "sklearn_meta_model": True,
                 "_estimator": clf}
    else:
        raise ValueError(f"Unknown late-fusion method: {method}")
    return ids, y_true, fused, model, base_mats


def train_late_fusion(export_files: list[str], output, method="validation_weighted_late_fusion",
                      calibrated: bool = False) -> dict:
    if len(export_files) < 2:
        raise ValueError("Late fusion requires at least two base-model exports")
    exports = [load_probability_export(p) for p in export_files]
    class_order = exports[0]["class_order"]

    ids, y_true, fused, model, base_mats = _fuse_valid(exports, method, calibrated)
    metrics = compute_metrics(y_true, fused.argmax(1), fused, class_names=class_order)
    per_sample = _per_sample_uncertainty(base_mats, calibrated)

    out = Path(output); out.mkdir(parents=True, exist_ok=True)

    # Save the fitted fusion model so it can be reloaded/applied later.
    model_meta = {k: v for k, v in model.items() if k != "_estimator"}
    model_meta.update({
        "class_order": class_order,
        "base_model_ids": [e.get("model_id") for e in exports],
        "base_checkpoint_hashes": [e.get("checkpoint_hash") for e in exports],
        "base_calibration_status": [e.get("calibration_status") for e in exports],
        "test_used": False, "selection_split": "valid",
    })
    if "_estimator" in model:
        import joblib
        meta_path = out / "stacking_meta_model.joblib"
        joblib.dump({"estimator": model["_estimator"], "class_order": class_order}, meta_path)
        model_meta["meta_model_path"] = str(meta_path)
    (out / "late_fusion_model.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    result = {
        "method": method, "selection_split": "valid", "test_used": False,
        "weights": model_meta.get("weights"),
        "validation_macro_f1": metrics["macro_f1"],
        "validation_metrics": metrics,
        "n_valid": len(ids),
        "per_sample_uncertainty_example": per_sample[0] if per_sample else None,
        "fusion_model": str(out / "late_fusion_model.json"),
    }
    (out / "late_fusion_result.json").write_text(json.dumps(result, indent=2, default=float),
                                                 encoding="utf-8")
    # Persist per-sample uncertainty aligned by sample_id.
    (out / "per_sample_uncertainty.json").write_text(
        json.dumps({sid: u for sid, u in zip(ids, per_sample)}, indent=2), encoding="utf-8")
    return result


def load_late_fusion(model_json: str) -> dict:
    return json.loads(Path(model_json).read_text(encoding="utf-8"))


def apply_late_fusion(model_meta: dict, export_files: list[str], split: str):
    """Apply a saved fusion model to any split (validation/test/inference).
    Returns (sample_ids, fused_probabilities). Aligns by sample_id."""
    exports = [load_probability_export(p) for p in export_files]
    ids, _y, base_mats, _folds = align_exports(exports, split)
    method = model_meta["method"]
    if method == "equal_late_fusion":
        fused = np.mean(base_mats, axis=0)
    elif method == "validation_weighted_late_fusion":
        w = model_meta["weights"]
        fused = w[0] * base_mats[0] + w[1] * base_mats[1]
    elif method == "logistic_probability_meta":
        import joblib
        payload = joblib.load(model_meta["meta_model_path"])
        meta = probability_meta_features(*base_mats)
        fused = payload["estimator"].predict_proba(meta)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    fused = np.asarray(fused, dtype=float)
    fused = fused / fused.sum(axis=1, keepdims=True)
    return ids, fused
