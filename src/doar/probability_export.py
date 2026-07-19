"""
probability_export.py — universal probability exporter (B1 + B2).

Supports every model family into one common export schema:
  * objective-feature .joblib models,
  * primary-fusion .joblib bundles,
  * calibrated fusion bundles,
  * deep .pt / .pth checkpoints.

Rules:
  * probabilities used for evaluation are the CALIBRATED ones when a bundle has a
    validation-fitted temperature; raw_probabilities are exported separately (B1).
  * fusion export raises a clear alignment error listing sample IDs that lack an
    embedding rather than silently skipping them.
  * deep export uses the checkpoint preprocessing spec, no augmentation, aligns by
    sample_id, keeps fixed class order, records checkpoint SHA-256, and fails on
    missing/unreadable samples.
  * test split export is guarded by the shared final-test guard (B3), applied at
    the CLI layer.
"""

from __future__ import annotations
import csv
import hashlib
from pathlib import Path

import numpy as np

from .dataset import CLASSES
from .evaluation import save_probability_export

META = {"image_id", "path", "split", "class"}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _apply_calibration(bundle, raw_proba):
    """Return (used_proba, raw_proba, status, temperature). Calibrated when the
    bundle carries a validation-fitted temperature (B1)."""
    cal = bundle.get("calibration") if isinstance(bundle.get("calibration"), dict) else None
    if cal and cal.get("status") == "calibrated" and cal.get("temperature"):
        from .fusion.calibrate import apply_temperature
        used = apply_temperature(raw_proba, cal["temperature"])
        return used, raw_proba, "calibrated", float(cal["temperature"])
    return raw_proba, raw_proba, (cal or {}).get("status", "uncalibrated"), None


def _load_feature_rows(features_csv):
    with open(features_csv, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    if not rows:
        raise ValueError("Empty feature table")
    names = [n for n in rows[0] if n not in META]
    return names, rows


# ---------------------------------------------------------------------------
# sklearn export (objective-feature or fusion .joblib)
# ---------------------------------------------------------------------------

def _export_sklearn(model_path, features_csv, embeddings_npz, output, splits):
    import joblib
    bundle = joblib.load(model_path)
    model = bundle["model"]
    model_id = bundle.get("model_name") or bundle.get("method") or Path(model_path).stem
    is_fusion = bundle.get("checkpoint_type") == "doar_fusion_bundle_v1" or embeddings_npz

    feature_names, rows = _load_feature_rows(features_csv)
    embed_by_id = {}
    if embeddings_npz:
        cache = np.load(embeddings_npz, allow_pickle=True)
        for sid, vec in zip(cache["image_ids"].astype(str), cache["embeddings"]):
            embed_by_id[sid] = np.asarray(vec, dtype=float)

    sample_ids, split_list, y_true, X, fold_ids = [], [], [], [], []
    has_fold = "fold_id" in rows[0]
    missing_embeddings = []
    for row in rows:
        if row["split"] not in splits:
            continue
        feats = [float(row[n]) if row[n] not in ("", "nan", "NaN") else np.nan
                 for n in feature_names]
        if is_fusion and embeddings_npz:
            sid = row["image_id"]
            if sid not in embed_by_id:
                missing_embeddings.append(sid)      # B2: do not silently skip
                continue
            vec = np.concatenate((np.asarray(feats), embed_by_id[sid]))
        else:
            vec = np.asarray(feats)
        sample_ids.append(row["image_id"]); split_list.append(row["split"])
        y_true.append(CLASSES.index(row["class"])); X.append(vec)
        if has_fold:
            fold_ids.append(int(row["fold_id"]))
    if is_fusion and missing_embeddings:
        raise ValueError(
            f"Fusion export alignment error: {len(missing_embeddings)} feature rows "
            f"lack an embedding, e.g. {missing_embeddings[:5]}. Refusing to silently "
            f"skip them.")
    if not X:
        raise ValueError("No rows matched the requested splits")

    raw = model.predict_proba(np.vstack(X))
    used, raw, status, temperature = _apply_calibration(bundle, raw)
    path = save_probability_export(
        output, sample_ids=sample_ids, splits=split_list, y_true=y_true, proba=used,
        raw_proba=raw, temperature=temperature, model_id=model_id,
        checkpoint_hash=_sha256(model_path), calibration_status=status,
        class_order=list(CLASSES), fold_ids=(fold_ids if has_fold else None))
    return {"export": path, "model_id": model_id, "family": "fusion" if is_fusion else "objective_feature",
            "count": len(sample_ids), "calibration_status": status,
            "splits": sorted(set(split_list)), "has_oof_folds": has_fold}


# ---------------------------------------------------------------------------
# deep export (.pt / .pth checkpoint)
# ---------------------------------------------------------------------------

def _export_deep(checkpoint, manifest, output, splits, device="auto"):
    import torch
    from PIL import Image
    from .deep.registry import build_model
    from .deep.preprocessing import (resolve_preprocessing, build_eval_transform,
                                     assert_preprocessing_compatible)

    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    payload = torch.load(checkpoint, map_location=selected, weights_only=False)
    if tuple(payload["classes"]) != CLASSES:
        raise ValueError("Checkpoint class mapping mismatch")
    model = build_model(payload["model_name"], len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model_state"]); model.to(selected).eval()

    spec = payload.get("preprocessing_spec") or resolve_preprocessing(
        payload["model_name"], payload["image_size"])
    assert_preprocessing_compatible(payload.get("preprocessing_hash"), spec)
    tf = build_eval_transform(spec)

    with open(manifest, newline="", encoding="utf-8") as h:
        rows = [r for r in csv.DictReader(h) if r["split"] in splits]
    if not rows:
        raise ValueError("No manifest rows matched the requested splits")

    sample_ids, split_list, y_true, tensors = [], [], [], []
    for r in rows:
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Unreadable sample {r['image_id']} ({r['path']}): {exc}")
        tensors.append(tf(img)); sample_ids.append(r["image_id"])
        split_list.append(r["split"]); y_true.append(CLASSES.index(r["class"]))

    probs = []
    with torch.no_grad():
        for i in range(0, len(tensors), 32):
            batch = torch.stack(tensors[i:i + 32]).to(selected)
            probs.append(torch.softmax(model(batch), 1).cpu().numpy())
    raw = np.concatenate(probs, axis=0)

    temperature = float(payload.get("temperature", 1.0))
    if temperature != 1.0:
        from .fusion.calibrate import apply_temperature
        used, status = apply_temperature(raw, temperature), "calibrated"
    else:
        used, status, temperature = raw, payload.get("calibration_status", "uncalibrated"), None

    path = save_probability_export(
        output, sample_ids=sample_ids, splits=split_list, y_true=y_true, proba=used,
        raw_proba=raw, temperature=temperature, model_id=payload["model_name"],
        checkpoint_hash=_sha256(checkpoint), calibration_status=status,
        class_order=list(CLASSES))
    return {"export": path, "model_id": payload["model_name"], "family": "deep",
            "count": len(sample_ids), "calibration_status": status,
            "splits": sorted(set(split_list))}


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def export_probabilities(model_path, features_csv=None, embeddings_npz=None,
                         output=None, splits=("train", "valid"),
                         manifest=None, device="auto") -> dict:
    """Universal exporter. Deep checkpoints use `manifest`; sklearn models use
    `features_csv` (+ `embeddings_npz` for fusion)."""
    suffix = Path(model_path).suffix.lower()
    if suffix in (".pt", ".pth"):
        if not manifest:
            raise ValueError("Deep checkpoint export requires --manifest")
        return _export_deep(model_path, manifest, output, splits, device)
    if suffix in (".joblib", ".pkl"):
        if not features_csv:
            raise ValueError("sklearn model export requires --features")
        return _export_sklearn(model_path, features_csv, embeddings_npz, output, splits)
    raise ValueError(f"Unsupported model file: {model_path}")
