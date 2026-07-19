from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .dataset import CLASSES


def unavailable(reason: str = "No emotion checkpoint was supplied.") -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "probabilities": {name: None for name in CLASSES},
        "top_class": None,
        "confidence": None,
        "top_two_margin": None,
        "entropy": None,
        "uncertainty": "unavailable",
        "calibration_status": "unavailable",
        "model_name": None,
        "model_family": None,
        "checkpoint": None,
        "model_version": None,
        "preprocessing_version": None,
        "evidence_id": None,
    }


def _summary(probabilities: np.ndarray, metadata: dict, checkpoint: Path) -> dict:
    if probabilities.shape != (len(CLASSES),) or not np.isfinite(probabilities).all():
        raise ValueError("Emotion model returned invalid probabilities")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("Emotion probabilities sum to zero")
    probabilities = probabilities / total
    order = np.argsort(probabilities)[::-1]
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    confidence = float(probabilities[order[0]])
    uncertainty = "low" if confidence >= .75 else "moderate" if confidence >= .50 else "high"
    return {
        "status": "available",
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
        "top_class": CLASSES[int(order[0])],
        "confidence": confidence,
        "top_two_margin": float(probabilities[order[0]] - probabilities[order[1]]),
        "entropy": entropy,
        "uncertainty": uncertainty,
        "calibration_status": metadata.get("calibration_status", "uncalibrated"),
        "model_name": metadata.get("model_name", metadata.get("model", "unknown")),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_version": metadata.get("model_version", "unknown"),
        "preprocessing_version": metadata.get("preprocessing_version", "unknown"),
        "evidence_id": "ev_emotion_prediction",
    }


def predict(
    image_path: str | Path, checkpoint: str | Path | None, analysis_context: dict | None = None
) -> dict:
    if not checkpoint:
        return unavailable()
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return {**unavailable(f"Checkpoint does not exist: {checkpoint}"), "status": "failed"}
    try:
        if checkpoint.suffix.lower() in (".pt", ".pth"):
            from .deep.inference import predict_image
            result = predict_image(str(image_path), str(checkpoint))
            result["model_family"] = "deep_image"
            return result
        if checkpoint.suffix.lower() not in (".joblib", ".pkl", ".pickle"):
            raise ValueError(
                f"Unsupported checkpoint type {checkpoint.suffix!r}; "
                "expected .joblib/.pkl or .pt/.pth"
            )
        import joblib
        from .models import _image_features

        payload = joblib.load(checkpoint)
        if tuple(payload.get("classes", ())) != CLASSES:
            raise ValueError("Checkpoint class mapping is not Angry/Fear/Happy/Sad")
        if payload.get("checkpoint_type") == "doar_fusion_bundle_v1":
            if analysis_context is None:
                raise ValueError("Fusion inference requires objective analysis context")
            from .deep.embeddings import embed_image
            from .features import objective_feature_row
            expected = list(payload["objective_feature_names"])
            extracted = objective_feature_row(image_path, analysis_context)
            missing = [name for name in expected if name not in extracted]
            if missing:
                raise ValueError(f"Fusion feature schema mismatch; missing features: {missing}")
            objective = np.asarray([extracted[name].value for name in expected], dtype=np.float32)
            embedding = embed_image(image_path, payload["embedding_backbone"])
            if len(embedding) != int(payload["embedding_dimension"]):
                raise ValueError(
                    f"Fusion embedding dimension mismatch: expected "
                    f"{payload['embedding_dimension']}, got {len(embedding)}"
                )
            probabilities = np.asarray(
                payload["model"].predict_proba(np.concatenate((objective, embedding))[None])[0]
            )
            metadata = {
                "model_name": payload["method"],
                "model_family": "primary_multimodal_fusion",
                "model_version": payload["checkpoint_type"],
                "preprocessing_version": payload["embedding_preprocessing_hash"],
                "calibration_status": payload["calibration"]["status"],
            }
            result = _summary(probabilities, metadata, checkpoint)
            result["model_family"] = "primary_multimodal_fusion"
            result["fusion_method"] = payload["method"]
            result["feature_order_validated"] = True
            result["embedding_backbone"] = payload["embedding_backbone"]
            return result
        model = payload["model"]
        probabilities = np.asarray(model.predict_proba(_image_features(str(image_path))[None, :])[0])
        result = _summary(probabilities, payload, checkpoint)
        result["model_family"] = payload.get("model_family", "whole_image_statistical")
        return result
    except Exception as exc:
        failed = unavailable(str(exc))
        failed["status"] = "failed"
        failed["checkpoint"] = checkpoint.name
        return failed
