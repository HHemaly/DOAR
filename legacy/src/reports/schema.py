"""
schema.py — the documented analysis record that ALL report modules consume.

This is the single contract between the analysis pipeline and the reporting
layer. Real model outputs (Level 3) are plugged in later through
`model_prediction` without touching any report code.

The record keeps the five scientific levels strictly separate and NEVER merges
their confidence values.

────────────────────────────────────────────────────────────────────────────
AnalysisRecord
{
  "schema_version": "1.0",
  "is_synthetic": bool,               # True => clearly-labelled placeholder data
  "image": {
      "id": str, "path": str, "filename": str,
      "width": int, "height": int, "quality_score": float
  },

  # ── Level 2: dataset ground truth (never reinterpreted) ──
  "ground_truth": { "label": str | None, "source": "dataset_folder" | None },

  # ── Level 3: ML prediction (filled by src/models/evaluate or predict) ──
  "model_prediction": {
      "available": bool,              # False until a model is run
      "model_name": str | None,
      "predicted_class": str | None,
      "confidence": float | None,     # softmax probability, NOT psychological confidence
      "top_k": [ {"class": str, "probability": float}, ... ],
      "correct": bool | None,         # vs ground_truth.label when available
      "note": str
  },

  # ── Level 1: objective visual observations ──
  "objective_features": { ... free-form measured values ... },

  # ── Detections (each is fully traceable; see DetectionRecord) ──
  "detections": [ DetectionRecord, ... ],

  "ocr": [ {"text": str, "confidence": float, "semantic_tag": str}, ... ],

  # ── Level 4: rule-based indicators (non-diagnostic) ──
  "rules": [ RuleRecord, ... ],
  "themes": [ {"theme": str, "score": float, "supporting_rules": [str]}, ... ],

  # emotion is a heuristic, kept separate from model_prediction on purpose
  "emotion_heuristic": {
      "estimated_emotion": str, "confidence_label": str,
      "method": "heuristic", "note": str
  },

  "claims": [ ClaimRecord, ... ],     # validated claims (validator_status set)

  # ── Parent-facing outputs (EN + AR) ──
  "parent_output_en": { "parent_answer": str, "gentle_questions": [str],
                        "safety_note": str, "disclaimer": str },
  "parent_output_ar": { ... same shape, Arabic ... } | None,

  "final_judgment": { "status": str, "safe_to_show": bool,
                      "checks_passed": int, "checks_total": int, "issues": [str] },

  "artifacts": {                      # file paths produced for this image
      "original": str, "annotated": str | None,
      "crops": [str], "gradcam": str | None,
      "technical_html": str | None, "parent_en_html": str | None,
      "parent_ar_html": str | None, "psychologist_html": str | None,
      "analysis_json": str | None
  }
}

────────────────────────────────────────────────────────────────────────────
DetectionRecord — one detected object/symbol, fully traceable
{
  "label": str,
  "category": str,
  "bbox": [x1, y1, x2, y2] | None,    # pixel coords in the ORIGINAL image
  "source_detector": str,             # e.g. "sam_clip", "synthetic"
  "detector_score": float | None,     # raw detector confidence (NOT CLIP, NOT probability)
  "crop_path": str | None,            # saved crop used for validation
  "clip_similarity": float | None,    # raw CLIP cosine similarity (NOT a probability)
  "validation_method": str,           # "clip_crop" | "clip_fullimage_fallback" | "none"
  "validation_threshold": float | None,
  "validator_status": str             # "verified" | "uncertain" | "rejected" | "unavailable"
}

────────────────────────────────────────────────────────────────────────────
RuleRecord — one activated rule (Level 4)
{
  "rule_id": str, "rule_name": str, "evidence_level": str,
  "input_features": dict, "thresholds": dict,
  "score": float, "confidence": float,
  "status": "verified" | "uncertain" | "rejected",
  "interpretation": str, "caveat": str, "reference": str
}
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0"


def empty_model_prediction(note: str = "No trained model has been run yet.") -> dict:
    """Level-3 placeholder used until a real classifier is evaluated."""
    return {
        "available": False,
        "model_name": None,
        "predicted_class": None,
        "confidence": None,
        "top_k": [],
        "correct": None,
        "note": note,
    }


def make_detection(label, *, category="object", bbox=None,
                   source_detector="none", detector_score=None,
                   crop_path=None, clip_similarity=None,
                   validation_method="none", validation_threshold=None,
                   validator_status="unavailable") -> dict:
    """Construct a fully-traceable DetectionRecord (keeps detector score,
    CLIP similarity, threshold, crop path and status SEPARATE)."""
    return {
        "label": label,
        "category": category,
        "bbox": list(bbox) if bbox else None,
        "source_detector": source_detector,
        "detector_score": detector_score,
        "crop_path": crop_path,
        "clip_similarity": clip_similarity,
        "validation_method": validation_method,
        "validation_threshold": validation_threshold,
        "validator_status": validator_status,
    }


def build_analysis_record(doc: dict, parent_en: dict, judgment: dict,
                          *, parent_ar: dict | None = None,
                          model_prediction: dict | None = None,
                          detections: list | None = None,
                          is_synthetic: bool = False) -> dict:
    """
    Assemble an AnalysisRecord from a pipeline `doc` + parent output + judgment.

    `model_prediction` defaults to the empty (not-yet-run) placeholder so reports
    render before any classifier exists. Pass a real prediction dict later.
    """
    from pathlib import Path
    img_path = doc.get("source_image", "")
    iq = doc.get("image_quality", {})

    v1 = [r for r in doc.get("psychological_rule_activations", []) if r.get("activated")]
    v2 = [r for r in doc.get("psychological_rule_activations_v2", []) if r.get("activated")]
    rules = [_rule_to_record(r) for r in (v1 + v2)]

    gt = doc.get("metadata", {}).get("label_from_dataset")

    record = {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": is_synthetic,
        "image": {
            "id": Path(img_path).stem if img_path else "unknown",
            "path": img_path,
            "filename": Path(img_path).name if img_path else "",
            "width": iq.get("width_px", 0),
            "height": iq.get("height_px", 0),
            "quality_score": iq.get("quality_score", 0.0),
        },
        "ground_truth": {
            "label": gt,
            "source": "dataset_folder" if gt else None,
        },
        "model_prediction": model_prediction or empty_model_prediction(),
        "objective_features": {
            "color": doc.get("color_features", {}),
            "composition": doc.get("composition_features", {}),
            "stroke": doc.get("stroke_features", {}),
        },
        "detections": detections if detections is not None else _detections_from_doc(doc),
        "ocr": doc.get("ocr_results", []),
        "rules": rules,
        "themes": doc.get("theme_scores_v2", []),
        "emotion_heuristic": doc.get("feature_based_emotional_tendency", {}),
        "claims": doc.get("_validated_claims", []),
        "parent_output_en": {
            "parent_answer": parent_en.get("parent_answer", ""),
            "gentle_questions": parent_en.get("gentle_questions", []),
            "safety_note": parent_en.get("safety_note", ""),
            "disclaimer": parent_en.get("disclaimer", ""),
        },
        "parent_output_ar": parent_ar,
        "final_judgment": {
            "status": judgment.get("final_answer_status", ""),
            "safe_to_show": judgment.get("safe_to_show", False),
            "checks_passed": judgment.get("checks_passed", 0),
            "checks_total": judgment.get("checks_total", 10),
            "issues": judgment.get("issues", []),
        },
        "artifacts": {
            "original": img_path, "annotated": None, "crops": [],
            "gradcam": None, "technical_html": None, "parent_en_html": None,
            "parent_ar_html": None, "psychologist_html": None, "analysis_json": None,
        },
    }
    return record


def _rule_to_record(r: dict) -> dict:
    return {
        "rule_id": r.get("rule_id", ""),
        "rule_name": r.get("rule_id", ""),
        "evidence_level": r.get("evidence_strength", r.get("evidence_level", "unspecified")),
        "input_features": r.get("input_features", {}),
        "thresholds": r.get("thresholds", {}),
        "score": r.get("rule_confidence", r.get("score", 0.0)),
        "confidence": r.get("rule_confidence", 0.0),
        "status": r.get("status", "uncertain"),
        "interpretation": r.get("interpretation", ""),
        "caveat": r.get("caution", r.get("caveat", "")),
        "reference": "; ".join(r.get("sources", [])) if r.get("sources") else r.get("reference", ""),
    }


def _detections_from_doc(doc: dict) -> list:
    """Convert doc.detected_objects into DetectionRecords (bbox-aware)."""
    out = []
    for obj in doc.get("detected_objects", []):
        top = obj.get("top_match", {})
        out.append(make_detection(
            top.get("label", "unknown"),
            category=top.get("category", "object"),
            bbox=obj.get("bbox"),
            source_detector=obj.get("source_detector", "sam_clip"),
            detector_score=obj.get("detector_score"),
            crop_path=obj.get("crop_path"),
            clip_similarity=top.get("clip_similarity"),
            validation_method=obj.get("validation_method", "none"),
            validation_threshold=obj.get("validation_threshold"),
            validator_status=obj.get("validator_status", "unavailable"),
        ))
    return out
