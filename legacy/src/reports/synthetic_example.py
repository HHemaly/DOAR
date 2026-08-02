"""
synthetic_example.py — build a clearly-labelled SYNTHETIC AnalysisRecord.

Used ONLY to test the report structure before real dataset/model outputs exist.
Every record has is_synthetic=True, which renders a visible banner in all HTML
reports. The report/per-image code automatically replaces these with real
records once the pipeline + model produce them — no report code changes needed.
"""

from __future__ import annotations
import os

from src.reports.schema import (SCHEMA_VERSION, make_detection,
                                 empty_model_prediction)


def make_synthetic_image(path: str):
    """Create a simple synthetic drawing so reports have a real image to embed."""
    import numpy as np
    import cv2
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.circle(img, (300, 90), 45, (0, 200, 255), -1)      # sun (bbox ~255,45..345,135)
    cv2.circle(img, (150, 160), 35, (0, 0, 0), 3)          # head
    cv2.line(img, (150, 195), (150, 300), (0, 0, 0), 4)    # body
    cv2.line(img, (0, 340), (400, 340), (0, 150, 0), 4)    # ground
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)
    return path


def build_synthetic_record(image_path: str, *, with_model: bool = True) -> dict:
    """Return a fully-populated SYNTHETIC AnalysisRecord (is_synthetic=True)."""
    detections = [
        make_detection("sun", category="symbols_positive", bbox=[255, 45, 345, 135],
                       source_detector="synthetic", detector_score=0.81,
                       clip_similarity=0.78, validation_method="clip_crop",
                       validation_threshold=0.70, validator_status="verified"),
        make_detection("person", category="people", bbox=[110, 120, 190, 305],
                       source_detector="synthetic", detector_score=0.44,
                       clip_similarity=0.61, validation_method="clip_crop",
                       validation_threshold=0.70, validator_status="uncertain"),
    ]

    if with_model:
        model_prediction = {
            "available": True,
            "model_name": "SYNTHETIC-resnet18",
            "predicted_class": "Happy",
            "confidence": 0.82,
            "top_k": [{"class": "Happy", "probability": 0.82},
                      {"class": "Sad", "probability": 0.11},
                      {"class": "Angry", "probability": 0.07}],
            "correct": True,
            "note": "SYNTHETIC placeholder prediction — replace with real model output.",
        }
    else:
        model_prediction = empty_model_prediction()

    record = {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": True,
        "image": {"id": "synthetic_case_001", "path": image_path,
                  "filename": os.path.basename(image_path),
                  "width": 400, "height": 400, "quality_score": 0.9},
        "ground_truth": {"label": "Happy", "source": "dataset_folder"},
        "model_prediction": model_prediction,
        "objective_features": {
            "color": {"dominant_color": "yellow", "color_diversity_count": 3,
                      "dark_dominance": 0.08, "warm_dominance": 0.22},
            "composition": {"empty_space_ratio": 0.61, "quadrant_bias": "upper_right"},
            "stroke": {"fragmentation_ratio": 0.18, "stroke_darkness_estimate": 0.7},
        },
        "detections": detections,
        "ocr": [{"text": "SUN", "confidence": 0.88, "semantic_tag": "neutral"}],
        "rules": [{
            "rule_id": "ALSCHULER_RED_WARM", "rule_name": "ALSCHULER_RED_WARM",
            "evidence_level": "literature_supported",
            "input_features": {"warm_dominance": 0.22}, "thresholds": {"warm_dominance": 0.25},
            "score": 0.55, "confidence": 0.55, "status": "verified",
            "interpretation": "Warm colours are present, which may reflect emotional expressivity.",
            "caveat": "Warm colours are also typical of cheerful drawings; not diagnostic.",
            "reference": "Alschuler & Hattwick (1947); Malchiodi (1998)",
        }],
        "themes": [{"theme": "possible_positive_expression", "score": 0.45,
                    "supporting_rules": ["ALSCHULER_RED_WARM", "SYM_SHAPE_FLOWERS_CLOUDS_SUN"]}],
        "emotion_heuristic": {"estimated_emotion": "happy", "confidence_label": "low",
                              "method": "heuristic",
                              "note": "Feature-based heuristic, not a trained model."},
        "claims": [
            {"claim_type": "visual_symbol", "claim": "A sun was detected (upper area).",
             "validator_status": "verified"},
            {"claim_type": "visual_object", "claim": "A person may be present (uncertain).",
             "validator_status": "uncertain"},
        ],
        "parent_output_en": {
            "parent_answer": ("Thank you for sharing your child's drawing. The drawing "
                              "uses warm colours and includes a sun in the upper area. "
                              "These are gentle observations, not conclusions."),
            "gentle_questions": ["Can you tell me about the sun in your drawing?",
                                 "How does this person feel?"],
            "safety_note": ("This interpretation is not diagnostic and is a supportive "
                            "observation only."),
            "disclaimer": ("Drawing-based psychological indicators are not diagnostic on "
                           "their own and should be interpreted alongside the child's "
                           "explanation, age, culture, and professional assessment."),
        },
        "parent_output_ar": {
            "parent_answer": "شكراً لمشاركتك رسم طفلك. يستخدم الرسم ألواناً دافئة ويتضمن شمساً في الأعلى. هذه ملاحظات لطيفة وليست استنتاجات.",
            "gentle_questions": ["هل يمكنك أن تخبرني عن الشمس في رسمتك؟",
                                 "كيف يشعر هذا الشخص؟"],
            "safety_note": "هذا التفسير ليس تشخيصاً وهو ملاحظة داعمة فقط.",
            "disclaimer": "مؤشرات الرسم النفسية ليست تشخيصية بحد ذاتها.",
        },
        "final_judgment": {"status": "PASS", "safe_to_show": True,
                           "checks_passed": 10, "checks_total": 10, "issues": []},
        "artifacts": {"original": image_path, "annotated": None, "crops": [],
                      "gradcam": None, "technical_html": None, "parent_en_html": None,
                      "parent_ar_html": None, "psychologist_html": None,
                      "analysis_json": None},
    }
    return record
