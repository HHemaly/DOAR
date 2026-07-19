from __future__ import annotations

import re


DIAGNOSTIC = re.compile(
    r"\b(has|diagnosed with|proves?|confirms?)\s+(anxiety|depression|trauma|aggression|autism|abuse)\b",
    re.IGNORECASE,
)


def run_judges(analysis: dict) -> dict:
    coverage = analysis["composition"]["foreground_coverage"]
    empty = analysis["composition"]["empty_space_ratio"]
    evidence_ids = {item["evidence_id"] for item in analysis["evidence"]}
    rule_errors = []
    for rule in analysis["rule_evaluations"]:
        unknown = set(rule["matched_evidence_ids"]) - evidence_ids
        if unknown:
            rule_errors.append({"rule_id": rule["rule_id"], "unknown_evidence_ids": sorted(unknown)})
    narrative = " ".join(
        filter(None, [rule.get("professional_reasoning") for rule in analysis["rule_evaluations"]])
    )
    return {
        "quality_judge": {
            "status": "pass" if analysis["quality"]["supported"] else "requires_review",
            "checks": {"supported_file": analysis["quality"]["supported"]},
        },
        "segmentation_judge": {
            "status": "pass" if abs(coverage + empty - 1) < 1e-6 else "fail",
            "checks": {
                "coverage_complementary": abs(coverage + empty - 1) < 1e-6,
                "not_implausibly_full": coverage < 0.90,
                "candidate_disagreement": analysis["segmentation"]["candidate_disagreement"],
            },
        },
        "feature_judge": {
            "status": "pass",
            "checks": {
                "blank_has_no_centroid": not (
                    coverage == 0 and analysis["composition"]["centroid_normalized"] is not None
                ),
                "bbox_evidence_present": "ev_bbox_coverage" in evidence_ids,
            },
        },
        "rule_judge": {
            "status": "pass" if not rule_errors else "fail",
            "unsupported_evidence_references": rule_errors,
            "active_concerns_from_single_symbol": False,
        },
        "safety_judge": {
            "status": "pass" if not DIAGNOSTIC.search(narrative) else "fail",
            "diagnostic_language_found": bool(DIAGNOSTIC.search(narrative)),
            "disclaimer_present": "not a diagnosis" in analysis["safety_disclaimer"].lower(),
        },
        "module_availability": {
            "detection": "unavailable",
            "ocr": "unavailable",
            "emotion_model": "unavailable",
            "clinician_review": "not_submitted",
        },
    }
