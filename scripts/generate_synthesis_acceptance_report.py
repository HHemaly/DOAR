#!/usr/bin/env python
"""Reproducible acceptance report for the unified drawing-synthesis
milestone (`drawing_synthesis.py` + `clinician_review_app.py`).

Runs the SAME pipeline the clinician app uses -- `clinician_review_app.
load_case_bundle` (cache-first, `drawing_synthesis.synthesize_drawing`
under the hood) -- across the 15 cached development cases, with zero
live Gemini calls, and writes a machine-readable JSON + CSV report plus
a concise terminal summary. This is a REPORTING tool only: it invents no
new scoring, aggregation, or rule logic -- every field it prints is read
directly off `DrawingSynthesisResult`/`build_parent_friendly_profile`/
`build_technical_feature_rows`, all already exercised by the test suite.

For each case, reports:
  - synthesis level, summary, supporting evidence families/rule ids
  - concern domains touched
  - page-reference assessability (page-frame safety)
  - number of candidate hypotheses
  - whether the Parent View is populated (non-empty)
  - whether any internal rule_id leaked into the Parent View text
    (`\\b(PSY_AR_|EN_COMPILED_)[A-Z_0-9]+\\b`, the same pattern
    `tests/test_clinician_review_app.py::ParentViewSafetyTests` already
    enforces)

Usage:
    python scripts/generate_synthesis_acceptance_report.py
    python scripts/generate_synthesis_acceptance_report.py --image-id p2b_0003
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "clinician_review_app", ROOT / "scripts" / "clinician_review_app.py")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "prototype_cases" / "synthesis_acceptance_reports"

# Same pattern tests/test_clinician_review_app.py::ParentViewSafetyTests
# already uses to guarantee no internal rule identifier reaches parent-
# facing text -- reused verbatim, not redefined.
_RULE_ID_PATTERN = re.compile(r"\b(PSY_AR_|EN_COMPILED_)[A-Z_0-9]+\b")

CSV_COLUMNS = (
    "image_id", "synthesis_level", "positive_evidence_families", "positive_rule_ids",
    "concern_evidence_families", "concern_rule_ids", "concern_domains_touched",
    "page_reference_mode", "page_relative_features_assessable",
    "candidate_hypothesis_count", "parent_view_populated", "parent_view_rule_id_leak",
)


def build_case_report(case_id: str) -> dict | None:
    """One case's full acceptance record, or None if no cached
    Observer/Verifier data exists for this case_id (never makes a live
    call to find out more -- same guarantee as the app itself)."""
    bundle = app.load_case_bundle(case_id)
    if bundle is None or bundle["rows"] is None:
        return None

    synthesis = bundle["synthesis"]
    overall = synthesis.overall_synthesis
    page_reference = (bundle["deterministic_features"] or {}).get("page_reference") or {}

    parent_profile = app.build_parent_friendly_profile(bundle)
    parent_blob = json.dumps(parent_profile)
    # `objects_seen` may legitimately be empty (a sparse case with zero
    # verified entities -- the UI substitutes a fallback sentence), so
    # "populated" checks the sections that are always non-empty instead.
    parent_populated = all(bool(parent_profile.get(k)) for k in (
        "visual_style_notes", "evidence_text", "questions", "uncertainty_note"))
    parent_rule_id_leak = bool(_RULE_ID_PATTERN.search(parent_blob))

    technical_rows = app.build_technical_feature_rows(bundle)

    return {
        "image_id": case_id,
        "source_path": str(bundle["source_path"]),
        "objective_profile_categories": {cat: len(items) for cat, items in synthesis.objective_profile.items()},
        "unified_evidence_count": len(synthesis.unified_evidence),
        "technical_feature_row_count": len(technical_rows),
        "literature_linked_associations": synthesis.literature_linked_associations,
        "overall_synthesis": overall,
        "candidate_hypotheses": [
            {"concern_domain": h.concern_domain, "hypothesis_label": h.hypothesis_label,
             "support_level": h.support_level, "supporting_rule_ids": list(h.supporting_rule_ids),
             "evidence_families_represented": list(h.evidence_families_represented)}
            for h in synthesis.candidate_hypotheses
        ],
        "page_reference": {
            "page_reference_mode": page_reference.get("page_reference_mode"),
            "page_relative_features_assessable": page_reference.get("page_relative_features_assessable"),
            "confidence": page_reference.get("confidence"),
        },
        "parent_view": {
            "populated": parent_populated,
            "rule_id_leak": parent_rule_id_leak,
            "profile": parent_profile,
        },
    }


def to_csv_row(case_report: dict) -> dict:
    overall = case_report["overall_synthesis"]
    page_reference = case_report["page_reference"]
    return {
        "image_id": case_report["image_id"],
        "synthesis_level": overall["level"],
        "positive_evidence_families": "|".join(overall["positive_evidence_families"]),
        "positive_rule_ids": "|".join(overall["positive_rule_ids"]),
        "concern_evidence_families": "|".join(overall["concern_evidence_families"]),
        "concern_rule_ids": "|".join(overall["concern_rule_ids"]),
        "concern_domains_touched": "|".join(overall["domains_touched"]),
        "page_reference_mode": page_reference["page_reference_mode"],
        "page_relative_features_assessable": page_reference["page_relative_features_assessable"],
        "candidate_hypothesis_count": len(case_report["candidate_hypotheses"]),
        "parent_view_populated": case_report["parent_view"]["populated"],
        "parent_view_rule_id_leak": case_report["parent_view"]["rule_id_leak"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "Reproducible acceptance report for the unified drawing-synthesis pipeline -- "
        "cached development set only, zero live Gemini calls."))
    parser.add_argument("--image-id", default=None, help="Restrict to a single image_id.")
    parser.add_argument("--output-dir", default=None, help="Directory to write report.json/report.csv into.")
    args = parser.parse_args()

    case_ids = [args.image_id] if args.image_id else app.list_case_ids()
    reports = []
    for case_id in case_ids:
        report = build_case_report(case_id)
        if report is None:
            print(f"[skip] {case_id}: no cached Observer/Verifier data")
            continue
        reports.append(report)

    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for report in reports:
            writer.writerow(to_csv_row(report))

    print(f"\n{'image_id':12s} {'synthesis_level':32s} {'families(+/-)':14s} {'domains':10s} "
          f"{'page_ok':8s} {'hyps':5s} {'parent_ok':10s} {'rule_leak':10s}")
    for report in reports:
        overall = report["overall_synthesis"]
        page_reference = report["page_reference"]
        families = f"{len(overall['positive_evidence_families'])}/{len(overall['concern_evidence_families'])}"
        print(f"{report['image_id']:12s} {overall['level']:32s} {families:14s} "
              f"{len(overall['domains_touched']):<10d} "
              f"{str(page_reference['page_relative_features_assessable']):8s} "
              f"{len(report['candidate_hypotheses']):<5d} "
              f"{str(report['parent_view']['populated']):10s} "
              f"{str(report['parent_view']['rule_id_leak']):10s}")

    n_assessable = sum(1 for r in reports if r["page_reference"]["page_relative_features_assessable"])
    n_hypotheses = sum(len(r["candidate_hypotheses"]) for r in reports)
    n_parent_unpopulated = sum(1 for r in reports if not r["parent_view"]["populated"])
    n_rule_leaks = sum(1 for r in reports if r["parent_view"]["rule_id_leak"])
    print(f"\n{len(reports)} cases reported. Page-assessable: {n_assessable}/{len(reports)}. "
          f"Total candidate hypotheses: {n_hypotheses}. "
          f"Parent View unpopulated: {n_parent_unpopulated}. Parent View rule-ID leaks: {n_rule_leaks}.")
    print(f"\nWrote {out_dir / 'report.json'}")
    print(f"Wrote {out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
