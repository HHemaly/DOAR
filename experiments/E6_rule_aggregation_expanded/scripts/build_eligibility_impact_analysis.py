#!/usr/bin/env python
"""E6X eligibility-fix before-vs-after impact analysis.

Uses ONLY the 34 already-cached E6-EXPANDED cases -- NO API calls, NO new
perception. "BEFORE" is reconstructed by replicating exactly what the
UNFIXED `reasoning_chain.build_eligible_matches` /
`drawing_synthesis.build_deterministic_eligible_matches` did (promote
every `satisfied` precondition check regardless of `allowed_output_level`)
without touching or reverting the now-fixed production code. "AFTER" calls
the real, currently-fixed production functions directly. Both use the SAME
cached Observer/Verifier + deterministic-feature evidence -- only the
eligibility gate differs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation_expanded"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(EXP_DIR / "scripts"))

import run_development_benchmark as rdb  # noqa: E402
from doar import reasoning_chain as rc  # noqa: E402
from doar import drawing_synthesis as ds  # noqa: E402


def build_matches_unfiltered(entities, det):
    """Exact reconstruction of the UNFIXED (pre-eligibility-gate) match
    construction: every `satisfied` check (semantic + deterministic)
    promoted to an EligibleAtomicRuleMatch regardless of
    allowed_output_level -- byte-for-byte the same field-copy logic the
    real builders used before the gate was added, just without the
    `if row["allowed_output_level"] != ELIGIBLE_OUTPUT_LEVEL: continue`
    line. Never reverts or monkeypatches the real (now-fixed) production
    code -- this is an independent, additive reconstruction for
    comparison purposes only."""
    matrix = rc.load_rule_matrix()
    checks = list(rc.check_visual_preconditions(entities)) + list(
        ds.check_deterministic_preconditions(det["composition"], det["objective_features"], det.get("page_reference")))
    matches = []
    for check in checks:
        if check.status != "satisfied":
            continue
        row = matrix[check.rule_id]
        matches.append(rc.EligibleAtomicRuleMatch(
            rule_id=check.rule_id, evidence_family=row["evidence_family"], concern_domain=row["concern_domain"],
            allowed_output_level=row["allowed_output_level"], source_claim=row["source_claim"],
            possible_interpretation=row["possible_interpretation_as_written"],
            alternative_explanations=tuple(row["alternative_explanations"].split(" | ")) if row["alternative_explanations"] else (),
            matched_entity_ids=check.matched_entity_ids, evidence_direction=row["evidence_direction"],
            context_transfer_justification=row["context_transfer_justification"]))
    return matches


def build_matches_fixed(entities, det):
    """The real, currently-fixed production construction -- calls the
    actual (patched) functions directly, no reimplementation."""
    return list(rc.build_eligible_matches(entities)) + list(
        ds.build_deterministic_eligible_matches(det["composition"], det["objective_features"], det.get("page_reference")))


def main():
    manifest_path = EXP_DIR / "raw" / "cohort_manifest.csv"
    with open(manifest_path, encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    matrix = rc.load_rule_matrix()
    per_case_rows = []
    n_disabled_matches_removed_total = 0
    n_level_changed = 0
    n_hyp_changed = 0
    n_cases_affected = 0
    n_cases_total = 0

    for row in manifest_rows:
        if row["cohort_role"] == "locked_benchmark_reserved_do_not_process":
            continue
        image_id = row["image_id"]
        rows_cached, _ = rdb.find_saved_verification_rows(image_id)
        if rows_cached is None:
            continue
        n_cases_total += 1
        entities = rdb.entities_from_verification_rows(rows_cached)
        image_path = ROOT / row["relative_path"]
        det = ds.load_or_compute_deterministic_features(image_id, image_path)

        matches_before = build_matches_unfiltered(entities, det)
        matches_after = build_matches_fixed(entities, det)

        rule_ids_before = sorted({m.rule_id for m in matches_before})
        rule_ids_after = sorted({m.rule_id for m in matches_after})
        disabled_removed = sorted(set(rule_ids_before) - set(rule_ids_after))
        # Sanity: every removed rule_id really was disabled, nothing else changed.
        for rid in disabled_removed:
            assert matrix[rid]["allowed_output_level"] != rc.ELIGIBLE_OUTPUT_LEVEL, (
                f"{image_id}: {rid} removed by the fix but is not actually disabled -- investigate")
        assert set(rule_ids_after) <= set(rule_ids_before), f"{image_id}: fix somehow ADDED a match -- investigate"

        assoc_before = ds.build_literature_linked_associations(matches_before)
        assoc_after = ds.build_literature_linked_associations(matches_after)

        domains_before = sorted({m.concern_domain for m in matches_before})
        domains_after = sorted({m.concern_domain for m in matches_after})

        synthesis_before = ds.build_overall_synthesis(matches_before)
        synthesis_after = ds.build_overall_synthesis(matches_after)

        hyps_before = rc.build_candidate_hypotheses(matches_before)
        hyps_after = rc.build_candidate_hypotheses(matches_after)

        level_changed = synthesis_before["level"] != synthesis_after["level"]
        hyp_changed = len(hyps_before) != len(hyps_after) or (
            {h.support_level for h in hyps_before} != {h.support_level for h in hyps_after})
        case_affected = bool(disabled_removed)

        if case_affected:
            n_cases_affected += 1
        n_disabled_matches_removed_total += len(disabled_removed)
        if level_changed:
            n_level_changed += 1
        if hyp_changed:
            n_hyp_changed += 1

        per_case_rows.append({
            "case_id": image_id,
            "cohort_role": row["cohort_role"],
            "matched_rules_before": ";".join(rule_ids_before),
            "matched_rules_after": ";".join(rule_ids_after),
            "disabled_rules_removed": ";".join(disabled_removed),
            "n_matched_before": len(rule_ids_before),
            "n_matched_after": len(rule_ids_after),
            "n_disabled_removed": len(disabled_removed),
            "n_literature_associations_before": len(assoc_before),
            "n_literature_associations_after": len(assoc_after),
            "domains_touched_before": ";".join(domains_before),
            "domains_touched_after": ";".join(domains_after),
            "overall_synthesis_level_before": synthesis_before["level"],
            "overall_synthesis_level_after": synthesis_after["level"],
            "overall_synthesis_level_changed": level_changed,
            "n_candidate_hypotheses_before": len(hyps_before),
            "n_candidate_hypotheses_after": len(hyps_after),
            "candidate_hypotheses_changed": hyp_changed,
            "case_affected_by_fix": case_affected,
        })

    raw_out = EXP_DIR / "raw" / "eligibility_before_after_per_case.csv"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_case_rows)

    # E6X_T9: same data, in tables/, with a compact summary row appended.
    t9_out = EXP_DIR / "tables" / "E6X_T9_eligibility_fix_impact.csv"
    with open(t9_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_case_rows)

    summary = {
        "n_cases_total": n_cases_total,
        "n_cases_affected_by_fix": n_cases_affected,
        "pct_cases_affected": round(100 * n_cases_affected / n_cases_total, 1) if n_cases_total else 0.0,
        "n_disabled_matches_removed_total": n_disabled_matches_removed_total,
        "n_cases_overall_synthesis_level_changed": n_level_changed,
        "n_cases_candidate_hypotheses_changed": n_hyp_changed,
    }
    summary_out = EXP_DIR / "eligibility_fix" / "IMPACT_SUMMARY.csv"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Wrote {raw_out} ({len(per_case_rows)} cases)")
    print(f"Wrote {t9_out}")
    print(f"Wrote {summary_out}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return per_case_rows, summary


if __name__ == "__main__":
    main()
