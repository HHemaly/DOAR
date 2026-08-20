#!/usr/bin/env python
"""E6-EXPANDED -- corrected rule/evidence aggregation ablation.

Corrects 6 methodological issues found in E6-PILOT (experiments/
E6_rule_aggregation/, preserved unmodified -- see that directory's own
README for the provisional pilot result). See PROTOCOL.md for full
rationale of every fix; summarized here:

1. ELIGIBILITY GOVERNANCE DEFECT (HIGH PRIORITY, reported not silently
   fixed): the frozen production pipeline (`drawing_synthesis.
   synthesize_drawing` -> `build_overall_synthesis`) does NOT filter
   matched rules by `allowed_output_level` anywhere -- traced explicitly
   in PROTOCOL.md Section 1. Rules with `allowed_output_level=="disabled"`
   (31/41 rules) CAN and DO become EligibleAtomicRuleMatch objects,
   literature associations, and contribute to synthesis in production
   TODAY (confirmed: 13/24, 54%, of E6-PILOT's own matched associations
   were "disabled" rules). This script does NOT touch
   drawing_synthesis.py/reasoning_chain.py to fix this (production
   remains unmodified, defect remains present and unaddressed pending
   separate remediation) -- but E6's OWN re-aggregation layer applies the
   correct filter (`allowed_output_level == "individual_heuristic_only"`
   only) so the aggregation-STRATEGY comparison itself is not built on a
   scientifically contaminated evidence set. Both the raw (unfiltered)
   and filtered evidence are saved so the defect's exact impact remains
   auditable.
2. Contradiction redefinition: separates true literature contradictions
   (`literature_level_contradictions`), contextual/cultural limitations
   (`context_limited_by_edges`), precondition/requires limitations
   (`requires_edges`), and duplicate/same-family relationships
   (`same_evidence_family_edges`) -- E6-PILOT incorrectly conflated the
   second with the first.
3. Fair domain scope: R1/R2 now have a PRIMARY variant restricted to the
   same positive/concern-domain scope as R3/R4/R5 (isolating raw-count
   vs. family-dedup as the only variable), plus a clearly-labeled
   secondary "_broad" variant (includes neutral domains) as a stress-test
   baseline only, never the primary comparison.
4. Non-circular outcomes: "insufficient independent evidence" (a POLICY
   choice R4/R1/R5 differ on) is no longer counted as a "structural
   violation" -- only genuine, strategy-independent errors are (duplicate
   counting, cross-domain pooling, disabled-rule use, unavailable/page-
   dependent evidence use, unverified-evidence misuse, ignored TRUE
   contradiction). Policy differences are tracked separately in their own
   "policy_disagreement_vs_r4" column/table.
5. Page assessability: uses the real per-case
   `page_relative_features_assessable` value (already correctly fixed
   within E6-PILOT's own final run; carried forward unchanged here).
6. Figure seeds: fixed explicit integers, never `hash()`.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation_expanded"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_development_benchmark as rdb  # noqa: E402
from doar import reasoning_chain as rc  # noqa: E402
from doar import drawing_synthesis as ds  # noqa: E402

MIN_EVIDENCE = 2  # concerns.py's own frozen constant, reused verbatim, never redefined
R5_RATIO_THRESHOLD = 0.5

POSITIVE_DOMAINS = frozenset({"positive_affect_or_social_engagement"})
DISTRESS_DOMAINS = frozenset({
    "depressive_or_low_mood_related", "anxiety_or_stress_related", "aggression_or_threat_related",
    "social_withdrawal_related", "developmental_or_attention_related",
})
ASSESSABLE_STATUSES = {"satisfied", "not_satisfied"}
ELIGIBLE_OUTPUT_LEVEL = "individual_heuristic_only"  # E6's own corrected eligibility filter -- see module docstring

PRIMARY_STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]
ALL_STRATEGIES = ["R1", "R1_broad", "R2", "R2_broad", "R3", "R4", "R5"]


def direction_of(domain: str) -> str | None:
    if domain in POSITIVE_DOMAINS:
        return "positive"
    if domain in DISTRESS_DOMAINS:
        return "concern"
    return None


# ---------------------------------------------------------------------------
# 0. E6-specific bundle loader -- same computation load_case_bundle()
#    performs, but sourcing image metadata from THIS experiment's own
#    cohort_manifest.csv instead of requiring DEVELOPMENT_SET_15.json
#    membership (which stays frozen/untouched). Cache-only: never makes
#    a live call.
# ---------------------------------------------------------------------------


def load_bundle(image_id: str, relative_path: str) -> dict | None:
    rows, source_path = rdb.find_saved_verification_rows(image_id)
    if rows is None:
        return None
    entities = rdb.entities_from_verification_rows(rows)
    checks = rc.check_visual_preconditions(entities)
    image_path = ROOT / relative_path
    deterministic_features = ds.load_or_compute_deterministic_features(image_id, image_path)
    synthesis = ds.synthesize_drawing(image_id, entities, deterministic_features)
    return {"rows": rows, "source_path": source_path, "entities": entities, "checks": checks,
            "deterministic_features": deterministic_features, "synthesis": synthesis}


# ---------------------------------------------------------------------------
# 1. Load the frozen cohort manifest, build every case's bundle.
# ---------------------------------------------------------------------------


def load_cohort() -> dict:
    manifest_path = EXP_DIR / "raw" / "cohort_manifest.csv"
    with open(manifest_path, encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    matrix = rc.load_rule_matrix()
    graph = json.load(open(ROOT / "RULE_RELATIONSHIP_GRAPH.json", encoding="utf-8"))
    contextual_limitation_rule_ids = set()
    for edge in graph["context_limited_by_edges"]:
        contextual_limitation_rule_ids.update(edge["rule_ids"])
    true_contradiction_rule_ids = set()  # from literature_level_contradictions -- none reference rule_ids
    # directly today (only EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038 is named, in the note text, and it is
    # disabled) -- resolved by name match against the one recorded topic's note, not invented.
    for topic in graph["literature_level_contradictions"]:
        if "EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038" in topic.get("note", ""):
            true_contradiction_rule_ids.add("EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038")

    cases = {}
    for row in manifest_rows:
        if row["cohort_role"] == "locked_benchmark_reserved_do_not_process":
            continue  # NEVER processed, per the locked-benchmark reservation
        if row["has_cached_evidence"] != "True":
            cases[row["image_id"]] = None  # pending live perception -- explicitly recorded as excluded, not silently dropped
            continue
        bundle = load_bundle(row["image_id"], row["relative_path"])
        if bundle is None or bundle["synthesis"] is None:
            cases[row["image_id"]] = None
            continue

        entities_by_id = {e.entity_id: e for e in (bundle["entities"] or [])}
        det = bundle["deterministic_features"]
        det_checks = ds.check_deterministic_preconditions(det["composition"], det["objective_features"],
                                                            det.get("page_reference"))
        semantic_status = {c.rule_id: c.status for c in (bundle["checks"] or [])}
        det_status = {c.rule_id: c.status for c in det_checks}
        all_status = {**semantic_status, **det_status}
        page_ref = det.get("page_reference") or {}

        cases[row["image_id"]] = {
            "assocs_all": bundle["synthesis"].literature_linked_associations,  # UNFILTERED (includes disabled rules)
            "all_status": all_status, "matrix": matrix, "entities_by_id": entities_by_id,
            "contextual_limitation_rule_ids": contextual_limitation_rule_ids,
            "true_contradiction_rule_ids": true_contradiction_rule_ids,
            "page_assessable": bool(page_ref.get("page_relative_features_assessable")),
            "cohort_role": row["cohort_role"],
        }
    return cases


# ---------------------------------------------------------------------------
# 2. Raw atomic evidence -- BOTH filtered and unfiltered rows saved, so the
#    eligibility-governance defect's exact impact stays auditable.
# ---------------------------------------------------------------------------


def build_atomic_evidence_rows(cases: dict) -> list[dict]:
    rows = []
    for cid, data in cases.items():
        if data is None:
            continue
        for a in data["assocs_all"]:
            matched_entity_ids = a["matched_entity_ids"]
            statuses = [data["entities_by_id"][eid].case_verification_status
                        for eid in matched_entity_ids if eid in data["entities_by_id"]]
            verification_status = ("deterministic_measurement" if not statuses
                                    else ("verified" if all(s == "verified" for s in statuses)
                                          else "/".join(sorted(set(statuses)))))
            rows.append({
                "case_id": cid, "rule_id": a["rule_id"], "evidence_family": a["evidence_family"],
                "concern_domain": a["concern_domain"], "direction_bucket": direction_of(a["concern_domain"]) or "neutral",
                "evidence_direction": a["evidence_direction"], "allowed_output_level": a["allowed_output_level"],
                "eligible_for_e6": a["allowed_output_level"] == ELIGIBLE_OUTPUT_LEVEL,
                "matched_entity_ids": ";".join(matched_entity_ids), "verification_status": verification_status,
                "is_true_contradiction": a["rule_id"] in data["true_contradiction_rule_ids"],
                "is_contextual_limitation": a["rule_id"] in data["contextual_limitation_rule_ids"],
            })
    return rows


# ---------------------------------------------------------------------------
# 3. Per (case, domain, strategy) aggregation.
# ---------------------------------------------------------------------------


def _assessable_families_for_domain(all_status: dict, matrix: dict, domain: str) -> set[str]:
    return {row["evidence_family"] for rule_id, row in matrix.items()
            if row["concern_domain"] == domain and all_status.get(rule_id) in ASSESSABLE_STATUSES}


def build_case_domain_rows(cases: dict) -> list[dict]:
    rows = []
    for cid, data in cases.items():
        if data is None:
            continue
        matrix = data["matrix"]
        all_status = data["all_status"]
        entities_by_id = data["entities_by_id"]

        # E6's own corrected eligibility filter, applied here (not in the frozen pipeline).
        assocs_eligible = [a for a in data["assocs_all"] if a["allowed_output_level"] == ELIGIBLE_OUTPUT_LEVEL]
        assocs_broad = data["assocs_all"]  # unfiltered -- ONLY used for the _broad stress-test variants' evidence

        by_domain_eligible: dict[str, list[dict]] = {}
        for a in assocs_eligible:
            by_domain_eligible.setdefault(a["concern_domain"], []).append(a)
        by_domain_broad: dict[str, list[dict]] = {}
        for a in assocs_broad:
            by_domain_broad.setdefault(a["concern_domain"], []).append(a)

        pooled_families = {"positive": set(), "concern": set()}
        pooled_domains_contributing = {"positive": set(), "concern": set()}
        for a in assocs_eligible:
            direction = direction_of(a["concern_domain"])
            if direction is None:
                continue
            pooled_families[direction].add(a["evidence_family"])
            pooled_domains_contributing[direction].add(a["concern_domain"])
        pooled_triggered = {d: len(pooled_families[d]) >= MIN_EVIDENCE for d in ("positive", "concern")}

        all_domains_touched = set(by_domain_eligible) | set(by_domain_broad)
        for domain in all_domains_touched:
            direction = direction_of(domain)
            elig_matches = by_domain_eligible.get(domain, [])
            broad_matches = by_domain_broad.get(domain, [])

            rule_ids_elig = sorted({m["rule_id"] for m in elig_matches})
            fam_counts_elig = Counter(m["evidence_family"] for m in elig_matches)
            unique_families_elig = sorted(fam_counts_elig)
            duplicate_family_rule_count = sum(c - 1 for c in fam_counts_elig.values() if c > 1)

            rule_ids_broad = sorted({m["rule_id"] for m in broad_matches})
            fam_counts_broad = Counter(m["evidence_family"] for m in broad_matches)

            assessable_families = _assessable_families_for_domain(all_status, matrix, domain)
            support_ratio = (len(unique_families_elig) / len(assessable_families)) if assessable_families else None

            true_contradictions = sorted({m["rule_id"] for m in elig_matches} & data["true_contradiction_rule_ids"])
            contextual_limitations = sorted({m["rule_id"] for m in elig_matches} & data["contextual_limitation_rule_ids"])
            disabled_rules_present = sorted({m["rule_id"] for m in broad_matches} - set(rule_ids_elig))

            unverified_used = any(
                entities_by_id.get(eid) is not None and entities_by_id[eid].case_verification_status != "verified"
                for m in elig_matches for eid in m["matched_entity_ids"])

            base_common = {
                "case_id": cid, "concern_domain": domain, "direction_bucket": direction or "neutral",
                "eligible_rule_ids": ";".join(rule_ids_elig), "matched_rule_count": len(rule_ids_elig),
                "unique_evidence_families": ";".join(unique_families_elig), "unique_family_count": len(unique_families_elig),
                "assessable_family_count": len(assessable_families), "support_ratio": support_ratio,
                "verification_status": "uses_unverified_evidence" if unverified_used else "all_verified_or_deterministic",
                "true_contradictions": ";".join(true_contradictions), "has_true_contradiction": bool(true_contradictions),
                "contextual_limitations": ";".join(contextual_limitations),
                "has_contextual_limitation": bool(contextual_limitations),
                "disabled_rules_present_but_excluded": ";".join(disabled_rules_present),
                "page_assessability": data["page_assessable"],
            }

            for strategy in ALL_STRATEGIES:
                row = dict(base_common)
                row["strategy"] = strategy
                disabled_rule_used = False  # by construction, always False for every non-_broad strategy

                if strategy == "R1":  # primary: any eligible rule, concern-domain scope only
                    triggered = direction is not None and len(rule_ids_elig) >= 1
                    duplicate_inflation = False
                    cross_domain = False
                elif strategy == "R1_broad":  # secondary stress-test: any eligible rule, ALL domains, UNFILTERED evidence
                    triggered = len(rule_ids_broad) >= 1
                    duplicate_inflation = False
                    cross_domain = False
                    disabled_rule_used = triggered and bool(disabled_rules_present)
                elif strategy == "R2":  # primary: raw rule count, no family dedup, concern-domain scope only
                    triggered = direction is not None and len(rule_ids_elig) >= MIN_EVIDENCE
                    duplicate_inflation = triggered and duplicate_family_rule_count > 0
                    cross_domain = False
                elif strategy == "R2_broad":  # secondary stress-test: raw count, ALL domains, UNFILTERED evidence
                    triggered = len(rule_ids_broad) >= MIN_EVIDENCE
                    dup_broad = sum(c - 1 for c in fam_counts_broad.values() if c > 1)
                    duplicate_inflation = triggered and dup_broad > 0
                    cross_domain = False
                    disabled_rule_used = triggered and bool(disabled_rules_present)
                elif strategy == "R3":  # pooled cross-domain family convergence
                    triggered = direction is not None and pooled_triggered[direction]
                    contributing = pooled_domains_contributing[direction] if direction else set()
                    cross_domain = triggered and len(contributing) > 1
                    duplicate_inflation = False
                elif strategy == "R4":  # current DOAR: same-domain family convergence
                    triggered = direction is not None and len(unique_families_elig) >= MIN_EVIDENCE
                    duplicate_inflation = False
                    cross_domain = False
                elif strategy == "R5":  # support ratio, same domain scope as R4
                    triggered = (direction is not None and support_ratio is not None
                                 and support_ratio >= R5_RATIO_THRESHOLD)
                    duplicate_inflation = False
                    cross_domain = False
                else:  # pragma: no cover
                    raise ValueError(strategy)

                # OBJECTIVE STRUCTURAL FAILURES ONLY (Section B.4) -- never includes
                # "insufficient independent evidence", which is a POLICY difference,
                # tracked separately below.
                unavailable_page_dependent_evidence = False  # structurally impossible -- see PROTOCOL.md
                ineligible_evidence_used = disabled_rule_used
                inappropriate_unverified_use = triggered and unverified_used
                ignored_true_contradiction = triggered and bool(true_contradictions)
                structural_violation = any([
                    duplicate_inflation, cross_domain, ineligible_evidence_used,
                    unavailable_page_dependent_evidence, inappropriate_unverified_use, ignored_true_contradiction,
                ])

                # POLICY DIFFERENCE (not an error) -- how this strategy's own trigger
                # decision compares to R4's >=2-independent-family, same-domain rule.
                meets_r4_family_floor = len(unique_families_elig) >= MIN_EVIDENCE
                policy_disagreement_vs_r4 = triggered != (
                    direction is not None and meets_r4_family_floor)

                row.update({
                    "triggered": triggered, "abstained": not triggered,
                    "duplicate_inflation": duplicate_inflation, "cross_domain_convergence": cross_domain,
                    "ineligible_evidence_used": ineligible_evidence_used,
                    "unavailable_page_dependent_evidence": unavailable_page_dependent_evidence,
                    "inappropriate_unverified_use": inappropriate_unverified_use,
                    "ignored_true_contradiction": ignored_true_contradiction,
                    "structural_violation": structural_violation,
                    "meets_r4_family_floor": meets_r4_family_floor,
                    "policy_disagreement_vs_r4": policy_disagreement_vs_r4,
                    "candidate_interpretation": (f"{domain}: {len(unique_families_elig)} family(ies) via {strategy}"
                                                  if triggered else None),
                })
                rows.append(row)
    return rows


def main() -> None:
    cases = load_cohort()
    n_total = len(cases)
    n_usable = sum(1 for v in cases.values() if v is not None)
    print(f"Cohort manifest: {n_total} non-locked-benchmark cases; {n_usable} usable (have cached synthesis).")

    # Case-level summary -- covers EVERY usable case, including the ones with
    # zero matched associations (which never appear in aggregation_per_domain.csv),
    # so page_assessability/eligible-rule-count are always recoverable per case.
    case_summary_rows = []
    for cid, data in cases.items():
        if data is None:
            continue
        n_eligible = sum(1 for a in data["assocs_all"] if a["allowed_output_level"] == ELIGIBLE_OUTPUT_LEVEL)
        n_disabled = sum(1 for a in data["assocs_all"] if a["allowed_output_level"] != ELIGIBLE_OUTPUT_LEVEL)
        case_summary_rows.append({
            "case_id": cid, "cohort_role": data["cohort_role"], "page_assessable": data["page_assessable"],
            "n_eligible_associations": n_eligible, "n_disabled_associations_excluded": n_disabled,
        })
    pd.DataFrame(case_summary_rows).sort_values("case_id").to_csv(
        EXP_DIR / "raw" / "case_level_summary.csv", index=False)
    print(f"Wrote case_level_summary.csv ({len(case_summary_rows)} rows)")

    atomic_rows = build_atomic_evidence_rows(cases)
    raw_dir = EXP_DIR / "raw"
    pd.DataFrame(atomic_rows).to_csv(raw_dir / "input_atomic_evidence.csv", index=False)
    print(f"Wrote input_atomic_evidence.csv ({len(atomic_rows)} rows)")

    case_domain_rows = build_case_domain_rows(cases)
    pd.DataFrame(case_domain_rows).to_csv(raw_dir / "aggregation_per_domain.csv", index=False)
    print(f"Wrote aggregation_per_domain.csv ({len(case_domain_rows)} rows)")

    df = pd.DataFrame(case_domain_rows)
    per_case_rows = []
    usable_ids = sorted([cid for cid, v in cases.items() if v is not None])
    for cid in usable_ids:
        case_df = df[df["case_id"] == cid]
        for strategy in ALL_STRATEGIES:
            sdf = case_df[case_df["strategy"] == strategy]
            trig = sdf[sdf["triggered"]]
            per_case_rows.append({
                "case_id": cid, "strategy": strategy, "any_domain_triggered": bool(len(trig) > 0),
                "n_domains_triggered": int(len(trig)), "triggered_domains": ";".join(sorted(trig["concern_domain"])),
                "any_structural_violation": bool(sdf["structural_violation"].any()),
                "any_duplicate_inflation": bool(sdf["duplicate_inflation"].any()),
                "any_cross_domain_convergence": bool(sdf["cross_domain_convergence"].any()),
                "any_ineligible_evidence_used": bool(sdf["ineligible_evidence_used"].any()),
                "any_ignored_true_contradiction": bool(sdf["ignored_true_contradiction"].any()),
                "any_policy_disagreement_vs_r4": bool((sdf["triggered"] & sdf["policy_disagreement_vs_r4"]).any()),
                "total_domains_considered": int(len(sdf)),
            })
    # NOTE: cases with zero matched associations at all (never appear in
    # aggregation_per_domain.csv) are ALREADY correctly represented above --
    # for such a case, `case_df`/`sdf` are simply empty DataFrames, and every
    # `.any()`/`len()` on them naturally evaluates to False/0, so the loop
    # above already emits the correct all-False row for every strategy. An
    # earlier version of this script ALSO appended a second explicit
    # "zero-association" block for these cases, which silently double-
    # counted them (a genuine bug, caught by pivot() raising on duplicate
    # index entries when building E6X_T3) -- removed, not worked around.

    per_case_df = pd.DataFrame(per_case_rows).sort_values(["case_id", "strategy"])
    per_case_df.to_csv(raw_dir / "aggregation_per_case.csv", index=False)
    print(f"Wrote aggregation_per_case.csv ({len(per_case_df)} rows, {per_case_df['case_id'].nunique()} cases)")
    print("\nDone with extraction.")


if __name__ == "__main__":
    main()
